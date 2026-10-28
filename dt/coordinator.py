"""CLI orchestrator + state twin. Runs in the central container/EC2."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import time
import uuid
import httpx
import torch
from dt.codec import pack, unpack, sha
from dt.planner import plans
from src.fl.fedavg_core import aggregate


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def validate_pool(case, headers):
    for c in case['clients']:
        r = httpx.get(c['url']+'/state', headers=headers, timeout=15, trust_env=False)
        r.raise_for_status()
        live = r.json()
        if any(live[k] != c[k] for k in ['id','n','hist','data_version']) or live['busy']:
            raise ValueError(f'Worker drift/busy: {c["id"]}')


def one_round(weights, case, ids, run_id, round_id, headers):
    started = time.perf_counter()
    base = sha(pack(weights, {}))
    byid = {c['id']: c for c in case['clients']}
    def call(cid):
        c = byid[cid]
        meta = {'job_id': f'{run_id}_r{round_id}_c{cid}', 'base_hash': base,
                'data_version': c['data_version'], 'input_dim': case['input_dim'],
                'hidden': case['hidden'], 'fl': case['fl'],
                'seed': case['seed']+round_id*10000+cid*100}
        request = pack(weights, meta)
        t = time.perf_counter()
        with httpx.Client(timeout=300, trust_env=False) as client:
            response = client.post(c['url']+'/train', content=request, headers=headers)
            response.raise_for_status()
        update, log = unpack(response.content)
        log.update({'elapsed_seconds': time.perf_counter()-t,
                    'body_bytes': len(request)+len(response.content), 'received_at': time.time()})
        if (log['base_hash'] != base or log['job_id'] != meta['job_id'] or
            log['data_version'] != c['data_version'] or log['n'] != c['n'] or log['hist'] != c['hist']):
            raise ValueError('Stale update or wrong retained counts')
        if set(update) != set(weights) or any(update[k].shape != weights[k].shape or
            update[k].dtype != weights[k].dtype or not torch.isfinite(update[k]).all() for k in weights):
            raise ValueError('Invalid model schema or nonfinite update')
        return update, log
    # Barrier: failed participant aborts the round. Never silently average a partial set.
    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        futures = [pool.submit(call, cid) for cid in ids]
        replies = [future.result() for future in futures]
    t = time.perf_counter()
    result = aggregate([u for u,_ in replies], [log['n'] for _,log in replies])
    agg_seconds = time.perf_counter()-t
    elapsed = time.perf_counter()-started
    logs = [log for _, log in replies]
    return result, {'round': round_id, 'clients': logs, 'wall_seconds': elapsed,
                    'agg_seconds': agg_seconds,
                    'round_overhead_seconds': max(0.,elapsed-max(x['elapsed_seconds'] for x in logs))}


def feedback(twin, log, alpha=.3):
    for observed in log['clients']:
        c = dict(observed)
        key = str(c['client_id'])
        old = twin['clients'].get(key)
        if old:
            c['elapsed_seconds'] = (1-alpha)*old['elapsed_seconds']+alpha*c['elapsed_seconds']
        twin['clients'][key] = dict(c)
    twin['round_overhead_seconds'] = (1-alpha)*twin.get('round_overhead_seconds', 0.)+alpha*log['round_overhead_seconds']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['calibrate','run'])
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--twin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--budget-seconds', type=float, default=60)
    p.add_argument('--budget-bytes', type=int, default=100000000)
    p.add_argument('--policy', choices=['dt_adaptive','dt_static','all','target','random_k','fastest_k','largest_k'], default='dt_adaptive')
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--max-js', type=float, default=.05)
    p.add_argument('--min-coverage', type=float, default=.2)
    p.add_argument('--margin', type=float, default=1.25)
    p.add_argument('--ttl', type=float, default=600)
    p.add_argument('--overhead-seconds', type=float, default=.5)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    case = json.loads(args.case.read_text())
    weights, _ = unpack(args.initial.read_bytes())
    headers = {'X-DT-Token': os.environ['DT_TOKEN']}
    run_id = uuid.uuid4().hex
    report = {'run_id': run_id, 'mode': args.mode, 'policy': args.policy,
              'budget_scope': 'recovery_only_plus_control_overhead; FU stage excluded',
              'initial_hash': sha(args.initial.read_bytes()), 'case_hash': sha(args.case.read_bytes()),
              'logs': [], 'status': 'RUNNING', 'research_claim': 'infrastructure_only',
              'settings': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}
    save(args.output/'report.json', report)
    try:
        validate_pool(case, headers)
        if args.mode == 'calibrate':
            if args.twin.exists():
                raise FileExistsError('Use a fresh twin calibration path')
            twin = {'clients': {}, 'round_overhead_seconds': 0., 'case_hash': report['case_hash']}
            for r in range(3):
                _, log = one_round(weights, case, [c['id'] for c in case['clients']], run_id, r, headers)
                report['logs'].append(log)
                feedback(twin, log)
            save(args.twin, twin)
            report['status'] = 'CALIBRATED'
        else:
            twin = json.loads(args.twin.read_text())
            if twin['case_hash'] != report['case_hash']:
                raise ValueError('Case/worker environment changed: recalibrate')
            snapshot = json.loads(json.dumps(twin))
            save(args.output/'twin_before.json', snapshot)
            t = time.perf_counter()
            candidates = plans(case, twin, args.budget_seconds, args.budget_bytes,
                               args.max_js, args.min_coverage, args.margin, args.overhead_seconds, args.ttl)
            if args.policy.startswith('dt_'):
                plan = candidates[0] if candidates else None
            else:
                target = case['target']
                rest = [c['id'] for c in case['clients'] if c['id'] != target]
                if args.policy == 'all': ids = [c['id'] for c in case['clients']]
                elif args.policy == 'target': ids = [target]
                else:
                    if not 1 <= args.k <= len(case['clients']): raise ValueError('Invalid k')
                    if args.policy == 'random_k': random.Random(case['seed']).shuffle(rest)
                    elif args.policy == 'fastest_k': rest.sort(key=lambda i: twin['clients'][str(i)]['elapsed_seconds'])
                    else: rest.sort(key=lambda i: -next(c['n'] for c in case['clients'] if c['id']==i))
                    ids = [target]+rest[:args.k-1]
                # Baselines deliberately execute even when budget/representation constraints fail.
                item = next((v for v in candidates if set(v['clients'])==set(ids)), None)
                plan = item or {'clients': sorted(ids), 'predicted_feasible': False}
            report['planning_seconds'] = time.perf_counter()-t
            report['plan'] = plan
            if plan is None:
                report['status'] = 'NO_FEASIBLE_PLAN'
            else:
                for r in range(case['rounds']):
                    weights, log = one_round(weights, case, plan['clients'], run_id, r, headers)
                    report['logs'].append(log)
                    if args.policy == 'dt_adaptive': feedback(twin, log)
                    save(args.output/'report.json', report)
                (args.output/'recovered.npz').write_bytes(pack(weights, {'case_hash': report['case_hash']}))
                save(args.output/'twin_after.json', twin)
                if args.policy == 'dt_adaptive': save(args.twin, twin)
                report['status'] = 'COMPLETED'
    except Exception as e:
        report['status'] = 'FAILED'
        report['error'] = f'{type(e).__name__}: {e}'
        raise
    finally:
        report['actual_seconds'] = time.perf_counter()-started
        report['actual_body_bytes'] = sum(c['body_bytes'] for r in report['logs'] for c in r['clients'])
        report['bytes_scope'] = 'HTTP model request/response bodies only; excludes headers/TCP/TLS/state probes'
        report['budget_violated'] = (report['actual_seconds'] > args.budget_seconds or report['actual_body_bytes'] > args.budget_bytes) if args.mode=='run' and report['status']=='COMPLETED' else None
        save(args.output/'report.json', report)
        print(json.dumps({k:report[k] for k in ['status','actual_seconds','actual_body_bytes','budget_violated']}))


if __name__ == '__main__':
    main()
