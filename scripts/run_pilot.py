"""Run the no-DT research pilot. A completed run can still fail the FU gate."""
import argparse
import copy
import json
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.data.prepare import digest, write_json
from src.data.federated import partition, forget_request, retained_partition, ids_hash
from src.fl.fedavg_core import ClientWorker, model, state, fedavg, predict
from src.fu.conpfu_inspired import alternating, distill
from src.eval.metrics import utility, similarity, calibrate_mia, matched_indices, mia


def load_data(path):
    audit = json.loads((path/'audit.json').read_text())
    if not audit['training_ready']:
        raise ValueError('Preprocessing gate is closed')
    if 'cross_split_encoded_overlap' not in audit or any(audit['cross_split_encoded_overlap'].values()):
        raise ValueError('Missing/failed encoded feature leakage audit; regenerate data')
    manifest = json.loads((path/'manifest.json').read_text())
    for name, expected in manifest['artifacts_sha256'].items():
        if digest(path/name) != expected:
            raise ValueError(f'Data artifact hash mismatch: {name}')
    data = {}
    for split in ['train', 'validation', 'calibration', 'test']:
        with np.load(path/f'{split}.npz', allow_pickle=False) as arrays:
            data[split] = dict(arrays)
    for a in data:
        for b in data:
            if a < b and (set(data[a]['row_id']) & set(data[b]['row_id']) or set(data[a]['group_id']) & set(data[b]['group_id'])):
                raise ValueError('Overlapping split IDs/groups')
    return data


def run(data_path, config_path, output, smoke=False):
    if output.exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    config = json.loads(config_path.read_text())
    if smoke:
        config['fl']['rounds'] = 1
        config['recovery_rounds'] = 1
        config['fractions'] = [.1]
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    data = load_data(data_path)
    train = data['train']
    if smoke:
        # Explicit smoke-only stratified cap, never used as full-data experimental evidence.
        indices = np.concatenate([np.flatnonzero(train['y'] == c)[:1000] for c in [0,1]])
        train = {k: v[indices] for k,v in train.items()}
    clients = partition(train['y'], config['seed'], config['alpha'], config['clients'])
    target = int(np.argmax([len(idx) for idx in clients]))
    requests = {}
    for fraction in config['fractions']:
        f, r = forget_request(train['y'], clients[target], fraction, config['seed']+1)
        requests[str(fraction)] = (f, r, retained_partition(clients, target, f))
    output.mkdir(parents=True)
    (output/'checkpoints').mkdir()
    started = time.perf_counter()
    manifest = {'config': config, 'smoke_only': smoke, 'data_manifest_sha256': digest(data_path/'manifest.json'),
                'source_hashes': {str(p): digest(p) for root in ['src','scripts'] for p in Path(root).rglob('*.py')},
                'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                'platform': platform.platform(), 'processor': platform.processor(), 'device': 'cpu',
                'timing': 'time.perf_counter; sequential logical workers', 'code_revision': None,
                'target_client': target, 'algorithm': 'conpfu_inspired',
                'planner': 'all_clients_fixed', 'threshold': .5,
                'transform_scope': 'fixed_original_train_transform_conditional_retraining',
                'optimizer_policy': 'Adam_reset_each_client_round; fresh_retrain_same_initial_seed',
                'mia': 'fixed negative-CE score; class-matched samples; thresholds from original on calibration only; strict >',
                'clients': [{'client_id': i, 'count': len(idx), 'labels': np.bincount(train['y'][idx], minlength=2).tolist(),
                             'row_ids_hash': ids_hash(train['row_id'][idx]), 'groups': len(set(train['group_id'][idx]))}
                            for i,idx in enumerate(clients)]}
    np.savez_compressed(output/'partitions.npz', **{f'client_{i}': train['row_id'][idx] for i,idx in enumerate(clients)})
    for name, (f,r,retained) in requests.items():
        write_json(output/f'request_{name}.json', {'fraction': float(name), 'target': target,
                   'forget_row_ids': train['row_id'][f].tolist(), 'retain_row_ids': train['row_id'][r].tolist(),
                   'forget_hash': ids_hash(train['row_id'][f]), 'retain_hash': ids_hash(train['row_id'][r]),
                   'forget_labels': np.bincount(train['y'][f], minlength=2).tolist(),
                   'forget_groups': sorted(set(train['group_id'][f])),
                   'rounding': 'floor_per_class; nested class permutation'})
    manifest['input_hashes'] = {p.name: digest(p) for p in output.glob('*.json')}
    manifest['input_hashes']['partitions.npz'] = digest(output/'partitions.npz')
    write_json(output/'manifest.json', manifest)
    coverage, flat, metrics, timing = [], [], {}, {}
    def workers(parts):
        return [ClientWorker(i, train['X'][idx], train['y'][idx], train['row_id'][idx]) for i,idx in enumerate(parts)]
    def save(name, m):
        torch.save({'state_dict': state(m), 'input_dim': train['X'].shape[1], 'hidden': config['hidden'],
                    'manifest_sha256': digest(output/'manifest.json')}, output/'checkpoints'/f'{name}.pt')
    try:
        initial = model(train['X'].shape[1], config['hidden'], config['seed'])
        save('initial', initial)
        t = time.perf_counter()
        original, logs = fedavg(initial, workers(clients), config['fl'], config['fl']['rounds'], config['seed'])
        timing['original_seconds'] = time.perf_counter()-t
        write_json(output/'original_training.json', logs)
        save('original', original)
        print('Original trained; evaluating fixed validation protocol', flush=True)
        original_prediction = {s: predict(original, d['X']) for s,d in data.items() if s != 'train'}
        original_train = predict(original, train['X'])
        thresholds = calibrate_mia(data['calibration']['y'], original_prediction['calibration'])
        write_json(output/'mia_protocol.json', {'thresholds': thresholds, 'calibration_hash': ids_hash(data['calibration']['row_id']),
                   'matching': 'exact binary class balance; group covariates unmatched; distribution shift is a confound',
                   'attack_training': 'none; fixed loss score, calibration only sets operating thresholds'})
        for name,(f,r,retained) in requests.items():
            print(f'Request {name}: {len(f)} forgotten records', flush=True)
            case_start = time.perf_counter()
            retained_ids = np.concatenate(retained)
            if np.intersect1d(retained_ids, f).size:
                raise AssertionError('Forgotten samples in recovery')
            rw = workers(retained)
            fresh = model(train['X'].shape[1], config['hidden'], config['seed'])
            assert all(torch.equal(state(initial)[k], state(fresh)[k]) for k in state(initial))
            t=time.perf_counter()
            retrain, log = fedavg(fresh, rw, config['fl'], config['fl']['rounds'], config['seed'])
            timing[f'{name}_retrain_seconds'] = time.perf_counter()-t
            write_json(output/f'{name}_retrain_training.json', log)
            models = {'original': original, 'no_op': copy.deepcopy(original), 'retrain': retrain}
            t=time.perf_counter()
            models['retain_only'], log = fedavg(original, rw, config['fl'], config['recovery_rounds'], config['seed']+900000)
            timing[f'{name}_retain_only_seconds'] = time.perf_counter()-t
            write_json(output/f'{name}_retain_only_recovery.json',log)
            fw = ClientWorker(target, train['X'][f],train['y'][f],train['row_id'][f])
            t=time.perf_counter()
            local, log = alternating(original, fw, rw[target], config['fu'], config['seed'])
            timing[f'{name}_alternating_seconds']=time.perf_counter()-t
            models['after_local']=local
            write_json(output/f'{name}_local_losses.json',log)
            t=time.perf_counter()
            kd, losses = distill(original, local, rw[target], config['fu'], config['seed'])
            timing[f'{name}_kd_seconds']=time.perf_counter()-t
            models['after_kd']=kd
            write_json(output/f'{name}_kd_losses.json',losses)
            for method, start_model in [('fu_no_kd',local),('fu_retain_kd',kd)]:
                t=time.perf_counter()
                models[method],log=fedavg(start_model,rw,config['fl'],config['recovery_rounds'],config['seed']+900000)
                timing[f'{name}_{method}_recovery_seconds']=time.perf_counter()-t
                write_json(output/f'{name}_{method}_recovery.json',log)
            for method,m in models.items(): save(f'{name}_{method}',m)
            # Retained controls exclude every component represented in F.
            control_pool = retained_ids[~np.isin(train['group_id'][retained_ids],train['group_id'][f])]
            fm,fn = matched_indices(train['y'][f], data['test']['y'],config['seed'])
            rm,rn = matched_indices(train['y'][control_pool],data['test']['y'],config['seed']+2)
            np.savez_compressed(output/f'{name}_evaluation_ids.npz', forget=train['row_id'][f[fm]],
                                forget_nonmember=data['test']['row_id'][fn], retained=train['row_id'][control_pool[rm]],
                                retained_nonmember=data['test']['row_id'][rn])
            predictions = {method: {'train': predict(m,train['X']), 'test': predict(m,data['test']['X']),
                                   'validation': predict(m,data['validation']['X'])} for method,m in models.items()}
            if not np.array_equal(predictions['original']['train'], predictions['no_op']['train']):
                raise AssertionError('No-op changed predictions')
            for method,p in predictions.items():
                ref=predictions['retrain']
                result={'utility_test':utility(data['test']['y'],p['test']),
                        'utility_validation':utility(data['validation']['y'],p['validation']),
                        'similarity_forget':similarity(p['train'][f],ref['train'][f]),
                        'similarity_retain':similarity(p['train'][retained_ids],ref['train'][retained_ids]),
                        'similarity_test':similarity(p['test'],ref['test']),
                        'mia_forget':mia(train['y'][f[fm]],p['train'][f[fm]],data['test']['y'][fn],p['test'][fn],thresholds),
                        'mia_retained_control':mia(train['y'][control_pool[rm]],p['train'][control_pool[rm]],data['test']['y'][rn],p['test'][rn],thresholds)}
                metrics[f'{name}/{method}']=result
                flat.append({'fraction':float(name),'method':method,'macro_f1':result['utility_test']['macro_f1'],
                             'pr_auc_ap':result['utility_test']['pr_auc_ap'],'fpr':result['utility_test']['fpr'],
                             'prediction_mae_F':result['similarity_forget']['prediction_mae'],
                             'prediction_mae_test':result['similarity_test']['prediction_mae'],
                             'mia_auc_F':result['mia_forget']['mia_auc'],
                             'mia_auc_R':result['mia_retained_control']['mia_auc']})
            np.savez_compressed(output/f'{name}_predictions.npz', **{f'{m}_{s}':p for m,v in predictions.items() for s,p in v.items()})
            for method in models:
                coverage.append({'fraction':float(name),'method':method,'status':'completed'})
            timing[f'{name}_case_seconds']=time.perf_counter()-case_start
            write_json(output/'metrics.json',metrics)
            pd.DataFrame(flat).to_csv(output/'metrics.csv',index=False)
        gate=config['gate']; decisions=[]
        for name in requests:
            base=metrics[f'{name}/original']; ret=metrics[f'{name}/retrain']
            for method in ['fu_no_kd','fu_retain_kd']:
                candidate=metrics[f'{name}/{method}']
                separation=base['similarity_forget']['prediction_mae']
                improvement=separation-candidate['similarity_forget']['prediction_mae']
                checks={'identifiable':separation>=gate['minimum_original_retrain_mae'],
                        'closer_to_retrain':improvement>0,
                        'utility_floor':candidate['utility_test']['macro_f1']>=gate['minimum_macro_f1'],
                        'utility_preserved':candidate['utility_test']['macro_f1']>=max(base['utility_test']['macro_f1'],ret['utility_test']['macro_f1'])-gate['maximum_f1_drop'],
                        'mia_positive_control':base['mia_retained_control']['mia_auc']>=gate['minimum_retained_mia_auc'],
                        'mia_closer_to_retrain':abs(candidate['mia_forget']['mia_auc']-ret['mia_forget']['mia_auc'])<=abs(base['mia_forget']['mia_auc']-ret['mia_forget']['mia_auc'])}
                decisions.append({'fraction':float(name),'method':method,'checks':checks,'signed_mae_improvement':improvement,'passed':all(checks.values()) and not smoke})
        passed=not smoke and any(all(d['passed'] for d in decisions if d['method']==m) for m in ['fu_no_kd','fu_retain_kd'])
        write_json(output/'gate.json',{'pilot_passed':passed,'dt_allowed':False,
                   'reason':'Single-seed pilot requires review/replication before DT' if passed else 'FU evidence insufficient; do not implement DT',
                   'decisions':decisions})
    except Exception:
        (output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        coverage.append({'fraction':None,'method':'pipeline','status':'failed'})
        raise
    finally:
        timing['wall_time_total_seconds']=time.perf_counter()-started
        write_json(output/'timings.json',timing)
        pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False)
        write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps({'output':str(output),'seconds':timing['wall_time_total_seconds'],'pilot_passed':passed}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/processed/connected_v3_seed42'))
    parser.add_argument('--config',type=Path,default=Path('configs/pilot.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    run(args.data,args.config,args.output,args.smoke)
