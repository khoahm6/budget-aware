"""Exhaustive fixed-round planner. Representation proxy is not an FU guarantee."""
import itertools
import math
import time


def js(p, q):
    p = [v/sum(p) for v in p]
    q = [v/sum(q) for v in q]
    m = [(a+b)/2 for a,b in zip(p,q)]
    return .5 * sum(a*math.log(a/c) if a else 0 for a,c in zip(p,m)) + .5 * sum(b*math.log(b/c) if b else 0 for b,c in zip(q,m))


def plans(case, twin, budget_seconds, budget_bytes, max_js=.05, min_coverage=.2,
          margin=1.25, overhead_seconds=.5, ttl=600):
    if budget_seconds <= 0 or budget_bytes <= 0 or margin < 1:
        raise ValueError('Invalid budget/margin')
    clients = case['clients']
    target = case['target']
    others = [c['id'] for c in clients if c['id'] != target]
    byid = {c['id']: c for c in clients}
    global_hist = [sum(c['hist'][j] for c in clients) for j in range(2)]
    total = sum(c['n'] for c in clients)
    feasible = []
    for k in range(len(others)+1):
        for rest in itertools.combinations(others, k):
            ids = sorted([target, *rest])
            if any(str(i) not in twin['clients'] or time.time()-twin['clients'][str(i)]['received_at'] > ttl for i in ids):
                continue
            hist = [sum(byid[i]['hist'][j] for i in ids) for j in range(2)]
            coverage = sum(byid[i]['n'] for i in ids)/total
            divergence = js(hist, global_hist)
            predicted = overhead_seconds + case['rounds'] * margin * (
                max(twin['clients'][str(i)]['elapsed_seconds'] for i in ids) + twin['round_overhead_seconds'])
            size = case['rounds'] * sum(twin['clients'][str(i)]['body_bytes'] for i in ids)
            if predicted <= budget_seconds and size <= budget_bytes and divergence <= max_js and coverage >= min_coverage:
                feasible.append({'clients': ids, 'predicted_seconds': predicted, 'predicted_body_bytes': size,
                                 'coverage': coverage, 'js': divergence})
    return sorted(feasible, key=lambda p: (p['predicted_seconds'], -p['coverage'], p['clients']))
