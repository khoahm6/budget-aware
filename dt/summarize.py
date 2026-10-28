"""Aggregate completed-run prediction errors; do not hide failures/rejections."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for f in sorted(a.runs.rglob('report.json')):
        r=json.loads(f.read_text())
        if r.get('mode')!='run': continue
        predicted=(r.get('plan') or {}).get('predicted_seconds')
        actual=r.get('actual_seconds')
        completed=r['status']=='COMPLETED'
        rows.append({'path':str(f),'status':r['status'],'policy':r.get('policy'),
                     'scope':r.get('budget_scope'),'predicted_seconds':predicted,
                     'actual_seconds':actual,'body_bytes':r.get('actual_body_bytes'),
                     'budget_violated':r.get('budget_violated'),
                     'absolute_error':abs(actual-predicted) if completed and predicted is not None else None})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists(): raise FileExistsError(a.output)
    with a.output.open('w',newline='',encoding='utf-8-sig') as stream:
        if rows:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'requests':len(rows),'completed':sum(r['status']=='COMPLETED' for r in rows),
             'no_feasible':sum(r['status']=='NO_FEASIBLE_PLAN' for r in rows),
             'failed_or_incomplete':sum(r['status'] not in ['COMPLETED','NO_FEASIBLE_PLAN'] for r in rows)}
    valid=[r for r in rows if r['absolute_error'] is not None]
    done=[r for r in rows if r['status']=='COMPLETED']
    summary['prediction_cases']=len(valid)
    summary['mae_seconds']=sum(r['absolute_error'] for r in valid)/len(valid) if valid else None
    denom=sum(r['actual_seconds'] for r in valid)
    summary['wape']=sum(r['absolute_error'] for r in valid)/denom if denom else None
    summary['violation_rate_completed']=sum(bool(r['budget_violated']) for r in done)/len(done) if done else None
    print(json.dumps(summary,indent=2))
    print('Group by seed/profile/policy/budget scope before making scientific comparisons.')


if __name__=='__main__': main()
