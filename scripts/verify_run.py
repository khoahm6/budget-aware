"""Independently verify run artifact hashes, ownership, deletion and FedAvg counts."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.data.prepare import digest
from src.data.federated import ids_hash


def verify(run, data):
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name, expected in hashes.items():
        if digest(run/name)!=expected: raise ValueError(f'Artifact modified: {name}')
    manifest=json.loads((run/'manifest.json').read_text())
    if digest(data/'manifest.json')!=manifest['data_manifest_sha256']:
        raise ValueError('Wrong data manifest')
    with np.load(data/'train.npz',allow_pickle=False) as arrays:
        train_ids=set(arrays['row_id'])
        row_groups=dict(zip(arrays['row_id'],arrays['group_id']))
    with np.load(run/'partitions.npz',allow_pickle=False) as arrays:
        clients=[set(arrays[f'client_{i}']) for i in range(5)]
    if sum(map(len,clients))!=len(set.union(*clients)): raise ValueError('Client overlap')
    if not set.union(*clients)<=train_ids: raise ValueError('Non-training client rows')
    if not manifest['smoke_only'] and set.union(*clients)!=train_ids: raise ValueError('Incomplete training partition')
    for split in ['validation','calibration','test']:
        with np.load(data/f'{split}.npz',allow_pickle=False) as arrays:
            if set.union(*clients)&set(arrays['row_id']): raise ValueError('Holdout in clients')
    checkpoints=run/'checkpoints'
    def load(name):
        return torch.load(checkpoints/f'{name}.pt',map_location='cpu',weights_only=True)['state_dict']
    original=load('original')
    total_up,total_down,compute=0,0,0.
    for logs_file in run.glob('*training.json'):
        for entry in json.loads(logs_file.read_text()):
            total_up+=entry['payload_bytes_up']; total_down+=entry['payload_bytes_down']
            compute+=sum(c['compute_seconds'] for c in entry['clients'])
    previous=set()
    for fraction in sorted(manifest['config']['fractions']):
        name=str(fraction)
        request=json.loads((run/f'request_{name}.json').read_text())
        f,r=set(request['forget_row_ids']),set(request['retain_row_ids'])
        if not previous<=f: raise ValueError('Non-nested forget requests')
        previous=f
        if f&r or f|r!=clients[request['target']]: raise ValueError('Invalid F/R partition')
        if ids_hash(f)!=request['forget_hash'] or ids_hash(r)!=request['retain_hash']:
            raise ValueError('Request hash mismatch')
        expected=[len(c-f) for c in clients]
        for log_path in [run/f'{name}_retrain_training.json',*run.glob(f'{name}_*recovery.json')]:
            for entry in json.loads(log_path.read_text()):
                if entry['counts']!=expected: raise ValueError(f'Wrong retained counts: {log_path}')
                np.testing.assert_allclose(entry['weights'],np.array(expected)/sum(expected))
                if 'recovery' in log_path.name:
                    total_up+=entry['payload_bytes_up']; total_down+=entry['payload_bytes_down']
                    compute+=sum(c['compute_seconds'] for c in entry['clients'])
        noop=load(f'{name}_no_op')
        if any(not torch.equal(original[k],noop[k]) for k in original): raise ValueError('No-op differs')
        with np.load(run/f'{name}_evaluation_ids.npz',allow_pickle=False) as ev:
            fg={row_groups[i] for i in ev['forget']}
            rg={row_groups[i] for i in ev['retained']}
            if fg&rg: raise ValueError('MIA retained control overlaps forget groups')
    return {'verified':True,'verified_artifacts':len(hashes),'nested_requests':True,'no_op_exact':True,
            'retained_counts_correct':True,'fl_payload_bytes_up_all_comparators':total_up,
            'fl_payload_bytes_down_all_comparators':total_down,'fl_client_compute_seconds_all_comparators':compute,
            'cost_scope':'FL training/recovery only, aggregated across comparisons; excludes FU transfer, Python/process startup, preprocessing, actual network and peak memory'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=Path('data/processed/connected_v3_seed42'))
    args=parser.parse_args()
    print(json.dumps(verify(args.run,args.data),indent=2))
