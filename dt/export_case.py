"""Offline research provisioning. Never run on the central server."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.run_pilot import load_data
from src.data.federated import ids_hash
from dt.codec import pack, sha


def export(data, run, fraction, output):
    output.mkdir(parents=True, exist_ok=False)
    train = load_data(data)['train']
    manifest = json.loads((run/'manifest.json').read_text())
    config = manifest['config']
    request = json.loads((run/f'request_{fraction}.json').read_text())
    target = request['target']
    forgotten = set(map(str, request['forget_row_ids']))
    mapping = {str(r): i for i, r in enumerate(train['row_id'])}
    if not forgotten or not forgotten <= set(mapping):
        raise ValueError('Unknown/empty forget request')
    (output/'central').mkdir()
    for variant in ['original', 'after_local', 'after_kd']:
        name = 'original' if variant == 'original' else f'{fraction}_{variant}'
        checkpoint = torch.load(run/'checkpoints'/f'{name}.pt', map_location='cpu', weights_only=True)
        (output/'central'/f'{variant}.npz').write_bytes(pack(checkpoint['state_dict'], {}))
    clients, seen = [], set()
    with np.load(run/'partitions.npz', allow_pickle=False) as z:
        for cid in range(config['clients']):
            owned = list(map(str, z[f'client_{cid}']))
            if seen.intersection(owned) or not set(owned) <= set(mapping):
                raise ValueError('Invalid ownership')
            seen.update(owned)
            if cid == target and not forgotten <= set(owned):
                raise ValueError('Forget IDs do not belong to target')
            retained = [r for r in owned if r not in forgotten]
            if not retained:
                raise ValueError('Empty retained worker')
            idx = np.array([mapping[r] for r in retained])
            dest = output/f'client{cid}'
            dest.mkdir()
            np.savez(dest/'local.npz', X=train['X'][idx], y=train['y'][idx], row_id=train['row_id'][idx])
            version = sha((dest/'local.npz').read_bytes())
            clients.append({'id': cid, 'url': f'http://client{cid}:8000', 'n': len(idx),
                            'hist': np.bincount(train['y'][idx], minlength=2).tolist(),
                            'data_version': version})
    if len(seen) != len(train['y']):
        raise ValueError('Partition is not complete; use full pilot, not smoke subset')
    case = {'schema': 1, 'target': target, 'clients': clients, 'fl': config['fl'],
            'hidden': config['hidden'], 'input_dim': train['X'].shape[1],
            'seed': config['seed'] + 900000, 'rounds': config['recovery_rounds'],
            'source_run': str(run), 'request_hash': ids_hash(request['forget_row_ids']),
            'mode': 'infrastructure_only_until_fu_gate_passes',
            'gate': json.loads((run/'gate.json').read_text()) if (run/'gate.json').exists() else None}
    (output/'central'/'case.json').write_text(json.dumps(case, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--fraction', default='0.1')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    export(a.data, a.run, a.fraction, a.output)
