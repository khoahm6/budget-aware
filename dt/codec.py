import hashlib
import io
import json
import numpy as np
import torch


def pack(state, meta):
    stream = io.BytesIO()
    np.savez(stream, __meta__=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
             **{k: v.detach().cpu().numpy() for k, v in state.items()})
    return stream.getvalue()


def unpack(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        meta = json.loads(z['__meta__'].tobytes())
        state = {k: torch.from_numpy(z[k].copy()) for k in z.files if k != '__meta__'}
    return state, meta


def sha(payload):
    return hashlib.sha256(payload).hexdigest()
