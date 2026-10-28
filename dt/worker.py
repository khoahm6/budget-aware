"""One retained-data worker per container/EC2; one uvicorn process only."""
import asyncio
import os
import re
import threading
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from fastapi import FastAPI, Header, HTTPException, Request, Response
from src.fl.fedavg_core import ClientWorker, model
from dt.codec import pack, unpack, sha

torch.set_num_threads(int(os.getenv('OMP_NUM_THREADS', '1')))
torch.use_deterministic_algorithms(True)
app = FastAPI()
lock = threading.Lock()
data = Path(os.getenv('DATA_DIR', '/data'))
cache = Path(os.getenv('CACHE_DIR', '/cache'))
cache.mkdir(parents=True, exist_ok=True)
cid = int(os.getenv('CLIENT_ID', '0'))
token = os.environ['DT_TOKEN']
version = sha((data/'local.npz').read_bytes())
with np.load(data/'local.npz', allow_pickle=False) as z:
    worker = ClientWorker(cid, z['X'], z['y'], z['row_id'])
hist = np.bincount(worker.y.numpy(), minlength=2).tolist()


def auth(value):
    import secrets
    if not secrets.compare_digest(value, token):
        raise HTTPException(401, 'Invalid token')


@app.get('/health')
def health():
    return {'ok': True, 'client_id': cid}


@app.get('/state')
def get_state(x_dt_token: str = Header(default='')):
    auth(x_dt_token)
    return {'id': cid, 'n': len(worker.y), 'hist': hist, 'data_version': version,
            'busy': lock.locked(), 'timestamp': time.time(), 'rss_bytes': psutil.Process().memory_info().rss}


@app.post('/train')
async def train(request: Request, x_dt_token: str = Header(default='')):
    auth(x_dt_token)
    payload = await request.body()
    if len(payload) > 16 * 1024 * 1024:
        raise HTTPException(413, 'Payload too large')
    weights, m = unpack(payload)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', m['job_id']):
        raise HTTPException(422, 'Invalid job ID')
    if m['data_version'] != version:
        raise HTTPException(409, 'Data version mismatch')
    if not lock.acquire(blocking=False):
        raise HTTPException(409, 'Busy: do not overlap experiments')
    try:
        return await asyncio.to_thread(execute, payload, weights, m)
    finally:
        lock.release()


def execute(payload, weights, m):
    prefix = cache/m['job_id']
    fingerprint = sha(payload)
    result, signature = prefix.with_suffix('.npz'), prefix.with_suffix('.sha')
    if result.exists() and signature.exists():
        if signature.read_text() != fingerprint:
            raise HTTPException(409, 'Job ID reused with different content')
        return Response(result.read_bytes(), media_type='application/octet-stream')
    started = time.perf_counter()
    stop = threading.Event()
    peak = [psutil.Process().memory_info().rss]
    def sample():
        while not stop.wait(.02):
            peak[0] = max(peak[0], psutil.Process().memory_info().rss)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        local = model(m['input_dim'], m['hidden'], 0)
        local.load_state_dict(weights, strict=True)
        update, n, log = worker.train(local, m['fl'], m['seed'])
    finally:
        stop.set()
        monitor.join()
    log.update({'n': n, 'hist': hist, 'data_version': version,
                'base_hash': m['base_hash'], 'job_id': m['job_id'],
                'worker_seconds': time.perf_counter()-started, 'sampled_process_peak_rss': peak[0]})
    out = pack(update, log)
    tmp = prefix.with_suffix('.tmp')
    tmp.write_bytes(out)
    tmp.replace(result)
    signature.write_text(fingerprint)
    return Response(out, media_type='application/octet-stream')
