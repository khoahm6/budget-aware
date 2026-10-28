"""Deterministic disjoint client ownership and nested sample deletion requests."""
import hashlib
import numpy as np


def ids_hash(ids):
    return hashlib.sha256('\n'.join(sorted(map(str, ids))).encode()).hexdigest()


def partition(y, seed, alpha=.5, clients=5):
    if alpha <= 0 or clients < 2 or set(np.unique(y)) != {0, 1}:
        raise ValueError('Invalid partition parameters or labels')
    rng = np.random.default_rng(seed)
    result = [[] for _ in range(clients)]
    for label in [0, 1]:
        idx = rng.permutation(np.flatnonzero(y == label))
        lengths = rng.multinomial(len(idx), rng.dirichlet(np.full(clients, alpha)))
        for dest, part in zip(result, np.split(idx, np.cumsum(lengths)[:-1])):
            dest.extend(part.tolist())
    result = [np.array(sorted(x), dtype=np.int64) for x in result]
    for i, idx in enumerate(result):
        if len(idx) < 4 or set(y[idx]) != {0, 1}:
            raise ValueError(f'Degenerate client {i}; no automatic repartition')
    assert len(np.unique(np.concatenate(result))) == len(y)
    return result


def forget_request(y, client_indices, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError('Fraction must be strictly between zero and one')
    rng = np.random.default_rng(seed)
    forgotten = []
    for label in [0, 1]:
        idx = rng.permutation(client_indices[y[client_indices] == label])
        count = int(np.floor(len(idx) * fraction))
        if count < 1 or count >= len(idx):
            raise ValueError('Stratified request leaves an empty class in F or R')
        forgotten.extend(idx[:count])
    f = np.array(sorted(forgotten), dtype=np.int64)
    r = np.setdiff1d(client_indices, f)
    return f, r


def retained_partition(clients, target, f):
    if target not in range(len(clients)) or len(f) != len(np.unique(f)) or not len(f):
        raise ValueError('Invalid target/forget IDs')
    if not np.isin(f, clients[target]).all():
        raise ValueError('Forget records must belong to target client')
    result = [np.setdiff1d(idx, f) if i == target else idx.copy() for i, idx in enumerate(clients)]
    if any(len(idx) == 0 or np.intersect1d(idx, f).size for idx in result):
        raise ValueError('Invalid retained partition')
    return result
