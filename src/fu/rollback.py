"""Checkpointed FedAvg and partial-rollback retraining.

The server stores only the global model of every round (O(T*d); ~18 KB per round for the pilot MLP),
never per-client updates. When a sample-deletion request arrives, rolling back to round T-R and
re-running R rounds on retained data gives a knob between
    R = 0  -> the original model (no work, no unlearning)
    R = T  -> exact retraining (identical seeds, identical result)
so cost and closeness-to-retrain trade off smoothly. Seeds follow fedavg()s own schedule
(seed + round * 10000), so the trajectory reproduces fedavg() bit-for-bit on the same machine."""
import copy
from src.fl.fedavg_core import fedavg


def fedavg_trajectory(initial, workers, config, rounds, seed):
    """Same result as fedavg(initial, workers, config, rounds, seed) but keeps every round model."""
    models, logs = [copy.deepcopy(initial)], []
    for r in range(rounds):
        model, log = fedavg(models[-1], workers, config, 1, seed + r * 10000)
        log[0]["round"] = r
        models.append(model)
        logs.extend(log)
    return models, logs


def rollback_retrain(trajectory, workers, config, start_round, seed):
    """Restart from trajectory[start_round]; re-run rounds start_round..T-1 on `workers` (retained data)."""
    total = len(trajectory) - 1
    if not 0 <= start_round <= total:
        raise ValueError("start_round outside the stored trajectory")
    model, logs = copy.deepcopy(trajectory[start_round]), []
    for r in range(start_round, total):
        model, log = fedavg(model, workers, config, 1, seed + r * 10000)
        log[0]["round"] = r
        logs.extend(log)
    return model, logs
