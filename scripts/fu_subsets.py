"""Rollback with only a SUBSET of clients re-training: the decision space of a budget-aware planner.

The target client (owner of the deleted records) always participates. For every subset of the other
clients it runs, with R rounds each,
  rollback     : restart from the stored global model of round T-R, FedAvg over the subset on retained data
  retain_only  : restart from the final original model, same subset and R rounds
and reports distance to the retrain reference (on F and on test), utility and cost.
Rollback uses independent shuffle seeds (seed + offset), so it does NOT share the reference's seeds.
Reads a finished run (read-only); writes a NEW output directory.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from scripts.fu_study import cost
from scripts.run_pilot import load_data
from src.data.federated import partition, forget_request, retained_partition
from src.eval.metrics import utility, similarity
from src.fl.fedavg_core import ClientWorker, model as make_model, fedavg, predict
from src.fu.rollback import fedavg_trajectory, rollback_retrain


def main(data_path, config_path, output, fraction, rounds, offset):
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")
    cfg = json.loads(config_path.read_text())
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    data = load_data(data_path)
    train, test = data["train"], data["test"]
    seed, total, fl = cfg["seed"], cfg["fl"]["rounds"], cfg["fl"]
    clients = partition(train["y"], seed, cfg["alpha"], cfg["clients"])
    target = int(np.argmax([len(i) for i in clients]))
    f, _ = forget_request(train["y"], clients[target], fraction, seed + 1)
    retained = retained_partition(clients, target, f)

    def workers(parts):
        return [ClientWorker(i, train["X"][idx], train["y"][idx], train["row_id"][idx])
                for i, idx in enumerate(parts)]

    all_w, rw = workers(clients), workers(retained)
    dim = train["X"].shape[1]
    initial = make_model(dim, cfg["hidden"], seed)

    def fresh():
        return make_model(dim, cfg["hidden"], seed)

    traj, _ = fedavg_trajectory(initial, all_w, fl, total, seed)
    original = traj[-1]
    retrain, _ = fedavg(fresh(), rw, fl, total, seed)
    xf = train["X"][f]
    ref_f, ref_t = predict(retrain, xf), predict(retrain, test["X"])
    rows = []

    def add(method, ids, m, logs):
        p_f, p_t = predict(m, xf), predict(m, test["X"])
        sim_f, sim_t = similarity(p_f, ref_f), similarity(p_t, ref_t)
        u = utility(test["y"], p_t)
        rows.append({"method": method, "clients": "".join(str(i) for i in ids), "size": len(ids),
                     "mae_F_to_retrain": sim_f["prediction_mae"], "mae_test_to_retrain": sim_t["prediction_mae"],
                     "disagree_F_to_retrain": sim_f["disagreement"], "macro_f1_test": u["macro_f1"],
                     **cost(logs)})

    add("original", [], original, None)
    floor, flogs = fedavg(fresh(), rw, fl, total, seed + 777)
    add("floor_retrain_reshuffle", list(range(len(clients))), floor, flogs)
    others = [i for i in range(len(clients)) if i != target]
    for k in range(len(others) + 1):
        for sub in itertools.combinations(others, k):
            ids = sorted((target,) + sub)
            ws = [rw[i] for i in ids]
            m, logs = rollback_retrain(traj, ws, fl, total - rounds, seed + offset)
            add("rollback", ids, m, logs)
            m, logs = fedavg(original, ws, fl, rounds, seed + 900000)
            add("retain_only", ids, m, logs)
            print(f"done clients {ids}", flush=True)

    frame = pd.DataFrame(rows)
    full = frame[(frame.method == "rollback") & (frame["size"] == len(clients))].fl_client_steps.iloc[0]
    frame["cost_share_vs_full_rollback"] = frame.fl_client_steps / full
    output.mkdir(parents=True)
    frame.to_csv(output / "subsets.csv", index=False)
    (output / "subsets_meta.json").write_text(json.dumps({
        "fraction": fraction, "rounds": rounds, "target": target, "seed_offset": offset,
        "note": "target always participates; rollback uses independent shuffle seeds"}, indent=2))
    show = frame[frame.method.isin(["rollback", "retain_only"])].sort_values(["method", "size", "mae_F_to_retrain"])
    print(show[["method", "clients", "size", "mae_F_to_retrain", "mae_test_to_retrain",
                "macro_f1_test", "cost_share_vs_full_rollback"]].to_string(index=False))
    print(f"target client = {target}; wrote {output / 'subsets.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=555)
    args = ap.parse_args()
    started = time.perf_counter()
    main(args.data, args.config, args.output, args.fraction, args.rounds, args.seed_offset)
    print(f"done in {time.perf_counter() - started:.1f}s")
