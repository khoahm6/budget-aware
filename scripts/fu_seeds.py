"""Seed-robust comparison of rollback vs retain_only (and the retrain noise floor).

One run of fu_study.py has one shuffle seed per method, and the seed-to-seed spread in
mae_F_to_retrain is of order 5e-4..9e-4, as large as some gaps being interpreted. This script repeats
the comparison with K independent seed offsets against ONE fixed retrain reference (seed = config seed):
  floor     : retrain on retained data with a different shuffle seed (pure training noise)
  rollback  : restart from stored round T-R model, R rounds, independent seeds
  retain_only: original -> R rounds on retained data, independent seeds
and reports mean +- std, plus a paired win rate (same seed index) of rollback over retain_only.
Reads only data + config (no run directory needed); writes a NEW output directory.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from scripts.run_pilot import load_data
from src.data.federated import partition, forget_request, retained_partition
from src.fl.fedavg_core import ClientWorker, model as make_model, fedavg, predict
from src.fu.rollback import fedavg_trajectory, rollback_retrain


def main(data_path, config_path, output, fraction, rounds_list, repeats):
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

    def mae(m):
        return (float(np.mean(np.abs(predict(m, xf) - ref_f))),
                float(np.mean(np.abs(predict(m, test["X"]) - ref_t))))

    records = [{"method": "original", "R": 0, "rep": 0, **dict(zip(("mae_F", "mae_test"), mae(original)))}]
    for rep in range(repeats):
        shift = 5000 + 97 * rep          # never collides with retrain seeds seed + r*10000 + cid*100
        floor, _ = fedavg(fresh(), rw, fl, total, seed + shift)
        records.append({"method": "floor_retrain_reshuffle", "R": total, "rep": rep,
                        **dict(zip(("mae_F", "mae_test"), mae(floor)))})
        for rounds in rounds_list:
            m, _ = rollback_retrain(traj, rw, fl, total - rounds, seed + shift)
            records.append({"method": "rollback", "R": rounds, "rep": rep,
                            **dict(zip(("mae_F", "mae_test"), mae(m)))})
            m, _ = fedavg(original, rw, fl, rounds, seed + 900000 + 97 * rep)
            records.append({"method": "retain_only", "R": rounds, "rep": rep,
                            **dict(zip(("mae_F", "mae_test"), mae(m)))})
        print(f"seed repeat {rep + 1}/{repeats} done", flush=True)

    frame = pd.DataFrame(records)
    summary = (frame.groupby(["method", "R"]).mae_F.agg(["mean", "std", "count"]).reset_index())
    base = float(frame[frame.method == "original"].mae_F.iloc[0])
    floor_mean = float(frame[frame.method == "floor_retrain_reshuffle"].mae_F.mean())
    print(f"\nfraction {fraction}: original gap {base:.6f}, noise floor mean {floor_mean:.6f} "
          f"(signal/noise {base / floor_mean:.1f}x)")
    lines = []
    for rounds in rounds_list:
        rb = frame[(frame.method == "rollback") & (frame.R == rounds)].sort_values("rep").mae_F.to_numpy()
        ro = frame[(frame.method == "retain_only") & (frame.R == rounds)].sort_values("rep").mae_F.to_numpy()
        lines.append({"R": rounds,
                      "rollback_mean": rb.mean(), "rollback_std": rb.std(ddof=1) if len(rb) > 1 else float("nan"),
                      "retain_only_mean": ro.mean(), "retain_only_std": ro.std(ddof=1) if len(ro) > 1 else float("nan"),
                      "rollback_remaining_gap": rb.mean() / base, "retain_only_remaining_gap": ro.mean() / base,
                      "paired_win_rate_rollback": float(np.mean(rb < ro))})
    table = pd.DataFrame(lines)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    output.mkdir(parents=True)
    frame.to_csv(output / "seeds_raw.csv", index=False)
    table.to_csv(output / "seeds_summary.csv", index=False)
    summary.to_csv(output / "seeds_by_method.csv", index=False)
    (output / "seeds_meta.json").write_text(json.dumps({
        "fraction": fraction, "repeats": repeats, "rounds": rounds_list, "target": target,
        "original_gap": base, "floor_mean": floor_mean}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    started = time.perf_counter()
    main(args.data, args.config, args.output, args.fraction, args.rounds, args.repeats)
    print(f"done in {time.perf_counter() - started:.1f}s")
