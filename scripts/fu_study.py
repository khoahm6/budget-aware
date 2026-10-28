"""How close to retraining can cheap FU get?  Reads a FINISHED pilot run (read-only), writes a NEW output dir.

Everything is evaluated on the same forget request, same seeds and same fixed protocol as run_pilot.py:
  floor_*        training-noise floors (same data, different shuffle seed). A gap smaller than these is
                 not a real difference between two models.
  continue_full  F kept, R extra rounds: control for "extra training alone moves the model".
  retain_only    current recovery (original -> R rounds on retained data).
  rollback_R     restart from the stored global model of round T-R, R rounds on retained data.
  ga2_*_start    bounded ascent (+KD) BEFORE recovery;  ga2_*_rec = followed by recovery rounds.
The ga2 radius is calibrated with the retrain reference (||theta_retrain - theta_orig||). That is an ORACLE,
used only to see whether the mechanism can work at all; never report it as a deployable method.
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
from src.eval.metrics import utility, similarity
from src.fl.fedavg_core import ClientWorker, model as make_model, fedavg, predict
from src.fu import fu_v2
from src.fu.rollback import fedavg_trajectory, rollback_retrain


def cost(logs):
    if not logs:
        return {"fl_rounds": 0, "fl_client_steps": 0, "fl_payload_mb": 0.0}
    return {"fl_rounds": len(logs),
            "fl_client_steps": int(sum(c["steps"] for e in logs for c in e["clients"])),
            "fl_payload_mb": sum(e["payload_bytes_up"] + e["payload_bytes_down"] for e in logs) / 1e6}


def main(run, data_path, config_path, output, fraction, rounds_list, ga_steps, ga_lr, kd_steps):
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
    fw = ClientWorker(target, train["X"][f], train["y"][f], train["row_id"][f])
    dim = train["X"].shape[1]
    initial = make_model(dim, cfg["hidden"], seed)

    def fresh():
        return make_model(dim, cfg["hidden"], seed)

    traj, _ = fedavg_trajectory(initial, all_w, fl, total, seed)
    original = traj[-1]
    saved = torch.load(Path(run) / "checkpoints" / "original.pt", map_location="cpu",
                       weights_only=True)["state_dict"]
    gap = max((original.state_dict()[k] - saved[k]).abs().max().item() for k in saved)
    print(f"[check] re-created original vs saved original: max abs diff {gap:.3e}", flush=True)
    retrain, rlog = fedavg(fresh(), rw, fl, total, seed)
    rb0, _ = rollback_retrain(traj, rw, fl, 0, seed)
    gap0 = max((retrain.state_dict()[k] - rb0.state_dict()[k]).abs().max().item() for k in saved)
    print(f"[check] rollback to round 0 vs retrain: max abs diff {gap0:.3e} (must be ~0)", flush=True)

    xf = train["X"][f]
    ref_f, ref_t = predict(retrain, xf), predict(retrain, test["X"])
    org_f = predict(original, xf)
    rows = []

    def add(name, m, logs=None, extra_steps=0):
        p_f, p_t = predict(m, xf), predict(m, test["X"])
        u = utility(test["y"], p_t)
        sim_f, sim_t = similarity(p_f, ref_f), similarity(p_t, ref_t)
        row = {"method": name, "mae_F_to_retrain": sim_f["prediction_mae"],
               "mae_test_to_retrain": sim_t["prediction_mae"],
               "mae_F_to_original": float(np.mean(np.abs(p_f - org_f))),
               "disagree_F_to_retrain": sim_f["disagreement"],
               "macro_f1_test": u["macro_f1"], "fpr_test": u["fpr"], "ap_test": u["pr_auc_ap"],
               "extra_client_steps": int(extra_steps), **cost(logs)}
        rows.append(row)
        print(f"{name:30s} MAE_F={row['mae_F_to_retrain']:.6f} MAE_test={row['mae_test_to_retrain']:.6f} "
              f"F1={row['macro_f1_test']:.4f} rounds={row['fl_rounds']}", flush=True)

    add("original", original)
    add("retrain_reference", retrain, rlog)
    m, logs = fedavg(fresh(), all_w, fl, total, seed + 777)
    add("floor_full_reshuffle", m, logs)
    m, logs = fedavg(fresh(), rw, fl, total, seed + 777)
    add("floor_retrain_reshuffle", m, logs)
    for rounds in rounds_list:
        m, logs = fedavg(original, all_w, fl, rounds, seed + 900000)
        add(f"continue_full_R{rounds}", m, logs)
        m, logs = fedavg(original, rw, fl, rounds, seed + 900000)
        add(f"retain_only_R{rounds}", m, logs)
        m, logs = rollback_retrain(traj, rw, fl, total - rounds, seed)
        add(f"rollback_R{rounds}", m, logs)
        # Control: rollback_R reuses the retrain reference's shuffle seeds (common random numbers), which
        # can make it look closer than it would be in deployment. Independent seeds remove that advantage.
        m, logs = rollback_retrain(traj, rw, fl, total - rounds, seed + 555)
        add(f"rollback_indep_R{rounds}", m, logs)

    xcal = torch.as_tensor(data["calibration"]["X"], dtype=torch.float32)
    ycal = torch.as_tensor(data["calibration"]["y"], dtype=torch.long)
    xr, yr = rw[target].x, rw[target].y
    dist = fu_v2.radius_from_reference(fu_v2.flat(original), fu_v2.flat(retrain))
    print(f"[info] ||theta_retrain - theta_orig|| = {dist:.4f}", flush=True)
    recovery = cfg["recovery_rounds"]
    for scale in (0.5, 1.0, 2.0):
        for role in ("none", "repair"):
            start, _ = fu_v2.unlearn_v2(original, fw.x, fw.y, xr, yr, xcal, ycal, radius=scale * dist,
                                        steps=ga_steps, lr=ga_lr, kd_role=role, kd_steps=kd_steps, seed=seed)
            tag = f"ga2_r{scale}_{role}"
            extra = ga_steps + (kd_steps if role != "none" else 0)
            add(f"{tag}_start", start, None, extra)
            m, logs = fedavg(start, rw, fl, recovery, seed + 900000)
            add(f"{tag}_rec{recovery}", m, logs, extra)

    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output / "study.csv", index=False)
    (output / "study_meta.json").write_text(json.dumps({
        "run": str(run), "fraction": fraction, "seed": seed, "rounds": total, "recovery_rounds": recovery,
        "orig_reproduction_max_abs_diff": gap, "rollback0_vs_retrain_max_abs_diff": gap0,
        "oracle_radius_reference_norm": dist, "ga_steps": ga_steps, "ga_lr": ga_lr, "kd_steps": kd_steps,
        "note": "ga2 radius is oracle-calibrated with the retrain reference; pilot diagnostic only"}, indent=2))
    print(f"wrote {output / 'study.csv'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10])
    ap.add_argument("--ga-steps", type=int, default=100)
    ap.add_argument("--ga-lr", type=float, default=1e-3)
    ap.add_argument("--kd-steps", type=int, default=100)
    args = ap.parse_args()
    started = time.perf_counter()
    main(args.run, args.data, args.config, args.output, args.fraction, args.rounds,
         args.ga_steps, args.ga_lr, args.kd_steps)
    print(f"done in {time.perf_counter() - started:.1f}s")
