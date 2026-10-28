"""Which cheap-to-know subset properties explain closeness to retrain?  (pandas/numpy only, no training)

Reads subsets.csv from fu_subsets.py and the pilot run's manifest.json (client label counts). For every
subset S of participating clients (target always included) it computes, from retained data only:
  sample_share     fraction of all retained samples held by S
  missing_mass     1 - sample_share
  attack_ratio_gap |attack ratio of pooled S  -  attack ratio of all retained data|
  js_label         Jensen-Shannon divergence between the two label distributions
and reports Spearman rank correlation with mae_F_to_retrain, separately for rollback and retain_only.
A feature that explains the error is something a planner can know WITHOUT training; one that does not
means the planner needs another signal. Correlation on 16 subsets is descriptive, not a proof.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def js(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return (kl(p, m) + kl(q, m)) / 2


def retained_labels(manifest, target, fraction):
    labels = {}
    for c in manifest["clients"]:
        n = np.array(c["labels"], dtype=float)
        if c["client_id"] == target:
            n = n - np.floor(n * fraction)      # forget_request removes floor(n_c * fraction) per class
        labels[c["client_id"]] = n
    return labels


def main(subsets_csv, manifest_path, output):
    subsets_csv = Path(subsets_csv)
    meta = json.loads((subsets_csv.parent / "subsets_meta.json").read_text())
    target, fraction = meta["target"], meta["fraction"]
    manifest = json.loads(Path(manifest_path).read_text())
    labels = retained_labels(manifest, target, fraction)
    total = sum(labels.values())
    overall = total / total.sum()
    frame = pd.read_csv(subsets_csv, dtype={"clients": str})
    frame = frame[frame.method.isin(["rollback", "retain_only"]) & (frame["size"] >= 1)].copy()
    feats = []
    for ids in frame["clients"]:
        pooled = sum(labels[int(ch)] for ch in ids)
        share = pooled.sum() / total.sum()
        dist = pooled / pooled.sum()
        feats.append({"sample_share": share, "missing_mass": 1 - share,
                      "attack_ratio_gap": abs(dist[1] - overall[1]), "js_label": js(dist, overall)})
    frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    cols = ["sample_share", "missing_mass", "attack_ratio_gap", "js_label", "cost_share_vs_full_rollback"]
    print(f"target client {target}, fraction {fraction}, retained label shares (normal, attack): "
          f"{np.round(overall, 4).tolist()}")
    for client_id, n in labels.items():
        print(f"  client {client_id}: retained {int(n.sum())} samples, attack share {n[1] / n.sum():.3f}")
    rows = []
    for method, part in frame.groupby("method"):
        for col in cols:
            rho = part[[col, "mae_F_to_retrain"]].corr(method="spearman").iloc[0, 1]
            rows.append({"method": method, "feature": col, "spearman_vs_mae_F": rho})
    corr = pd.DataFrame(rows)
    print("\nSpearman correlation with mae_F_to_retrain (positive = larger feature, larger error):")
    print(corr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    show = frame[frame.method == "rollback"].sort_values("mae_F_to_retrain")
    print("\nrollback subsets, best to worst:")
    print(show[["clients", "mae_F_to_retrain", "cost_share_vs_full_rollback", "sample_share",
                "attack_ratio_gap", "js_label"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Refusing overwrite: {out}")
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsets", type=Path, required=True, help="runs/subsets_*/subsets.csv")
    ap.add_argument("--manifest", type=Path, required=True, help="runs/reproduction_pilot/manifest.json")
    ap.add_argument("--output", type=Path, required=True, help="new CSV path")
    a = ap.parse_args()
    main(a.subsets, a.manifest, a.output)
