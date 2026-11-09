"""Software-environment check: the seed-0 two-epoch Qwen3-8B adapter, trained on a second machine, re-evaluated in
the main environment (gpu/run_envcheck.py) on the identical test file. Compares the two prediction files row by row.

Writes results/envcheck.txt.

Usage: python envcheck.py
"""
import numpy as np
import pandas as pd

from _paths import RESULTS as RES, RUNS

run = RUNS / "acc_predict_envcheck/Qwen3-8B-unsloth-bnb-4bit_seed0"
a, b = pd.read_csv(run / "orig56_test_predictions.csv"), pd.read_csv(run / "test_predictions.csv")
assert len(a) == len(b) and (a.dataset == b.dataset).all() and (a.gt_best_accuracy == b.gt_best_accuracy).all()
lines = [f"test rows: {len(a)}"]
for col in ("pred_best_accuracy", "pred_best_epoch"):
    x, y = a[col].to_numpy(float), b[col].to_numpy(float)
    lines.append(f"{col}: identical {int((x == y).sum())}/{len(x)}, max |difference| {np.nanmax(np.abs(x - y)):g}")
g = a.gt_best_accuracy.to_numpy(float)
for name, d in (("second machine", a), ("main environment", b)):
    p = d.pred_best_accuracy.to_numpy(float)
    lines.append(f"{name}: R2 {1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum():.4f}, MAE {np.abs(g - p).mean():.4f}")
(RES / "envcheck.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
