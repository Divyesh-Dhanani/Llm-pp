"""Source-code runs trained before the sequence-length fix, when training sequences were cut at 1,024 tokens
(runs/acc_predict_ablation/*_trunc1024), next to the same configurations trained with full sequences.

Writes results/truncation_runs.csv (per run) and prints the mean over seeds.

Usage: python truncation_runs.py
"""
import numpy as np
import pandas as pd

from _paths import RESULTS as RES, RUNS

RUN = "Qwen3-8B-unsloth-bnb-4bit_seed{}"
rows = []
for arm in ("code2p", "code3p", "code3p_ro"):
    for version, folder in (("truncated to 1,024 tokens", f"{arm}_trunc1024"), ("full sequences", arm)):
        for seed in (0, 42, 123):
            d = pd.read_csv(RUNS / "acc_predict_ablation" / folder / RUN.format(seed) / "test_predictions.csv")
            g, p = d.gt_best_accuracy.to_numpy(float), d.pred_best_accuracy.to_numpy(float)
            rows.append({"arm": arm, "training": version, "seed": seed, "parse_failures": int(np.isnan(p).sum()),
                         "r2": 1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum(), "mae": np.abs(g - p).mean()})
df = pd.DataFrame(rows)
df.to_csv(RES / "truncation_runs.csv", index=False)
print(df.groupby(["arm", "training"], sort=False)[["r2", "mae"]].agg(["mean", "std"]).round(3).to_string())
