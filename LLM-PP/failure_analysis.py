"""Where do the LLMs fail? Large-error analysis on the shared 657-row test set.

For each LLM run and the XGBoost reference: share of items with |error| > 0.2 / 0.3, which datasets they
come from, how much of the squared error they carry, and how consistent the large misses are across seeds
(a miss repeated in all 3 seeds points at the input; a miss in one seed points at training instability).
Also explains the mnist pattern (high Spearman, low Pearson) by listing its largest misses.

Usage: python failure_analysis.py
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from _paths import RESULTS as RES, RUNS as OUT
MODELS = {"Qwen3-8B": "Qwen3-8B-unsloth-bnb-4bit", "Llama-3.1-8B": "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
          "Gemma-2-9B": "gemma-2-9b-it-bnb-4bit"}
CONFIGS = {"e2+P": OUT / "acc_predict", "e3+P": OUT / "acc_predict_3epoch"}
SEEDS = [0, 42, 123]

tab = pd.read_csv(OUT / "paper/tabular_3epoch.csv")
test = tab[tab.split == "test"].reset_index(drop=True)
xgb = np.mean([pd.read_csv(OUT / f"paper/xgboost_grid/xgb_e3_proxies_seed{s}_test_predictions.csv").pred_best_accuracy
               for s in [0, 42, 123, 7, 2024]], 0)

def miss_summary(datasets):
    """'dataset:count' of the large misses, most frequent first; ties by name, so the order does not depend on pandas."""
    counts = datasets.value_counts().sort_index().sort_values(ascending=False, kind="stable")
    return "; ".join(f"{k}:{v}" for k, v in counts.items())


rows, miss_rows = [], []
for (model, mdir), (config, root) in [(m, c) for m in MODELS.items() for c in CONFIGS.items()]:
    errs = []
    for s in SEEDS:
        df = pd.read_csv(root / f"{mdir}_seed{s}/test_predictions.csv")
        gt = df.gt_best_accuracy.to_numpy(float)
        err = df.pred_best_accuracy.to_numpy(float) - gt
        errs.append(np.abs(err))
        big = np.abs(err) > 0.2
        sq = err ** 2
        rows.append({"model": model, "config": config, "seed": s,
                     "share_err_gt_0.2": big.mean(), "share_err_gt_0.3": (np.abs(err) > 0.3).mean(),
                     "share_of_sq_error_from_err_gt_0.2": sq[big].sum() / sq.sum(),
                     "r2_without_err_gt_0.2": 1 - sq[~big].sum() / ((gt[~big] - gt[~big].mean()) ** 2).sum(),
                     "big_miss_datasets": miss_summary(df.dataset[big])})
    E = np.array(errs) > 0.2
    n_seeds_missing = E.sum(0)
    for k in (1, 2, 3):
        miss_rows.append({"model": model, "config": config, "missed_in_exactly_k_seeds": k, "items": int((n_seeds_missing == k).sum())})

big_xgb = np.abs(xgb - test.best_accuracy.round(2)) > 0.2
rows.append({"model": "XGBoost (5-seed mean)", "config": "e3+P", "seed": -1, "share_err_gt_0.2": big_xgb.mean(),
             "share_err_gt_0.3": (np.abs(xgb - test.best_accuracy.round(2)) > 0.3).mean(),
             "big_miss_datasets": miss_summary(test.dataset[big_xgb])})

big = pd.DataFrame(rows)
big.to_csv(RES / "large_errors.csv", index=False)
consistency = pd.DataFrame(miss_rows)
consistency.to_csv(RES / "large_error_seed_consistency.csv", index=False)

# mnist: which items drive the low Pearson? (3-epoch Gemma and Qwen, seed mean)
lines = []
for model in ["Gemma-2-9B", "Qwen3-8B"]:
    P = np.mean([pd.read_csv(CONFIGS["e3+P"] / f"{MODELS[model]}_seed{s}/test_predictions.csv").pred_best_accuracy for s in SEEDS], 0)
    m = (test.dataset == "mnist").to_numpy()
    g, p = test.best_accuracy.round(2).to_numpy()[m], P[m]
    order = np.argsort(-np.abs(p - g))
    lines.append(f"{model} mnist (n={m.sum()}): pearson={pearsonr(p, g)[0]:.3f} spearman={spearmanr(p, g)[0]:.3f}; "
                 f"gt range {g.min():.2f}-{g.max():.2f}, median {np.median(g):.2f}")
    keep = np.ones(m.sum(), bool)
    keep[order[:3]] = False
    lines.append(f"  pearson without its 3 largest misses: {pearsonr(p[keep], g[keep])[0]:.3f}")
    sub = test[m].reset_index(drop=True)
    for i in order[:5]:
        lines.append(f"  gt={g[i]:.2f} pred={p[i]:.2f} | ep1={sub.accuracy_epoch_1[i]:.3f} ep2={sub.accuracy_epoch_2[i]:.3f} "
                     f"ep3={sub.accuracy_epoch_3[i]:.3f} | {sub.nn[i][:60]}")
(RES / "mnist_outliers.txt").write_text("\n".join(lines) + "\n")

pd.set_option("display.width", 250, "display.max_colwidth", 90)
summary = big.groupby(["model", "config"])[["share_err_gt_0.2", "share_err_gt_0.3", "share_of_sq_error_from_err_gt_0.2", "r2_without_err_gt_0.2"]].mean()
print(summary.round(3).to_string())
print()
print(consistency.pivot_table(index=["model", "config"], columns="missed_in_exactly_k_seeds", values="items").to_string())
print()
print(big[big.config == "e3+P"][["model", "seed", "big_miss_datasets"]].to_string(index=False))
print()
print("\n".join(lines))
