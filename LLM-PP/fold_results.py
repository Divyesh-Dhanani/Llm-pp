"""Results on five held-out families: the main split ("other", 15 test architectures) and the alternative splits of
folds.FOLDS (transformer, 35; rnn, 9; detector, 9; segmentation, 18), so 86 test architectures in total.

Every split is scored with the same methods where their runs exist: Qwen3-8B (3 ep. + proxies, answer-only loss,
per-device batch 1 x 32) with and without source code, Gemma-2-9B (2 and 3 ep. + proxies), all seed 0; XGBoost (5 seeds);
LC-PFN; the per-dataset training mean; and the untuned average of Gemma-2-9B and XGBoost (3 ep. + proxies). Per split:
R2 with a cluster-bootstrap CI over test architectures, and MAE. Pooled over the test architectures of every split
where both methods have runs: paired comparisons of per-architecture MAE (bootstrap CI over architectures, Wilcoxon
signed-rank, Holm correction over the pooled comparisons).
A prediction that failed to parse is replaced by the training-split mean and counted.

Writes results/{fold_metrics,fold_comparisons,fold_per_architecture}.csv.

Usage: python fold_results.py
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from _paths import RESULTS as RES, RUNS as OUT
RNG = np.random.default_rng(0)
N_BOOT = 10000
QWEN, GEMMA = "Qwen3-8B-unsloth-bnb-4bit_seed0", "gemma-2-9b-it-bnb-4bit_seed0"
XGB_SEEDS = (0, 42, 123, 7, 2024)
SPLITS = {  # name -> (held-out family, tabular export, LLM run roots, XGBoost dir, LC-PFN dir)
    "main": ("other", OUT / "paper/tabular_3epoch.csv",
             {"qwen": OUT / "acc_predict_ablation/e3p_ro_b1", "code": OUT / "acc_predict_ablation/code3p_ro",
              "gemma": OUT / "acc_predict_3epoch", "gemma2": OUT / "acc_predict"}, OUT / "paper/xgboost_grid", OUT / "paper/lcpfn"),
    **{f: (fam, OUT / f"paper/folds/tabular_{f}.csv",
           {"qwen": OUT / f"acc_predict_folds/{f}/e3p_ro_b1", "code": OUT / f"acc_predict_folds/{f}/code3p_ro",
            "gemma": OUT / f"acc_predict_folds/{f}/e3p", "gemma2": OUT / f"acc_predict_folds/{f}/e2p"},
           OUT / f"paper/folds/{f}/xgboost_grid", OUT / f"paper/folds/{f}/lcpfn")
       for f, fam in (("tf", "transformer"), ("rnn", "rnn"), ("det", "detector"), ("seg", "segmentation"))},
}
METHODS = {"qwen": "Qwen3-8B, 3 ep. + proxies", "code": "Qwen3-8B, 3 ep. + proxies + code", "gemma": "Gemma-2-9B, 3 ep. + proxies",
           "gemma2": "Gemma-2-9B, 2 ep. + proxies", "xgb_p": "XGBoost, 3 ep. + proxies", "xgb": "XGBoost, 3 ep.",
           "xgb2_p": "XGBoost, 2 ep. + proxies", "lcpfn": "LC-PFN, 3 ep.", "lcpfn2": "LC-PFN, 2 ep.",
           "dsmean": "Per-dataset training mean", "ens": "Gemma-2-9B + XGBoost, 3 ep. + proxies"}


def r2(p, g):
    return 1 - ((g - p) ** 2).sum(-1) / ((g - g.mean()) ** 2).sum()


metric_rows, arch_rows = [], []
for split, (family, table, llm, xgb_dir, lcpfn_dir) in SPLITS.items():
    tab = pd.read_csv(table)
    train, test = tab[tab.split == "train"], tab[tab.split == "test"].reset_index(drop=True)
    gt = test.best_accuracy.astype(float).round(2).to_numpy()
    arch = test.nn.to_numpy()

    def load(path, fill):
        d = pd.read_csv(path)
        assert len(d) == len(test) and (d.dataset.to_numpy() == test.dataset.to_numpy()).all(), f"rows differ: {path}"
        assert np.allclose(d.gt_best_accuracy.astype(float), gt), f"labels differ: {path}"
        p = d.pred_best_accuracy.astype(float)
        return p.fillna(fill).to_numpy(), int(p.isna().sum())

    preds, failures = {}, {}
    mean_acc = train.best_accuracy.mean()
    for key, root in llm.items():
        path = root / (GEMMA if key.startswith("gemma") else QWEN) / "test_predictions.csv"
        if not path.exists():  # not every LLM configuration was run on every split
            continue
        preds[key], failures[key] = load(path, mean_acc)
        preds[key] = preds[key][None]
    for key, stem in (("xgb_p", "xgb_e3_proxies"), ("xgb", "xgb_e3"), ("xgb2_p", "xgb_e2_proxies")):
        preds[key] = np.array([load(xgb_dir / f"{stem}_seed{s}_test_predictions.csv", mean_acc)[0] for s in XGB_SEEDS])
    preds["lcpfn"] = load(lcpfn_dir / "lcpfn_e3_seed0_test_predictions.csv", mean_acc)[0][None]
    preds["lcpfn2"] = load(lcpfn_dir / "lcpfn_e2_seed0_test_predictions.csv", mean_acc)[0][None]
    preds["dsmean"] = test.dataset.map(train.groupby("dataset").best_accuracy.mean()).fillna(mean_acc).to_numpy()[None]
    if "gemma" in preds:
        preds["ens"] = (preds["gemma"][0] + preds["xgb_p"].mean(0))[None] / 2

    clusters = [np.flatnonzero(arch == a) for a in pd.unique(arch)]
    boot = [np.concatenate([clusters[k] for k in RNG.integers(0, len(clusters), len(clusters))]) for _ in range(2000)]
    for key, P in preds.items():
        ci = np.percentile([r2(P[:, i], gt[i]).mean() for i in boot], [2.5, 97.5])
        metric_rows.append({"split": split, "held_out_family": family, "n_runs": len(test), "n_arch": len(clusters),
                            "method": METHODS[key], "n_seeds": len(P), "r2": r2(P, gt).mean(), "r2_ci_lo": ci[0], "r2_ci_hi": ci[1],
                            "mae": np.abs(P - gt).mean(), "parse_failures": failures.get(key, 0)})
        err = np.abs(P - gt).mean(0)
        for a in pd.unique(arch):
            arch_rows.append({"split": split, "architecture": a, "method": METHODS[key], "mae": err[arch == a].mean()})

metrics = pd.DataFrame(metric_rows)
per_arch = pd.DataFrame(arch_rows).pivot_table(index=["split", "architecture"], columns="method", values="mae")
RES.mkdir(parents=True, exist_ok=True)
metrics.to_csv(RES / "fold_metrics.csv", index=False)
per_arch.to_csv(RES / "fold_per_architecture.csv")

# pooled over the test architectures of every split where both methods have runs: B minus A in per-architecture MAE
comparisons = [("3rd epoch (XGBoost)", METHODS["xgb2_p"], METHODS["xgb_p"]),
               ("3rd epoch (LC-PFN)", METHODS["lcpfn2"], METHODS["lcpfn"]),
               ("3rd epoch (Gemma-2-9B)", METHODS["gemma2"], METHODS["gemma"]),
               ("Gemma-2-9B + XGBoost vs XGBoost", METHODS["xgb_p"], METHODS["ens"]),
               ("source code (Qwen3-8B)", METHODS["qwen"], METHODS["code"]),
               ("Qwen3-8B vs XGBoost", METHODS["xgb_p"], METHODS["qwen"]),
               ("Qwen3-8B + code vs XGBoost", METHODS["xgb_p"], METHODS["code"]),
               ("Gemma-2-9B vs XGBoost", METHODS["xgb_p"], METHODS["gemma"]),
               ("Gemma-2-9B vs Qwen3-8B", METHODS["qwen"], METHODS["gemma"]),
               ("proxies (XGBoost)", METHODS["xgb"], METHODS["xgb_p"]),
               ("XGBoost vs LC-PFN", METHODS["lcpfn"], METHODS["xgb_p"])]
cmp_rows = []
for name, a, b in comparisons:
    for scope, sub in [("all splits", per_arch)] + [(s, per_arch.loc[s]) for s in SPLITS]:
        d = (sub[b] - sub[a]).dropna().to_numpy()
        if len(d) == 0:
            continue
        ci = np.percentile([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)], [2.5, 97.5])
        cmp_rows.append({"comparison": name, "A": a, "B": b, "scope": scope, "n_arch": len(d), "delta_mae": d.mean(),
                         "ci_lo": ci[0], "ci_hi": ci[1], "arch_better_B": int((d < 0).sum()),
                         "wilcoxon_p": wilcoxon(d).pvalue if len(d) > 5 else np.nan})
cmp_ = pd.DataFrame(cmp_rows)
# Holm over the pooled comparisons (scope "all splits")
pooled = cmp_.index[cmp_.scope == "all splits"]
order = cmp_.loc[pooled, "wilcoxon_p"].sort_values().index
adj = np.minimum(1, np.maximum.accumulate(cmp_.loc[order, "wilcoxon_p"].to_numpy() * np.arange(len(order), 0, -1)))
cmp_.loc[order, "holm_p"] = adj
cmp_.to_csv(RES / "fold_comparisons.csv", index=False)

pd.set_option("display.width", 220)
print(metrics.round(4).to_string(index=False))
print()
print(cmp_.round(4).to_string(index=False))
