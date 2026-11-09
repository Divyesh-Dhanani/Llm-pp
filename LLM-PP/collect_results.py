"""Single source of truth for every number in the manuscript.

Recomputes all metrics from raw test_predictions.csv files (never from logs), on the shared 657-row
cross-family test set, and writes tables under results/:
  runs.csv            one row per run (LLM, XGBoost, trivial baselines, ensembles)
  configs.csv         per configuration: mean +/- SD over seeds, bootstrap 95% CI of the seed-mean
  per_dataset.csv     per configuration x dataset (mean over seeds)
  comparisons.csv     paired comparisons: delta R2 / delta MAE with bootstrap CI, Wilcoxon on |error|, Holm
The 657 test rows are runs of only 15 architectures, so rows are not independent: the primary
uncertainty is a CLUSTER bootstrap over architectures (and a Wilcoxon over per-architecture mean
|error|); the row-level bootstrap/Wilcoxon are kept for reference only.
Ensembles use untuned equal weights only (validation and test families differ, so no weight tuning).

Usage: python collect_results.py
"""
from __future__ import annotations

import csv
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

from _paths import RESULTS as RES, RUNS as OUT
SEEDS = [0, 42, 123]
MODELS = {"Qwen3-8B": "Qwen3-8B-unsloth-bnb-4bit", "Llama-3.1-8B": "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
          "Gemma-2-9B": "gemma-2-9b-it-bnb-4bit"}
# configuration -> directory holding <model_dir>_seed<seed>/test_predictions.csv
LLM_CONFIGS = {"e2+P": OUT / "acc_predict", "e3+P": OUT / "acc_predict_3epoch",
               "e2+P+code": OUT / "acc_predict_ablation/code2p", "e3+P+code": OUT / "acc_predict_ablation/code3p",
               "e3": OUT / "acc_predict_ablation/e3np",
               "e3+P (answer-only loss)": OUT / "acc_predict_ablation/e3p_ro",
               "e3+P+code (answer-only loss)": OUT / "acc_predict_ablation/code3p_ro",
               # controls trained after the switch to SFTConfig, like every code run: data order fixed across seeds
               "e3+P [rerun]": OUT / "acc_predict_ablation/e3p_all",
               "e3+P (answer-only loss) [rerun]": OUT / "acc_predict_ablation/e3p_ro_all",
               "e3+P (answer-only loss) [batch 1x32]": OUT / "acc_predict_ablation/e3p_ro_b1"}
XGB_DIR, XGB_SEEDS = OUT / "paper/xgboost_grid", [0, 42, 123, 7, 2024]
XGB_CONFIGS = {"e2": "xgb_e2", "e2+P": "xgb_e2_proxies", "e3": "xgb_e3", "e3+P": "xgb_e3_proxies"}
TABLE = OUT / "paper/tabular_3epoch.csv"
N_BOOT, RNG = 10000, np.random.default_rng(0)


def metrics(p, g):
    ss = np.sum((g - g.mean()) ** 2)
    return {"r2": 1 - np.sum((g - p) ** 2) / ss, "pearson_r": pearsonr(p, g)[0], "spearman_r": spearmanr(p, g)[0],
            "rmse": float(np.sqrt(np.mean((p - g) ** 2))), "mae": float(np.mean(np.abs(p - g)))}


def r2_rows(P, g, idx):
    """R2 of each prediction row (seed) on the resampled items idx; P: seeds x items."""
    gi = g[idx]
    return 1 - ((P[:, idx] - gi) ** 2).sum(1) / ((gi - gi.mean()) ** 2).sum()


# ---------------------------------------------------------------- load predictions
runs = {}  # (method, config) -> dict(seeds -> (acc_pred, epoch_pred))
gt_acc = gt_epoch = datasets = None


def add(method, config, seed, path, acc_col="pred_best_accuracy", ep_col="pred_best_epoch"):
    global gt_acc, gt_epoch, datasets
    df = pd.read_csv(path)
    if gt_acc is None:
        gt_acc, gt_epoch = df.gt_best_accuracy.to_numpy(float), df.gt_best_epoch.to_numpy(float)
        datasets = df.dataset.to_numpy()
    assert len(df) == 657 and (df.dataset.to_numpy() == datasets).all(), f"test rows differ: {path}"
    assert np.allclose(df.gt_best_accuracy.to_numpy(float), gt_acc), f"labels differ: {path}"
    assert df[acc_col].notna().all(), f"parse failures in {path}"
    runs.setdefault((method, config), {})[seed] = (df[acc_col].to_numpy(float), df[ep_col].to_numpy(float))


for (model, mdir), (config, root) in product(MODELS.items(), LLM_CONFIGS.items()):
    for s in SEEDS:
        f = root / f"{mdir}_seed{s}/test_predictions.csv"
        if (root / f"{mdir}_seed{s}/test_metrics.log").exists():
            add(model, config, s, f)
for config, stem in XGB_CONFIGS.items():
    for s in XGB_SEEDS:
        add("XGBoost", config, s, XGB_DIR / f"{stem}_seed{s}_test_predictions.csv")
# learning-curve extrapolation baseline (LC-PFN, pre-trained; deterministic, one "seed")
for config, k in (("e2", 2), ("e3", 3)):
    f = OUT / f"paper/lcpfn/lcpfn_e{k}_seed0_test_predictions.csv"
    if f.exists():
        add("LC-PFN", config, 0, f)

# trivial baselines from the training split (deterministic, one "seed")
tab = pd.read_csv(TABLE)
train, test = tab[tab.split == "train"], tab[tab.split == "test"].reset_index(drop=True)
assert (test.dataset.to_numpy() == datasets).all()
clusters = [np.flatnonzero(test.nn.to_numpy() == a) for a in pd.unique(test.nn)]
arch_of = test.nn.to_numpy()


def cluster_sample():
    pick = RNG.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[k] for k in pick])


cluster_idx = [cluster_sample() for _ in range(N_BOOT)]
mean_acc, mean_ep = train.best_accuracy.mean(), train.epochs_to_best.mean()
ds_acc, ds_ep = train.groupby("dataset").best_accuracy.mean(), train.groupby("dataset").epochs_to_best.mean()
trivial = {
    "train mean": (np.full(657, mean_acc), np.full(657, mean_ep)),
    "per-dataset train mean": (test.dataset.map(ds_acc).fillna(mean_acc).to_numpy(), test.dataset.map(ds_ep).fillna(mean_ep).to_numpy()),
    "last early accuracy (epoch 2)": (test.accuracy_epoch_2.to_numpy(float), np.full(657, 2.0)),
    "last early accuracy (epoch 3)": (test.accuracy_epoch_3.to_numpy(float), np.full(657, 3.0)),
}
for name, pr in trivial.items():
    runs[("Trivial", name)] = {0: pr}


# ---------------------------------------------------------------- ensembles (untuned, equal weights)
def seed_mean(key):
    s = runs[key]
    return np.mean([v[0] for v in s.values()], 0), np.mean([v[1] for v in s.values()], 0)


def ensemble(name, keys):
    if all(k in runs for k in keys):
        parts = [seed_mean(k) for k in keys]
        runs[("Ensemble", name)] = {0: (np.mean([p[0] for p in parts], 0), np.mean([p[1] for p in parts], 0))}


for model in MODELS:
    for c in ("e2+P", "e3+P"):
        ensemble(f"{model} {c}: seed average", [(model, c)])
    ensemble(f"{model}: e2+P & e3+P", [(model, "e2+P"), (model, "e3+P")])
    ensemble(f"{model} e3+P & XGBoost e3+P", [(model, "e3+P"), ("XGBoost", "e3+P")])
    # compute-matched: one LLM run (single seed) plus XGBoost
    for s in runs.get((model, "e3+P"), {}):
        llm_run, xgb = runs[(model, "e3+P")][s], seed_mean(("XGBoost", "e3+P"))
        runs[("Ensemble", f"{model} e3+P seed {s} & XGBoost e3+P")] = {0: ((llm_run[0] + xgb[0]) / 2, (llm_run[1] + xgb[1]) / 2)}
for c in ("e2+P", "e3+P"):
    ensemble(f"3 LLMs {c}", [(m, c) for m in MODELS])
    ensemble(f"3 LLMs {c} & XGBoost {c}", [(m, c) for m in MODELS] + [("XGBoost", c)])

# ---------------------------------------------------------------- per-run and per-config tables
RES.mkdir(parents=True, exist_ok=True)
run_rows, cfg_rows, ds_rows = [], [], []
boot_idx = RNG.integers(0, 657, size=(N_BOOT, 657))
for (method, config), seeds in runs.items():
    A = np.array([v[0] for v in seeds.values()])
    E = np.array([v[1] for v in seeds.values()])
    for s, (a, e) in seeds.items():
        run_rows.append({"method": method, "config": config, "seed": s,
                         **{f"acc_{k}": v for k, v in metrics(a, gt_acc).items()},
                         **{f"epoch_{k}": v for k, v in metrics(e, gt_epoch).items()}})
    per_seed = pd.DataFrame([metrics(a, gt_acc) for a in A])
    per_seed_ep = pd.DataFrame([metrics(e, gt_epoch) for e in E])
    boot = np.array([r2_rows(A, gt_acc, i).mean() for i in boot_idx[:2000]])
    cboot = np.array([r2_rows(A, gt_acc, i).mean() for i in cluster_idx[:2000]])
    row = {"method": method, "config": config, "n_seeds": len(A)}
    for k in per_seed:
        row[f"acc_{k}_mean"], row[f"acc_{k}_sd"] = per_seed[k].mean(), per_seed[k].std(ddof=1) if len(A) > 1 else 0.0
        row[f"epoch_{k}_mean"] = per_seed_ep[k].mean()
    row["acc_r2_ci_lo"], row["acc_r2_ci_hi"] = np.percentile(cboot, [2.5, 97.5])
    row["acc_r2_rowci_lo"], row["acc_r2_rowci_hi"] = np.percentile(boot, [2.5, 97.5])
    cfg_rows.append(row)
    for d in sorted(set(datasets)):
        m = datasets == d
        per = pd.DataFrame([metrics(a[m], gt_acc[m]) for a in A])
        ds_rows.append({"method": method, "config": config, "dataset": d, "n": int(m.sum()),
                        **{f"{k}_mean": per[k].mean() for k in per}, **{f"{k}_sd": (per[k].std(ddof=1) if len(A) > 1 else 0.0) for k in per}})

pd.DataFrame(run_rows).to_csv(RES / "runs.csv", index=False)
cfg = pd.DataFrame(cfg_rows).sort_values("acc_r2_mean", ascending=False)
cfg.to_csv(RES / "configs.csv", index=False)
pd.DataFrame(ds_rows).to_csv(RES / "per_dataset.csv", index=False)


# ---------------------------------------------------------------- paired comparisons
def compare(name, a_key, b_key):
    """B minus A. Bootstrap over items of the seed-averaged R2 / MAE; Wilcoxon on per-item |error| (seed mean)."""
    if a_key not in runs or b_key not in runs:
        return None
    A = np.array([v[0] for v in runs[a_key].values()])
    B = np.array([v[0] for v in runs[b_key].values()])
    d_r2 = np.array([r2_rows(B, gt_acc, i).mean() - r2_rows(A, gt_acc, i).mean() for i in boot_idx])
    ea, eb = np.abs(A - gt_acc).mean(0), np.abs(B - gt_acc).mean(0)
    d_mae = np.array([(eb[i] - ea[i]).mean() for i in boot_idx])
    c_r2 = np.array([r2_rows(B, gt_acc, i).mean() - r2_rows(A, gt_acc, i).mean() for i in cluster_idx])
    c_mae = np.array([(eb[i] - ea[i]).mean() for i in cluster_idx])
    arch = pd.DataFrame({"a": arch_of, "ea": ea, "eb": eb}).groupby("a")[["ea", "eb"]].mean()
    point = r2_rows(B, gt_acc, np.arange(657)).mean() - r2_rows(A, gt_acc, np.arange(657)).mean()
    return {"comparison": name, "A": " ".join(a_key), "B": " ".join(b_key),
            "delta_r2": point, "delta_r2_ci_lo": np.percentile(c_r2, 2.5), "delta_r2_ci_hi": np.percentile(c_r2, 97.5),
            "delta_mae": float((eb - ea).mean()), "delta_mae_ci_lo": np.percentile(c_mae, 2.5), "delta_mae_ci_hi": np.percentile(c_mae, 97.5),
            "arch_better_B": int((arch.eb < arch.ea).sum()), "n_arch": len(arch),
            "wilcoxon_p_arch": wilcoxon(arch.ea, arch.eb).pvalue,
            "row_delta_r2_ci_lo": np.percentile(d_r2, 2.5), "row_delta_r2_ci_hi": np.percentile(d_r2, 97.5),
            "wilcoxon_p_rows": wilcoxon(ea, eb).pvalue}


# Runs trained after the switch to SFTConfig (all code runs and the [rerun]/[batch 1x32] controls) share one data
# order across seeds; earlier runs shuffle per seed. Comparisons marked "(mixed order)" cross the two regimes.
comps = []
for m in MODELS:
    comps += [compare(f"{m}: 3rd epoch", (m, "e2+P"), (m, "e3+P")),
              compare(f"{m}: source code (3 epochs, mixed order)", (m, "e3+P"), (m, "e3+P+code")),
              compare(f"{m} vs XGBoost (e3+P)", ("XGBoost", "e3+P"), (m, "e3+P")),
              compare(f"{m} vs XGBoost (e2+P)", ("XGBoost", "e2+P"), (m, "e2+P"))]
q = "Qwen3-8B"
comps += [compare(f"{q}: source code (2 epochs, mixed order)", (q, "e2+P"), (q, "e2+P+code")),
          compare(f"{q}: source code (3 epochs)", (q, "e3+P [rerun]"), (q, "e3+P+code")),
          compare(f"{q}: source code (3 epochs, answer-only loss)", (q, "e3+P (answer-only loss) [batch 1x32]"),
                  (q, "e3+P+code (answer-only loss)")),
          compare(f"{q}: proxies (3 epochs)", (q, "e3"), (q, "e3+P")),
          compare(f"{q}: answer-only loss (no code)", (q, "e3+P"), (q, "e3+P (answer-only loss)")),
          compare(f"{q}: answer-only loss (no code, rerun)", (q, "e3+P [rerun]"), (q, "e3+P (answer-only loss) [rerun]")),
          compare(f"{q}: answer-only loss (code)", (q, "e3+P+code"), (q, "e3+P+code (answer-only loss)")),
          compare(f"{q}: batch 1x32 instead of 4x8", (q, "e3+P (answer-only loss) [rerun]"), (q, "e3+P (answer-only loss) [batch 1x32]")),
          compare(f"{q}: rerun with fixed data order", (q, "e3+P"), (q, "e3+P [rerun]"))]
comps += [compare("XGBoost: 3rd epoch (+P)", ("XGBoost", "e2+P"), ("XGBoost", "e3+P")),
          compare("XGBoost: proxies (3 epochs)", ("XGBoost", "e3"), ("XGBoost", "e3+P")),
          compare("Qwen vs Llama (e2+P)", ("Llama-3.1-8B", "e2+P"), ("Qwen3-8B", "e2+P")),
          compare("Gemma vs Qwen (e3+P)", ("Qwen3-8B", "e3+P"), ("Gemma-2-9B", "e3+P")),
          compare("Ensemble 3 LLMs+XGB vs XGBoost (e3+P)", ("XGBoost", "e3+P"), ("Ensemble", "3 LLMs e3+P & XGBoost e3+P"))]
comps = pd.DataFrame([c for c in comps if c])
order = np.argsort(comps.wilcoxon_p_arch.to_numpy())
holm = np.empty(len(comps))
running = 0.0
for rank, i in enumerate(order):  # Holm step-down adjusted p-values
    running = max(running, min(1.0, (len(comps) - rank) * comps.wilcoxon_p_arch.iloc[i]))
    holm[i] = running
comps["wilcoxon_p_arch_holm"] = holm
comps.to_csv(RES / "comparisons.csv", index=False)

pd.set_option("display.width", 220)
print(cfg[["method", "config", "n_seeds", "acc_r2_mean", "acc_r2_sd", "acc_r2_ci_lo", "acc_r2_ci_hi",
           "acc_pearson_r_mean", "acc_mae_mean", "acc_rmse_mean", "epoch_r2_mean"]].round(4).to_string(index=False))
print()
print(comps[["comparison", "delta_r2", "delta_r2_ci_lo", "delta_r2_ci_hi", "delta_mae", "delta_mae_ci_lo", "delta_mae_ci_hi",
             "arch_better_B", "wilcoxon_p_arch", "wilcoxon_p_arch_holm", "wilcoxon_p_rows"]].round(4).to_string(index=False))
print("\nwritten: runs.csv, configs.csv, per_dataset.csv, comparisons.csv")
