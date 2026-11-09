"""Robustness analyses requested in review round 1.

R7  Sensitivity to the two dominant test architectures (RNN on WikiText, ResNet-LSTM on COCO: 42% of the test set):
    metrics and key paired effects on the full test set and with each / both removed; per-architecture error table.
R8  Within-dataset ranking, the quantity that matters for model selection: Kendall tau between predicted and true best
    accuracy inside each dataset, top-5 recall, and the selection regret of picking the predicted-best run.

Reads the same raw prediction files as collect_results.py. Writes results/{sensitivity,per_architecture,
ranking}.csv.

Usage: python robustness.py
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from _paths import RESULTS as RES, RUNS as OUT
SEEDS = [0, 42, 123]
LLM_DIRS = {"Qwen3-8B": "Qwen3-8B-unsloth-bnb-4bit", "Llama-3.1-8B": "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "Gemma-2-9B": "gemma-2-9b-it-bnb-4bit"}
ROOTS = {"e2+P": OUT / "acc_predict", "e3+P": OUT / "acc_predict_3epoch",
         "e3+P+code": OUT / "acc_predict_ablation/code3p",
         # matched pair for the code effect: same loss (answer-only), batch layout and data order
         "e3+P (answer-only, 1x32)": OUT / "acc_predict_ablation/e3p_ro_b1",
         "e3+P+code (answer-only)": OUT / "acc_predict_ablation/code3p_ro"}

test = pd.read_csv(OUT / "paper/tabular_3epoch.csv").query("split == 'test'").reset_index(drop=True)
arch = test.nn.to_numpy()
dataset = test.dataset.to_numpy()


def load(path):
    df = pd.read_csv(path)
    assert (df.dataset.to_numpy() == dataset).all()
    return df.pred_best_accuracy.to_numpy(float), df.gt_best_accuracy.to_numpy(float)


# method -> list of prediction vectors (one per seed); gt shared
methods, gt = {}, None
for m, d in LLM_DIRS.items():
    for c in ("e2+P", "e3+P"):
        methods[f"{m}, {c}"] = [load(ROOTS[c] / f"{d}_seed{s}/test_predictions.csv")[0] for s in SEEDS]
code_files = [ROOTS["e3+P+code"] / f"Qwen3-8B-unsloth-bnb-4bit_seed{s}/test_predictions.csv" for s in SEEDS]
if all(f.exists() for f in code_files):  # source-code runs are optional (reruns with the fixed sequence limit)
    methods["Qwen3-8B, e3+P+code"] = [load(f)[0] for f in code_files]
for c in ("e3+P (answer-only, 1x32)", "e3+P+code (answer-only)"):
    methods[f"Qwen3-8B, {c}"] = [load(ROOTS[c] / f"Qwen3-8B-unsloth-bnb-4bit_seed{s}/test_predictions.csv")[0] for s in SEEDS]
for c, stem in (("e3", "xgb_e3"), ("e3+P", "xgb_e3_proxies")):
    methods[f"XGBoost, {c}"] = [load(OUT / f"paper/xgboost_grid/{stem}_seed{s}_test_predictions.csv")[0] for s in (0, 42, 123, 7, 2024)]
p, gt = load(OUT / "paper/lcpfn/lcpfn_e3_seed0_test_predictions.csv")
methods["LC-PFN, e3"] = [p]
ens_llm = np.mean([np.mean(methods[f"{m}, e3+P"], 0) for m in LLM_DIRS], 0)
methods["Ensemble: 3 LLMs + XGBoost, e3+P"] = [(3 * ens_llm + np.mean(methods["XGBoost, e3+P"], 0)) / 4]


def r2(p, g):
    return 1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum()


# ---------------------------------------------------------------- R7: subsets
subsets = {"all 15 architectures": np.ones(len(gt), bool),
           "without RNN (WikiText)": arch != "RNN",
           "without ResNet-LSTM (COCO)": arch != "RESNETLSTM",
           "without both": (arch != "RNN") & (arch != "RESNETLSTM")}
rows = []
for name, preds in methods.items():
    for sub, m in subsets.items():
        rows.append({"method": name, "subset": sub, "n_runs": int(m.sum()), "n_arch": len(set(arch[m])),
                     "r2": np.mean([r2(p[m], gt[m]) for p in preds]), "mae": np.mean([np.abs(p[m] - gt[m]).mean() for p in preds])})
sens = pd.DataFrame(rows)
effects = {"source code, Qwen3-8B (3 ep., answer-only)": ("Qwen3-8B, e3+P (answer-only, 1x32)", "Qwen3-8B, e3+P+code (answer-only)"),
           "3rd epoch, Gemma-2-9B": ("Gemma-2-9B, e2+P", "Gemma-2-9B, e3+P"),
           "3rd epoch, Qwen3-8B": ("Qwen3-8B, e2+P", "Qwen3-8B, e3+P"),
           "Gemma-2-9B vs XGBoost (e3+P)": ("XGBoost, e3+P", "Gemma-2-9B, e3+P"),
           "Ensemble vs XGBoost (e3+P)": ("XGBoost, e3+P", "Ensemble: 3 LLMs + XGBoost, e3+P")}
eff_rows = []
for ename, (a, b) in effects.items():
    for sub in subsets:
        ra = sens[(sens.method == a) & (sens.subset == sub)].iloc[0]
        rb = sens[(sens.method == b) & (sens.subset == sub)].iloc[0]
        eff_rows.append({"effect": ename, "subset": sub, "delta_r2": rb.r2 - ra.r2, "delta_mae": rb.mae - ra.mae})
eff = pd.DataFrame(eff_rows)
sens.to_csv(RES / "sensitivity.csv", index=False)
eff.to_csv(RES / "sensitivity_effects.csv", index=False)

# per-architecture MAE (seed mean)
pa = []
for a in pd.unique(arch):
    m = arch == a
    row = {"architecture": a, "n_runs": int(m.sum()), "datasets": len(set(dataset[m]))}
    for name in [n for n in ("XGBoost, e3+P", "Gemma-2-9B, e3+P", "Qwen3-8B, e3+P", "Qwen3-8B, e3+P+code",
                             "Ensemble: 3 LLMs + XGBoost, e3+P") if n in methods]:
        row[name] = np.mean([np.abs(p[m] - gt[m]).mean() for p in methods[name]])
    pa.append(row)
pd.DataFrame(pa).sort_values("n_runs", ascending=False).to_csv(RES / "per_architecture.csv", index=False)

# ---------------------------------------------------------------- R8: within-dataset ranking
K = 5
rank_rows = []
for name, preds in methods.items():
    per_ds = []
    for d in sorted(set(dataset)):
        m = dataset == d
        if m.sum() < 2 * K:
            continue
        g = gt[m]
        taus, recalls, regrets = [], [], []
        for p in preds:
            q = p[m]
            taus.append(kendalltau(q, g)[0])
            # true accuracies are rounded, so the true top-K is tie-aware: every run at least as good as the K-th best;
            # ties among predictions are broken by run order (a stable sort), so the result does not depend on the platform
            recalls.append(np.mean(g[np.argsort(-q, kind="stable")[:K]] >= np.sort(g)[-K]))
            regrets.append(g.max() - g[np.argmax(q)])
        per_ds.append({"dataset": d, "n": int(m.sum()), "tau": np.nanmean(taus), "top5_recall": np.mean(recalls),
                       "selection_regret": np.mean(regrets)})
    per_ds = pd.DataFrame(per_ds)
    w = per_ds.n / per_ds.n.sum()
    rank_rows.append({"method": name, "datasets": len(per_ds), "kendall_tau": (w * per_ds.tau).sum(),
                      "top5_recall": (w * per_ds.top5_recall).sum(), "selection_regret_pts": 100 * (w * per_ds.selection_regret).sum()})
    slug = re.sub(r"[^A-Za-z0-9.]+", "_", name.replace("+", "p")).strip("_")
    per_ds.assign(method=name).to_csv(RES / f"ranking_per_dataset_{slug}.csv", index=False)
rank = pd.DataFrame(rank_rows).sort_values("kendall_tau", ascending=False)
rank.to_csv(RES / "ranking.csv", index=False)

pd.set_option("display.width", 220)
print(sens.pivot_table(index="method", columns="subset", values="r2").round(3).to_string())
print()
print(eff.pivot_table(index="effect", columns="subset", values="delta_r2").round(3).to_string())
print()
print(pd.DataFrame(pa).sort_values("n_runs", ascending=False).round(3).to_string(index=False))
print()
print(rank.round(3).to_string(index=False))
