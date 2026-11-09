"""best_epoch evaluated by what it is used for: does stopping at the predicted epoch (keeping the best checkpoint so
far) land on the plateau? Compared with fixed stopping fractions and with patience-based early stopping.

For each test network, its full validation curve comes from runs/paper/test_curves.csv (exported from LEMUR by
export_curves.py, with the same grouping and filtering as the pipeline). A predicted epoch "hits" if the true accuracy there is within tol of the run's peak; "regret"
is the accuracy given up by stopping at the predicted epoch instead of the true best one.
Compared against fixed-epoch and dataset-mean baselines, for every LLM/XGBoost run and ensemble.

Usage: python plateau_metric.py
"""
import numpy as np
import pandas as pd

from _paths import RESULTS as RES, RUNS as OUT

TOLS = (0.005, 0.01, 0.02)  # 0.5 / 1 / 2 accuracy points, on the 0-1 scale used by the records
MODELS = {"Qwen3-8B": "Qwen3-8B-unsloth-bnb-4bit", "Llama-3.1-8B": "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
          "Gemma-2-9B": "gemma-2-9b-it-bnb-4bit"}
CONFIGS = {"e2+P": OUT / "acc_predict", "e3+P": OUT / "acc_predict_3epoch"}

# test runs in pipeline order, with their full validation curves (exported from LEMUR by export_curves.py)
long = pd.read_csv(OUT / "paper/test_curves.csv", float_precision="round_trip")  # exact floats, as exported
test = long.drop_duplicates("test_index").sort_values("test_index").to_dict("records")
curves = {rid: g.set_index("epoch")["accuracy"].astype(float) for rid, g in long.groupby("id", sort=False)}
print("test records:", len(test), "curves:", len(curves))
for r in test:  # sanity: every curve reproduces the run's targets
    c = curves[r["id"]]
    assert abs(c.max() - r["best_accuracy"]) < 1e-9 and int(c.idxmax()) == r["epochs_to_best"], r["id"]


RNG = np.random.default_rng(0)
ARCH = np.array([r["nn"] for r in test])
CLUSTERS = [np.flatnonzero(ARCH == a) for a in pd.unique(ARCH)]
BOOT = [np.concatenate([CLUSTERS[k] for k in RNG.integers(0, len(CLUSTERS), len(CLUSTERS))]) for _ in range(2000)]


def evaluate(pred_epochs):
    """Stop at the predicted epoch and keep the best checkpoint seen so far (standard practice). Regret is the run's peak
    minus that best-so-far accuracy. CIs: cluster bootstrap over test architectures."""
    gap, used = np.empty(len(test)), np.empty(len(test))
    for i, (r, p) in enumerate(zip(test, pred_epochs)):
        c = curves[r["id"]]
        e = int(np.clip(round(float(p)), c.index.min(), c.index.max()))
        gap[i] = c.max() - c.loc[:e].max()
        used[i] = e / c.index.max()
    hit1 = gap <= 0.01
    out = {f"hit@{t*100:g}pt": float(np.mean(gap <= t)) for t in TOLS}
    out.update({"mean_regret_pts": 100 * float(gap.mean()), "median_regret_pts": 100 * float(np.median(gap)),
                "budget_used": float(used.mean())})
    out["hit@1pt_ci_lo"], out["hit@1pt_ci_hi"] = np.percentile([hit1[b].mean() for b in BOOT], [2.5, 97.5])
    lo, hi = np.percentile([100 * gap[b].mean() for b in BOOT], [2.5, 97.5])
    out["mean_regret_ci_lo"], out["mean_regret_ci_hi"] = lo, hi
    return out


def patience_stop(c, patience):
    """Classic early stopping on validation accuracy: stop once `patience` epochs pass without a new best."""
    best, since = -1.0, 0
    for e, acc in c.items():
        if acc > best:
            best, since = acc, 0
        else:
            since += 1
            if since >= patience:
                return e
    return c.index.max()


rows = []
gt = np.array([r["epochs_to_best"] for r in test])
total = np.array([r["total_epochs"] for r in test])
train_mean = pd.read_csv(OUT / "paper/tabular_3epoch.csv").query("split == 'train'")
ds_mean = train_mean.groupby("dataset").epochs_to_best.mean()
baselines = {"oracle (true best epoch)": gt, "always last epoch": total, "always epoch 25": np.full(len(test), 25),
             "train mean epoch": np.full(len(test), train_mean.epochs_to_best.mean()),
             "per-dataset train mean": np.array([ds_mean.get(r["dataset"], train_mean.epochs_to_best.mean()) for r in test])}
for name, p in baselines.items():
    rows.append({"method": "Baseline", "config": name, "seed": "-", **evaluate(p)})
# standard early stopping (observes the curve online, unlike the predictors, which decide after epoch 2 or 3)
for pat in (3, 5, 10, 20):
    rows.append({"method": "Early stopping", "config": f"patience {pat}", "seed": "-",
                 **evaluate([patience_stop(curves[r["id"]], pat) for r in test])})
# fixed stopping rules: stop at a fixed fraction of each run's budget -> the compute/regret frontier
for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    rows.append({"method": "Fixed fraction", "config": f"stop at {frac:.0%} of budget", "seed": "-", **evaluate(frac * total)})

seed_means = {}
for (model, mdir) in MODELS.items():
    for c, root in CONFIGS.items():
        P = []
        for s in (0, 42, 123):
            p = pd.read_csv(root / f"{mdir}_seed{s}/test_predictions.csv").pred_best_epoch.to_numpy(float)
            P.append(p)
            rows.append({"method": model, "config": c, "seed": s, **evaluate(p)})
        seed_means[(model, c)] = np.mean(P, 0)
        rows.append({"method": model, "config": c, "seed": "mean of 3", **evaluate(seed_means[(model, c)])})
xgb = {c: np.mean([pd.read_csv(OUT / f"paper/xgboost_grid/{stem}_seed{s}_test_predictions.csv").pred_best_epoch
                   for s in (0, 42, 123, 7, 2024)], 0) for c, stem in (("e2+P", "xgb_e2_proxies"), ("e3+P", "xgb_e3_proxies"))}
for c, p in xgb.items():
    rows.append({"method": "XGBoost", "config": c, "seed": "mean of 5", **evaluate(p)})
for c in ("e2+P", "e3+P"):
    llm = np.mean([seed_means[(m, c)] for m in MODELS], 0)
    rows.append({"method": "Ensemble", "config": f"3 LLMs {c}", "seed": "-", **evaluate(llm)})
    rows.append({"method": "Ensemble", "config": f"3 LLMs {c} & XGBoost", "seed": "-", **evaluate((llm + xgb[c]) / 2)})

res = pd.DataFrame(rows)
res.to_csv(RES / "best_epoch_plateau.csv", index=False)
pd.set_option("display.width", 200)
show = res[res.seed.isin(["-", "mean of 3", "mean of 5"])]
show = show[["method", "config", "seed", "budget_used", "hit@1pt", "hit@1pt_ci_lo", "hit@1pt_ci_hi", "mean_regret_pts",
             "mean_regret_ci_lo", "mean_regret_ci_hi", "median_regret_pts"]]
print(show.round(3).to_string(index=False))
per_seed = res[res.seed.isin([0, 42, 123])].groupby(["method", "config"])[["hit@1pt", "mean_regret_pts"]].agg(["mean", "std"])
print("\nsingle seeds (mean, sd):\n", per_seed.round(3).to_string())
