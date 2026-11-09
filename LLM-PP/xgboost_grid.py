"""XGBoost baselines on the exported cross-family table: {2,3 early epochs} x {no proxies, proxies} x seeds.

Same features as the paper's XGBoost arm (early accuracies + task/dataset/metric/max_epochs), optionally
plus the 5 normalized zero-cost proxies (missing values left as NaN; XGBoost handles them natively).
Hyperparameters match the earlier project baseline (ab/gpt/train_xgboost_baseline.py). Early stopping
uses the validation split, exactly like the LLM runs. Scored against targets rounded to 2 decimals,
because the LLM ground truth is rounded the same way.

Usage: python xgboost_grid.py <tabular.csv> <out_dir> [<config to skip> ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor

SEEDS = [0, 42, 123, 7, 2024]
PROXIES = ["proxy_synflow", "proxy_nwot", "proxy_grad_norm", "proxy_log_params", "proxy_depth"]
CATEGORICAL = ["task", "dataset", "metric"]


def metrics(pred, gt):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    ss_tot = np.sum((gt - gt.mean()) ** 2)
    return {"n": len(gt), "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))), "mae": float(np.mean(np.abs(pred - gt))),
            "pearson_r": float(pearsonr(pred, gt)[0]), "spearman_r": float(spearmanr(pred, gt)[0]),
            "r2": float(1 - np.sum((gt - pred) ** 2) / ss_tot)}


def features(df, epochs, proxies, columns=None):
    numeric = ["total_epochs"] + [f"accuracy_epoch_{e}" for e in range(1, epochs + 1)] + (PROXIES if proxies else [])
    x = pd.concat([df[numeric].apply(pd.to_numeric, errors="coerce"),
                   pd.get_dummies(df[CATEGORICAL].astype(str), prefix=CATEGORICAL)], axis=1).astype(float)
    return x if columns is None else x.reindex(columns=columns, fill_value=0.0)


src, out = Path(sys.argv[1]), Path(sys.argv[2])
skip = set(sys.argv[3:])
out.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(src, keep_default_na=True)
parts = {s: data[data.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
print({s: len(d) for s, d in parts.items()})
gt = {"best_accuracy": parts["test"]["best_accuracy"].astype(float).round(2).to_numpy(),
      "best_epoch": parts["test"]["epochs_to_best"].astype(float).to_numpy()}
target_col = {"best_accuracy": "best_accuracy", "best_epoch": "epochs_to_best"}

summary = []
for epochs in (2, 3):
    for proxies in (False, True):
        name = f"xgb_e{epochs}{'_proxies' if proxies else ''}"
        if name in skip:
            continue
        x_tr = features(parts["train"], epochs, proxies)
        x_va = features(parts["val"], epochs, proxies, list(x_tr.columns))
        x_te = features(parts["test"], epochs, proxies, list(x_tr.columns))
        for seed in SEEDS:
            preds = {"dataset": parts["test"]["dataset"].to_numpy()}
            row = {"config": name, "epochs": epochs, "proxies": proxies, "seed": seed}
            for target in ("best_accuracy", "best_epoch"):
                model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, early_stopping_rounds=20, eval_metric="rmse", random_state=seed)
                model.fit(x_tr, parts["train"][target_col[target]].astype(float),
                          eval_set=[(x_va, parts["val"][target_col[target]].astype(float))], verbose=False)
                p = model.predict(x_te)
                preds[f"pred_{target}"], preds[f"gt_{target}"] = p, gt[target]
                for k, v in metrics(p, gt[target]).items():
                    row[f"{target}_{k}"] = v
            pd.DataFrame(preds).to_csv(out / f"{name}_seed{seed}_test_predictions.csv", index=False)
            summary.append(row)
            print(f"{name:16s} seed={seed:5d} acc R2={row['best_accuracy_r2']:.4f} r={row['best_accuracy_pearson_r']:.4f} "
                  f"MAE={row['best_accuracy_mae']:.4f} | epoch R2={row['best_epoch_r2']:.4f}")

df = pd.DataFrame(summary)
agg = df.groupby("config")[["best_accuracy_r2", "best_accuracy_pearson_r", "best_accuracy_mae", "best_epoch_r2"]].agg(["mean", "std"])
print("\n", agg.round(4).to_string())
