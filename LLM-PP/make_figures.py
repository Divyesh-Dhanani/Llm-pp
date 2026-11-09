"""Generate every manuscript figure from results/*.csv and the raw prediction files.

Writes results/figures/*.pdf. Print-oriented: validated three-slot categorical palette (blue, orange, aqua)
plus neutral gray for baselines, with a distinct marker per method family so identity never rests on
colour alone (each figure also has a matching table in the paper).

Usage: python make_figures.py
"""

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # no timestamp in the PDFs, so they reproduce byte for byte
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from _paths import FIGURES as FIG, RESULTS as RES, RUNS as OUT

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, GRAY = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#8a8984"
FAMILY = {"XGBoost": (BLUE, "s"), "LLM": (ORANGE, "o"), "Ensemble": (AQUA, "D"), "Trivial": (GRAY, "^")}
FAMILY_OF = {"LC-PFN": "Trivial"}  # training-free baselines share the neutral style
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
                     "pdf.fonttype": 42, "lines.linewidth": 1.5})

cfg = pd.read_csv(RES / "configs.csv")
LLMS = ("Qwen3-8B", "Llama-3.1-8B", "Gemma-2-9B")
NICE = {"e2": "2 ep.", "e2+P": "2 ep. + proxies", "e3": "3 ep.", "e3+P": "3 ep. + proxies",
        "e2+P+code": "2 ep. + proxies + code", "e3+P+code": "3 ep. + proxies + code",
        "e3+P (answer-only loss)": "3 ep. + proxies, answer-only", "e3+P+code (answer-only loss)": "3 ep. + proxies + code, answer-only",
        "e3+P [rerun]": "3 ep. + proxies (rerun)\u2020", "e3+P (answer-only loss) [batch 1x32]": "3 ep. + proxies, answer-only, 1\u00d732\u2020"}


def family(method):
    return "LLM" if method in LLMS else FAMILY_OF.get(method, method)


# ---------------------------------------------------------------- Fig. 1: main results dot plot
rows = [("Trivial", "per-dataset train mean", "Per-dataset training mean"),
        ("Trivial", "last early accuracy (epoch 3)", "Accuracy at epoch 3"),
        ("LC-PFN", "e3", "LC-PFN (curve extrapolation), 3 ep.")]
rows += [("XGBoost", c, f"XGBoost, {NICE[c]}") for c in ("e2", "e2+P", "e3", "e3+P")]
rows += [(m, c, f"{m}, {NICE[c]}") for c in ("e2+P", "e3+P") for m in LLMS]
rows += [("Qwen3-8B", c, f"Qwen3-8B, {NICE[c]}" + ("\u2020" if "code" in c else ""))
         for c in ("e2+P+code", "e3+P [rerun]", "e3+P+code", "e3+P (answer-only loss)", "e3+P (answer-only loss) [batch 1x32]",
                   "e3+P+code (answer-only loss)")]
rows += [("Llama-3.1-8B", "e3+P+code", "Llama-3.1-8B, 3 ep. + proxies + code\u2020")]
rows += [("Ensemble", "3 LLMs e3+P", "Ensemble: 3 LLMs, 3 ep. + proxies"),
         ("Ensemble", "3 LLMs e3+P & XGBoost e3+P", "Ensemble: 3 LLMs + XGBoost, 3 ep. + proxies")]
rows = [r for r in rows if not cfg[(cfg.method == r[0]) & (cfg.config == r[1])].empty]
fig, ax = plt.subplots(figsize=(6.8, 0.24 * len(rows) + 0.9))
for i, (m, c, lab) in enumerate(rows):
    r = cfg[(cfg.method == m) & (cfg.config == c)].iloc[0]
    col, mk = FAMILY[family(m)]
    y = len(rows) - 1 - i
    ax.plot([r.acc_r2_ci_lo, r.acc_r2_ci_hi], [y, y], color=col, alpha=0.35, linewidth=3, solid_capstyle="round")
    if r.n_seeds > 1:
        ax.errorbar(r.acc_r2_mean, y, xerr=r.acc_r2_sd, fmt="none", ecolor=col, elinewidth=1.2, capsize=2)
    ax.plot(r.acc_r2_mean, y, mk, color=col, markersize=5, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax.text(1.005, y, f"{r.acc_r2_mean:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[2] for r in rows][::-1])
ax.set_xlabel("Best-accuracy $R^2$ on the cross-family test set")
ax.set_xlim(0, 1)
ax.grid(axis="y", visible=False)
handles = [plt.Line2D([], [], marker=mk, color=col, linestyle="", markersize=5, label="Baseline" if k == "Trivial" else k)
           for k, (col, mk) in FAMILY.items()]
ax.legend(handles=handles, loc="lower left", ncol=4, fontsize=7, bbox_to_anchor=(0, 1.0))
fig.text(0.01, 0.005, "Dot: mean over seeds; dark bar: $\\pm$1 SD over seeds; light band: 95% cluster-bootstrap CI over test architectures. "
         "\u2020 data order fixed across seeds.",
         fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 0.97, 1))
fig.savefig(FIG / "main_results.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 2: forest plot of paired effects
cmp_ = pd.read_csv(RES / "comparisons.csv")
keep = [c for c in cmp_.comparison if not c.startswith("Qwen vs") and not c.startswith("Gemma vs Qwen")]
cmp_ = cmp_[cmp_.comparison.isin(keep)].reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(6.8, 0.22 * len(cmp_) + 0.9), sharey=True)
for ax, (est, lo, hi, xlabel, better) in zip(axes, [("delta_r2", "delta_r2_ci_lo", "delta_r2_ci_hi", "$\\Delta R^2$ (B $-$ A)", "B better $\\rightarrow$"),
                                                     ("delta_mae", "delta_mae_ci_lo", "delta_mae_ci_hi", "$\\Delta$MAE (B $-$ A)", "$\\leftarrow$ B better")]):
    for i, r in cmp_.iterrows():
        y = len(cmp_) - 1 - i
        sig = (r[lo] > 0) or (r[hi] < 0)
        col = INK if sig else MUTED
        ax.plot([r[lo], r[hi]], [y, y], color=col, linewidth=1.2)
        ax.plot(r[est], y, "o", color=col, markersize=4 if sig else 3.5, markerfacecolor=col if sig else "white")
    ax.axvline(0, color=AXIS, linewidth=1, zorder=0)
    ax.set_xlabel(xlabel)
    ax.text(1, 1.01, better, transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(cmp_)))
axes[0].set_yticklabels(list(cmp_.comparison)[::-1])
fig.text(0.01, 0.005, "95% cluster-bootstrap CIs over the 15 test architectures; filled = CI excludes 0.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "forest.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig.: pooled comparisons over the held-out families
fc = pd.read_csv(RES / "fold_comparisons.csv")
fc = fc[fc.scope == "all splits"].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(3.4, 0.24 * len(fc) + 0.8))
for i, r in fc.iterrows():
    y = len(fc) - 1 - i
    sig = r.holm_p < 0.05
    col = INK if sig else MUTED
    ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, linewidth=1.2)
    ax.plot(r.delta_mae, y, "o", color=col, markersize=4 if sig else 3.5, markerfacecolor=col if sig else "white")
ax.axvline(0, color=AXIS, linewidth=1, zorder=0)
ax.set_yticks(range(len(fc)))
ax.set_yticklabels([f"{c} ({n})" for c, n in zip(fc.comparison, fc.n_arch)][::-1], fontsize=6.5)
ax.set_xlabel("Change in MAE (B $-$ A)")
ax.set_xticks([-0.04, -0.02, 0.0])
ax.tick_params(axis="x", labelsize=6.5)
ax.text(1, 1.01, "$\\leftarrow$ B better", transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIG / "forest_pooled.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: per-dataset R2 heatmap
pdset = pd.read_csv(RES / "per_dataset.csv")
cols = [("XGBoost", "e3+P", "XGBoost"), ("Qwen3-8B", "e3+P", "Qwen3-8B"), ("Llama-3.1-8B", "e3+P", "Llama-3.1-8B"),
        ("Gemma-2-9B", "e3+P", "Gemma-2-9B"), ("Qwen3-8B", "e3+P+code", "Qwen3-8B + code"),
        ("Ensemble", "3 LLMs e3+P & XGBoost e3+P", "3 LLMs + XGBoost")]
cols = [c for c in cols if not pdset[(pdset.method == c[0]) & (pdset.config == c[1])].empty]
dsets = sorted(pdset.dataset.unique())
M = np.array([[pdset[(pdset.method == m) & (pdset.config == c) & (pdset.dataset == d)].r2_mean.iloc[0] for m, c, _ in cols] for d in dsets])
cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
fig, ax = plt.subplots(figsize=(4.6, 3.2))
# rounded values and no resampling: the embedded image is then the same on every platform
im = ax.imshow(np.clip(M, -1, 1).round(9), cmap=cmap, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto",
               interpolation="none")
for i in range(len(dsets)):
    for j in range(len(cols)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(M[i, j]) > 0.6 else INK)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([c[2] for c in cols], rotation=30, ha="right")
n = pdset.drop_duplicates("dataset").set_index("dataset").n
ax.set_yticks(range(len(dsets)))
ax.set_yticklabels([f"{d} (n={int(n[d])})" for d in dsets])
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("$R^2$ (clipped to [$-$1, 1])", color=INK2)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "per_dataset.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: best epoch, compute vs regret
plat = pd.read_csv(RES / "best_epoch_plateau.csv")
fixed = plat[plat.method == "Fixed fraction"].sort_values("budget_used")
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(fixed.budget_used, fixed.mean_regret_pts, color=GRAY, linewidth=1.5, marker=".", markersize=4, label="Fixed stop at k% of budget")
pat = plat[plat.method == "Early stopping"].sort_values("budget_used")
ax.plot(pat.budget_used, pat.mean_regret_pts, color=INK2, linewidth=1.2, linestyle="--", marker="v", markersize=4,
        label="Early stopping (patience 3, 5, 10, 20)")
# (method, config, label, family, label offset in points)
pts = [("XGBoost", "e3+P", "XGBoost", "XGBoost", (-8, 7)), ("Qwen3-8B", "e3+P", "Qwen", "LLM", (-6, 7)),
       ("Llama-3.1-8B", "e3+P", "Llama", "LLM", (5, 5)), ("Gemma-2-9B", "e3+P", "Gemma", "LLM", (6, 6)),
       ("Ensemble", "3 LLMs e3+P & XGBoost", "3 LLMs + XGB", "Ensemble", (-80, -30)),
       ("Ensemble", "3 LLMs e3+P", "3 LLMs", "Ensemble", (14, 34))]
for m, c, lab, fam, off in pts:
    r = plat[(plat.method == m) & (plat.config == c) & (plat.seed.astype(str).isin(["-", "mean of 3", "mean of 5"]))]
    if r.empty:
        continue
    r = r.iloc[0]
    col, mk = FAMILY[fam]
    ax.plot(r.budget_used, r.mean_regret_pts, mk, color=col, markersize=6, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    leader = dict(arrowstyle="-", color=MUTED, linewidth=0.6) if abs(off[1]) > 20 else None
    ax.annotate(lab, (r.budget_used, r.mean_regret_pts), xytext=off, textcoords="offset points", fontsize=6.5, color=INK2,
                arrowprops=leader)
last = plat[(plat.method == "Baseline") & (plat.config == "always last epoch")].iloc[0]
ax.plot(last.budget_used, last.mean_regret_pts, "^", color=GRAY, markersize=6)
ax.annotate("train to\nthe end", (last.budget_used, last.mean_regret_pts), xytext=(-12, 8), textcoords="offset points", fontsize=6.5, color=INK2)
ax.set_xlabel("Share of training budget used")
ax.set_ylabel("Mean regret (accuracy points)")
ax.set_xlim(0.5, 1.04)
ax.set_ylim(0, 2.2)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(loc="upper right", fontsize=6.5)
fig.tight_layout()
fig.savefig(FIG / "best_epoch_frontier.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 5: predicted vs true
gt = None
panels = []
for title, files in [("XGBoost, 3 ep. + proxies", [OUT / f"paper/xgboost_grid/xgb_e3_proxies_seed{s}_test_predictions.csv" for s in (0, 42, 123, 7, 2024)]),
                     ("Gemma-2-9B, 3 ep. + proxies", [OUT / f"acc_predict_3epoch/gemma-2-9b-it-bnb-4bit_seed{s}/test_predictions.csv" for s in (0, 42, 123)]),
                     ("Qwen3-8B, 3 ep. + proxies + code", [OUT / f"acc_predict_ablation/code3p/Qwen3-8B-unsloth-bnb-4bit_seed{s}/test_predictions.csv" for s in (0, 42, 123)])]:
    if not all(p.exists() for p in files):
        continue
    dfs = [pd.read_csv(p) for p in files]
    gt = dfs[0].gt_best_accuracy.to_numpy(float)
    panels.append((title, np.mean([d.pred_best_accuracy for d in dfs], 0), dfs[0].dataset.to_numpy()))
fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.4), sharey=True)
for ax, (title, pred, ds) in zip(np.atleast_1d(axes), panels):
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, zorder=0)
    ax.scatter(gt, pred, s=6, color=BLUE, alpha=0.45, linewidths=0)
    big = np.abs(pred - gt) > 0.2
    ax.scatter(gt[big], pred[big], s=8, facecolors="none", edgecolors=ORANGE, linewidths=0.6)
    ax.set_title(f"{title}\n|error| > 0.2: {big.mean():.0%} of runs", fontsize=7, color=INK)
    ax.set_xlabel("True best accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
np.atleast_1d(axes)[0].set_ylabel("Predicted (seed mean)")
fig.tight_layout()
fig.savefig(FIG / "pred_vs_true.pdf")
plt.close(fig)
print("figures:", sorted(p.name for p in FIG.iterdir()))
