"""Generate every LaTeX table of the manuscript from results/*.csv (no hand-typed numbers).

Writes results/tables/*.tex (booktabs). Rows whose runs do not exist yet are skipped, so the script can
be rerun as results arrive.

Usage: python make_tables.py
"""

import numpy as np
import pandas as pd

from _paths import RESULTS as RES, RUNS, TABLES as TAB

cfg = pd.read_csv(RES / "configs.csv")
cmp_ = pd.read_csv(RES / "comparisons.csv")
plat = pd.read_csv(RES / "best_epoch_plateau.csv")

LABEL = {"e2": "2 ep.", "e2+P": "2 ep. + proxies", "e3": "3 ep.", "e3+P": "3 ep. + proxies",
         "e2+P+code": "2 ep. + proxies + code", "e3+P+code": "3 ep. + proxies + code",
         "e3+P (answer-only loss)": "3 ep. + proxies, answer-only loss",
         "e3+P+code (answer-only loss)": "3 ep. + proxies + code, answer-only loss",
         "e3+P [rerun]": "3 ep. + proxies (rerun)",
         "e3+P (answer-only loss) [rerun]": "3 ep. + proxies, answer-only loss (rerun)",
         "e3+P (answer-only loss) [batch 1x32]": "3 ep. + proxies, answer-only loss, batch $1\\times32$"}


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}".replace("-", "$-$")


def pm(m, s, d=3):
    return f(m, d) if (pd.isna(s) or s == 0) else f"{f(m, d)}\\,$\\pm$\\,{f(s, d)}"


def write(name, body, cols, caption, label, size="\\small", fit=False, place="t"):
    """fit=True scales the table to the text width (for tables that are wider than a page). Every table is a table*, which
    spans both columns of the two-column main paper and is an ordinary table in the one-column supplement."""
    tabular = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", *body, "\\bottomrule", "\\end{tabular}"]
    if fit:
        tabular = ["\\resizebox{\\textwidth}{!}{%", *tabular, "}"]
    place = "!htbp" if place == "H" else place  # [H] is not available for table*
    tex = [f"\\begin{{table*}}[{place}]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           *tabular, "\\end{table*}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(tex))
    print("wrote", name)


def row_of(method, config):
    r = cfg[(cfg.method == method) & (cfg.config == config)]
    return None if r.empty else r.iloc[0]


# ---------------------------------------------------------------- Table: main results
groups = [
    ("Trivial baselines", [("Trivial", "train mean", "Training-set mean"),
                           ("Trivial", "per-dataset train mean", "Per-dataset training mean"),
                           ("Trivial", "last early accuracy (epoch 3)", "Accuracy at epoch 3")]),
    ("Learning-curve extrapolation (LC-PFN, pre-trained)", [("LC-PFN", c, LABEL[c]) for c in ("e2", "e3")]),
    ("XGBoost (5 seeds)", [("XGBoost", c, LABEL[c]) for c in ("e2", "e2+P", "e3", "e3+P")]),
    ("Fine-tuned LLMs (3 seeds)", [(m, c, f"{m}, {LABEL[c]}") for m in ("Qwen3-8B", "Llama-3.1-8B", "Gemma-2-9B")
                                   for c in ("e2+P", "e3+P")]),
    ("Untuned equal-weight ensembles", [("Ensemble", "Gemma-2-9B e3+P: seed average", "Gemma-2-9B, 3 ep. + proxies, 3 seeds averaged"),
                                        ("Ensemble", "3 LLMs e3+P", "3 LLMs, 3 ep. + proxies"),
                                        ("Ensemble", "3 LLMs e3+P & XGBoost e3+P", "3 LLMs + XGBoost, 3 ep. + proxies")]),
]
body = ["Method & $R^2$ & 95\\% CI & Pearson $r$ & MAE & RMSE \\\\", "\\midrule"]
for gname, rows in groups:
    present = [(m, c, lab) for m, c, lab in rows if row_of(m, c) is not None]
    if not present:
        continue
    body.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{gname}}}}} \\\\")
    for m, c, lab in present:
        r = row_of(m, c)
        body.append(f"\\quad {lab} & {pm(r.acc_r2_mean, r.acc_r2_sd)} & [{f(r.acc_r2_ci_lo)}, {f(r.acc_r2_ci_hi)}] & "
                    f"{f(r.acc_pearson_r_mean)} & {f(r.acc_mae_mean)} & {f(r.acc_rmse_mean)} \\\\")
    body.append("\\midrule")
body = body[:-1]
write("main_results", body, "lccccc",
      "Best-accuracy prediction on the cross-family test set (657 runs of 15 unseen architectures). $R^2$: mean\\,$\\pm$\\,SD "
      "over seeds; other metrics: seed mean (per-seed values in \\cref{S-tab:per-seed}). The 95\\% CI is a cluster bootstrap over "
      "test architectures of the seed-averaged $R^2$. Accuracy is on a 0--1 scale.",
      "tab:main", size="\\footnotesize", fit=True)

# ---------------------------------------------------------------- Table: input ablation (Qwen3-8B, all arms; Llama code)
# (model, config, loss, per-device batch x accumulation, data order across seeds)
arms = [("Qwen3-8B", "e2+P", "full", "4$\\times$8", "per seed"), ("Qwen3-8B", "e2+P+code", "full", "1$\\times$32", "fixed"),
        ("Qwen3-8B", "e3", "full", "4$\\times$8", "per seed"), ("Qwen3-8B", "e3+P", "full", "4$\\times$8", "per seed"),
        ("Qwen3-8B", "e3+P [rerun]", "full", "4$\\times$8", "fixed"), ("Qwen3-8B", "e3+P+code", "full", "1$\\times$32", "fixed"),
        ("Qwen3-8B", "e3+P (answer-only loss)", "answer", "4$\\times$8", "per seed"),
        ("Qwen3-8B", "e3+P (answer-only loss) [rerun]", "answer", "4$\\times$8", "fixed"),
        ("Qwen3-8B", "e3+P (answer-only loss) [batch 1x32]", "answer", "1$\\times$32", "fixed"),
        ("Qwen3-8B", "e3+P+code (answer-only loss)", "answer", "1$\\times$32", "fixed"),
        ("Llama-3.1-8B", "e3+P", "full", "4$\\times$8", "per seed"), ("Llama-3.1-8B", "e3+P+code", "full", "1$\\times$32", "fixed")]
INPUTS = {"e2+P": "2 ep. + proxies", "e2+P+code": "2 ep. + proxies + code", "e3": "3 ep.", "e3+P+code": "3 ep. + proxies + code",
          "e3+P+code (answer-only loss)": "3 ep. + proxies + code"}
body = ["Model & Inputs & Loss & Batch & Data order & $R^2$ & MAE & RMSE & best-epoch $R^2$ \\\\", "\\midrule"]
prev = None
for m, c, loss, batch, order in arms:
    r = row_of(m, c)
    if r is None:
        continue
    if prev and m != prev:
        body.append("\\midrule")
    prev = m
    body.append(f"{m} & {INPUTS.get(c, '3 ep. + proxies')} & {loss} & {batch} & {order} & {pm(r.acc_r2_mean, r.acc_r2_sd)} & "
                f"{pm(r.acc_mae_mean, r.acc_mae_sd)} & {pm(r.acc_rmse_mean, r.acc_rmse_sd)} & {f(r.epoch_r2_mean, 2)} \\\\")
write("qwen_ablation", body, "llllccccc",
      "Input and training ablation (mean\\,$\\pm$\\,SD over seeds 0, 42, 123). \\emph{Loss}: over the full chat text or the "
      "answer only. \\emph{Batch}: per-device batch $\\times$ accumulation steps (effective batch 32 throughout). \\emph{Data "
      "order}: shuffled per seed, or identical for all seeds (every run trained after the switch to an explicit trainer "
      "configuration, see the main paper); with a fixed order the seeds differ only in initialisation and dropout, so their SD "
      "understates run-to-run variation.", "tab:qwen-ablation", size="\\footnotesize", fit=True)

# ---------------------------------------------------------------- Table: paired comparisons
body = ["Comparison (B $-$ A) & $\\Delta R^2$ & 95\\% CI & $\\Delta$MAE & 95\\% CI & Arch.\\ improved & $p$ & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule"]
for _, r in cmp_.iterrows():
    body.append(f"{r.comparison} & {f(r.delta_r2)} & [{f(r.delta_r2_ci_lo)}, {f(r.delta_r2_ci_hi)}] & {f(r.delta_mae)} & "
                f"[{f(r.delta_mae_ci_lo)}, {f(r.delta_mae_ci_hi)}] & {int(r.arch_better_B)}/{int(r.n_arch)} & "
                f"{f(r.wilcoxon_p_arch)} & {f(r.wilcoxon_p_arch_holm)} \\\\")
write("comparisons", body, "lccccccc",
      "Paired comparisons on best-accuracy prediction. $\\Delta$ is B minus A (positive $\\Delta R^2$ and negative "
      "$\\Delta$MAE favour B). CIs are cluster bootstraps over the 15 test architectures; $p$ is a Wilcoxon signed-rank test on "
      "per-architecture mean absolute error, Holm-corrected across all comparisons.", "tab:comparisons", size="\\scriptsize", fit=True)

# ---------------------------------------------------------------- Table: per-dataset R2
pdset = pd.read_csv(RES / "per_dataset.csv")
cols = [("XGBoost", "e3+P", "XGB"), ("Qwen3-8B", "e3+P", "Qwen"), ("Llama-3.1-8B", "e3+P", "Llama"),
        ("Gemma-2-9B", "e3+P", "Gemma"), ("Qwen3-8B", "e3+P+code", "Qwen+code"), ("Ensemble", "3 LLMs e3+P & XGBoost e3+P", "Ens.")]
cols = [c for c in cols if not pdset[(pdset.method == c[0]) & (pdset.config == c[1])].empty]
arch = pd.read_csv(RUNS / "paper/tabular_3epoch.csv").query("split == 'test'").groupby("dataset").nn.nunique()
body = ["Dataset & $n$ & Arch. & " + " & ".join(c[2] for c in cols) + " \\\\", "\\midrule"]
for d in sorted(pdset.dataset.unique()):
    vals = []
    for m, c, _ in cols:
        r = pdset[(pdset.method == m) & (pdset.config == c) & (pdset.dataset == d)].iloc[0]
        vals.append(f(r.r2_mean, 2))
    n = int(pdset[pdset.dataset == d].n.iloc[0])
    body.append(f"{d} & {n} & {arch.get(d, 0)} & " + " & ".join(vals) + " \\\\")
write("per_dataset", body, "lcc" + "c" * len(cols),
      "Per-dataset best-accuracy $R^2$ (3 ep.\\ + proxies unless marked; LLMs averaged over seeds). "
      "\\emph{Arch.} is the number of distinct test architectures behind each dataset: coco and wikitext are a single "
      "architecture each, and places365 has only 15 runs. The Qwen and Qwen+code columns come from different data-order "
      "regimes (\\cref{tab:qwen-ablation}) and are not a matched comparison.", "tab:per-dataset")

# ---------------------------------------------------------------- Table: best epoch as a stopping decision
want = [("Baseline", "oracle (true best epoch)", "Oracle (true best epoch)"), ("Baseline", "always last epoch", "Always train to the end"),
        ("Baseline", "per-dataset train mean", "Per-dataset training mean"),
        ("Fixed fraction", "stop at 70% of budget", "Stop at 70\\% of budget"), ("Fixed fraction", "stop at 80% of budget", "Stop at 80\\% of budget"),
        ("Fixed fraction", "stop at 90% of budget", "Stop at 90\\% of budget"),
        ("Early stopping", "patience 5", "Early stopping, patience 5"), ("Early stopping", "patience 10", "Early stopping, patience 10"),
        ("Early stopping", "patience 20", "Early stopping, patience 20"),
        ("XGBoost", "e3+P", "XGBoost, 3 ep. + proxies"), ("Qwen3-8B", "e3+P", "Qwen3-8B, 3 ep. + proxies"),
        ("Llama-3.1-8B", "e3+P", "Llama-3.1-8B, 3 ep. + proxies"), ("Gemma-2-9B", "e3+P", "Gemma-2-9B, 3 ep. + proxies"),
        ("Ensemble", "3 LLMs e3+P", "3 LLMs"), ("Ensemble", "3 LLMs e3+P & XGBoost", "3 LLMs + XGBoost")]
body = ["Stopping rule & Budget used & Hit within 1 pt & 95\\% CI & Mean regret (pts) & 95\\% CI & Median regret \\\\", "\\midrule"]
for m, c, lab in want:
    r = plat[(plat.method == m) & (plat.config == c) & (plat.seed.astype(str).isin(["-", "mean of 3", "mean of 5"]))]
    if r.empty:
        continue
    r = r.iloc[0]
    pct = lambda x: f"{x:.0%}".replace("%", "\\%")
    ci_h = f"[{pct(r['hit@1pt_ci_lo'])}, {pct(r['hit@1pt_ci_hi'])}]"
    ci_r = f"[{f(r.mean_regret_ci_lo, 2)}, {f(r.mean_regret_ci_hi, 2)}]"
    body.append(f"{lab} & {pct(r.budget_used)} & {pct(r['hit@1pt'])} & {ci_h} & {f(r.mean_regret_pts, 2)} & {ci_r} & "
                f"{f(r.median_regret_pts, 2)} \\\\")
write("best_epoch", body, "lcccccc",
      "Best-epoch prediction evaluated as an early-stopping decision. Training stops at the predicted epoch and the best "
      "checkpoint seen so far is kept. \\emph{Budget used} is the stopping epoch as a share of the run's budget; a \\emph{hit} "
      "keeps an accuracy within one point of the run's peak; \\emph{regret} is the accuracy given up. Early stopping observes "
      "the curve online; the predictors decide after epoch 2 or 3. LLM and XGBoost predictions are averaged over seeds; CIs are "
      "cluster bootstraps over test architectures.", "tab:best-epoch", size="\\footnotesize", fit=True)

# ---------------------------------------------------------------- Table: protocol / split
fam = pd.read_csv(RES / "split_by_family.csv", index_col=0)
cov = pd.read_csv(RES / "proxy_coverage_pct.csv", index_col=0)
uniq = pd.read_csv(RES / "unique_architectures.csv", index_col=0)
body = ["Split & Families & Runs & Architectures & " + " & ".join(c.replace("proxy_", "").replace("_", "\\_") for c in cov.columns) + " \\\\", "\\midrule"]
for s in ("train", "val", "test"):
    fams = ", ".join(f"{k} ({int(v)})" for k, v in fam[s].items() if v > 0)
    body.append(f"{s} & {fams} & {int(fam[s].sum())} & {int(uniq.loc[s].iloc[0])} & " + " & ".join(f"{v:.0f}\\%" for v in cov.loc[s]) + " \\\\")
write("split", body, "llcc" + "c" * len(cov.columns),
      "Cross-family split (runs per family) and zero-cost proxy coverage (share of runs with the proxy available).",
      "tab:split", size="\\scriptsize", fit=True)

# ---------------------------------------------------------------- Appendix: per-seed results
runs = pd.read_csv(RES / "runs.csv")
llm = runs[runs.method.isin(["Qwen3-8B", "Llama-3.1-8B", "Gemma-2-9B"])].copy()
llm["order"] = llm.config.map({c: i for i, c in enumerate(LABEL)})
body = ["Model & Inputs & Seed & $R^2$ & Pearson $r$ & MAE & RMSE & best-epoch $R^2$ \\\\", "\\midrule"]
for (m, c), g in llm.sort_values(["method", "order", "seed"]).groupby(["method", "order"], sort=False):
    for _, r in g.iterrows():
        body.append(f"{r.method} & {LABEL[r.config]} & {r.seed} & {f(r.acc_r2)} & {f(r.acc_pearson_r)} & {f(r.acc_mae)} & "
                    f"{f(r.acc_rmse)} & {f(r.epoch_r2, 2)} \\\\")
    body.append("\\midrule")
write("per_seed", body[:-1], "llcccccc", "Every LLM run, per seed.", "tab:per-seed", size="\\scriptsize", place="H", fit=True)

# ---------------------------------------------------------------- Appendix: proxy transfer across families
from scipy.stats import spearmanr

tabular = pd.read_csv(RUNS / "paper/tabular_3epoch.csv")
body = ["Proxy & Train: $\\rho$ & Train: coverage & Test: $\\rho$ & Test: coverage \\\\", "\\midrule"]
for p in ["synflow", "nwot", "grad_norm", "log_params", "depth"]:
    cells = []
    for split in ("train", "test"):
        d = tabular[tabular.split == split]
        m = d[f"proxy_{p}"].notna()
        cells += [f(spearmanr(d.loc[m, f"proxy_{p}"], d.loc[m, "best_accuracy"])[0], 2), f"{m.mean():.0%}".replace("%", "\\%")]
    body.append(f"{p.replace('_', chr(92) + '_')} & " + " & ".join(cells) + " \\\\")
write("proxy_transfer", body, "lcccc",
      "Spearman correlation of each (normalised) zero-cost proxy with best accuracy, and the share of runs for which the proxy "
      "could be computed, on the training and test families.", "tab:proxy-transfer", place="H")

# ---------------------------------------------------------------- Appendix: single-seed exploratory runs (second machine)
EXP = RUNS / "paper/exploratory_56"
# (log path, inputs, change relative to the reference) -- all: Qwen3-8B, LoRA rank 16, alpha 32, at most 7 epochs, seed 42
EXPLORATORY = [
    ("training_recipe_fix/test_metrics_recipe_fix_baseline.log", "2 ep. + code", "--"),
    ("early_signals_only/test_metrics_early_signals_only.log", "2 ep.", "--"),
    ("nocode_plus_extras/test_metrics_nocode_proxies.log", "2 ep. + proxies", "reference"),
    ("nocode_proxies_reproduce/test_metrics.log", "2 ep. + proxies", "re-run of the reference"),
    ("nocode_plus_extras/test_metrics_nocode_retrieval.log", "2 ep. + retrieval", "3 similar training architectures"),
    ("three_early_epochs/test_metrics_three_early_epochs.log", "3 ep.", "--"),
    ("three_epoch_proxies/test_metrics_three_epoch_proxies.log", "3 ep. + proxies", "--"),
    ("proxies_plus_fisher/test_metrics_proxies_plus_fisher.log", "2 ep. + proxies + Fisher", "sixth proxy"),
    ("proxies_swap_fisher/test_metrics_proxies_swap_fisher.log", "2 ep. + proxies (Fisher for depth)", "proxy swap"),
    ("nn_stat_flops/test_metrics_nn_stat_flops.log", "2 ep. + proxies + FLOPs", "structural statistic"),
    ("cot/test_metrics_cot.log", "2 ep. + proxies, chain of thought", "templated rationale before answer"),
    ("recipe_sweep_lr_half/test_metrics_lr_half.log", "2 ep. + proxies", "learning rate halved"),
    ("recipe_sweep_lora_rank8/test_metrics_lora_rank8.log", "2 ep. + proxies", "LoRA rank 8"),
    ("recipe_sweep_alpha64/test_metrics_alpha64.log", "2 ep. + proxies", "LoRA $\\alpha$ 64"),
    ("recipe_sweep_warmup10/test_metrics_warmup10.log", "2 ep. + proxies", "warm-up 10\\%"),
    ("recipe_sweep_batch8/test_metrics_batch8.log", "2 ep. + proxies", "batch size 8"),
    ("llama_base_model/test_metrics_llama_base_model.log", "2 ep. + proxies", "Llama-3.1-8B backbone"),
]


def parse_log(path):
    vals = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            try:
                vals[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return vals


body = ["Inputs & Change & Valid & $R^2$ & Pearson $r$ & MAE & best-epoch $R^2$ \\\\", "\\midrule"]
for rel, inputs, change in EXPLORATORY:
    v = parse_log(EXP / rel)
    body.append(f"{inputs} & {change} & {int(v['valid_predictions'])} & {f(v['best_accuracy_R2'])} & "
                f"{f(v['best_accuracy_Pearson_r'])} & {f(v['best_accuracy_MAE'])} & {f(v['best_epoch_R2'], 2)} \\\\")
write("exploratory", body, "llccccc",
      "Single-seed exploratory runs that preceded the controlled study (Qwen3-8B unless noted; LoRA rank 16, $\\alpha = 32$, at "
      "most seven epochs; second machine). Not replicated; shown for completeness only.", "tab:exploratory",
      size="\\scriptsize", place="H", fit=True)


# ---------------------------------------------------------------- Robustness: sensitivity to dominant architectures (R7)
eff = pd.read_csv(RES / "sensitivity_effects.csv")
subsets = ["all 15 architectures", "without RNN (WikiText)", "without ResNet-LSTM (COCO)", "without both"]
heads = ["All", "w/o RNN (WikiText)", "w/o ResNet-LSTM (COCO)", "w/o both"]
body = ["Effect ($\\Delta R^2$) & " + " & ".join(heads) + " \\\\", "\\midrule"]
for e in pd.unique(eff.effect):
    vals = [f(eff[(eff.effect == e) & (eff.subset == sb)].delta_r2.iloc[0]) for sb in subsets]
    body.append(f"{e} & " + " & ".join(vals) + " \\\\")
write("sensitivity", body, "lcccc",
      "Paired effects on best-accuracy $R^2$ with the two dominant test architectures removed (together 42\\% of the test "
      "set). Seed-averaged; positive values favour the second-named configuration.", "tab:sensitivity", size="\\footnotesize",
      fit=True)

# ---------------------------------------------------------------- Ranking within datasets (R8)
rank = pd.read_csv(RES / "ranking.csv")
NAMES = {"XGBoost, e3": "XGBoost, 3 ep.", "XGBoost, e3+P": "XGBoost, 3 ep. + proxies", "LC-PFN, e3": "LC-PFN, 3 ep.",
         "Ensemble: 3 LLMs + XGBoost, e3+P": "3 LLMs + XGBoost, 3 ep. + proxies"}
body = ["Method & Kendall $\\tau$ & Top-5 recall & Selection regret (pts) \\\\", "\\midrule"]
for _, r in rank.iterrows():
    name = NAMES.get(r.method, r.method.replace("e2+P", "2 ep. + proxies").replace("e3+P+code", "3 ep. + proxies + code")
                     .replace("e3+P", "3 ep. + proxies").replace("1x32", "$1\\times32$"))
    body.append(f"{name} & {f(r.kendall_tau)} & {f(r.top5_recall, 2)} & {f(r.selection_regret_pts, 2)} \\\\")
write("ranking", body, "lccc",
      "Ranking runs within each dataset, the use case of model selection. Kendall $\\tau$ between predicted and true best "
      "accuracy, share of the five runs ranked highest by the predictor whose true accuracy is at least the fifth-highest "
      "(ties in the true accuracy count as top-5), and the accuracy lost by picking the predicted-best "
      "run; averaged over datasets weighted by size (all 9 test datasets) and over seeds.", "tab:ranking", size="\\footnotesize")

# ---------------------------------------------------------------- Compute cost per run (S1)
import datetime as dt
import json
import re

MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def ts(text):
    _, mon, day, hms, _, year = text.split()
    h, mi, sec = map(int, hms.split(":"))
    return dt.datetime(int(year), MON[mon], int(day), h, mi, sec)


durations = {}
for status in (RUNS / "logs/queue_3epoch.status", RUNS / "logs/ablation.status"):
    start, fixed = {}, False
    for line in open(status):
        # source-code runs before the sequence-limit fix trained on truncated prompts: do not report their cost
        fixed = fixed or "fixed max_length" in line
        m = re.match(r"(\w{3} \w{3} +\d+ [\d:]+ CES?T \d{4}) (START|END) (\S+) (\S+)( seed\d+)?( EXIT_CODE=(\d+))?", line.strip())
        if not m:
            continue
        key = (m.group(3), m.group(4), (m.group(5) or "").strip())
        if m.group(2) == "START":
            start[key] = ts(m.group(1))
        elif m.group(7) == "0" and key in start and "(train)" not in line:
            if key[0].startswith("code") and not fixed:
                continue
            durations.setdefault(key[:2], []).append((ts(m.group(1)) - start[key]).total_seconds() / 3600)
LABELS = {("qwen3-8b", "seed42"): None}
GROUPS = [("Qwen3-8B, 3 ep. + proxies", [("qwen3-8b", "seed42"), ("qwen3-8b", "seed123")]),
          ("Llama-3.1-8B, 3 ep. + proxies", [("llama3.1-8b", s) for s in ("seed0", "seed42", "seed123")]),
          ("Gemma-2-9B, 3 ep. + proxies", [("gemma2-9b", s) for s in ("seed0", "seed42", "seed123")]),
          ("Qwen3-8B, 3 ep. (no proxies)", [("e3np", "qwen3-8b")]),
          ("Qwen3-8B, 3 ep. + proxies, answer-only loss", [("e3p_ro", "qwen3-8b")]),
          ("Qwen3-8B, 3 ep. + proxies, answer-only loss$^\\dagger$", [("e3p_ro_b1", "qwen3-8b")]),
          ("Qwen3-8B, 2 ep. + proxies + code$^\\dagger$", [("code2p", "qwen3-8b")]),
          ("Qwen3-8B, 3 ep. + proxies + code$^\\dagger$", [("code3p", "qwen3-8b")]),
          ("Qwen3-8B, 3 ep. + proxies + code, answer-only loss$^\\dagger$", [("code3p_ro", "qwen3-8b")]),
          ("Llama-3.1-8B, 3 ep. + proxies + code$^\\dagger$", [("code3p", "llama3.1-8b")])]
tim = json.loads((RUNS / "paper/timings_cpu.json").read_text())
body = ["Predictor & Hardware & Time per run & Runs \\\\", "\\midrule",
        f"XGBoost & CPU (8 cores) & {tim['xgboost_grid_wall_seconds'] / tim['xgboost_grid_fits'] * 2:.1f} s & 5 per input set \\\\",
        f"LC-PFN (pre-trained, inference only) & CPU & {tim['lcpfn_wall_seconds'] / tim['lcpfn_predictions'] * 657:.1f} s & 1 \\\\"]
for lab, keys in GROUPS:
    hrs = [h for k in keys for h in durations.get(k, [])]
    if hrs:
        body.append(f"{lab} & RTX 4090 & {np.median(hrs):.1f} h ({min(hrs):.1f}--{max(hrs):.1f}) & {len(hrs)} \\\\")
write("cost", body, "llcc",
      "Wall-clock cost per run on the hardware used (median and range over the timed runs). LLM times include training with "
      "early stopping and test inference; the GPU was at times shared with other users' jobs, so they are indicative. "
      "$^\\dagger$Per-device batch 1 with 32 accumulation steps instead of 4 with 8 (same effective batch).",
      "tab:cost", size="\\footnotesize", fit=True)

# ---------------------------------------------------------------- Checkpoint selection (full-text vs answer-only loss)
sel = pd.read_csv(RES / "checkpoint_selection.csv")
body = ["Checkpoint rule & \\multicolumn{3}{c}{Full-text loss} & \\multicolumn{3}{c}{Answer-only loss} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", " & Epochs & $R^2$ & MAE & Epochs & $R^2$ & MAE \\\\", "\\midrule"]
for rule in pd.unique(sel.rule):
    cells = []
    for loss in ("full-text loss", "answer-only loss"):
        r = sel[(sel.rule == rule) & (sel.loss == loss)].iloc[0]
        cells += [r.epochs.replace(" ", ", "), pm(r.test_r2_mean, r.test_r2_sd), pm(r.test_mae_mean, r.test_mae_sd)]
    body.append(f"{rule[0].upper() + rule[1:]} & " + " & ".join(cells) + " \\\\")
write("ckpt_selection", body, "lcccccc",
      "Separating the training objective from checkpoint selection (Qwen3-8B, 3 ep.\\ + proxies, every epoch's checkpoint kept; "
      "data order fixed across seeds). \\emph{Epochs}: the epoch chosen for seeds 0, 42 and 123. Test $R^2$ and MAE: mean\\,$\\pm$\\,SD "
      "over seeds. The fixed-epoch rows are shown for comparison and are not available to a practitioner in advance.",
      "tab:ckpt-selection", size="\\footnotesize", fit=True)

# ---------------------------------------------------------------- Further held-out families
fm = pd.read_csv(RES / "fold_metrics.csv")
fc = pd.read_csv(RES / "fold_comparisons.csv")
SPLIT_NAME = {"main": "other (main)", "tf": "transformer", "rnn": "rnn", "det": "detector", "seg": "segmentation"}
ORDER = ["Qwen3-8B, 3 ep. + proxies", "Qwen3-8B, 3 ep. + proxies + code", "Gemma-2-9B, 2 ep. + proxies",
         "Gemma-2-9B, 3 ep. + proxies", "Gemma-2-9B + XGBoost, 3 ep. + proxies", "XGBoost, 2 ep. + proxies", "XGBoost, 3 ep.",
         "XGBoost, 3 ep. + proxies", "LC-PFN, 2 ep.", "LC-PFN, 3 ep.", "Per-dataset training mean"]
methods = [m for m in ORDER if m in set(fm.method)] + [m for m in pd.unique(fm.method) if m not in ORDER]


def fold_row(s, m):
    r = fm[(fm.split == s) & (fm.method == m)]
    return None if r.empty else r.iloc[0]


def fail_mark(r):
    return f"$^{{{int(r.parse_failures)}}}$" if r.parse_failures else ""


# main paper: R2 per split
body = ["Method & " + " & ".join(SPLIT_NAME.values()) + " \\\\",
        " & " + " & ".join(f"{int(fm[fm.split == s].n_arch.iloc[0])} arch." for s in SPLIT_NAME) + " \\\\", "\\midrule"]
for m in methods:
    cells = [("--" if (r := fold_row(s, m)) is None else f"{f(r.r2)}{fail_mark(r)}") for s in SPLIT_NAME]
    body.append(f"{m} & " + " & ".join(cells) + " \\\\")
write("folds", body, "l" + "c" * len(SPLIT_NAME),
      "Best-accuracy $R^2$ on five held-out families (86 test architectures). LLMs: seed 0; Qwen3-8B with answer-only loss and "
      "batch $1\\times32$; Gemma-2-9B with the full-text loss. Data order is fixed for all LLM runs except on the main split "
      "(\\cref{sec:data-order}). XGBoost: mean over 5 seeds; ensemble: unweighted average of Gemma-2-9B and XGBoost. "
      "--: not run. A superscript counts answers that failed to parse (scored as the training mean). Confidence intervals and "
      "MAE: \\cref{S-tab:folds-full}.", "tab:folds", size="\\footnotesize", fit=True)

# supplement: R2 with CI and MAE, one block per split
body = ["Method & $R^2$ & 95\\% CI & MAE \\\\", "\\midrule"]
for s in SPLIT_NAME:
    first = fm[fm.split == s].iloc[0]
    body.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{SPLIT_NAME[s]}: {int(first.n_arch)} architectures, "
                f"{int(first.n_runs)} runs}}}} \\\\")
    for m in methods:
        if (r := fold_row(s, m)) is not None:
            body.append(f"\\quad {m} & {f(r.r2)}{fail_mark(r)} & [{f(r.r2_ci_lo, 2)}, {f(r.r2_ci_hi, 2)}] & {f(r.mae)} \\\\")
    body.append("\\midrule")
write("folds_full", body[:-1], "lccc",
      "Best-accuracy prediction on the five held-out families, with cluster-bootstrap CIs over test architectures. Methods as "
      "in the held-out family table of the main paper.", "tab:folds-full", size="\\scriptsize")

body = ["Comparison (B $-$ A) & Arch. & $\\Delta$MAE & 95\\% CI & B better & $p$ & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule"]
for _, r in fc[fc.scope == "all splits"].iterrows():
    body.append(f"{r.comparison} & {int(r.n_arch)} & {f(r.delta_mae, 4)} & [{f(r.ci_lo, 4)}, {f(r.ci_hi, 4)}] & "
                f"{int(r.arch_better_B)} & {f(r.wilcoxon_p, 4)} & {f(r.holm_p, 4)} \\\\")
write("fold_comparisons", body, "lcccccc",
      "Paired comparisons of per-architecture MAE, pooled over the test architectures of every held-out family on which both "
      "methods were run, each architecture weighted equally. Negative $\\Delta$MAE favours B. CI: bootstrap over architectures; "
      "$p$: Wilcoxon signed-rank test; $p_{\\mathrm{Holm}}$: Holm-corrected across the comparisons in this table. LLM runs on "
      "the further families are single seeds, so the intervals do not include seed variation.", "tab:fold-comparisons",
      size="\\footnotesize", fit=True)
