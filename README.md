# What Helps Large Language Models Predict Neural-Network Performance Across Architecture Families?

**A controlled multi-seed study of early signals, source code, zero-cost proxies and ensembling**

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-blue.svg)](LICENSE-DATA)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Which information lets a fine-tuned large language model predict a network's final accuracy from its first training
epochs, for architecture families it has never seen? This repository contains the analysis code and the committed
results behind a controlled study of that question on the LEMUR benchmark: three LLMs (Qwen3-8B, Llama-3.1-8B,
Gemma-2-9B) fine-tuned with QLoRA over three seeds, one input or training choice (early epochs, zero-cost proxies,
source code, loss, checkpoint selection) varied at a time, XGBoost, LC-PFN and trivial baselines given the same
information, and uncertainty measured over test architectures, on five held-out architecture families (86 test
architectures).

Every number, table and figure in the accompanying manuscript is produced by a script in `LLM-PP/` and stored under
`LLM-PP/results/`. Reproducing them needs no GPU, because the predictions of every LLM run are committed under
`LLM-PP/runs/`. One command retrains those runs on a GPU (see Full rebuild).

---

## What is in this repository

```
LLM-PP/
├── run_all.sh              driver: default (analysis + tables + figures) | inputs | baselines | analysis | tables | figures | all | llm
├── verify_results.py       re-runs the CPU pipeline in a scratch copy and diffs it against results/
├── requirements.txt        dependencies for the default run (and what the other stages need)
├── _paths.py               every input/output location, relative to this folder
├── folds.py                definition of the four further held-out families
├── <analysis scripts>      statistics, robustness, stopping, folds, tables, figures (see the map below)
├── <export scripts>        export_tabular.py, fold_tabular.py, export_curves.py (stage "inputs", need nn-gpt)
├── xgboost_grid.py, lcpfn_baseline.py   baselines (stage "baselines")
├── gpu/                    all 66 LLM runs in one script (run_llm_experiments.sh), runners, patches to nn-gpt, pinned
│                           GPU requirements, and how each run was made (gpu/README.md)
├── runs/                   inputs of the analysis
│   ├── acc_predict/                2 early epochs + proxies, 3 LLMs x 3 seeds (Gemma-2-9B: matched reruns, gpu/README.md)
│   ├── acc_predict_3epoch/         3 early epochs + proxies, 3 LLMs x 3 seeds
│   ├── acc_predict_ablation/<arm>/ Qwen3-8B ablations, controls, Llama-3.1-8B with code, truncated-sequence runs
│   ├── acc_predict_folds/<fold>/   runs on the transformer (tf), rnn, detector (det) and segmentation (seg) families
│   ├── acc_predict_envcheck/       one adapter evaluated in two software environments
│   ├── paper/                      frozen inputs: the LEMUR export of the study, the zero-cost proxy cache and
│   │                               its statistics, the test learning curves; tabular exports, XGBoost and LC-PFN
│   │                               predictions, the single-seed exploratory runs, CPU timings
│   ├── logs/                       queue logs (run start/end times, used for the cost table)
│   └── superseded/                 earlier two-epoch Gemma-2-9B runs (other library version), not used by any script
└── results/                every output of the analysis: CSV/TXT, tables/ (LaTeX), figures/ (PDF)
```

Each LLM run folder (`<model>_seed<seed>/`) holds `test_predictions.csv` (one row per test run: predicted and true best
accuracy and best epoch), `test_metrics.log`, and `trainer_state.json` (training and validation loss per epoch) where
the training log was retained; the checkpoint-selection runs also hold `ckpt_eval/` with predictions of every epoch's
checkpoint on validation and test. Arm names:

| Arm | Configuration | Data order across seeds |
|---|---|---|
| `acc_predict/`, `acc_predict_3epoch/` | 2 or 3 early epochs + proxies, full-text loss, batch 4×8 | per seed |
| `e3np` | 3 early epochs, no proxies | per seed |
| `e3p_ro` | 3 early epochs + proxies, answer-only loss | per seed |
| `code2p`, `code3p`, `code3p_ro` | + source code (2 or 3 epochs; answer-only loss for `_ro`), batch 1×32 | fixed |
| `*_trunc1024` | the same, trained before the sequence-length fix (sequences cut at 1,024 tokens) | per seed |
| `e3p_all`, `e3p_ro_all` | reruns of `acc_predict_3epoch` and `e3p_ro` keeping every epoch's checkpoint | fixed |
| `e3p_ro_b1` | `e3p_ro` with batch 1×32 | fixed |
| `acc_predict_folds/<fold>/` | `e2p` and `e3p` (Gemma-2-9B) on all four further families; `e3p_ro_b1` and `code3p_ro` (Qwen3-8B) on `tf` and `rnn`; seed 0 | fixed |

"Fixed" marks every run trained after the sequence-length fix, which as a side effect made the trainer use one data
order for all seeds (Section VI-E of the paper, `gpu/README.md`).

The manuscript sources are **not** part of this repository.

---

## Quick start

```bash
git clone https://github.com/Divyesh-Dhanani/Llm-pp.git && cd Llm-pp/LLM-PP
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python3 verify_results.py       # re-runs everything in a scratch copy and diffs it against results/ (~1 min)
bash run_all.sh                 # the same, in place: rewrites results/ (~1 min)
bash run_all.sh tables          # only the LaTeX tables -> results/tables/
bash run_all.sh figures         # only the figures      -> results/figures/
```

Run `verify_results.py` first: `run_all.sh` overwrites the committed files under `results/`, so a later verification
would compare your outputs with themselves. `git checkout -- results` restores the committed versions.

`verify_results.py` copies the folder to a temporary directory, empties `results/` there, runs every analysis, table and
figure script, and compares each output with the committed file. Nothing in your checkout is modified. Exit status is 0
when everything matches. Add `--baselines` to also re-fit XGBoost on the five splits (needs `xgboost==3.2.0` and
`scikit-learn`; on macOS, XGBoost also needs the OpenMP runtime: `brew install libomp`).

---

## Full rebuild

The default run starts from the committed predictions. The earlier stages recompute those inputs:

```bash
bash run_all.sh baselines LCPFN=/path/to/lcpfn       # XGBoost (5 seeds; 4 input sets on the main split, 3 on the others) and LC-PFN
bash run_all.sh inputs NNGPT=/path/to/nn-gpt         # tabular exports from the frozen LEMUR export
bash run_all.sh llm                                  # all 66 LLM runs, on a GPU (about two weeks)
```

- **baselines** needs `xgboost==3.2.0` and `scikit-learn` (macOS: `brew install libomp`). The XGBoost version matters:
  3.4.1 changes XGBoost's $R^2$ by up to 0.008 (e.g. 0.852 to 0.860 on the main split), while 3.2.0 reproduces every
  committed prediction byte for byte. LC-PFN needs PyTorch, `requests` (it downloads its pre-trained weights on first
  use) and a checkout of [`automl/lcpfn`](https://github.com/automl/lcpfn) at commit `ba892f6`, passed as `LCPFN=` and
  **not** installed with pip (its package metadata pins `torch<=1.11` and `numpy<2`; `lcpfn_baseline.py` patches the
  two incompatibilities with current PyTorch at run time). With PyTorch 2.9–2.15 its predictions agree with the
  committed ones to float32 precision (largest difference 2.4e-7). Without `LCPFN=`, only XGBoost is re-fitted.
- **inputs** runs inside a patched [`nn-gpt`](https://github.com/ABrain-One/nn-gpt) checkout (`gpu/README.md`) and
  starts from the frozen LEMUR export (`runs/paper/lemur_export.jsonl.gz`; `RAW=` selects another). The five tabular
  exports are rebuilt byte for byte. The full learning curves of the test runs are not part of the export and come
  from the local LEMUR database, which no longer contains 33 of the 657 test runs; `runs/paper/test_curves.csv` is
  then kept as committed (see Data sources).
- **llm** (about two weeks and 110 GB on one RTX 4090) downloads the pinned `nn-gpt`, applies the patches, trains and
  tests all 66 runs, and copies their outputs into `runs/`. `bash run_all.sh llm --dry-run` lists every job first. It
  needs the exact package versions of `gpu/requirements-gpu.txt`; see `gpu/README.md`.

Set `PYTHON=/path/to/venv/bin/python` if the default interpreter lacks the dependencies.

---

## From the paper to the scripts

| Paper | Script | Output |
|---|---|---|
| Table I, Fig. S2 (main results) | `collect_results.py` | `results/configs.csv`, `results/tables/main_results.tex`, `results/figures/main_results.pdf` |
| Table II, Table S9 (five held-out families) | `fold_results.py` | `results/fold_metrics.csv`, `results/tables/folds.tex`, `results/tables/folds_full.tex` |
| Fig. 1, Table S10 (pooled comparisons, Holm) | `fold_results.py`, `make_figures.py` | `results/fold_comparisons.csv`, `results/figures/forest_pooled.pdf`, `results/tables/fold_comparisons.tex` |
| Table III (checkpoint selection) | `checkpoint_selection.py` | `results/checkpoint_selection.csv`, `results/tables/ckpt_selection.tex` |
| Fig. 2, Table S6 (best epoch as a stopping decision) | `plateau_metric.py` | `results/best_epoch_plateau.csv`, `results/figures/best_epoch_frontier.pdf`, `results/tables/best_epoch.tex` |
| Table S1 (split, proxy coverage) | `split_tables.py`, `make_tables.py` | `results/split_by_family.csv`, `results/tables/split.tex` |
| Table S2 (cost) | `make_tables.py` (from `runs/logs/`, `runs/paper/timings_cpu.json`) | `results/tables/cost.tex` |
| Table S3 (input and training ablation) | `collect_results.py` | `results/configs.csv`, `results/tables/qwen_ablation.tex` |
| Table S4, Fig. S1 (main-split comparisons, Holm over 25) | `collect_results.py` | `results/comparisons.csv`, `results/tables/comparisons.tex`, `results/figures/forest.pdf` |
| Table S5, Fig. S4 (per dataset) | `collect_results.py` | `results/per_dataset.csv`, `results/tables/per_dataset.tex`, `results/figures/per_dataset.pdf` |
| Fig. S3 (large errors) | `failure_analysis.py`, `make_figures.py` | `results/large_errors.csv`, `results/mnist_outliers.txt`, `results/figures/pred_vs_true.pdf` |
| Table S7 (dominant architectures), Table S8 (ranking) | `robustness.py` | `results/sensitivity_effects.csv`, `results/ranking.csv` |
| Table S11 (per seed) | `collect_results.py` | `results/runs.csv`, `results/tables/per_seed.tex` |
| Table S12 (proxy transfer), Table S13 (exploratory runs) | `make_tables.py` | `results/tables/proxy_transfer.tex`, `results/tables/exploratory.tex` |
| Sec. VI-C (selected epochs: 42 of 45 main-split runs pick epoch 1 or 2) | `selected_epochs.py` | `results/selected_epochs.csv` |
| Sec. VI-E (runs with truncated sequences) | `truncation_runs.py` | `results/truncation_runs.csv` |
| Section S8 (environment check) | `envcheck.py` | `results/envcheck.txt` |
| Discussion (one LLM run + XGBoost) | `collect_results.py` | `results/configs.csv` (rows `... seed <s> & XGBoost e3+P`) |

Each script's console output is kept as `results/<script>_stdout.txt`.

---

## Data sources

No measurements are taken by the default run; all inputs are committed or public.

| Input | Location | Used for |
|---|---|---|
| Training records of LEMUR, as used by every run | `runs/paper/lemur_export.jsonl.gz` (frozen export, MIT-licensed records of [`ABrain-One/nn-dataset`](https://github.com/ABrain-One/nn-dataset), `nn-dataset` 2.2.13) | prompts, targets, early accuracies, tabular exports |
| Full learning curves of the test runs | `runs/paper/test_curves.csv` (exported from the same LEMUR snapshot) | best epoch as a stopping decision |
| Zero-cost proxies | `runs/paper/proxy_cache.jsonl`, `proxy_norm_stats.json` (computed once, at random initialisation) | proxy inputs of LLMs and XGBoost |
| Prompt construction, fine-tuning, inference | [`ABrain-One/nn-gpt`](https://github.com/ABrain-One/nn-gpt), commit **`e3269ce`**, plus `gpu/*.patch` | all LLM runs |
| Base LLMs | `unsloth/Qwen3-8B-unsloth-bnb-4bit`, `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, `unsloth/gemma-2-9b-it-bnb-4bit` (Hugging Face) | all LLM runs |
| LC-PFN | [`automl/lcpfn`](https://github.com/automl/lcpfn), commit `ba892f6`, pre-trained weights as released | learning-curve baseline |

LEMUR is a living dataset, and pinning the package version is not enough: nn-dataset rebuilds its local database from
the current records, and an export made today differs from the study's (638 instead of 657 test runs on the main split;
33 test runs no longer present). The three files above freeze the inputs of the study. The proxies are frozen for a
different reason: recomputing them gives identical depth and parameter counts, but gradient-based scores that differ
by up to 77%, because GPU gradient computations are not deterministic.

---

## Reproducibility

**What is guaranteed.** Every bootstrap and every model fitted on a CPU is seeded, and no code path depends on
`PYTHONHASHSEED`. With the committed environment (Python 3.12.3, numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, matplotlib
3.11.2, tabulate 0.10.0), `verify_results.py` reproduces all committed outputs byte for byte, including the PDFs (their
timestamp is fixed through `SOURCE_DATE_EPOCH`). With other versions and platforms (tested: Linux x86-64 with numpy
2.5.4, pandas 3.1.0, XGBoost 3.2.0 and scikit-learn 1.9.1; macOS on Apple silicon with Python 3.14), every table, every
figure and every XGBoost prediction is still byte-identical; up to thirteen CSV and JSON files then differ only in the
last digits of floats (relative difference below 1e-9), which `verify_results.py` reports separately and counts as
reproduced. Ties are broken explicitly wherever they could change a result (top-5 recall, dataset order in the
large-error summary), so no number depends on the platform's sort routine.

**What is not.** LLM fine-tuning on a GPU is not deterministic, so re-running `gpu/` gives close but not identical
predictions. Everything before training is reproducible: from the frozen inputs, a fresh `nn-gpt` checkout at the
pinned commit plus the patches rebuilds the train, validation and test prompts of every run byte for byte. Two properties of the training stack matter for anyone re-running it, and both are documented in the
paper:

- Before `gpu/02-sequence-length-fix.patch`, TRL/Unsloth silently truncate training sequences to 1,024 tokens. Prompts
  without code (at most 434 tokens) are unaffected; 96% of training prompts with code lost their answer. The runs made
  that way are kept as `*_trunc1024` for reference only.
- After that patch, Unsloth's configuration subclass fixes `data_seed=3407`, so the seeds of those runs differ only in
  initialisation and dropout. Comparisons in the paper are made within one regime unless labelled "mixed order".

**Statistical conventions.** Bracketed quantities are 95% percentile intervals from a bootstrap that resamples **test
architectures** (the 657 main-split test runs come from 15 architectures): 10,000 resamples for paired comparisons,
2,000 for per-configuration intervals. Paired tests are Wilcoxon signed-rank tests on per-architecture mean absolute
error. The primary tests pool the architectures of every held-out family on which both methods were run (up to 86) and
are Holm-corrected across the pooled comparisons; the 25 main-split comparisons are Holm-corrected among themselves. "Supported" means the interval excludes zero, "significant" that the Holm-corrected
p-value is below 0.05.

---

## Scope of the evidence

One benchmark (LEMUR), one fine-tuning method (QLoRA with the recipe of the NNGPT accuracy predictor), three 8–9B LLMs
and five held-out families: `other` (main split; 657 runs, 15 architectures), `transformer` (254 runs, 35), `rnn`
(201 runs, 9), `detector` (145 runs, 9) and `segmentation` (1,356 runs, 18). The LLMs on the four further families are
single runs. Results are specific to this setting; generalisation beyond
it is not established here.

---

## Citation

The article is not yet published. Until then, please cite it as:

```bibtex
@misc{dhanani2026llmpp,
  title        = {What Helps Large Language Models Predict Neural-Network Performance Across Architecture Families?},
  author       = {Dhanani, Divyesh Rameshbhai and Ilyas, Usama and Naseem, Hamza and Ignatov, Dmitry and Timofte, Radu},
  year         = {2026},
  note         = {Manuscript},
  howpublished = {\url{https://github.com/Divyesh-Dhanani/Llm-pp}}
}
```

Please also cite the LEMUR dataset ([`ABrain-One/nn-dataset`](https://github.com/ABrain-One/nn-dataset)).

---

## License

Code in `LLM-PP/` is released under the MIT License (`LICENSE`). The committed results under `LLM-PP/runs/` and
`LLM-PP/results/` are released under CC BY 4.0 (`LICENSE-DATA`). The upstream LEMUR records and model weights carry
their own terms.
