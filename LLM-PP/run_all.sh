#!/usr/bin/env bash
# =====================================================================
# Reproduce every number, table and figure of the manuscript from this folder.
#
# Usage (from any directory):
#   bash run_all.sh                        # analysis + tables + figures from the committed runs/ (~1 min, CPU)
#   bash run_all.sh analysis               # statistics only          -> results/*.csv, results/*.txt
#   bash run_all.sh tables                 # LaTeX tables              -> results/tables/
#   bash run_all.sh figures                # figures                   -> results/figures/
#   bash run_all.sh baselines [LCPFN=/p]   # XGBoost (+ LC-PFN if LCPFN points to a checkout of automl/lcpfn)
#   bash run_all.sh inputs NNGPT=/p        # re-export tables and curves from the frozen LEMUR export (needs nn-gpt)
#   bash run_all.sh all NNGPT=/p LCPFN=/p
#   bash run_all.sh llm [--dry-run]        # retrain all 66 LLM runs (about two weeks on one GPU, see gpu/README.md);
#                                          # afterwards run the default mode to recompute everything from them
#
# The outputs of the LLM runs are committed under runs/, so only the mode "llm" needs a GPU.
# Set PYTHON to use a specific interpreter, e.g. PYTHON=.venv/bin/python
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
PYTHON="${PYTHON:-python3}"
MODE="${1:-default}"
shift || true
NNGPT="" RAW="" LCPFN="" DRY_RUN=""
for arg in "$@"; do
  case "$arg" in
    NNGPT=*) NNGPT="${arg#NNGPT=}" ;;
    RAW=*) RAW="${arg#RAW=}" ;;
    LCPFN=*) LCPFN="${arg#LCPFN=}" ;;
    --dry-run) DRY_RUN=--dry-run ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

run() {  # run a script and keep what it prints next to its outputs
  local script=$1; shift
  echo "-- $script $*"
  "$PYTHON" "$script" "$@" > "results/${script%.py}_stdout.txt"
}

run_inputs() {
  if [[ -z "$NNGPT" ]]; then
    echo "inputs: set NNGPT=/path/to/nn-gpt (patched, see gpu/README.md)" >&2
    exit 2
  fi
  local here; here=$(pwd)
  # the LEMUR export every run used (runs/paper/lemur_export.jsonl.gz) unless RAW points elsewhere
  local raw="${RAW:-$here/runs/paper/lemur_export.jsonl.gz}" tmp=""
  if [[ "$raw" == *.gz ]]; then tmp=$(mktemp); gunzip -c "$raw" > "$tmp"; raw=$tmp; fi
  # proxy cache and normalisation statistics of the study, unless the checkout already has its own
  mkdir -p "$NNGPT/out/acc_predict"
  for f in proxy_cache.jsonl proxy_norm_stats.json; do
    [[ -f "$NNGPT/out/acc_predict/$f" ]] || cp "runs/paper/$f" "$NNGPT/out/acc_predict/$f"
  done
  for f in tf rnn det seg; do
    mkdir -p "$NNGPT/out/acc_predict_folds/$f"
    [[ -f "$NNGPT/out/acc_predict_folds/$f/proxy_norm_stats.json" ]] || \
      cp "runs/acc_predict_folds/$f/proxy_norm_stats.json" "$NNGPT/out/acc_predict_folds/$f/"
  done
  echo "== Inputs: tabular exports and test curves =="
  (cd "$NNGPT" && PYTHONPATH="$NNGPT" "$PYTHON" "$here/export_tabular.py" "$raw" "$here/runs/paper/tabular_3epoch.csv")
  for f in tf rnn det seg; do
    (cd "$NNGPT" && PYTHONPATH="$NNGPT" "$PYTHON" "$here/fold_tabular.py" $f "$raw" "$here/runs/paper/folds/tabular_$f.csv")
  done
  # full learning curves are not part of the export: they come from the local LEMUR database, which may no longer
  # contain every test run of the study; export_curves.py then leaves the committed file unchanged
  (cd "$NNGPT" && PYTHONPATH="$NNGPT" "$PYTHON" "$here/export_curves.py" "$raw" "$here/runs/paper/test_curves.csv") || \
    echo "   (test curves not re-exported; the committed runs/paper/test_curves.csv is kept)"
  [[ -n "$tmp" ]] && rm -f "$tmp"
}

run_baselines() {
  echo "== Baselines: XGBoost on the five splits =="
  "$PYTHON" xgboost_grid.py runs/paper/tabular_3epoch.csv runs/paper/xgboost_grid > results/xgboost_grid_stdout.txt
  for f in tf rnn det seg; do
    "$PYTHON" xgboost_grid.py runs/paper/folds/tabular_$f.csv runs/paper/folds/$f/xgboost_grid xgb_e2 > results/xgboost_grid_${f}_stdout.txt
  done
  if [[ -n "$LCPFN" ]]; then
    echo "== Baselines: LC-PFN on the five splits =="
    PYTHONPATH="$LCPFN" "$PYTHON" lcpfn_baseline.py runs/paper/tabular_3epoch.csv runs/paper/lcpfn > results/lcpfn_stdout.txt
    for f in tf rnn det seg; do
      PYTHONPATH="$LCPFN" "$PYTHON" lcpfn_baseline.py runs/paper/folds/tabular_$f.csv runs/paper/folds/$f/lcpfn \
        > results/lcpfn_${f}_stdout.txt
    done
  else
    echo "   (LC-PFN skipped: set LCPFN=/path/to/automl/lcpfn to re-run it; its committed predictions are used)"
  fi
}

run_analysis() {
  echo "== Analysis: statistics from the committed predictions =="
  run split_tables.py
  run collect_results.py
  run robustness.py
  run plateau_metric.py
  run failure_analysis.py
  run checkpoint_selection.py
  run fold_results.py
  run selected_epochs.py
  run envcheck.py
  run truncation_runs.py
}

run_tables() { echo "== Tables =="; run make_tables.py; }
run_figures() { echo "== Figures =="; run make_figures.py; }

case "$MODE" in
  default) run_analysis; run_tables; run_figures ;;
  inputs) run_inputs ;;
  baselines) run_baselines ;;
  analysis) run_analysis ;;
  tables) run_tables ;;
  figures) run_figures ;;
  all) run_inputs; run_baselines; run_analysis; run_tables; run_figures ;;
  llm) PYTHON="$PYTHON" bash gpu/run_llm_experiments.sh $DRY_RUN ;;
  *) echo "Unknown mode: $MODE (default | inputs | baselines | analysis | tables | figures | all | llm)" >&2; exit 2 ;;
esac
echo "== Done =="
