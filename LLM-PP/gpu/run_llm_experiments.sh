#!/bin/bash
# Re-run all 66 LLM trainings and tests of the paper from a fresh download of the code (about two weeks on one RTX 4090)
# and copy the results into ../runs/. Run it in an environment with requirements-gpu.txt installed. Runs that already
# finished are skipped, so it can simply be started again after an interruption.
#
#   bash run_llm_experiments.sh --dry-run    # list the jobs and where each result goes; changes nothing
#   bash run_llm_experiments.sh              # download the code, run every job, copy the results into ../runs/
#   ONLY=e3np bash run_llm_experiments.sh    # only the jobs whose --dry-run line matches this pattern (nothing is copied)
#
# Optional: WORK (code checkouts and outputs, default ../../work), PYTHON (default python3), ONLY.
set -eu
GPU_DIR=$(cd "$(dirname "$0")" && pwd)
LLMPP=$(dirname "$GPU_DIR")
. "$GPU_DIR/common.sh"
WORK=${WORK:-$(dirname "$LLMPP")/work}
PYTHON=${PYTHON:-python3}
ONLY=${ONLY:-}
NNGPT_URL=https://github.com/ABrain-One/nn-gpt.git
COMMIT=e3269ce515b2054b16e3dccce5bc64db08335da3
export PYTHON WORK
export RAW=$LLMPP/runs/paper/lemur_export.jsonl.gz
export TZ=Europe/Berlin LC_TIME=C   # make_tables.py reads the run times from the status logs in this format

mapfile -t JOBS < <(list_jobs | grep -E "${ONLY:-.}")

if [ "${1:-}" = --dry-run ]; then
  for job in "${JOBS[@]}"; do
    read -r co fold model arm seed dest <<< "$job"
    printf '%-3s %-4s %-11s %-10s seed %-3s -> runs/%s/%s_seed%s\n' $co $fold $model $arm $seed $dest ${DIR[$model]} $seed
  done
  echo "${#JOBS[@]} jobs"
  exit 0
fi

checkout() {  # checkout <folder> <patch> ...: the pinned commit, the patches, and the frozen inputs of the study
  local dir=$WORK/$1 tmp=$WORK/$1.tmp f p; shift
  [ -d "$dir" ] && return 0
  git clone -q "$NNGPT_URL" "$tmp"
  git -C "$tmp" checkout -q "$COMMIT"
  for p in "$@"; do git -C "$tmp" apply "$GPU_DIR/$p"; done
  mkdir -p "$tmp/out/acc_predict"
  cp "$LLMPP"/runs/paper/proxy_cache.jsonl "$LLMPP"/runs/paper/proxy_norm_stats.json "$tmp/out/acc_predict/"
  for f in tf rnn det seg; do
    mkdir -p "$tmp/out/acc_predict_folds/$f"
    cp "$LLMPP/runs/acc_predict_folds/$f/proxy_norm_stats.json" "$tmp/out/acc_predict_folds/$f/"
  done
  mv "$tmp" "$dir"
}

mkdir -p "$WORK"
P00=00-run-folders-model-presets-seed.patch P01=01-gemma-system-role-and-answer-only-loss.patch P02=02-sequence-length-fix.patch
checkout nn-gpt-p01 $P00 $P01
checkout nn-gpt $P00 $P01 $P02

for job in "${JOBS[@]}"; do
  read -r co fold model arm seed dest <<< "$job"
  if [ $arm = 3epoch ]; then NNGPT=$WORK/${CHECKOUT[$co]} bash "$GPU_DIR/run_3epoch_seeds.sh" $model $seed
  else NNGPT=$WORK/${CHECKOUT[$co]} FOLD=${fold/main/} bash "$GPU_DIR/run_ablation_seeds.sh" $model $arm:$seed; fi
done

missing=0
for job in "${JOBS[@]}"; do
  read -r co fold model arm seed dest <<< "$job"
  f=$WORK/${CHECKOUT[$co]}/$(run_folder $fold $model $arm $seed)/test_metrics.log
  [ -f "$f" ] || { echo "not finished: $job"; missing=$((missing + 1)); }
done
if [ $missing -gt 0 ]; then echo "$missing of ${#JOBS[@]} jobs did not finish; see $WORK/*/logs/. Start this script again to retry them."; exit 1; fi
if [ -n "$ONLY" ]; then echo "${#JOBS[@]} jobs finished in $WORK. Nothing was copied, because ONLY selects a part of the study."; exit 0; fi
bash "$GPU_DIR/collect_runs.sh"
echo "All runs are in $LLMPP/runs/. Now run: bash $LLMPP/run_all.sh"
