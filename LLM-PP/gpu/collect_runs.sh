#!/bin/bash
# Copy the small artefacts of finished runs from WORK into ../runs/ (the analysis inputs): test predictions and metrics,
# per-checkpoint evaluations, the trainer log of each run's last checkpoint, and the queue logs of the cost table.
# Absolute paths inside the logs are made relative. Adapters and prompt files are not copied.
# Usage: bash collect_runs.sh     Optional: WORK (as in run_llm_experiments.sh), DEST (default ../runs), ONLY.
set -eu
GPU_DIR=$(cd "$(dirname "$0")" && pwd)
LLMPP=$(dirname "$GPU_DIR")
. "$GPU_DIR/common.sh"
WORK=${WORK:-$(dirname "$LLMPP")/work}
DEST=${DEST:-$LLMPP/runs}
n=0
while read -r co fold model arm seed dest; do
  src=$WORK/${CHECKOUT[$co]}/$(run_folder $fold $model $arm $seed)
  [ -f "$src/test_metrics.log" ] || { echo "not finished, skipped: $src"; continue; }
  to=$DEST/$dest/${DIR[$model]}_seed$seed
  mkdir -p "$to"
  cp "$src/test_predictions.csv" "$src/test_metrics.log" "$to/"
  if [ -d "$src/ckpt_eval" ]; then cp -r "$src/ckpt_eval" "$to/"; fi
  last=$(ls -d "$src"/tuned_model/checkpoint-* | sort -t- -k2 -n | tail -1)
  cp "$last/trainer_state.json" "$to/"
  grep -rlZE "$WORK/nn-gpt(-p01)?/" "$to" | xargs -0 -r sed -i -E "s#$WORK/nn-gpt(-p01)?/##g"
  n=$((n + 1))
done < <(list_jobs | grep -E "${ONLY:-.}")
mkdir -p "$DEST/logs"
# the marker tells make_tables.py that the source-code runs after it have the sequence-length fix
{ cat "$WORK/nn-gpt-p01/logs/ablation.status" 2>/dev/null || true; echo "fixed max_length (from here on: patches 00, 01 and 02)"
  cat "$WORK/nn-gpt/logs/ablation.status" 2>/dev/null || true; } > "$DEST/logs/ablation.status"
if [ -f "$WORK/nn-gpt-p01/logs/queue_3epoch.status" ]; then cp "$WORK/nn-gpt-p01/logs/queue_3epoch.status" "$DEST/logs/"; fi
echo "collected $n runs into $DEST"
