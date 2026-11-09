#!/bin/bash
# 3-epoch input: one model, several seeds, one after another. Each: wait for a free GPU, train, then test in a fresh process.
# Usage: NNGPT=/path/to/nn-gpt bash run_3epoch_seeds.sh <ACC_MODEL> <seed> [<seed> ...]   (optional: PYTHON)
# Skips seeds that already have test_metrics.log; retests seeds whose training already finished.
set -u
model=$1; shift
GPU_DIR=$(cd "$(dirname "$0")" && pwd)
. "$GPU_DIR/common.sh"
: "${NNGPT:?set NNGPT=/path/to/nn-gpt (patched checkout, see gpu/README.md)}"
PYTHON=${PYTHON:-python3}
cd "$NNGPT" && mkdir -p logs
export PYTHONPATH=$NNGPT PYTHONUNBUFFERED=1 ACC_MODEL=$model
S=logs/queue_3epoch.status
for seed in "$@"; do
  run=out/acc_predict_3epoch/${DIR[$model]}_seed$seed
  log=logs/3epoch_${model}_seed${seed}.log
  if [ -f $run/test_metrics.log ]; then echo "$(date) SKIP $model seed$seed (done)" >> $S; continue; fi
  wait_for_gpu
  echo "$(date) START $model seed$seed" >> $S
  rc=0
  if [ ! -f $run/tuned_model/adapter_model.safetensors ]; then
    [ -d $run/tuned_model ] && mv $run/tuned_model $run/tuned_model.failed_$(date +%s)
    "$PYTHON" "$GPU_DIR/run_3epoch_phase.py" train $seed 2>&1 | tee -a $log
    rc=${PIPESTATUS[0]}
  fi
  if [ $rc -eq 0 ]; then
    "$PYTHON" "$GPU_DIR/run_3epoch_phase.py" test $seed 2>&1 | tee -a $log
    rc=${PIPESTATUS[0]}
  fi
  echo EXIT_CODE=$rc >> $log
  echo "$(date) END $model seed$seed EXIT_CODE=$rc" >> $S
done
echo "$(date) $model $* DONE" >> $S
