#!/bin/bash
# Input-ablation queue: for each "arm:seed" job, wait for a free GPU, train, then test in a fresh process.
# Usage: NNGPT=/path/to/nn-gpt bash run_ablation_seeds.sh <ACC_MODEL> <arm:seed> [<arm:seed> ...]
#   ACC_MODEL: qwen3-8b | llama3.1-8b | gemma2-9b.  Optional: PYTHON (default python3), FOLD (tf | rnn | det | seg).
# Skips jobs that already have test_metrics.log; retests jobs whose training already finished.
# Env FOLD=<name> runs on an alternative held-out family (out/acc_predict_folds/<fold>/). Arms ending in _all
# also evaluate every saved checkpoint on val and test (skipping evaluations that already exist).
set -u
model=$1; shift
GPU_DIR=$(cd "$(dirname "$0")" && pwd)
. "$GPU_DIR/common.sh"
: "${NNGPT:?set NNGPT=/path/to/nn-gpt (patched checkout, see gpu/README.md)}"
PYTHON=${PYTHON:-python3}
FOLD=${FOLD:-}
cd "$NNGPT" && mkdir -p logs
export PYTHONPATH=$NNGPT PYTHONUNBUFFERED=1 ACC_MODEL=$model
S=logs/ablation.status
for job in "$@"; do
  arm=${job%%:*}; seed=${job##*:}
  if [ -n "$FOLD" ]; then run=out/acc_predict_folds/$FOLD/$arm/${DIR[$model]}_seed$seed; tag="$arm@$FOLD"
  else run=out/acc_predict_ablation/$arm/${DIR[$model]}_seed$seed; tag=$arm; fi
  log=logs/ablation_${arm}${FOLD:+_$FOLD}_${model}_seed$seed.log
  if [ -f $run/test_metrics.log ] && [[ $arm != *_all ]]; then echo "$(date) SKIP $tag $model seed$seed (done)" >> $S; continue; fi
  echo "$(date) WAIT-GPU $tag $model seed$seed" >> $S; wait_for_gpu
  echo "$(date) START $tag $model seed$seed" >> $S
  if [ ! -f $run/tuned_model/adapter_model.safetensors ]; then
    [ -d $run/tuned_model ] && mv $run/tuned_model $run/tuned_model.failed_$(date +%s)
    FOLD=$FOLD "$PYTHON" "$GPU_DIR/run_ablation_phase.py" $arm train $seed 2>&1 | tee -a $log
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then echo EXIT_CODE=$rc >> $log; echo "$(date) END $tag $model seed$seed EXIT_CODE=$rc (train)" >> $S; continue; fi
  fi
  if [ ! -f $run/test_metrics.log ]; then
    FOLD=$FOLD "$PYTHON" "$GPU_DIR/run_ablation_phase.py" $arm test $seed 2>&1 | tee -a $log
    rc=${PIPESTATUS[0]}
  else rc=0; fi
  if [[ $arm == *_all ]]; then
    for ck in $(ls -d $run/tuned_model/checkpoint-* | xargs -n1 basename | sort -t- -k2 -n); do
      for split in val test; do
        [ -f $run/ckpt_eval/${ck}_${split}_metrics.log ] && continue
        FOLD=$FOLD "$PYTHON" "$GPU_DIR/run_ablation_phase.py" $arm ckpt $seed $ck $split 2>&1 | tee -a $log
        r=${PIPESTATUS[0]}; [ $r -ne 0 ] && { rc=$r; echo "$(date) CKPT-FAIL $tag $model seed$seed $ck $split rc=$r" >> $S; }
      done
    done
  fi
  echo EXIT_CODE=$rc >> $log
  echo "$(date) END $tag $model seed$seed EXIT_CODE=$rc" >> $S
done
echo "$(date) QUEUE DONE $model${FOLD:+ fold=$FOLD} $*" >> $S
