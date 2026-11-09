# Shared by the GPU scripts: the job list and the wait for a free GPU.
declare -A DIR=([qwen3-8b]=Qwen3-8B-unsloth-bnb-4bit [llama3.1-8b]=Meta-Llama-3.1-8B-Instruct-bnb-4bit [gemma2-9b]=gemma-2-9b-it-bnb-4bit)
declare -A CHECKOUT=([p01]=nn-gpt-p01 [p12]=nn-gpt)   # p01: patches 00 + 01; p12: patches 00 + 01 + 02
MIN_FREE_MB=20000

wait_for_gpu() {  # the GPU may be shared: start only when enough memory is free, twice in a row
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$MIN_FREE_MB" ]; then sleep 60
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$free" -ge "$MIN_FREE_MB" ] && return
    fi
    sleep 300
  done
}

list_jobs() {  # every LLM run of the paper, one per line: checkout fold model arm seed folder-under-runs
  local s m a f
  for s in 0 42 123; do
    for m in qwen3-8b llama3.1-8b gemma2-9b; do
      echo "p01 main $m e2p $s acc_predict"
      echo "p01 main $m 3epoch $s acc_predict_3epoch"
    done
    for a in e3np e3p_ro; do echo "p01 main qwen3-8b $a $s acc_predict_ablation/$a"; done
    for a in code2p code3p code3p_ro; do
      echo "p01 main qwen3-8b $a $s acc_predict_ablation/${a}_trunc1024"   # without the sequence-length fix
      echo "p12 main qwen3-8b $a $s acc_predict_ablation/$a"
    done
    echo "p12 main llama3.1-8b code3p $s acc_predict_ablation/code3p"
    for a in e3p_ro_b1 e3p_all e3p_ro_all; do echo "p12 main qwen3-8b $a $s acc_predict_ablation/$a"; done
  done
  for f in tf rnn; do
    for a in e3p_ro_b1 code3p_ro; do echo "p12 $f qwen3-8b $a 0 acc_predict_folds/$f/$a"; done
  done
  for f in tf rnn det seg; do
    for a in e2p e3p; do echo "p12 $f gemma2-9b $a 0 acc_predict_folds/$f/$a"; done
  done
}

run_folder() {  # where a job's outputs are, inside its checkout: run_folder <fold> <model> <arm> <seed>
  local name=${DIR[$2]}_seed$4
  if [ "$3" = 3epoch ]; then echo "out/acc_predict_3epoch/$name"
  elif [ "$1" = main ]; then echo "out/acc_predict_ablation/$3/$name"
  else echo "out/acc_predict_folds/$1/$3/$name"; fi
}
