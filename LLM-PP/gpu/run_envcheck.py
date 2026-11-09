"""Environment check: evaluate an existing adapter on its own test file in the current environment, so that its
predictions can be compared with those it produced elsewhere (the study re-evaluated the seed-0 two-epoch Qwen3-8B
adapter, trained on a second machine; envcheck.py compares the two prediction files).

Usage (from an nn-gpt checkout, PYTHONPATH set to it):
    python run_envcheck.py <run_dir>     # run_dir holds tuned_model/ and test_llm_dataset.jsonl
"""
import sys
from pathlib import Path

from ab.gpt.act.acc_predictor.eval import test_model

run = Path(sys.argv[1])
test_model(model_path=run / "tuned_model", data_path=run / "test_llm_dataset.jsonl",
           output_path=run / "test_predictions.csv", metrics_path=run / "test_metrics.log")
