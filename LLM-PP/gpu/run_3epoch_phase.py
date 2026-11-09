"""Run one phase of the 3-early-epoch experiment in its own process.

Warm-up is passed explicitly (64 steps, 2% of the 3,200 scheduled steps): the original runs used the default of that
time, which nn-gpt later changed to 2 steps.

phase "train": Steps 1-3 (preprocess, datasets, fine-tune) -- same calls as acc_predictor.__main__.main.
phase "test":  Step 4 only, in a fresh process so the training copy of the model is no longer on the GPU.
Usage (from an nn-gpt checkout, PYTHONPATH set to it): ACC_MODEL=<model> python run_3epoch_phase.py <train|test> <seed>
"""
import sys

from ab.gpt.act.acc_predictor import config

phase, seed = sys.argv[1], int(sys.argv[2])
config.SEED = seed
config.USE_THIRD_EARLY_EPOCH = True
config.ACC_DIR = config.ACC_DIR.parent / "acc_predict_3epoch"

from ab.gpt.act.acc_predictor.__main__ import _run_tag
from ab.gpt.act.acc_predictor.preprocess import data_preprocessing
from ab.gpt.act.acc_predictor.dataset import prepare_llm_datasets
from ab.gpt.act.acc_predictor.train import train_model
from ab.gpt.act.acc_predictor.eval import test_model
from frozen_export import write_export

run_dir = config.ACC_DIR / _run_tag()
run_dir.mkdir(parents=True, exist_ok=True)
print(f"Run: model={config.MODEL_NAME}  seed={config.SEED}  phase={phase}")
print(f"Outputs -> {run_dir}")

if phase == "train":
    print("=" * 80 + "\nStep 1/4: Data preprocessing (nn_dataset → JSONL)\n" + "=" * 80)
    if not write_export(run_dir / "llm_finetuning_data.jsonl"):  # RAW=<frozen export> (frozen_export.py)
        data_preprocessing(output_jsonl_path=run_dir / "llm_finetuning_data.jsonl",
                           output_csv_path=run_dir / "llm_finetuning_data.csv")
    print("=" * 80 + "\nStep 2/4: Prepare LLM training datasets\n" + "=" * 80)
    train_path, val_path, _ = prepare_llm_datasets(input_path=run_dir / "llm_finetuning_data.jsonl", output_dir=run_dir)
    print("=" * 80 + "\nStep 3/4: Fine-tune model\n" + "=" * 80)
    train_model(train_path=train_path, val_path=val_path, output_dir=run_dir / "tuned_model",
                model_name=config.MODEL_NAME, seed=config.SEED, warmup_steps=64)
elif phase == "test":
    print("=" * 80 + "\nStep 4/4: Test model on held-out test set\n" + "=" * 80)
    test_model(model_path=run_dir / "tuned_model", data_path=run_dir / "test_llm_dataset.jsonl",
               output_path=run_dir / "test_predictions.csv", metrics_path=run_dir / "test_metrics.log")
    print(f"\nDone. Pipeline complete. Results in {run_dir}")
else:
    sys.exit(f"unknown phase {phase}")
