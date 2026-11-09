"""Run one phase of an input-ablation arm in its own process (paper recipe, config defaults).

Arms (outputs under out/acc_predict_ablation/<arm>/<model>_seed<seed>/):
  e2p     -- 2 early epochs + proxies, no code (base configuration)
  code2p  -- 2 early epochs + proxies + source code
  code3p  -- 3 early epochs + proxies + source code
  e3np    -- 3 early epochs, no proxies, no source code
  e3p_ro    -- 3 early epochs + proxies, no code, answer-only loss
  code3p_ro -- 3 early epochs + proxies + source code, answer-only loss
  e3p       -- 3 early epochs + proxies, no code (same settings as out/acc_predict_3epoch)
  e3p_ro_b1 -- e3p_ro with the code arms' per-device batch 1 x 32 accumulation (batch-size control)
  e3p_all, e3p_ro_all -- e3p / e3p_ro keeping every epoch's checkpoint (checkpoint-selection control)

Env FOLD=<name> holds out a different architecture family (see FOLDS); outputs then go to
out/acc_predict_folds/<fold>/<arm>/, with proxy normalization refit on that fold's training families.

phase "prep":  Steps 1-2 only (data preprocessing + LLM datasets), for pre-launch checks.
phase "train": Steps 1-3 (same calls as acc_predictor.__main__.main).
phase "test":  Step 4 only, in a fresh process so the training copy of the model is off the GPU.
phase "ckpt":  evaluate one saved checkpoint on the val or test split -> <run>/ckpt_eval/<checkpoint>_<split>_*.
Usage (from an nn-gpt checkout, PYTHONPATH set to it):
       ACC_MODEL=<model> python run_ablation_phase.py <arm> <prep|train|test> <seed>
       ACC_MODEL=<model> python run_ablation_phase.py <arm> ckpt <seed> <checkpoint-N> <val|test>
"""
import os
import sys
from pathlib import Path

from ab.gpt.act.acc_predictor import config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # folds.py lives in LLM-PP/
from folds import FOLDS  # noqa: E402

ARMS = {
    # 2 early epochs + proxies, no code: the base configuration (the original runs used the package defaults)
    "e2p": dict(USE_SOURCE_CODE=False, USE_THIRD_EARLY_EPOCH=False, USE_ZERO_COST_PROXIES=True),
    "code2p": dict(USE_SOURCE_CODE=True, USE_THIRD_EARLY_EPOCH=False, USE_ZERO_COST_PROXIES=True),
    "code3p": dict(USE_SOURCE_CODE=True, USE_THIRD_EARLY_EPOCH=True, USE_ZERO_COST_PROXIES=True),
    "e3np": dict(USE_SOURCE_CODE=False, USE_THIRD_EARLY_EPOCH=True, USE_ZERO_COST_PROXIES=False),
    "e3p_ro": dict(USE_SOURCE_CODE=False, USE_THIRD_EARLY_EPOCH=True, USE_ZERO_COST_PROXIES=True, TRAIN_ON_RESPONSES_ONLY=True),
    "code3p_ro": dict(USE_SOURCE_CODE=True, USE_THIRD_EARLY_EPOCH=True, USE_ZERO_COST_PROXIES=True, TRAIN_ON_RESPONSES_ONLY=True),
    "e3p": dict(USE_SOURCE_CODE=False, USE_THIRD_EARLY_EPOCH=True, USE_ZERO_COST_PROXIES=True),
    "e3p_ro_b1": dict(USE_SOURCE_CODE=False, USE_THIRD_EARLY_EPOCH=True, USE_ZERO_COST_PROXIES=True, TRAIN_ON_RESPONSES_ONLY=True),
    "e3p_all": dict(USE_SOURCE_CODE=False, USE_THIRD_EARLY_EPOCH=True, USE_ZERO_COST_PROXIES=True, SAVE_TOTAL_LIMIT=None),
    "e3p_ro_all": dict(USE_SOURCE_CODE=False, USE_THIRD_EARLY_EPOCH=True, USE_ZERO_COST_PROXIES=True, TRAIN_ON_RESPONSES_ONLY=True,
                       SAVE_TOTAL_LIMIT=None),
}
# Arms without code that still use the code arms' batch layout.
BATCH1_ARMS = {"e3p_ro_b1"}
# Training settings shared by every arm, passed explicitly so later changes to repo defaults cannot leak in:
# warm-up = 2% of the 3,200 scheduled steps (the original recipe's ratio; the repo default later became 2 steps).
TRAIN_KWARGS = dict(warmup_steps=64)
# Source-code prompts reach 6,144 tokens: per-device batch 1 with 32 accumulation steps keeps the effective batch at 32.
CODE_TRAIN_KWARGS = dict(batch_size=1, gradient_accumulation_steps=32)

arm, phase, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
if arm not in ARMS:
    sys.exit(f"unknown arm {arm}; choose from {sorted(ARMS)}")
config.SEED = seed
for name, value in ARMS[arm].items():
    setattr(config, name, value)
fold = os.environ.get("FOLD")
if fold:
    for name, value in FOLDS[fold].items():
        setattr(config, name, value)
    config.PROXY_NORM_STATS_PATH = config.ACC_DIR.parent / "acc_predict_folds" / fold / "proxy_norm_stats.json"
    config.ACC_DIR = config.ACC_DIR.parent / "acc_predict_folds" / fold / arm
else:
    config.ACC_DIR = config.ACC_DIR.parent / "acc_predict_ablation" / arm
# Importing the package already ran `from .config import *` in its submodules, so names they read unqualified
# (family sets, PROXY_NORM_STATS_PATH, SAVE_TOTAL_LIMIT) must be overridden there too.
overrides = dict(ARMS[arm], **(FOLDS[fold] if fold else {}),
                 PROXY_NORM_STATS_PATH=config.PROXY_NORM_STATS_PATH, ACC_DIR=config.ACC_DIR)
for mod_name, mod in list(sys.modules.items()):
    if mod_name.startswith("ab.gpt.act.acc_predictor.") and mod is not config:
        for name, value in overrides.items():
            if hasattr(mod, name):
                setattr(mod, name, value)

from ab.gpt.act.acc_predictor.__main__ import _run_tag
from ab.gpt.act.acc_predictor.preprocess import data_preprocessing
from ab.gpt.act.acc_predictor.dataset import prepare_llm_datasets
from ab.gpt.act.acc_predictor.train import train_model
from ab.gpt.act.acc_predictor.eval import test_model
from frozen_export import write_export

run_dir = config.ACC_DIR / _run_tag()
run_dir.mkdir(parents=True, exist_ok=True)
print(f"Run: arm={arm} fold={fold or 'main'} model={config.MODEL_NAME} seed={config.SEED} phase={phase} {ARMS[arm]}")
print(f"Outputs -> {run_dir}")

if phase in ("prep", "train"):
    print("=" * 80 + "\nStep 1/4: Data preprocessing (nn_dataset → JSONL)\n" + "=" * 80)
    if not write_export(run_dir / "llm_finetuning_data.jsonl"):  # RAW=<frozen export> (frozen_export.py)
        data_preprocessing(output_jsonl_path=run_dir / "llm_finetuning_data.jsonl",
                           output_csv_path=run_dir / "llm_finetuning_data.csv")
    print("=" * 80 + "\nStep 2/4: Prepare LLM training datasets\n" + "=" * 80)
    train_path, val_path, _ = prepare_llm_datasets(input_path=run_dir / "llm_finetuning_data.jsonl", output_dir=run_dir)
if phase == "train":
    print("=" * 80 + "\nStep 3/4: Fine-tune model\n" + "=" * 80)
    extra = dict(TRAIN_KWARGS, **(CODE_TRAIN_KWARGS if ARMS[arm].get("USE_SOURCE_CODE") or arm in BATCH1_ARMS else {}))
    print(f"train_model overrides: {extra}")
    train_model(train_path=train_path, val_path=val_path, output_dir=run_dir / "tuned_model",
                model_name=config.MODEL_NAME, seed=config.SEED, **extra)
elif phase == "test":
    print("=" * 80 + "\nStep 4/4: Test model on held-out test set\n" + "=" * 80)
    test_model(model_path=run_dir / "tuned_model", data_path=run_dir / "test_llm_dataset.jsonl",
               output_path=run_dir / "test_predictions.csv", metrics_path=run_dir / "test_metrics.log")
    print(f"\nDone. Pipeline complete. Results in {run_dir}")
elif phase == "ckpt":
    ckpt, split = sys.argv[4], sys.argv[5]
    out = run_dir / "ckpt_eval"
    out.mkdir(exist_ok=True)
    test_model(model_path=run_dir / "tuned_model" / ckpt, data_path=run_dir / f"{split}_llm_dataset.jsonl",
               output_path=out / f"{ckpt}_{split}_predictions.csv", metrics_path=out / f"{ckpt}_{split}_metrics.log")
elif phase != "prep":
    sys.exit(f"unknown phase {phase}")
