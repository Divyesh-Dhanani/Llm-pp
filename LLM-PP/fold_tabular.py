"""Export the cross-family table (export_tabular.py) for an alternative held-out family from folds.FOLDS, with the
proxy normalization fitted on that fold's training families (<nn-gpt>/out/acc_predict_folds/<fold>/proxy_norm_stats.json, written by the LLM prep phase).

Usage: python fold_tabular.py <fold> <llm_finetuning_data.jsonl> <out.csv>
"""
import runpy
import sys
from pathlib import Path

from ab.gpt.act.acc_predictor import config
from folds import FOLDS

fold = sys.argv.pop(1)
for name, value in FOLDS[fold].items():
    setattr(config, name, value)
config.PROXY_NORM_STATS_PATH = config.ACC_DIR.parent / "acc_predict_folds" / fold / "proxy_norm_stats.json"
if not config.PROXY_NORM_STATS_PATH.exists():
    sys.exit(f"missing {config.PROXY_NORM_STATS_PATH}; run the fold's LLM prep phase first")
runpy.run_path(str(Path(__file__).with_name("export_tabular.py")), run_name="__main__")
