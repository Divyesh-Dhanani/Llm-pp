"""Export the cross-family dataset as one table for non-LLM baselines (XGBoost, trivial baselines).

One row per run, in the same order the LLM pipeline writes its split files, with: split, family,
task/dataset/metric, total_epochs, early accuracies (epochs 1-3), the 5 active zero-cost proxies
(normalized exactly as in the LLM prompt; missing = empty), and both targets.

Usage: python export_tabular.py <llm_finetuning_data.jsonl> <out.csv>
"""
import csv
import json
import sys

from ab.gpt.act.acc_predictor import config
from ab.gpt.act.acc_predictor.dataset import _attach_proxy_features, _load_proxy_cache, _load_proxy_norm_stats
from ab.gpt.act.acc_predictor.features import infer_arch_family

SPLIT_OF = {**{f: "train" for f in config.TRAIN_ARCH_FAMILIES},
            **{f: "val" for f in config.VAL_ARCH_FAMILIES},
            **{f: "test" for f in config.TEST_ARCH_FAMILIES}}
PROXIES = list(config.ACTIVE_PROXY_NAMES)
COLUMNS = (["split", "family", "nn", "task", "dataset", "metric", "total_epochs",
            "accuracy_epoch_1", "accuracy_epoch_2", "accuracy_epoch_3"]
           + [f"proxy_{p}" for p in PROXIES] + ["best_accuracy", "epochs_to_best"])

src, dst = sys.argv[1], sys.argv[2]
cache = _load_proxy_cache(config.PROXY_CACHE_PATH)
stats = _load_proxy_norm_stats(config.PROXY_NORM_STATS_PATH)
counts = {}
with open(src, encoding="utf-8") as f_in, open(dst, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=COLUMNS)
    writer.writeheader()
    for line in f_in:
        if not line.strip():
            continue
        rec = json.loads(line)
        family = infer_arch_family(rec.get("nn_code", "") or "", rec.get("task", "") or "")
        split = SPLIT_OF.get(family)
        if split is None:
            continue
        _attach_proxy_features(rec, cache, stats)
        row = {c: rec.get(c, "") for c in COLUMNS}
        row.update(split=split, family=family)
        for p in PROXIES:
            row[f"proxy_{p}"] = rec.get(f"proxy_{p}", "")
        writer.writerow(row)
        counts[split] = counts.get(split, 0) + 1
print("rows per split:", counts, "| proxies:", PROXIES, "| cache:", config.PROXY_CACHE_PATH)
