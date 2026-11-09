"""Export the full validation curve of every test run of the main split, for plateau_metric.py.

Rebuilds the curves from LEMUR with the same grouping and filtering as the training pipeline and checks that each
curve reproduces the run's targets. Needs the nn-gpt code base (stage "inputs"); writes one row per (run, epoch).

Usage (inside an nn-gpt checkout, PYTHONPATH set to it):
    python export_curves.py <llm_finetuning_data.jsonl> [out.csv]
"""
import json
import sys

import pandas as pd

from ab.gpt.act.acc_predictor import config
from ab.gpt.act.acc_predictor.features import infer_arch_family
from ab.gpt.act.acc_predictor.preprocess import (_DP_GROUP_KEYS, _dp_filter_long_runs, _dp_load_nn_data,
                                                  _dp_normalize_dataframe)
from _paths import RUNS

raw = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else RUNS / "paper/test_curves.csv"

# test records, in pipeline order
test = []
for line in open(raw, encoding="utf-8"):
    rec = json.loads(line)
    if infer_arch_family(rec.get("nn_code", "") or "", rec.get("task", "") or "") in config.TEST_ARCH_FAMILIES:
        test.append(rec)
need = {r["id"] for r in test}

# full curves, grouped exactly like _dp_build_records
df, _, _ = _dp_filter_long_runs(_dp_normalize_dataframe(_dp_load_nn_data()))
curves = {}
for _, g in df.groupby(list(_DP_GROUP_KEYS), observed=True):
    g = g.sort_values("epoch").drop_duplicates("epoch", keep="first")
    rid = str(g.iloc[0]["id"])
    if rid in need:
        curves[rid] = g.set_index("epoch")["accuracy"].astype(float)

missing = [r["id"] for r in test if r["id"] not in curves]
if missing:  # LEMUR has changed since the study: keep the committed curves rather than write an incomplete file
    sys.exit(f"{len(missing)} of {len(test)} test runs are not in the local LEMUR database (it is rebuilt from the current "
             f"records); {out} was not changed. The committed file holds the curves of the LEMUR snapshot of the study.")

rows = []
for i, r in enumerate(test):
    c = curves[r["id"]]
    assert abs(c.max() - r["best_accuracy"]) < 1e-9 and int(c.idxmax()) == r["epochs_to_best"], r["id"]
    rows += [{"test_index": i, "id": r["id"], "nn": r["nn"], "dataset": r["dataset"], "total_epochs": r["total_epochs"],
              "best_accuracy": r["best_accuracy"], "epochs_to_best": r["epochs_to_best"], "epoch": int(e), "accuracy": a}
             for e, a in c.items()]
pd.DataFrame(rows).to_csv(out, index=False)
print(f"test runs: {len(test)}, curves: {len(curves)}, rows written: {len(rows)} -> {out}")
