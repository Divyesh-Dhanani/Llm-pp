"""Which epoch's checkpoint each LLM run selected (lowest validation loss), from the trainer log of the run's last
checkpoint. Covers every main-split run whose trainer log is retained (the two-epoch Qwen3-8B and Llama-3.1-8B runs
were trained on a second machine and only their predictions are kept).

Writes results/selected_epochs.csv and prints the distribution.

Usage: python selected_epochs.py
"""
import json

import pandas as pd

from _paths import RESULTS as RES, RUNS

rows = []
for f in sorted(RUNS.glob("acc_predict*/**/trainer_state.json")):
    if "acc_predict_folds" in f.parts or "trunc1024" in str(f):
        continue
    state = json.loads(f.read_text())
    best = int(state["best_model_checkpoint"].rsplit("-", 1)[-1])
    evals = [l for l in state["log_history"] if "eval_loss" in l]
    rows.append({"run": str(f.parent.relative_to(RUNS)), "best_step": best,
                 "best_epoch": round([l["epoch"] for l in evals if l["step"] == best][0]), "epochs_trained": round(evals[-1]["epoch"])})
df = pd.DataFrame(rows)
df.to_csv(RES / "selected_epochs.csv", index=False)
counts = df.best_epoch.value_counts().sort_index()
print(f"runs with a trainer log: {len(df)}")
print("selected epoch:", ", ".join(f"{e}: {n}" for e, n in counts.items()))
print(f"epoch 1 or 2: {int(counts.get(1, 0) + counts.get(2, 0))} of {len(df)}")
