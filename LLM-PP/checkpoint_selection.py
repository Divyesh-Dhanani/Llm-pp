"""Checkpoint-selection control: does answer-only loss help, or does it only select a different epoch?

Qwen3-8B, 3 early epochs + proxies, trained with the full-text loss and with answer-only loss (seeds 0, 42, 123), keeping
every epoch's checkpoint (runs/acc_predict_ablation/{e3p_all,e3p_ro_all}); each checkpoint was evaluated on the validation
and the test split (<run>/ckpt_eval/). Selection rules: lowest validation loss (the training recipe), lowest validation
MAE of best accuracy, and a fixed epoch.

Writes results/checkpoint_selection.csv (one row per loss x rule: test R2/MAE mean and SD over seeds, epochs
chosen) and checkpoint_epochs.csv (every checkpoint).

Usage: python checkpoint_selection.py
"""
import glob
import json
import re

import numpy as np
import pandas as pd

from _paths import RESULTS as RES, RUNS as OUT
ARMS = {"full-text loss": "e3p_all", "answer-only loss": "e3p_ro_all"}
RUN = "Qwen3-8B-unsloth-bnb-4bit_seed{}"


def score(path):
    d = pd.read_csv(path)
    assert d.pred_best_accuracy.notna().all(), f"parse failures in {path}"
    g, p = d.gt_best_accuracy.to_numpy(float), d.pred_best_accuracy.to_numpy(float)
    return 1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum(), np.abs(g - p).mean()


rows = []
for loss, arm in ARMS.items():
    for seed in (0, 42, 123):
        run = OUT / "acc_predict_ablation" / arm / RUN.format(seed)
        state = json.loads((run / "trainer_state.json").read_text())  # trainer log of the run's last checkpoint
        val_loss = {l["step"]: l["eval_loss"] for l in state["log_history"] if "eval_loss" in l}
        steps = sorted(int(re.search(r"checkpoint-(\d+)_test", f).group(1)) for f in glob.glob(f"{run}/ckpt_eval/*_test_predictions.csv"))
        for step in steps:
            vr2, vmae = score(run / f"ckpt_eval/checkpoint-{step}_val_predictions.csv")
            tr2, tmae = score(run / f"ckpt_eval/checkpoint-{step}_test_predictions.csv")
            rows.append({"loss": loss, "seed": seed, "epoch": steps.index(step) + 1, "step": step, "val_loss": val_loss[step],
                         "val_r2": vr2, "val_mae": vmae, "test_r2": tr2, "test_mae": tmae})
ck = pd.DataFrame(rows)
RES.mkdir(parents=True, exist_ok=True)
ck.to_csv(RES / "checkpoint_epochs.csv", index=False)

rules = {"lowest validation loss (recipe)": lambda g: g.loc[g.val_loss.idxmin()],
         "lowest validation MAE": lambda g: g.loc[g.val_mae.idxmin()],
         "fixed epoch 1": lambda g: g[g.epoch == 1].iloc[0],
         "fixed epoch 2": lambda g: g[g.epoch == 2].iloc[0],
         "fixed epoch 3": lambda g: g[g.epoch == 3].iloc[0]}
summary = []
for loss in ARMS:
    for rule, pick in rules.items():
        chosen = pd.DataFrame([pick(g) for _, g in ck[ck.loss == loss].groupby("seed")])
        summary.append({"loss": loss, "rule": rule, "epochs": " ".join(str(int(e)) for e in chosen.epoch),
                        "test_r2_mean": chosen.test_r2.mean(), "test_r2_sd": chosen.test_r2.std(ddof=1),
                        "test_mae_mean": chosen.test_mae.mean(), "test_mae_sd": chosen.test_mae.std(ddof=1)})
summary = pd.DataFrame(summary)
summary.to_csv(RES / "checkpoint_selection.csv", index=False)
pd.set_option("display.width", 200)
print(ck.round(4).to_string(index=False))
print()
print(summary.round(4).to_string(index=False))
