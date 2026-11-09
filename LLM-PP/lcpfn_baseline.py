"""Learning-curve extrapolation baseline: LC-PFN (Adriaensen et al., NeurIPS 2023), pre-trained, no fine-tuning.

For every test run, LC-PFN sees the validation accuracy of the first 2 or 3 epochs (x = epoch index, as in the
authors' example) and extrapolates to the run's last epoch; the prediction of best accuracy is the extrapolated
value at the last epoch (LC-PFN's prior is over increasing curves, so this is also its predicted maximum). It does
not predict a best epoch; we record the last epoch, i.e. "train to the end".

Writes runs/paper/lcpfn/lcpfn_e{2,3}_seed0_test_predictions.csv in the same format as the other methods.
Requires the LC-PFN code (github.com/automl/lcpfn) on PYTHONPATH; nothing is installed into the shared environment.

Usage: PYTHONPATH=~/ext/lcpfn python lcpfn_baseline.py [<tabular.csv> <out_dir>]
"""
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# LC-PFN stores a full pickled model; PyTorch >= 2.6 refuses that unless weights_only=False.
torch.load = functools.partial(torch.load, weights_only=False)
import lcpfn  # noqa: E402
import importlib  # noqa: E402

# LC-PFN's custom encoder layer predates the is_causal argument that newer nn.TransformerEncoder passes; drop it.
for mod_name in ("lcpfn.layer", "layer"):
    try:
        layer_cls = importlib.import_module(mod_name).TransformerEncoderLayer
    except ImportError:
        continue
    if not getattr(layer_cls.forward, "_drops_is_causal", False):
        original = layer_cls.forward

        def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None, _orig=original):
            return _orig(self, src, src_mask, src_key_padding_mask)

        forward._drops_is_causal = True
        layer_cls.forward = forward

from _paths import RUNS as OUT
DEST = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "paper/lcpfn"
DEST.mkdir(parents=True, exist_ok=True)

tab = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else OUT / "paper/tabular_3epoch.csv")
test = tab[tab.split == "test"].reset_index(drop=True)
gt_acc = test.best_accuracy.astype(float).round(2).to_numpy()  # same rounding as the LLM ground truth
gt_ep = test.epochs_to_best.astype(float).to_numpy()
model = lcpfn.LCPFN()
# Modules pickled under PyTorch <= 1.11 lack attributes added later; fill them with the defaults that reproduce the
# old behaviour (GELU without approximation).
for m in model.modules():
    if isinstance(m, torch.nn.GELU) and "approximate" not in m.__dict__:
        m.approximate = "none"

for k in (2, 3):
    preds = []
    for _, r in test.iterrows():
        y = torch.tensor([[float(r[f"accuracy_epoch_{e}"])] for e in range(1, k + 1)]).clamp(0, 1)
        x = torch.arange(1, k + 1, dtype=torch.float32).unsqueeze(1)
        last = int(r.total_epochs)
        x_test = torch.tensor([[float(last)]])
        preds.append(float(model.predict_mean(x_train=x, y_train=y, x_test=x_test).squeeze()))
    p = np.clip(np.array(preds), 0, 1)
    df = pd.DataFrame({"dataset": test.dataset, "pred_best_accuracy": p, "gt_best_accuracy": gt_acc,
                       "pred_best_epoch": test.total_epochs.astype(float), "gt_best_epoch": gt_ep})
    df.to_csv(DEST / f"lcpfn_e{k}_seed0_test_predictions.csv", index=False)
    ss = ((gt_acc - gt_acc.mean()) ** 2).sum()
    print(f"LC-PFN, {k} epochs: R2={1 - ((gt_acc - p) ** 2).sum() / ss:.4f}  MAE={np.abs(gt_acc - p).mean():.4f}  "
          f"pearson={np.corrcoef(p, gt_acc)[0, 1]:.4f}")
