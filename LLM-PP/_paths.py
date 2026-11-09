"""Locations of every input and output, relative to this folder, so that each script runs from any directory.

runs/     mirror of the training pipeline's out/ directory, reduced to small files: per-run test predictions and
          metrics, checkpoint evaluations, trainer logs, the tabular exports and the baseline predictions
results/  everything the analysis scripts write (CSV/TXT), plus results/tables (LaTeX) and results/figures (PDF)
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RESULTS = HERE / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
for _d in (RESULTS, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
