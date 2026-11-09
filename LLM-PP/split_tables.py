"""Protocol tables: split sizes by family, dataset and task; proxy coverage per split; label statistics.

Usage: python split_tables.py
"""

import pandas as pd

from _paths import RESULTS as RES, RUNS as OUT
tab = pd.read_csv(OUT / "paper/tabular_3epoch.csv")
order = ["train", "val", "test"]
tables = {
    "split_by_family": pd.crosstab(tab.family, tab.split)[order],
    "split_by_dataset": pd.crosstab(tab.dataset, tab.split)[order],
    "split_by_task": pd.crosstab(tab.task, tab.split)[order],
    "unique_architectures": tab.groupby("split").nn.nunique().reindex(order).to_frame("unique nn"),
    "proxy_coverage_pct": (100 * tab.filter(like="proxy_").notna().groupby(tab.split).mean()).reindex(order).round(1),
    "labels": tab.groupby("split")[["best_accuracy", "epochs_to_best", "total_epochs"]].agg(["mean", "median", "std"]).reindex(order).round(3),
}
with open(RES / "split_tables.md", "w") as f:
    for name, t in tables.items():
        t.to_csv(RES / f"{name}.csv")
        f.write(f"## {name}\n\n{t.to_markdown()}\n\n")
        print(f"== {name}\n{t.to_string()}\n")
