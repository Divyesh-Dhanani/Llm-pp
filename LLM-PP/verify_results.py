#!/usr/bin/env python3
"""Re-run the CPU part of the pipeline and diff it against the committed outputs. Nothing in this checkout is
modified: the folder is copied to a temporary directory, results/ is emptied there, every analysis, table and figure
script runs, and each file it produces is compared byte for byte with the committed one under results/. With other
numpy/pandas versions a few CSV and JSON files can differ in the last digits of floats; those are compared numerically
(relative tolerance 1e-9) and listed, but count as reproduced. Tables and figures are always compared byte for byte.

    python verify_results.py               # analysis, tables, figures (~1 min)
    python verify_results.py --baselines   # also re-fit XGBoost on the five splits and compare its predictions
    python verify_results.py --keep        # leave the scratch copy in place

Exit status is 0 when every output matches, 1 otherwise. The LLM runs (gpu/) and the LEMUR exports (stage "inputs")
need a GPU or an nn-gpt checkout and are out of scope here; LC-PFN needs an external checkout and is not re-run.
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from folds import FOLDS

HERE = Path(__file__).resolve().parent
RTOL = 1e-9  # CSVs that differ only in the last digits of floats (other numpy/pandas versions) count as reproduced


def same_json(x, y) -> bool:
    """True if two parsed JSON values are equal, with floats equal up to RTOL."""
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(same_json(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(same_json(u, v) for u, v in zip(x, y))
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        return math.isclose(x, y, rel_tol=RTOL, abs_tol=1e-12) or (math.isnan(x) and math.isnan(y))
    return x == y


def same_numbers(a: Path, b: Path) -> bool:
    """True if two CSV or JSON files hold the same data, with floats equal up to RTOL."""
    if a.suffix == ".json":
        try:
            return same_json(json.loads(a.read_text()), json.loads(b.read_text()))
        except ValueError:
            return False
    try:
        x, y = pd.read_csv(a), pd.read_csv(b)
    except Exception:
        return False
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return False
    for c in x.columns:
        u, v = x[c], y[c]
        if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
            if not np.allclose(u.to_numpy(float), v.to_numpy(float), rtol=RTOL, atol=1e-12, equal_nan=True):
                return False
        elif not u.astype(str).equals(v.astype(str)):
            return False
    return True


def compare(old_root: Path, new_root: Path, pattern: str = "**/*", rounding: list | None = None) -> list[str]:
    """Relative paths whose bytes differ, are missing, or are new, with a short diff for text files. CSVs that match
    up to float rounding are not problems; they are appended to `rounding` instead."""
    old = {p.relative_to(old_root) for p in old_root.glob(pattern) if p.is_file()}
    new = {p.relative_to(new_root) for p in new_root.glob(pattern) if p.is_file()}
    problems = []
    for rel in sorted(old | new):
        a, b = old_root / rel, new_root / rel
        if rel not in new:
            problems.append(f"MISSING  {rel}")
        elif rel not in old:
            problems.append(f"NEW      {rel}")
        elif a.read_bytes() != b.read_bytes():
            if rel.suffix in {".csv", ".json"} and same_numbers(a, b):
                if rounding is not None:
                    rounding.append(str(rel))
                continue
            problems.append(f"DIFFERS  {rel}")
            if rel.suffix in {".csv", ".txt", ".tex", ".md", ".json"}:
                diff = difflib.unified_diff(a.read_text().splitlines(), b.read_text().splitlines(), "committed", "rerun",
                                            lineterm="", n=0)
                problems += [f"           {line}" for line in list(diff)[2:8]]
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", action="store_true", help="also re-fit XGBoost (needs xgboost)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch copy")
    args = ap.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="llmpp_verify_"))
    work = scratch / "LLM-PP"
    shutil.copytree(HERE, work, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".git"))
    shutil.rmtree(work / "results")
    env = {**os.environ, "PYTHON": sys.executable}
    problems, rounding = [], []
    try:
        if args.baselines:
            for d in [work / "runs/paper/xgboost_grid"] + [work / f"runs/paper/folds/{f}/xgboost_grid" for f in FOLDS]:
                shutil.rmtree(d)
            (work / "results").mkdir()
            subprocess.run(["bash", str(work / "run_all.sh"), "baselines"], check=True, env=env)
            for rel in ["runs/paper/xgboost_grid"] + [f"runs/paper/folds/{f}/xgboost_grid" for f in FOLDS]:
                problems += compare(HERE, work, f"{rel}/*", rounding)
        subprocess.run(["bash", str(work / "run_all.sh")], check=True, env=env)
        # console logs of the baseline stage are not committed
        skip = {f"{stem}_{f}_stdout.txt" if f else f"{stem}_stdout.txt"
                for stem in ("xgboost_grid", "lcpfn") for f in ("", *FOLDS)}
        problems += [p for p in compare(HERE / "results", work / "results", rounding=rounding)
                     if not any(p.endswith(s) for s in skip)]
    finally:
        if args.keep:
            print(f"scratch copy kept at {work}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)

    n = sum(1 for p in (HERE / "results").rglob("*") if p.is_file())
    if rounding:
        print(f"\n{len(rounding)} CSV/JSON files match only up to float rounding (relative difference < {RTOL:g}), as expected "
              "with other numpy/pandas versions:\n  " + "\n  ".join(rounding))
    if problems:
        print(f"\n{sum(1 for p in problems if not p.startswith(' '))} of {n} committed outputs do not match:")
        print("\n".join(problems))
        return 1
    print(f"\nAll {n} committed outputs under results/ reproduced" + (" (byte for byte unless listed above)" if rounding else " byte for byte")
          + (", and all XGBoost predictions." if args.baselines else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
