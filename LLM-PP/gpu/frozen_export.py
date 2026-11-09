"""Use the frozen LEMUR export of the study instead of rebuilding it from the local LEMUR database.

LEMUR is a living dataset: nn-dataset builds its local database (db/ab.nn.db) from the current records, so an export
made today differs from the one every run of the study used. That export ships as runs/paper/lemur_export.jsonl.gz
(md5 of the uncompressed file: caedbd7723f6...). Set RAW=/path/to/lemur_export.jsonl.gz (or an uncompressed .jsonl)
and the runners copy it into the run folder instead of calling data_preprocessing().
"""
import gzip
import os
import shutil
from pathlib import Path


def write_export(jsonl_path: Path) -> bool:
    """Copy the frozen export to jsonl_path if RAW is set; return whether it was used."""
    raw = os.environ.get("RAW")
    if not raw:
        return False
    opener = gzip.open if raw.endswith(".gz") else open
    with opener(raw, "rb") as src, open(jsonl_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"Frozen LEMUR export {raw} -> {jsonl_path}")
    return True
