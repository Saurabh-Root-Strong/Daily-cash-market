"""
Append-only forward-test log (CSV).

A forward test is only worth anything if nobody can edit it after the fact, so
this file has exactly two operations: append a row ONCE per key, and read
everything back. There is no update and no delete. A second write for the same
(trade_date, symbol) is ignored -- the first one, stamped with the moment it was
written, is the record. That is the property the Index Prediction "live track
record" lacked: 86% of its rows were back-filled in one batch months later.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORWARD_DIR = PROJECT_ROOT / "data" / "forward_log"


def log_path(name: str) -> Path:
    return FORWARD_DIR / f"{name}.csv"


def read_log(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def append_once(path: Path, row: dict, key: tuple = ("trade_date", "symbol")) -> bool:
    """Append `row` unless a row with the same key already exists.

    Returns True if written. Columns are fixed by the first row ever written; a
    later row may not add or drop columns (it would silently misalign the file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: ("" if v is None else v) for k, v in row.items()}
    existing = read_log(path)
    if not existing.empty:
        if list(existing.columns) != list(row.keys()):
            raise ValueError(f"{path.name}: columns {list(row.keys())} do not match the "
                             f"log's {list(existing.columns)}")
        hit = existing
        for k in key:
            hit = hit[hit[k] == str(row[k])]
        if not hit.empty:
            return False
    new = not path.exists() or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow({k: str(v) for k, v in row.items()})
    return True
