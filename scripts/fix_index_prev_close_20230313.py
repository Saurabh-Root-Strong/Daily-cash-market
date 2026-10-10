"""Repair index_data for 13 Mar 2023: prev_close was stored as the 09 Mar close.

Found 10 Oct 2026 while auditing the Index & Large Cap match table. On that one
session NSE's file carried the close of TWO sessions earlier as "previous close",
so points_chg and pct_chg span two days: Nifty 50 reads -2.47% where the real
move from 10 Mar (17,412.90) to 13 Mar (17,154.30) was -1.49%. 10 Mar and 13 Mar
both have rows for every index, so this is a wrong field, not a missing session.

Every panel that reads index_data.pct_chg inherits it: the carry spread for that
date is off by ~1pp, and the next-day / next-5-day outcomes of 10 Mar and the
four sessions before it are off in every base-rate and analogue table.

Idempotent: only rows whose prev_close disagrees with the 10 Mar close are
touched, and each is rewritten from the two closes. Run with --apply to write;
without it the script only reports. The dashboard must not hold the database.

    python scripts/fix_index_prev_close_20230313.py           # dry run
    python scripts/fix_index_prev_close_20230313.py --apply
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BAD, PRIOR = date(2023, 3, 13), date(2023, 3, 10)

_FIND = """
    SELECT b.index_name, b.close_val, b.prev_close AS stored_prev,
           p.close_val AS true_prev, b.pct_chg AS stored_pct,
           ROUND((b.close_val / p.close_val - 1) * 100, 2) AS true_pct
    FROM index_data b JOIN index_data p
      ON p.index_name = b.index_name AND p.trade_date = ?
    WHERE b.trade_date = ? AND p.close_val > 0
      AND ABS(b.prev_close - p.close_val) / p.close_val > 0.0005
    ORDER BY 1
"""

_FIX = """
    UPDATE index_data AS b
    SET prev_close = p.close_val,
        points_chg = ROUND(b.close_val - p.close_val, 2),
        pct_chg    = ROUND((b.close_val / p.close_val - 1) * 100, 2)
    FROM index_data AS p
    WHERE p.index_name = b.index_name AND p.trade_date = ? AND b.trade_date = ?
      AND p.close_val > 0
      AND ABS(b.prev_close - p.close_val) / p.close_val > 0.0005
"""


def main(apply: bool) -> int:
    import duckdb
    from src.core.config import get_config

    path = str(get_config().database.resolved_path)
    # no session may sit between the two -- otherwise 10 Mar is not "previous"
    con = duckdb.connect(path, read_only=not apply)
    try:
        between = con.execute(
            "SELECT COUNT(*) FROM index_data WHERE trade_date > ? AND trade_date < ?",
            [PRIOR, BAD]).fetchone()[0]
        if between:
            print(f"{between} index rows sit between {PRIOR} and {BAD} - refusing.")
            return 1
        rows = con.execute(_FIND, [PRIOR, BAD]).fetchdf()
        if rows.empty:
            print("Nothing to fix - every prev_close on 13 Mar 2023 matches 10 Mar.")
            return 0
        print(f"{len(rows)} index rows on {BAD} carry the wrong previous close:")
        print(rows.to_string(index=False))
        if not apply:
            print("\nDry run. Re-run with --apply to write.")
            return 0
        con.execute(_FIX, [PRIOR, BAD])
        left = con.execute(_FIND, [PRIOR, BAD]).fetchdf()
        print(f"\nFixed {len(rows) - len(left)} rows; {len(left)} still wrong.")
        return 0 if left.empty else 1
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main("--apply" in sys.argv))
