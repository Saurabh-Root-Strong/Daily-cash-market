"""Fill the SIX trading sessions missing from daily_data / index_data / fno_bhavcopy.

HOW THEY WERE FOUND, and why they are not a judgement call. Scanning every
session since 2018 for stocks whose `prev_close` does not match the previous
close we hold:

    date         prior in db    stocks   avg gap   what it is
    2020-02-01   2020-01-31      1,298     3.35%   Budget day (Sat)
    2022-08-08   2022-08-05      1,346     2.49%   an ORDINARY MONDAY
    2023-11-12   2023-11-10      1,341     1.96%   Muhurat / Diwali (Sun)
    2024-03-02   2024-03-01      1,166     9.55%   special live session (Sat)
    2025-02-01   2025-01-31      1,522     2.70%   Budget day (Sat)
    2026-02-01   2026-01-30      2,017     2.83%   Budget day (Sun)

NSE serves all six. Leaving them out is NOT the safe option: the session AFTER
each one carries a `prev_close` that exists nowhere in our table, so its return
is computed against a price we do not hold and every multi-day window spanning
the date is wrong. On 2026-02-02 that error is 3.6pp for RELIANCE alone
(1395.4 -> 1390.4 is -0.36% over two days; our rows imply +3.22%).

2022-08-08 is the one that settles the argument. That is a plain Monday - not
Budget, not Muhurat. 2022-08-09 was Muharram and correctly 404s. The 8th is an
ordinary trading day that was simply never ingested, so this is a gap to close,
not a decision about whether special sessions "count".

Uses the normal ingestion path (fetch_one_date -> bhavcopy + delivery,
IndexFetcher, FNOBhavCopyFetcher), so every write goes through the same guards
as the daily job. Reversible: DELETE those trade_dates to undo.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime as dt
import pandas as pd

from src.ingestion.http_client import NSEHttpClient
from src.ingestion.orchestrator import fetch_one_date
from src.ingestion.index_fetcher import IndexFetcher
from src.ingestion.fno_bhavcopy_fetcher import FNOBhavCopyFetcher
from src.data.repository import (get_repository, upsert_index_data,
                                 upsert_fno_bhavcopy, query_dataframe)

MISSING = [dt.date(2020, 2, 1), dt.date(2022, 8, 8), dt.date(2023, 11, 12),
           dt.date(2024, 3, 2), dt.date(2025, 2, 1), dt.date(2026, 2, 1)]


def chain_report(label: str) -> pd.DataFrame:
    """Sessions whose prev_close disagrees with the previous close we hold."""
    return query_dataframe("""
        WITH s AS (
          SELECT trade_date, symbol, prev_close,
                 LAG(close_price) OVER (PARTITION BY symbol ORDER BY trade_date) pc
          FROM daily_data WHERE series='EQ' AND close_price>0 AND prev_close>0
            AND trade_date >= '2018-01-01')
        SELECT trade_date, COUNT(*) AS broken_symbols,
               ROUND(AVG(ABS(prev_close-pc)/pc*100), 2) AS avg_gap_pct
        FROM s WHERE pc IS NOT NULL AND ABS(prev_close-pc)/pc > 0.005
        GROUP BY 1 HAVING COUNT(*) > 400 ORDER BY 1
    """)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    print("BEFORE — sessions with a broken prev_close chain:")
    before = chain_report("before")
    print(before.to_string(index=False) if not before.empty else "  none")

    if a.dry_run:
        return 0

    repo = get_repository()
    have = {d if isinstance(d, dt.date) else pd.to_datetime(d).date()
            for d in repo.get_distinct_dates("daily_data")}
    client = NSEHttpClient()
    idxf, fnof = IndexFetcher(), FNOBhavCopyFetcher(client)

    print("\nFILLING")
    for d in MISSING:
        if d in have:
            print(f"  {d} already present, skipped")
            continue
        try:
            status, rows = fetch_one_date(d, client, repo)
            print(f"  {d} {d:%a}  daily_data: {status}, {rows:,} rows")
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {d} daily_data FAILED: {exc}")
            continue
        try:
            idf = idxf.fetch(d)
            n = upsert_index_data(idf) if not idf.empty else 0
            print(f"           index_data: {n} indices")
        except Exception as exc:                                  # noqa: BLE001
            print(f"           index_data FAILED: {exc}")
        try:
            fdf = fnof.fetch(d)
            n = upsert_fno_bhavcopy(fdf) if fdf is not None and not fdf.empty else 0
            print(f"           fno_bhavcopy: {len(fdf) if fdf is not None else 0:,} rows")
        except Exception as exc:                                  # noqa: BLE001
            print(f"           fno_bhavcopy FAILED: {exc}")

    print("\nAFTER — sessions with a broken prev_close chain:")
    after = chain_report("after")
    print(after.to_string(index=False) if not after.empty else "  NONE — every chain repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
