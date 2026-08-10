"""
Repair index_data: day/month swap corruption + missing special sessions.

Two defects, found 2026-08-07:

1. DAY/MONTH SWAP. Three April-2023 snapshots were written under a swapped
   date (%m%d%Y instead of %d%m%Y), so the real session went missing and its
   data landed on a wrong date:

       2023-04-06  ->  2023-06-04  (Sunday   - visible, no stock data)
       2023-04-10  ->  2023-10-04  (Wednesday- SILENT, overwrote a real session)
       2023-04-11  ->  2023-11-04  (Saturday - visible, no stock data)

   2023-10-04 is the dangerous one: a real trading day whose Nifty 50 reads
   17624.05 between 19528.75 and 19545.75, i.e. a fake -9.7% crash and +10.9%
   snapback in the middle of October 2023.

2. MISSING SESSIONS. Special sessions (Muhurat, budget Saturdays, NSE
   disaster-recovery live sessions) plus a few ordinary weekdays never got an
   index snapshot, though most have full stock data.

Verification used: prev_close(t) must equal close_val(t-1) per index. Dates
immediately AFTER a corrupt date also show a break; those are collateral and
self-heal once the corrupt row is corrected -- they are not repaired directly.

--dry-run (default): fetch every target date, report what WOULD change. No DB.
--load             : delete the corrupt dates, then upsert all fetched dates.
                     Requires the Streamlit app to be stopped (DuckDB is
                     single-writer). Idempotent: upsert is ON CONFLICT DO UPDATE.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.index_fetcher import IndexFetcher  # noqa: E402

DB = r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"

# Rows to remove: two impossible weekend dates + the silently-overwritten
# 2023-10-04 (deleted rather than upserted so stale index names cannot survive
# -- the corrupt snapshot carried 107 indices, the true one carries fewer).
DELETE_DATES = ["2023-06-04", "2023-11-04", "2023-10-04"]

# Every date to (re)fetch from NiftyIndices.
FETCH_DATES = [
    # --- true dates behind the swap ---
    "2023-04-06", "2023-04-10", "2023-04-11",
    # --- the real session that was overwritten ---
    "2023-10-04",
    # --- missing special / ordinary sessions (stock data present) ---
    "2019-10-27",  # Sun - Muhurat
    "2020-11-14",  # Sat - Muhurat
    "2023-04-12",  # Wed - present but re-fetched to confirm chain
    "2024-01-20",  # Sat - NSE special live session
    "2024-05-18",  # Sat - NSE special live session
    "2025-03-26",  # Wed
    "2025-05-27",  # Tue
    "2025-05-28",  # Wed
    # --- budget sessions: absent from BOTH index_data and daily_data.
    #     Note 2026 traded on Sunday 02-01, not Saturday 01-31 (which 404s). ---
    "2020-02-01", "2025-02-01", "2026-02-01",
]


def fetch_all() -> dict[str, pd.DataFrame]:
    f = IndexFetcher()
    out: dict[str, pd.DataFrame] = {}
    for s in FETCH_DATES:
        df = f.fetch(dt.date.fromisoformat(s))
        out[s] = df
    return out


def report(fetched: dict[str, pd.DataFrame], con: duckdb.DuckDBPyConnection) -> None:
    rows = []
    for s, df in fetched.items():
        have = con.execute(
            "select count(*) from index_data where trade_date = ?", [s]
        ).fetchone()[0]
        n50 = pd.DataFrame()
        if not df.empty:
            n50 = df[df["index_name"].str.strip() == "Nifty 50"]
        rows.append({
            "date": s,
            "dow": dt.date.fromisoformat(s).strftime("%a"),
            "in_db_now": have,
            "nse_serves": 0 if df.empty else len(df),
            "nifty50_close": None if n50.empty else float(n50.iloc[0]["close_val"]),
        })
    print("\n=== fetch report ===")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== rows queued for DELETE ===")
    for d in DELETE_DATES:
        n = con.execute(
            "select count(*) from index_data where trade_date = ?", [d]
        ).fetchone()[0]
        print(f"  {d} ({dt.date.fromisoformat(d).strftime('%a')}): {n} rows")

    empties = [s for s, df in fetched.items() if df.empty]
    if empties:
        print("\n  WARNING - NSE served nothing for:", ", ".join(empties))


def load(fetched: dict[str, pd.DataFrame]) -> None:
    con = duckdb.connect(DB, read_only=False)
    try:
        con.execute("BEGIN TRANSACTION")

        deleted = 0
        for d in DELETE_DATES:
            n = con.execute(
                "select count(*) from index_data where trade_date = ?", [d]
            ).fetchone()[0]
            con.execute("delete from index_data where trade_date = ?", [d])
            deleted += n
            print(f"  deleted {n:4d} rows for {d}")

        inserted = 0
        for s, df in fetched.items():
            if df.empty:
                print(f"  skip {s}: nothing served")
                continue
            d = df.copy()
            d["index_name"] = d["index_name"].str.strip()
            con.register("_idx_df", d)
            con.execute("""
                INSERT INTO index_data
                    (trade_date, index_name, open_val, high_val, low_val, close_val,
                     prev_close, points_chg, pct_chg, volume, turnover_cr,
                     pe_ratio, pb_ratio, div_yield)
                SELECT
                    trade_date, index_name, open_val, high_val, low_val, close_val,
                    prev_close, points_chg, pct_chg, volume, turnover_cr,
                    pe_ratio, pb_ratio, div_yield
                FROM _idx_df
                ON CONFLICT (trade_date, index_name) DO UPDATE SET
                    open_val    = excluded.open_val,
                    high_val    = excluded.high_val,
                    low_val     = excluded.low_val,
                    close_val   = excluded.close_val,
                    prev_close  = excluded.prev_close,
                    points_chg  = excluded.points_chg,
                    pct_chg     = excluded.pct_chg,
                    volume      = excluded.volume,
                    turnover_cr = excluded.turnover_cr,
                    pe_ratio    = excluded.pe_ratio,
                    pb_ratio    = excluded.pb_ratio,
                    div_yield   = excluded.div_yield
            """)
            con.unregister("_idx_df")
            inserted += len(d)
            print(f"  upserted {len(d):4d} rows for {s}")

        con.execute("COMMIT")
        print(f"\ndeleted {deleted}, upserted {inserted}")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def verify() -> None:
    con = duckdb.connect(DB, read_only=True)
    print("\n=== post-fix chain check (prev_close vs prior close, >2% break, >=10 indices) ===")
    print(con.sql("""
        with s as (
          select index_name, trade_date, close_val, prev_close,
                 lag(close_val) over (partition by index_name order by trade_date) prev_actual
          from index_data where close_val is not null and prev_close is not null
        )
        select trade_date, dayname(trade_date) dow, count(*) n_broken,
               round(max(abs(prev_close-prev_actual)/nullif(prev_actual,0)*100),2) max_pct_gap
        from s where prev_actual is not null
          and abs(prev_close-prev_actual)/nullif(prev_actual,0) > 0.02
        group by 1,2 having count(*) >= 10 order by 1
    """).df().to_string(index=False))

    print("\n=== Nifty 50 around the repaired windows ===")
    for lo, hi in (("2023-04-04", "2023-04-14"), ("2023-09-29", "2023-10-06"),
                   ("2023-11-01", "2023-11-07")):
        print(f"\n-- {lo} .. {hi} --")
        print(con.sql(f"""
            select trade_date, dayname(trade_date) dow, open_val, high_val, low_val,
                   close_val, prev_close, pct_chg
            from index_data where index_name='Nifty 50'
              and trade_date between DATE '{lo}' and DATE '{hi}' order by 1
        """).df().to_string(index=False))

    print("\n=== coverage ===")
    print(con.sql("""
        select count(distinct trade_date) n_days, min(trade_date) d0, max(trade_date) d1
        from index_data""").df().to_string(index=False))
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="write to the database")
    ap.add_argument("--verify-only", action="store_true", help="run the chain check only")
    a = ap.parse_args()

    if a.verify_only:
        verify()
        return

    print("fetching", len(FETCH_DATES), "dates from NiftyIndices ...")
    fetched = fetch_all()

    con = duckdb.connect(DB, read_only=True)
    try:
        report(fetched, con)
    finally:
        con.close()

    if not a.load:
        print("\nDRY RUN - nothing written. Re-run with --load (stop Streamlit first).")
        return

    print("\nwriting ...")
    load(fetched)
    verify()


if __name__ == "__main__":
    main()
