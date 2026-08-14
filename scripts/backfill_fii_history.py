"""
Backfill NSE F&O participant-wise OI/Volume + FII Derivatives Statistics history.

WHY: fao_participant / fii_derivatives_stats only start 2024-12-25 (the day the live
pipeline was switched on), while NSE serves these files date-addressably from
2012-01-02 (participant) and 2014-12-01 (FII stats). Every FII-positioning study so
far has run on ~320 days of one bull regime. This closes that gap.

Sources (same nsearchives file endpoints the cash backfill used — the HTML site 403s,
the file endpoints do not):
  participant : /content/nsccl/fao_participant_oi_DDMMYYYY.csv   (+ _vol_)
  FII stats   : /content/fo/fii_stats_DD-Mon-YYYY.xls
404 = market holiday / weekend -> skipped and remembered.

ADDITIVE + RESUMABLE: skips any trade_date already present in the target table, so it
can be interrupted and re-run. Uses the SHIPPED fetchers + repository upserts, so the
backfilled rows are byte-identical in shape to what the nightly pipeline writes.

SCHEMA NOTE (honest): the FII stats file carries only 4 aggregate categories
(INDEX/STOCK FUTURES+OPTIONS) until Jan-2023; per-index rows (NIFTY / BANKNIFTY /
FINNIFTY / MIDCPNIFTY) start 2023-02. Participant OI/VOL has 15 identical columns
across the whole archive.

LEAKAGE NOTE: both files publish AFTER the close (~18:00-19:30 IST). A row dated D is
only actionable from D+1. Any backtest on this data must lag one day.

Usage:
  python scripts/backfill_fii_history.py                # 2018-01-01 -> existing min
  python scripts/backfill_fii_history.py --start 2012-01-02
  python scripts/backfill_fii_history.py --only fao     # or --only fii
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                          # noqa: BLE001
    pass

from src.data.repository import get_repository, query_dataframe          # noqa: E402
from src.ingestion.fao_fetcher import FAOParticipantFetcher              # noqa: E402
from src.ingestion.fii_stats_fetcher import FIIStatsFetcher              # noqa: E402
from src.ingestion.http_client import NSEHttpClient                      # noqa: E402

# Earliest date NSE actually serves each report (probed 2026-07-31).
FIRST_FAO = date(2012, 1, 2)
FIRST_FII = date(2014, 12, 1)
SLEEP = 0.35                       # polite; NSE tolerates this comfortably


def existing_dates(table: str) -> set[date]:
    df = query_dataframe(f"SELECT DISTINCT trade_date FROM {table}")
    if df.empty:
        return set()
    return {d.date() if hasattr(d, "date") else d for d in df["trade_date"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None, help="default = day before the earliest row already stored")
    ap.add_argument("--only", choices=["fao", "fii"], default=None)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    fao_have, fii_have = existing_dates("fao_participant"), existing_dates("fii_derivatives_stats")
    end = date.fromisoformat(args.end) if args.end else (min(fao_have | fii_have) - timedelta(days=1))

    do_fao = args.only in (None, "fao")
    do_fii = args.only in (None, "fii")

    print(f"backfill window {start} -> {end}   fao={do_fao} fii={do_fii}")
    print(f"already stored: fao {len(fao_have)} days, fii {len(fii_have)} days", flush=True)

    repo = get_repository()
    client = NSEHttpClient()
    fao_f, fii_f = FAOParticipantFetcher(client), FIIStatsFetcher(client)

    d = start
    n_fao = n_fii = n_hol = n_err = 0
    t0 = time.time()
    while d <= end:
        if d.weekday() >= 5:                               # Sat/Sun never publish
            d += timedelta(days=1)
            continue

        got_any = False

        if do_fao and d >= FIRST_FAO and d not in fao_have:
            try:
                df = fao_f.fetch(d)
                if df is not None and not df.empty:
                    repo.upsert_fao_data(df)
                    n_fao += 1
                    got_any = True
            except Exception as exc:                       # noqa: BLE001
                n_err += 1
                print(f"  ! fao {d}: {exc}", flush=True)
            time.sleep(SLEEP)

        if do_fii and d >= FIRST_FII and d not in fii_have:
            try:
                df = fii_f.fetch(d)
                if df is not None and not df.empty:
                    repo.upsert_fii_stats(df)
                    n_fii += 1
                    got_any = True
            except Exception as exc:                       # noqa: BLE001
                n_err += 1
                print(f"  ! fii {d}: {exc}", flush=True)
            time.sleep(SLEEP)

        if not got_any:
            n_hol += 1

        if d.day == 1 or d.weekday() == 0 and d.day <= 7:
            print(f"  {d}  fao+{n_fao} fii+{n_fii} holidays/empty {n_hol} err {n_err} "
                  f"{time.time()-t0:.0f}s", flush=True)
        d += timedelta(days=1)

    print(f"\nDONE  fao days +{n_fao}  fii days +{n_fii}  empty/holiday {n_hol}  errors {n_err}  "
          f"{time.time()-t0:.0f}s")
    for t in ("fao_participant", "fii_derivatives_stats"):
        r = query_dataframe(f"SELECT MIN(trade_date) mn, MAX(trade_date) mx, "
                            f"COUNT(DISTINCT trade_date) nd FROM {t}")
        print(f"  {t:24s} {str(r.mn[0])[:10]} -> {str(r.mx[0])[:10]}  days {int(r.nd[0])}")


if __name__ == "__main__":
    main()
