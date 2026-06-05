"""
Sector Memory Engine — one-time historical backfill.

Populates sector_rotation_log + sector_regime_log for all trading days in
[start_date, end_date] using a single efficient DuckDB window-function query,
then fills forward returns for records old enough to have measurable outcomes.

Usage
-----
  python scripts/sector_memory_backfill.py                   # backfill last 365 days
  python scripts/sector_memory_backfill.py --days 180        # last 180 days
  python scripts/sector_memory_backfill.py --from 2024-01-01 # from a specific date
  python scripts/sector_memory_backfill.py --no-fill         # skip outcome filling

Idempotent: uses INSERT OR IGNORE so re-running is safe.
Re-running with --fill-only only fills outcomes (no re-recording).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill sector rotation memory")
    p.add_argument("--days",      type=int,   default=365,
                   help="Number of calendar days to backfill (default: 365)")
    p.add_argument("--from",     dest="from_date", type=str, default=None,
                   help="Start date YYYY-MM-DD (overrides --days)")
    p.add_argument("--to",       dest="to_date",   type=str, default=None,
                   help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--no-fill",  dest="fill",      action="store_false",
                   help="Skip filling forward outcomes after recording")
    p.add_argument("--fill-only", action="store_true",
                   help="Only fill forward outcomes, do not re-record signals")
    p.add_argument("--min-turnover", type=float, default=1.0,
                   help="Minimum stock turnover filter in lacs (default: 1.0)")
    p.set_defaults(fill=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    today = date.today()

    to_date = (date.fromisoformat(args.to_date) if args.to_date else today)
    if args.from_date:
        from_date = date.fromisoformat(args.from_date)
    else:
        from_date = to_date - timedelta(days=args.days)

    print(f"[SectorMemory] Backfill: {from_date} → {to_date}")

    if args.fill_only:
        from src.analytics.sector_memory import fill_forward_outcomes
        n = fill_forward_outcomes(to_date, args.min_turnover)
        print(f"[SectorMemory] Filled outcomes for {n} records.")
        return

    from src.analytics.sector_memory import backfill_sector_memory
    result = backfill_sector_memory(
        end_date           = to_date,
        start_date         = from_date,
        min_turnover_lacs  = args.min_turnover,
        fill_outcomes      = args.fill,
        verbose            = True,
    )
    print(
        f"[SectorMemory] Done — "
        f"{result['dates_processed']} dates processed, "
        f"{result['rows_recorded']} rows recorded, "
        f"{result['outcomes_filled']} outcomes filled."
    )


if __name__ == "__main__":
    main()
