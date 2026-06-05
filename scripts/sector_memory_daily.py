"""
Sector Memory Engine — daily maintenance script.

Run after nightly_sync completes (7:30 PM IST) to:
  1. record   — snapshot today's sector rotation signals + regime fingerprint
  2. fill     — compute forward returns for records now old enough (≥7/14/30 days)

Usage
-----
  python -m scripts.sector_memory_daily record
  python -m scripts.sector_memory_daily fill
  python -m scripts.sector_memory_daily both       (default when no arg given)

Designed to be called from auto_update.py or a scheduled task after the
nightly bhavcopy fetch. Both operations are idempotent and safe to re-run.
"""
from __future__ import annotations

import sys
from datetime import date, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IST = timezone(timedelta(hours=5, minutes=30))


def _today_ist() -> date:
    return date.today()


def run_record(trade_date: date) -> None:
    from src.analytics.sector_memory import record_daily_snapshot
    n = record_daily_snapshot(trade_date)
    if n == 0:
        print(f"[SectorMemory] Already recorded for {trade_date} (or no data).")
    else:
        print(f"[SectorMemory] Recorded {n} sector rows for {trade_date}.")


def run_fill(trade_date: date) -> None:
    from src.analytics.sector_memory import fill_forward_outcomes
    n = fill_forward_outcomes(trade_date)
    print(f"[SectorMemory] Filled forward outcomes for {n} records.")


def main() -> None:
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "both"
    today = _today_ist()

    if mode in ("record", "both"):
        run_record(today)
    if mode in ("fill", "both"):
        run_fill(today)
    if mode not in ("record", "fill", "both"):
        print(f"Unknown mode '{mode}'. Use: record | fill | both")
        sys.exit(1)


if __name__ == "__main__":
    main()
