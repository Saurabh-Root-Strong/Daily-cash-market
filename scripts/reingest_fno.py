"""
One-off: re-ingest ALL F&O bhavcopy via the corrected UDiFF parser, OVERWRITING
existing rows (unlike backfill-fno which skips existing). Needed because:
  - pre-UDiFF rows had an inconsistent OI scale, and
  - the first UDiFF pass mapped settle_price to SttlmPric (same-day) instead of
    PrvsClsgPric (prev close).
Re-fetching every date makes OI scale consistent and settle_price = prev close.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.orchestrator import NSEHttpClient
from src.ingestion.fno_bhavcopy_fetcher import FNOBhavCopyFetcher
from src.data.repository import get_repository


def weekdays_back(n: int) -> list[date]:
    out, d = [], date.today()
    while len(out) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    client = NSEHttpClient()
    repo = get_repository()
    fetcher = FNOBhavCopyFetcher(client)
    dates = sorted(weekdays_back(n))
    ok = none = fail = 0
    for i, d in enumerate(dates, 1):
        try:
            df = fetcher.fetch(d)
            if df is not None and not df.empty:
                repo.upsert_fno_bhavcopy(df)
                ok += 1
                print(f"[{i}/{len(dates)}] {d}  OK {len(df)} rows", flush=True)
            else:
                none += 1
                print(f"[{i}/{len(dates)}] {d}  -- no data (holiday/none)", flush=True)
        except Exception as exc:
            fail += 1
            print(f"[{i}/{len(dates)}] {d}  !! {type(exc).__name__}: {exc}", flush=True)
    print(f"DONE reingest: OK={ok} none={none} fail={fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
