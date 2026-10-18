"""Write tonight's Index & Large Cap forward-test forecast (Nifty, Bank Nifty).

Runs from scripts/run_daily.bat after the evening fetch. Append-only: a second run
for the same session changes nothing. A row written at or after 09:15 on the next
session is kept but never scored (see src/analytics/ilc_forward.py).

usage: python scripts/ilc_forward_log.py [YYYY-MM-DD] [--dry-run]
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                            # noqa: E402

from src.analytics import ilc_forward as F                     # noqa: E402
from src.data.repository import query_dataframe                # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if args:
        d = date.fromisoformat(args[0])
    else:
        v = query_dataframe("SELECT MAX(trade_date) AS d FROM fno_bhavcopy "
                            "WHERE instrument = 'FUTSTK'")["d"].iloc[0]
        d = pd.to_datetime(v).date()          # a pandas Timestamp IS a date subclass; normalise
    for sym in ("NIFTY", "BANKNIFTY"):
        if dry:
            f = F.forecast(d, sym)
            print(f"[ilc-forward] {d} {sym}: {f['status']} score={f['score']} side={f['side']} (dry run, not written)")
            continue
        r = F.log_forecast(d, sym)
        what = ("written" if r["written"] else
                "already logged, left unchanged" if r["status"] == "ok" else "not written")
        print(f"[ilc-forward] {d} {sym}: {r['status']} score={r['score']} side={r['side']} -> {what}")


if __name__ == "__main__":
    main()
