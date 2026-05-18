"""
Exit 0  → market open today, proceed with fetch.
Exit 1  → market closed (weekend or NSE holiday), skip.

Holiday list is read from config/nse_holidays.yaml — update each January.
Source: https://www.nseindia.com/resources/exchange-communication-holidays
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOLIDAY_FILE = PROJECT_ROOT / "config" / "nse_holidays.yaml"


def _load_holidays() -> frozenset[datetime.date]:
    if not HOLIDAY_FILE.exists():
        print(f"[market_check] WARNING: {HOLIDAY_FILE} not found — assuming no holidays")
        return frozenset()
    with HOLIDAY_FILE.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    holidays: list[datetime.date] = []
    for year_entries in (data.get("holidays") or {}).values():
        for entry in year_entries:
            try:
                holidays.append(datetime.date.fromisoformat(entry["date"]))
            except (KeyError, ValueError):
                pass
    return frozenset(holidays)


def main() -> int:
    today = datetime.date.today()

    if today.weekday() >= 5:
        day_name = today.strftime("%A")
        print(f"[market_check] Skipping — {today.strftime('%d %b %Y')} is a {day_name}")
        return 1

    holidays = _load_holidays()
    if today in holidays:
        # Find the holiday name for the log
        with HOLIDAY_FILE.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        name = today.isoformat()
        for year_entries in (raw.get("holidays") or {}).values():
            for entry in year_entries:
                if entry.get("date") == today.isoformat():
                    name = entry.get("name", today.isoformat())
        print(f"[market_check] Skipping — {today.strftime('%d %b %Y')} is an NSE holiday: {name}")
        return 1

    print(
        f"[market_check] Market open — proceeding with fetch for "
        f"{today.strftime('%d %b %Y (%A)')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
