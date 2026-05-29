"""
Auto-update script — runs on VS Code folder open.

Two behaviours based on IST time:

  MARKET HOURS  (09:15 – 15:30, Mon–Fri)
    → Start the Streamlit dashboard if it is not already running,
      then open it in the default browser.

  AFTER 19:30   (7:30 PM IST onwards, any day)
    → Check if today's end-of-day bhavcopy has been fetched.
      If not, run `python -m src.cli daily` to fetch it.

Both checks run on every VS Code folder-open; each skips silently when its
condition is not met.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import webbrowser
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IST = timezone(timedelta(hours=5, minutes=30))

# Market hours (NSE): 09:15 – 15:30 IST, Mon–Fri
_MARKET_OPEN  = (9, 15)
_MARKET_CLOSE = (15, 30)

# Bhavcopy reliably published by 7:30 PM IST
_FETCH_AFTER  = (19, 30)

_DASHBOARD_PORT = 8501
_DASHBOARD_URL  = f"http://localhost:{_DASHBOARD_PORT}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hm(dt: datetime) -> tuple[int, int]:
    return (dt.hour, dt.minute)


def _dashboard_running() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("localhost", _DASHBOARD_PORT)) == 0


def _latest_db_date():
    try:
        from src.data.repository import query_dataframe
        df  = query_dataframe("SELECT MAX(trade_date) AS d FROM daily_data", [])
        val = df["d"].iloc[0]
        return val.date() if hasattr(val, "date") else None
    except Exception:
        return None


# ── Actions ───────────────────────────────────────────────────────────────────

def _open_dashboard() -> None:
    if not _dashboard_running():
        print("[auto-update] Starting Streamlit dashboard...", flush=True)
        streamlit_app = PROJECT_ROOT / "src" / "dashboard" / "streamlit_app.py"
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(streamlit_app),
             "--server.headless", "true"],
            cwd=str(PROJECT_ROOT),
        )
        # Wait up to 10 s for the server to become ready
        for _ in range(20):
            time.sleep(0.5)
            if _dashboard_running():
                break

    print(f"[auto-update] Opening dashboard at {_DASHBOARD_URL}", flush=True)
    webbrowser.open(_DASHBOARD_URL)


def _fetch_data(today) -> None:
    latest = _latest_db_date()
    if latest == today:
        print(f"[auto-update] DB already has {today} data. Nothing to do.", flush=True)
        return

    print(
        f"[auto-update] DB latest={latest}, today={today}. Fetching...",
        flush=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "daily"],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        print("[auto-update] Fetch completed successfully.", flush=True)
    else:
        print(f"[auto-update] Fetch exited with code {result.returncode}.", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    now  = datetime.now(IST)
    hm   = _hm(now)
    today     = now.date()
    is_weekday = now.weekday() < 5      # Mon=0 … Fri=4

    print(f"[auto-update] {now.strftime('%d %b %Y %I:%M %p')} IST", flush=True)

    # ── Market-hours: open dashboard ─────────────────────────────────────────
    if is_weekday and _MARKET_OPEN <= hm <= _MARKET_CLOSE:
        print("[auto-update] Market is open — launching dashboard.", flush=True)
        _open_dashboard()
    else:
        print(
            "[auto-update] Outside market hours"
            f" ({_MARKET_OPEN[0]:02d}:{_MARKET_OPEN[1]:02d}-"
            f"{_MARKET_CLOSE[0]:02d}:{_MARKET_CLOSE[1]:02d} IST Mon-Fri)."
            " Dashboard auto-launch skipped.",
            flush=True,
        )

    # ── After 7:30 PM: fetch end-of-day data ─────────────────────────────────
    if hm >= _FETCH_AFTER:
        print("[auto-update] After 19:30 IST - checking for new bhavcopy.", flush=True)
        _fetch_data(today)
    else:
        print(
            "[auto-update] Before 19:30 IST - bhavcopy fetch skipped.",
            flush=True,
        )


if __name__ == "__main__":
    main()
