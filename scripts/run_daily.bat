@echo off
cd /d "%~dp0.."
call venv\Scripts\activate.bat

REM ── DuckDB is single-writer; if the dashboard is open during this run it holds
REM    the file lock. Give the daily job a long retry budget so it patiently waits
REM    the dashboard out instead of failing (seen in scheduler.log at 23:30).
set PYTHONUTF8=1
set DUCKDB_LOCK_RETRIES=600
set DUCKDB_LOCK_BASE_DELAY=0.1

REM ── Check if today is an NSE trading day (Mon-Fri, not a holiday) ──────────
python scripts\market_open_check.py >> logs\scheduler.log 2>&1
if errorlevel 1 (
    echo [%date% %time%] Skipped — market closed today >> logs\scheduler.log
    goto write_timestamp
)

REM ── Fetch today's NSE data ────────────────────────────────────────────────
echo [%date% %time%] Starting daily fetch... >> logs\scheduler.log
python -m src.cli daily >> logs\scheduler.log 2>&1
echo [%date% %time%] Fetch complete. >> logs\scheduler.log

REM ── Apply sector overrides so new symbols get classified immediately ───────
echo [%date% %time%] Applying sector overrides... >> logs\scheduler.log
python -m src.cli reload-overrides >> logs\scheduler.log 2>&1
echo [%date% %time%] Sector overrides applied. >> logs\scheduler.log

REM ── Write timestamp from DB latest trade date (not wall clock) ───────────
:write_timestamp
echo [%date% %time%] Writing last_updated marker... >> logs\scheduler.log
python -c "from src.data.repository import query_dataframe; df=query_dataframe('SELECT MAX(trade_date) AS d FROM daily_data',[]); d=df['d'].iloc[0]; open('logs/last_updated.txt','w').write(str(d)[:10])" >> logs\scheduler.log 2>&1

REM ── Upload snapshot to GitHub Releases (for mobile / Streamlit Cloud access) ─
REM   Requires GITHUB_TOKEN and GITHUB_REPO in Windows Environment Variables.
REM   Skip silently if not configured (upload_snapshot.py handles missing vars).
echo [%date% %time%] Uploading slim snapshot to GitHub... >> logs\scheduler.log
python scripts\upload_snapshot.py --days 120 >> logs\scheduler.log 2>&1
echo [%date% %time%] Done. >> logs\scheduler.log
