@echo off
cd /d "%~dp0.."
call venv\Scripts\activate.bat

REM ── Check if today is an NSE trading day (Mon-Fri, not a holiday) ──────────
python scripts\market_open_check.py >> logs\scheduler.log 2>&1
if errorlevel 1 (
    echo [%date% %time%] Skipped — market closed today >> logs\scheduler.log
    exit /b 0
)

REM ── Fetch today's NSE data ────────────────────────────────────────────────
echo [%date% %time%] Starting daily fetch... >> logs\scheduler.log
python -m src.cli daily >> logs\scheduler.log 2>&1
echo [%date% %time%] Fetch complete. >> logs\scheduler.log

REM ── Apply sector overrides so new symbols get classified immediately ───────
echo [%date% %time%] Applying sector overrides... >> logs\scheduler.log
python -m src.cli reload-overrides >> logs\scheduler.log 2>&1
echo [%date% %time%] Sector overrides applied. >> logs\scheduler.log

REM ── Write fetch timestamp so dashboard can detect new data ────────────────
echo [%date% %time%] Writing last_updated marker... >> logs\scheduler.log
python -c "from datetime import datetime; open('logs/last_updated.txt','w').write(datetime.now().isoformat())"
echo [%date% %time%] Done. >> logs\scheduler.log
