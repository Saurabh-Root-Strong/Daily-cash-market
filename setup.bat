@echo off
echo ============================================================
echo  NSE Daily Cash Market Dashboard - Setup
echo ============================================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ and re-run.
    pause
    exit /b 1
)

REM Create venv
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate and install
echo Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

REM Create necessary directories
if not exist data mkdir data
if not exist logs mkdir logs

REM Init DB
echo Initializing database schema...
python -m src.cli init-db
if errorlevel 1 (
    echo ERROR: Database init failed.
    pause
    exit /b 1
)

REM Backfill 60 days
echo Backfilling 60 trading days of data (this may take 3-5 minutes)...
python -m src.cli backfill 100
if errorlevel 1 (
    echo WARNING: Backfill encountered some errors (weekends/holidays are normal).
)

REM Seed sectors
echo Seeding sector master...
python -m src.cli seed-sectors
if errorlevel 1 (
    echo WARNING: Sector seeding encountered some errors.
)

REM Register Task Scheduler
echo Registering Windows Task Scheduler (runs daily at 18:30 IST)...
set TASK_NAME=NSE_Daily_Fetch
set SCRIPT_PATH=%CD%\scripts\run_daily.bat
schtasks /create /tn "%TASK_NAME%" /tr "\"%SCRIPT_PATH%\"" /sc daily /st 18:30 /f >nul 2>&1
if errorlevel 1 (
    echo WARNING: Could not register Task Scheduler. Run as Administrator to enable auto-fetch.
) else (
    echo Task Scheduler registered successfully.
)

echo.
echo ============================================================
echo  Setup complete! Run dashboard.bat to launch the dashboard.
echo ============================================================
pause
