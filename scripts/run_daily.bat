@echo off
cd /d "%~dp0.."
call venv\Scripts\activate.bat
python -m src.cli daily >> logs\scheduler.log 2>&1

REM Auto-push code changes to GitHub
git add -A
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "auto: daily fetch %date%" >> logs\scheduler.log 2>&1
    git push origin main >> logs\scheduler.log 2>&1
)
