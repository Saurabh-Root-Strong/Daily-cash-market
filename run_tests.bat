@echo off
call venv\Scripts\activate.bat
pytest tests\ -v
pause
