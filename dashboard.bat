@echo off
call venv\Scripts\activate.bat
streamlit run src\dashboard\streamlit_app.py --server.port 8501
