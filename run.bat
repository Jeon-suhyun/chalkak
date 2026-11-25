@echo off
REM Chalkak launcher: double-click to run
cd /d "%~dp0"
"C:\venvs\chalkak\python.exe" -m streamlit run app.py
pause
