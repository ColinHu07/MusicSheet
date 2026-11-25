@echo off
setlocal
cd /d "%~dp0..\.."
".venv\Scripts\python.exe" "scripts\melody_studio\server.py" --open
if errorlevel 1 pause
