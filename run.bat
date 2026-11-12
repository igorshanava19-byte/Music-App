@echo off
rem Starts Music App and opens it in your browser. Close this window to stop it.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Music App is not installed yet. Run setup.bat first.
    pause
    exit /b 1
)
if exist settings.bat call settings.bat
set MUSICAPP_OPEN_BROWSER=1
.venv\Scripts\python.exe -m app
pause
