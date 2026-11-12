@echo off
rem Starts Music App with a live https link (Cloudflare quick tunnel) so you can open it from anywhere.
rem The link changes every time. Close this window to stop both the app and the link.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Music App is not installed yet. Run setup.bat first.
    goto :stop
)
if exist "%~dp0settings.bat" call "%~dp0settings.bat"
if "%MUSICAPP_PORT%"=="" set "MUSICAPP_PORT=8000"

rem Never share without a password: anyone with the link could use the app.
.venv\Scripts\python.exe -c "import os,sys; sys.exit(len(os.environ.get('MUSICAPP_PASSWORD', '').strip()) < 12)"
if errorlevel 1 (
    echo The live link needs a password of at least 12 characters.
    echo Open settings.bat, set the MUSICAPP_PASSWORD line, save, and run share.bat again.
    goto :stop
)
rem A copy already running (e.g. from run.bat) may have no password, so don't put a link in front of it.
.venv\Scripts\python.exe -c "import socket,sys; sys.exit(socket.socket().connect_ex(('127.0.0.1', int('%MUSICAPP_PORT%'))) == 0)"
if errorlevel 1 (
    echo Music App is already running on port %MUSICAPP_PORT%. Close that window first, then run share.bat again.
    goto :stop
)

if not exist .venv\cloudflared.exe (
    echo Downloading cloudflared from Cloudflare's GitHub releases ^(one time, about 60 MB^)...
    curl -fL --progress-bar -o .venv\cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
    if errorlevel 1 (
        del .venv\cloudflared.exe 2>nul
        echo Download failed. Check your internet connection and try again.
        goto :stop
    )
)

start /b "" .venv\Scripts\python.exe -m app
echo.
echo   Your live link is the https://....trycloudflare.com address in the box below.
echo   It is password protected and changes every time you run share.bat.
echo.
.venv\cloudflared.exe tunnel --no-autoupdate --url http://127.0.0.1:%MUSICAPP_PORT%

:stop
pause
