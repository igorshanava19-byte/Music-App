@echo off
rem One-time installer for Windows. Double-click it, or run it from a terminal.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo === Music App setup ===
echo.

where uv >nul 2>nul
if errorlevel 1 (
    echo Installing uv, a fast Python installer...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    where uv >nul 2>nul || goto :fail
)

echo.
echo [1/4] Creating the Python environment...
uv venv --python 3.12 --allow-existing .venv || goto :fail

rem Pick the PyTorch build for this GPU: RTX 50-series needs CUDA 13, others use CUDA 12.6.
set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
set "GPU_NAME="
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo No NVIDIA GPU driver found - installing the CPU version, processing will be slower.
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
) else (
    for /f "usebackq delims=" %%g in (`nvidia-smi --query-gpu^=name --format^=csv^,noheader`) do if not defined GPU_NAME set "GPU_NAME=%%g"
)
if defined GPU_NAME call :pick_torch

echo.
echo [2/4] Installing PyTorch from %TORCH_INDEX% ... (about 2-3 GB, this takes a while)
uv pip install --python .venv torch==2.14.1 --index-url %TORCH_INDEX% || goto :fail

echo.
echo [3/4] Installing the other packages...
uv pip install --python .venv -r requirements.txt || goto :fail
uv pip install --python .venv --no-deps basic-pitch==0.4.0 || goto :fail

echo.
echo [4/4] Downloading the AI models and checking everything works...
.venv\Scripts\python.exe -m app.check || goto :fail

if not exist settings.bat copy settings.example.bat settings.bat >nul

echo.
echo === Setup finished. Double-click run.bat to start Music App. ===
echo.
pause
exit /b 0

:pick_torch
echo Found GPU: %GPU_NAME%
echo %GPU_NAME%| findstr /r /c:"RTX 50[0-9][0-9]" /c:"RTX PRO" /c:"Blackwell" >nul
if not errorlevel 1 set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
exit /b 0

:fail
echo.
echo *** Setup failed. Scroll up for the error message. ***
echo.
pause
exit /b 1
