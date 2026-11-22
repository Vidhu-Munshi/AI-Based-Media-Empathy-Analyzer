@echo off
title AI Media Empathy & Censor Rating Analyzer (FastAPI)
echo =====================================================================
echo       Starting AI Media Empathy & Censor Rating Analyzer
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Check if virtual environment exists; if not, create it
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Virtual environment not found. Creating .venv with Python 3.13...
    py -3.13 -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create .venv with py -3.13. Trying python -m venv .venv ...
        python -m venv .venv
    )
    echo [SETUP] Installing dependencies into .venv...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo [SETUP] Environment setup complete!
    echo.
)

set PYTHON_EXE=.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=py -3.13
)

echo [INFO] Using interpreter: %PYTHON_EXE%
echo [INFO] Starting FastAPI server at http://localhost:8000 ...
echo [INFO] Your browser will open automatically in 3 seconds...
echo.

:: Launch browser in background after 3 seconds
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000"

:: Start Uvicorn FastAPI server
"%PYTHON_EXE%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server encountered an error.
    pause
)
