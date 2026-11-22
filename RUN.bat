@echo off
title AI Screenplay Censor Analyzer (FastAPI)
cd /d "%~dp0"

echo ===================================================================
echo     Starting FastAPI Server & Opening Web Browser...
echo ===================================================================

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe run.py
) else (
    py -3.13 run.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server stopped or encountered an error.
    pause
)
