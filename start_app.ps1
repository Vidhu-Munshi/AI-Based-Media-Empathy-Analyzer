# PowerShell One-Click Launcher for FastAPI App
$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectDir

$pythonExe = Join-Path $projectDir ".venv\Scripts\python.exe"

# 1. Create venv and install dependencies if missing
if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host "[SETUP] Virtual environment not found. Creating .venv..." -ForegroundColor Yellow
    py -3.13 -m venv .venv
    Write-Host "[SETUP] Installing dependencies into .venv..." -ForegroundColor Yellow
    & $pythonExe -m pip install --upgrade pip
    & $pythonExe -m pip install -r requirements.txt
    Write-Host "[SETUP] Environment setup complete!" -ForegroundColor Green
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "      Starting AI Media Empathy & Censor Rating Analyzer (FastAPI)" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[INFO] Opening browser at http://localhost:8000 ..." -ForegroundColor Green

# Open browser after 3 seconds
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:8000"
} | Out-Null

& $pythonExe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
