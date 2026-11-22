# =====================================================================
# AI Screenplay & Censor Rating Analyzer (FastAPI Launcher)
# Run with: .\run.ps1
# =====================================================================

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectDir

$pythonExe = Join-Path $projectDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "py -3.13"
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "      Starting AI Screenplay & Censor Rating Analyzer (FastAPI)" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[INFO] Opening browser at http://localhost:8000 ..." -ForegroundColor Green

# Launch browser in a background job
Start-Job -ScriptBlock {
    Start-Sleep -Milliseconds 1500
    Start-Process "http://localhost:8000"
} | Out-Null

# Start FastAPI server
if ($pythonExe -eq "py -3.13") {
    py -3.13 -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
} else {
    & $pythonExe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
}
