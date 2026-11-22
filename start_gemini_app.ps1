param(
    [string]$GeminiApiKey
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Project virtual environment was not found: $pythonExe"
}

Set-Location -LiteralPath $projectDir

if (-not $GeminiApiKey) {
    $secureKey = Read-Host "Paste your Gemini API key (it will not be saved)" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $GeminiApiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

if (-not $GeminiApiKey) {
    throw "A Gemini API key is required."
}

# This environment variable exists only for this PowerShell process and the
# Streamlit process it launches; it is never written to a project file.
$env:GEMINI_API_KEY = $GeminiApiKey

& $pythonExe -m pip install --upgrade -r requirements.txt
& $pythonExe -m pip install --upgrade google-genai
& $pythonExe -m streamlit run app.py
