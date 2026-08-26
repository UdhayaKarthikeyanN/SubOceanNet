# One-command bootstrap for OceanEmbed (Windows).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
param(
    [switch]$SkipData,
    [switch]$SkipTrain,
    [switch]$SkipNpm
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== OceanEmbed setup ==" -ForegroundColor Cyan
Write-Host "Project root: $Root"

# --- Python venv + backend deps ---
python --version
if (-not (Test-Path ".venv")) {
    Write-Host "-> creating virtualenv .venv"
    python -m venv .venv
}
$py = Join-Path $Root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip -q
Write-Host "-> installing python dependencies (torch may take a few minutes)"
& $py -m pip install -r requirements.txt

# --- Frontend deps ---
if (-not $SkipNpm) {
    Write-Host "-> installing frontend dependencies"
    Push-Location frontend
    & npm.cmd install
    Pop-Location
}

# --- Demo dataset + first-run model ---
if (-not $SkipData) {
    if (-not (Test-Path "data\synthetic\inputs.nc")) {
        Write-Host "-> generating synthetic North Indian Ocean dataset (~1-2 min)"
        & $py -m src.data.generate_synthetic
    } else {
        Write-Host "-> synthetic dataset found, skipping generation"
    }
}
if ((-not $SkipTrain) -and (-not $SkipData)) {
    if (-not (Test-Path "data\checkpoints\oceanembed_demo\checkpoint.pt")) {
        Write-Host "-> training quick demo model (~1 min on CPU)"
        & $py src\train.py --auto
    } else {
        Write-Host "-> checkpoint found, skipping training"
    }
}

Write-Host ""
Write-Host "== Setup complete ==" -ForegroundColor Green
Write-Host "Start the app:    bin\start.cmd      (stop with bin\stop.cmd)"
Write-Host "Or manually -     Terminal A:  .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
Write-Host "                  Terminal B:  cd frontend; npm.cmd run dev"
Write-Host "Then open http://localhost:5173"
