# OceanEmbed launcher: backend (uvicorn) + frontend (vite dev).
# Usage:
#   powershell -ExecutionPolicy Bypass -File bin\start.ps1 [-ApiPort 8000] [-WebPort 5173] [-NoFrontend]
param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 5173,
    [switch]$NoFrontend
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunDir = Join-Path $Root ".run"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Test-PortListening([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

# ---------- backend ----------
if (Test-PortListening $ApiPort) {
    Write-Host "[skip] port $ApiPort already listening - assuming backend is up" -ForegroundColor Yellow
} else {
    $Py = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path $Py)) {
        Write-Host "ERROR: $Py not found - run scripts\setup.ps1 first" -ForegroundColor Red
        exit 1
    }
    $proc = Start-Process -FilePath $Py `
        -ArgumentList "-m","uvicorn","backend.main:app","--host","127.0.0.1","--port",$ApiPort `
        -WorkingDirectory $Root -WindowStyle Minimized -PassThru
    $proc.Id | Set-Content (Join-Path $RunDir "backend.pid")
    Write-Host ("[ok]   backend  http://127.0.0.1:{0}  (docs /docs)  pid={1}" -f $ApiPort, $proc.Id) -ForegroundColor Green
}

# ---------- frontend ----------
if ($NoFrontend) {
    Write-Host "[skip] frontend disabled (-NoFrontend)"
} elseif (Test-PortListening $WebPort) {
    Write-Host "[skip] port $WebPort already listening - assuming frontend is up" -ForegroundColor Yellow
} else {
    $FeDir = Join-Path $Root "frontend"
    if (-not (Test-Path (Join-Path $FeDir "node_modules"))) {
        Write-Host "ERROR: frontend\node_modules missing - run scripts\setup.ps1 first" -ForegroundColor Red
        exit 1
    }
    # point the vite proxy at a non-default API port when needed
    $envLine = ""
    if ($ApiPort -ne 8000) { $envLine = "`$env:OCEANEMBED_API='http://127.0.0.1:$ApiPort'; " }
    $inner = "${envLine}Set-Location '$FeDir'; npm.cmd run dev -- --port $WebPort"
    $proc = Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command",$inner `
        -WindowStyle Minimized -PassThru
    $proc.Id | Set-Content (Join-Path $RunDir "frontend.pid")
    Write-Host ("[ok]   frontend http://localhost:{0}                pid={1}" -f $WebPort, $proc.Id) -ForegroundColor Green
}

Write-Host ""
Write-Host "Logs are NOT redirected (servers run in minimized windows)." 
Write-Host "Stop everything with:  powershell -ExecutionPolicy Bypass -File bin\stop.ps1$(if($ApiPort -ne 8000 -or $WebPort -ne 5173){' -ApiPort ' + $ApiPort + ' -WebPort ' + $WebPort})"
