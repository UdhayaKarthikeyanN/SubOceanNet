# OceanEmbed shutdown: stops backend/frontend recorded in .run\*.pid,
# then sweeps the ports as a fallback - ONLY killing processes that can be
# identified as ours (uvicorn / vite / npm inside this project tree),
# so unrelated services on the same ports are never touched.
# Usage:
#   powershell -ExecutionPolicy Bypass -File bin\stop.ps1 [-ApiPort 8000] [-WebPort 5173]
param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 5173
)
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunDir = Join-Path $Root ".run"

function Get-CmdLine([int]$ProcId) {
    try { return (Get-CimInstance Win32_Process -Filter "ProcessId=$ProcId" -ErrorAction Stop).CommandLine }
    catch { return $null }
}

function Remove-Ours([int]$ProcId, [string]$Tag) {
    if ($ProcId -le 0) { return }
    $cmd = Get-CmdLine $ProcId
    $isOurs = ($null -ne $cmd) -and (
        $cmd -match "uvicorn" -or $cmd -match "vite" -or $cmd -match "npm" -or
        $cmd -like "*$Root*"
    )
    if (-not $isOurs) {
        Write-Host ("[skip] pid {0} ({1}) does not look like OceanEmbed - not touching it" -f $ProcId, $Tag) -ForegroundColor DarkYellow
        return
    }
    # kill the whole tree (npm -> cmd -> node chains)
    & taskkill /PID $ProcId /T /F 2>$null | Out-Null
    Write-Host ("[stopped] {0} pid={1}" -f $Tag, $ProcId) -ForegroundColor Green
}

foreach ($name in "backend", "frontend") {
    $pidFile = Join-Path $RunDir "$name.pid"
    if (Test-Path $pidFile) {
        $procId = [int](Get-Content $pidFile -ErrorAction SilentlyContinue)
        Remove-Ours $procId $name
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
}

# port sweep fallback (children whose pids were never recorded)
Start-Sleep -Milliseconds 600
foreach ($pair in @(@("backend", $ApiPort), @("frontend", $WebPort))) {
    $tag = $pair[0]; $port = $pair[1]
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) { Remove-Ours ([int]$c.OwningProcess) "$tag(port $port)" }
}

Write-Host "OceanEmbed stopped." -ForegroundColor Cyan
