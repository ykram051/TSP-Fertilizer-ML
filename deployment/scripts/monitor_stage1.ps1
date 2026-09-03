$ErrorActionPreference = "Stop"
$StatusScript = Join-Path $PSScriptRoot "status_stage1.ps1"

Write-Host "Live monitor started. Press Ctrl+C to stop monitoring; containers will keep running."
Start-Sleep -Seconds 1

while ($true) {
    Clear-Host
    & $StatusScript
    Write-Host ""
    Write-Host "Refreshing every 2 seconds - Ctrl+C exits this screen only."
    Start-Sleep -Seconds 2
}
