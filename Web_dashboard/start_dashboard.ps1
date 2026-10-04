$ErrorActionPreference = "Stop"
$Dashboard = Split-Path -Parent $MyInvocation.MyCommand.Path
$Project = Split-Path -Parent $Dashboard

Write-Host "Starting TSP Process Intelligence..." -ForegroundColor Green
Start-Process -FilePath "python" -ArgumentList @("backend.py") -WorkingDirectory $Dashboard -WindowStyle Hidden
Start-Sleep -Seconds 2
Start-Process -FilePath "npm.cmd" -ArgumentList @("run", "dev") -WorkingDirectory $Dashboard -WindowStyle Hidden
Start-Sleep -Seconds 4

try {
    $health = Invoke-RestMethod "http://127.0.0.1:8000/api/health"
    Write-Host "Prediction service: $($health.status)" -ForegroundColor Green
} catch {
    Write-Warning "The prediction service is still loading. Refresh the page after a few seconds."
}

Start-Process "http://localhost:3000"
Write-Host "Dashboard opened at http://localhost:3000" -ForegroundColor Cyan

