$ErrorActionPreference = "Stop"
$DeploymentRoot = Split-Path -Parent $PSScriptRoot
Set-Location $DeploymentRoot

Write-Host "SQLite prediction database"
Write-Host "=========================="
docker compose exec -T inference-service `
    python -m industrial_inference.database_cli summary

Write-Host ""
Write-Host "Latest five predictions"
Write-Host "-----------------------"
docker compose exec -T inference-service `
    python -m industrial_inference.database_cli latest --limit 5
