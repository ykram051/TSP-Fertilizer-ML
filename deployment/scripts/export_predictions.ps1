$ErrorActionPreference = "Stop"
$DeploymentRoot = Split-Path -Parent $PSScriptRoot
Set-Location $DeploymentRoot

docker compose exec -T inference-service `
    python -m industrial_inference.database_cli export `
    --output /runtime/predictions_export.csv
Write-Host "Host file: $DeploymentRoot\runtime\predictions_export.csv"
