$ErrorActionPreference = "Stop"
$DeploymentRoot = Split-Path -Parent $PSScriptRoot
Set-Location $DeploymentRoot
docker compose down
Write-Host "Stage 1 containers stopped. Runtime evidence was preserved."
