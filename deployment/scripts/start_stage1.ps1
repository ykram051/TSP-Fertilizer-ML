$ErrorActionPreference = "Stop"

$DeploymentRoot = Split-Path -Parent $PSScriptRoot
Set-Location $DeploymentRoot

function Invoke-DockerChecked {
    $DockerArguments = @($args)
    & docker @DockerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed (exit $LASTEXITCODE): docker $($DockerArguments -join ' ')"
    }
}

function Pull-BaseImage {
    $BaseImage = "python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534"
    $Attempts = 3
    for ($Attempt = 1; $Attempt -le $Attempts; $Attempt++) {
        Write-Host "Checking/downloading the pinned Python 3.11 base image (attempt $Attempt of $Attempts)..."
        & docker pull $BaseImage
        if ($LASTEXITCODE -eq 0) { return }
        if ($Attempt -lt $Attempts) {
            Write-Warning "Docker Hub connection failed. Retrying in 10 seconds..."
            Start-Sleep -Seconds 10
        }
    }
    throw @"
The Python base image could not be downloaded from Docker Hub.
This is a registry/network problem, not a model or Compose problem.

Verify in Docker Desktop:
  1. Docker Desktop is fully running.
  2. Settings > Resources > Proxies matches the company network.
  3. VPN/firewall permits https://auth.docker.io and https://registry-1.docker.io.

Then rerun this script. The base-image version is pinned in the Dockerfiles.
"@
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install/start Docker Desktop, then rerun this script."
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created deployment/.env from the laboratory defaults."
}

New-Item -ItemType Directory -Force -Path "runtime" | Out-Null
Invoke-DockerChecked compose down
$GeneratedRuntimeFiles = @(
    "runtime/health.json",
    "runtime/latest_prediction.json",
    "runtime/predictions.jsonl",
    "runtime/service.log"
)
foreach ($RuntimeFile in $GeneratedRuntimeFiles) {
    if (Test-Path -LiteralPath $RuntimeFile) {
        Remove-Item -LiteralPath $RuntimeFile -Force
    }
}
Pull-BaseImage
Write-Host "Building OPC simulator first (network downloads are intentionally sequential)..."
Invoke-DockerChecked compose build opc-simulator
Write-Host "Building inference service..."
Invoke-DockerChecked compose build inference-service
Invoke-DockerChecked compose up -d --no-build
Invoke-DockerChecked compose ps

Write-Host ""
Write-Host "Stage 1 started. The CSV is replayed through OPC UA."
Write-Host ""
Write-Host "Readable live monitor: .\scripts\monitor_stage1.ps1"
Write-Host "One status check:      .\scripts\status_stage1.ps1"
Write-Host "Prediction history:    runtime\predictions.jsonl"
Write-Host "SQLite audit database: runtime\predictions.sqlite3"
Write-Host "Database summary:      .\scripts\database_status.ps1"
Write-Host "Technical logs only:   docker compose logs -f inference-service"
