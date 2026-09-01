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
    $Attempts = 3
    for ($Attempt = 1; $Attempt -le $Attempts; $Attempt++) {
        Write-Host "Checking/downloading python:3.11-slim (attempt $Attempt of $Attempts)..."
        & docker pull python:3.11-slim
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

Then test: docker pull python:3.11-slim
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
Pull-BaseImage
Invoke-DockerChecked compose build
Invoke-DockerChecked compose up -d --no-build
Invoke-DockerChecked compose ps

Write-Host ""
Write-Host "Stage 1 started. The CSV is replayed through OPC UA."
Write-Host "Health:      deployment/runtime/health.json"
Write-Host "Predictions: deployment/runtime/predictions.jsonl"
Write-Host "Logs:        docker compose logs -f inference-service"
