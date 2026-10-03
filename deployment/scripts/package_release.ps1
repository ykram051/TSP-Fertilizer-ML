param(
    [switch]$IncludeDockerImages
)

$ErrorActionPreference = "Stop"
$DeploymentRoot = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $DeploymentRoot
$ReleaseRoot = Join-Path $ProjectRoot "dist\TSP-Predictor-Advisory"
$ZipPath = Join-Path $ProjectRoot "dist\TSP-Predictor-Advisory.zip"

if (-not (Test-Path (Join-Path $DeploymentRoot "dashboard\tsp_dashboard.exe"))) {
    throw "dashboard\tsp_dashboard.exe is missing. Run scripts\build_dashboard.ps1 first."
}

if (Test-Path -LiteralPath $ReleaseRoot) {
    Remove-Item -LiteralPath $ReleaseRoot -Recurse -Force
}
if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null
foreach ($Directory in @("config", "database", "docs", "industrial_inference", "simulator")) {
    Copy-Item -Recurse -Force (Join-Path $DeploymentRoot $Directory) $ReleaseRoot
}
New-Item -ItemType Directory -Force -Path `
    (Join-Path $ReleaseRoot "dashboard"), `
    (Join-Path $ReleaseRoot "scripts"), `
    (Join-Path $ReleaseRoot "runtime"), `
    (Join-Path $ReleaseRoot "data"), `
    (Join-Path $ReleaseRoot "artifacts") | Out-Null

Copy-Item (Join-Path $DeploymentRoot "dashboard\tsp_dashboard.exe") (Join-Path $ReleaseRoot "dashboard")
Copy-Item (Join-Path $DeploymentRoot "dashboard\dashboard.py") (Join-Path $ReleaseRoot "dashboard")
Copy-Item (Join-Path $DeploymentRoot "dashboard\data_access.py") (Join-Path $ReleaseRoot "dashboard")
Copy-Item (Join-Path $DeploymentRoot "dashboard\interaction.py") (Join-Path $ReleaseRoot "dashboard")
Copy-Item (Join-Path $DeploymentRoot "dashboard\README.md") (Join-Path $ReleaseRoot "dashboard")
Copy-Item (Join-Path $DeploymentRoot "launch.bat") $ReleaseRoot
Copy-Item (Join-Path $DeploymentRoot "requirements.txt") $ReleaseRoot
Copy-Item (Join-Path $DeploymentRoot ".env.example") (Join-Path $ReleaseRoot ".env")
Copy-Item (Join-Path $DeploymentRoot "packaging\compose.yaml") (Join-Path $ReleaseRoot "compose.yaml")
Copy-Item (Join-Path $DeploymentRoot "packaging\Dockerfile.inference") $ReleaseRoot
Copy-Item (Join-Path $DeploymentRoot "packaging\Dockerfile.simulator") $ReleaseRoot
Copy-Item (Join-Path $DeploymentRoot "packaging\PACKAGE_README.txt") $ReleaseRoot
Copy-Item (Join-Path $ProjectRoot "tsp_1min.csv") (Join-Path $ReleaseRoot "data")

foreach ($Script in @("status_stage1.ps1", "monitor_stage1.ps1", "database_status.ps1",
                       "export_predictions.ps1", "stop_stage1.ps1")) {
    Copy-Item (Join-Path $DeploymentRoot "scripts\$Script") (Join-Path $ReleaseRoot "scripts")
}

$ArtifactCopies = @(
    @("feature_preparation_artifacts\variable_registry.csv", "feature_preparation_artifacts"),
    @("feature_preparation_artifacts\lag_selection_summary.csv", "feature_preparation_artifacts"),
    @("feature_preparation_artifacts\process_clipping_bounds.csv", "feature_preparation_artifacts"),
    @("exogenous_virtual_sensor_artifacts\models\target_free_virtual_sensor_bundle.joblib", "exogenous_virtual_sensor_artifacts\models"),
    @("target_free_upgrade_artifacts\models\target_free_upgrade_candidate.joblib", "target_free_upgrade_artifacts\models"),
    @("delta_transition_artifacts\models\delta_transition_research_candidates.joblib", "delta_transition_artifacts\models")
)
foreach ($Artifact in $ArtifactCopies) {
    $Source = Join-Path (Join-Path $ProjectRoot "reports") $Artifact[0]
    $Destination = Join-Path (Join-Path $ReleaseRoot "artifacts") $Artifact[1]
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Force $Source $Destination
}

New-Item -ItemType File -Force -Path (Join-Path $ReleaseRoot "runtime\.gitkeep") | Out-Null

if ($IncludeDockerImages) {
    New-Item -ItemType Directory -Force -Path (Join-Path $ReleaseRoot "images") | Out-Null
    Push-Location $DeploymentRoot
    try {
        docker compose build
        if ($LASTEXITCODE -ne 0) { throw "Docker images could not be built." }
        docker save -o (Join-Path $ReleaseRoot "images\tsp-docker-images.tar") `
            tsp-slurry-free-acid-lab-opc-simulator:latest `
            tsp-slurry-free-acid-lab-inference-service:latest
        if ($LASTEXITCODE -ne 0) { throw "Docker images could not be exported." }
    }
    finally { Pop-Location }
}

Compress-Archive -Path "$ReleaseRoot\*" -DestinationPath $ZipPath -CompressionLevel Optimal
Write-Host "Release folder: $ReleaseRoot"
Write-Host "ZIP package:    $ZipPath"
