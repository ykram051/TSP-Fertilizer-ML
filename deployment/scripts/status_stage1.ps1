$ErrorActionPreference = "Stop"
$DeploymentRoot = Split-Path -Parent $PSScriptRoot
Set-Location $DeploymentRoot
Write-Host "========================================"
Write-Host " SLURRY FREE ACID - OPC UA SIMULATION"
Write-Host "========================================"
$RunningServices = @(docker compose ps --status running --services)
Write-Host "Containers running: $($RunningServices.Count)/2"

if (Test-Path "runtime/health.json") {
    Write-Host ""
    $Health = Get-Content "runtime/health.json" -Raw | ConvertFrom-Json
    Write-Host "Status:             $($Health.status)"
    if ($Health.status -eq "WARMING_UP") {
        Write-Host "History collected: $($Health.buffer_rows)/$($Health.rows_required) rows"
        Write-Host "Rows remaining:    $($Health.rows_remaining)"
        Write-Host "Meaning: collecting the initial history required for temporal features."
    }
    elseif ($Health.status -eq "DATA_ERROR") {
        Write-Host "Reason: $($Health.reasons -join '; ')"
    }
    elseif ($Health.status -eq "RUNNING") {
        Write-Host "History buffer:     $($Health.buffer_rows) rows ($($Health.rows_required) required) - READY"
        Write-Host "Input quality:      $($Health.input_quality)"
    }
}

if ((Test-Path "runtime/latest_prediction.json") -and
    ((Get-Item "runtime/latest_prediction.json").Length -gt 0)) {
    Write-Host ""
    Write-Host "Latest successful result"
    Write-Host "------------------------"
    $Prediction = Get-Content "runtime/latest_prediction.json" -Raw | ConvertFrom-Json
    $ProcessTime = [DateTimeOffset]::Parse($Prediction.source_timestamp)
    Write-Host "Process timestamp:  $($ProcessTime.UtcDateTime.ToString('yyyy-MM-dd HH:mm:ss')) UTC"
    if ($Prediction.mode -eq "target_free") {
        Write-Host "Operating mode:     Current-value virtual sensor"
        Write-Host "Estimated free acid: $([math]::Round($Prediction.predicted_slurry_free_acid, 3))"
    }
    else {
        Write-Host "Operating mode:     $($Prediction.horizon_minutes)-minute target-anchored forecast"
        Write-Host "Forecast timestamp: $($Prediction.prediction_for)"
        Write-Host "Predicted free acid: $([math]::Round($Prediction.predicted_slurry_free_acid, 3))"
    }
    Write-Host "Input quality:      $($Prediction.input_quality)"
    if ($Prediction.imputed_inputs.Count -gt 0) {
        Write-Host "Imputed sensors:    $($Prediction.imputed_inputs -join ', ')"
    }
    else {
        Write-Host "Imputed sensors:    none"
    }
    Write-Host "Model:              $($Prediction.model_name)"
    if (Test-Path "runtime/predictions.sqlite3") {
        Write-Host "Audit storage:      SQLite + JSON mirror"
    }
}
else {
    Write-Host ""
    Write-Host "No prediction yet. This is expected until warm-up completes."
}
