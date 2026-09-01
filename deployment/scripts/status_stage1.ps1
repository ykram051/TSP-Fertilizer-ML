$ErrorActionPreference = "Stop"
$DeploymentRoot = Split-Path -Parent $PSScriptRoot
Set-Location $DeploymentRoot
docker compose ps

if (Test-Path "runtime/health.json") {
    Write-Host ""
    $Health = Get-Content "runtime/health.json" -Raw | ConvertFrom-Json
    Write-Host "Service status: $($Health.status)"
    if ($Health.status -eq "WARMING_UP") {
        Write-Host "History collected: $($Health.buffer_rows)/$($Health.rows_required) rows"
        Write-Host "Rows remaining:    $($Health.rows_remaining)"
        Write-Host "Meaning: the temporal model is collecting the history needed for its lags and rolling features."
    }
    elseif ($Health.status -eq "DATA_ERROR") {
        Write-Host "Reason: $($Health.reasons -join '; ')"
    }
    elseif ($Health.status -eq "RUNNING") {
        Write-Host "Prediction for:    $($Health.last_prediction_for)"
        Write-Host "Process familiarity: $($Health.process_familiarity)"
    }
}

if ((Test-Path "runtime/latest_prediction.json") -and
    ((Get-Item "runtime/latest_prediction.json").Length -gt 0)) {
    Write-Host ""
    Write-Host "Latest prediction:"
    $Prediction = Get-Content "runtime/latest_prediction.json" -Raw | ConvertFrom-Json
    Write-Host "Source time:       $($Prediction.source_timestamp)"
    Write-Host "Prediction for:    $($Prediction.prediction_for)"
    Write-Host "Free acid:         $([math]::Round($Prediction.predicted_slurry_free_acid, 3))"
    Write-Host "Model:             $($Prediction.model_name)"
    Write-Host "Familiarity:       $($Prediction.process_familiarity)"
}
else {
    Write-Host ""
    Write-Host "No prediction yet. This is expected until warm-up completes."
}
