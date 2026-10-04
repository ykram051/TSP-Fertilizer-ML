# TSP Process Intelligence Dashboard

Interactive dashboard for process exploration, target-free virtual sensing, target-anchored early warning, drift monitoring, and report generation.

## Start

From PowerShell, run:

```powershell
.\dashboard\start_dashboard.ps1
```

The launcher starts the local prediction service and opens the dashboard at `http://localhost:3000`.

## Prediction modes

1. **Target-free virtual sensor** — accepts process variables without `SLURRY_FREE_ACID`. It returns predicted acidity, an empirical 90% interval, and an input-drift flag.
2. **Target-anchored early warning** — requires current and historical `SLURRY_FREE_ACID` values and forecasts 1, 5, 10, or 15 minutes ahead.

## Uploaded CSV contract

- Timestamp column: `Date`
- Original process-column names must match the training dataset.
- Rows must represent chronological one-minute observations for the saved lag and rolling definitions to retain their meaning.
- Target-free mode does not require `SLURRY_FREE_ACID`.
- Early-warning mode requires `SLURRY_FREE_ACID`.

The first 30 rows are used as feature warm-up and do not receive predictions.

## Performance design

- Models and preprocessing objects are loaded once and cached.
- Default-data summaries are cached.
- Charts receive downsampled data while full predictions remain downloadable.
- Uploaded data are processed once per prediction request; the dashboard never retrains models.
- Prediction downloads are cached in memory for the active service session.

## Important limitation

The virtual sensor has within-month chronological validation. A new labeled month is still required to establish real cross-month accuracy. Drift flags identify unusual process inputs but cannot prove prediction correctness without reference acidity values.
