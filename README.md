# TSP Fertilizer Process Intelligence

An internship research project for exploring Triple Super Phosphate (TSP)
production data and evaluating machine-learning approaches for predicting
`SLURRY_FREE_ACID`. The repository contains the research notebooks, saved
research artifacts, an analytics dashboard, and an advisory-only OPC UA
laboratory prototype.

> **Data and safety:** Plant data is confidential and is not included. The
> inference prototype is read-only and advisory; it does not control PLC/DCS
> equipment or replace operator judgment.

## Project contents

| Path | Description |
| --- | --- |
| [`notebooks/`](notebooks/) | Chronological data preparation, feature engineering, model evaluation, forecasting, and model-upgrade research |
| [`reports/`](reports/) | Saved model artifacts, feature metadata, benchmark results, and research reports |
| [`Web_dashboard/`](Web_dashboard/) | Browser-based process analytics dashboard and local prediction API |
| [`deployment/`](deployment/) | OPC UA CSV-replay simulator, inference service, SQLite history, native Windows dashboard, tests, and deployment documentation |
| [`final_internship_report/`](final_internship_report/) | Final report sources and presentation material |

## Model and runtime contract

- The deployed feature builder produces **178 features** and requires **31
  contiguous one-minute rows** before its first prediction.
- The target-free virtual sensor can run without a current laboratory target.
  Target-anchored early warning requires current and historical
  `SLURRY_FREE_ACID` measurements.
- The research artifacts are loaded as saved; the runtime does not retrain
  models.
- The prototype rejects invalid OPC quality and real timestamp gaps. Isolated
  missing process values can be imputed by the saved pipeline.
- The virtual sensor's current validation is within-month chronological
  validation. Additional labeled months are needed to establish
  cross-month performance.

See [`deployment/README.md`](deployment/README.md) for the laboratory
architecture, operating instructions, and safety boundary.

## Run the deployment tests

The inference Docker image uses Python 3.11. Use the same Python version for
local test runs:

```powershell
cd deployment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
python -m pytest
```

The test requirements include the pinned runtime dependencies and `pytest`.
The model-runtime smoke tests exercise the checked-in artifacts and verify the
feature-count and warm-up contract. The same test command runs in
[`.github/workflows/tests.yml`](.github/workflows/tests.yml) on pushes and pull
requests.

## Run the browser dashboard

The dashboard requires Node.js 22.13 or later and Python 3.11:

```powershell
cd Web_dashboard
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm ci
.\start_dashboard.ps1
```

The launcher starts the local API and Next.js development server. The default
dashboard data is a demonstration sample, not confidential plant data. See
[`Web_dashboard/README.md`](Web_dashboard/README.md) for supported analysis
modes and CSV expectations.

## Run the OPC UA laboratory prototype

The Stage 1 lab replays a local CSV through a simulated OPC UA endpoint and
persists predictions to SQLite. It requires Docker Desktop and a local
one-minute CSV with the expected process columns. The source plant dataset is
intentionally excluded from Git; obtain an approved copy before starting the
simulator.

From PowerShell:

```powershell
cd deployment
Copy-Item .env.example .env
.\scripts\start_stage1.ps1
.\scripts\monitor_stage1.ps1
```

Stop the lab with `.\scripts\stop_stage1.ps1`. The full setup, status, export,
and native Windows dashboard instructions are in
[`deployment/README.md`](deployment/README.md).

## Research workflow

The notebooks progress from data exploration and feature preparation through
benchmarking, delta forecasting, target-free virtual sensing, and model
upgrades. Run them in order when reproducing the research. Source data is not
distributed, so notebook execution requires an authorized local dataset and
the input paths expected by each notebook.

## Limitations

- OPC UA node IDs and plant integration details remain environment-specific.
- The simulator and default configuration are for a lab environment; they are
  not production credentials or a production security configuration.
- Model output is predictive, not causal, and is not an operating instruction.
- Performance claims should be interpreted against the validation design and
  available labeled data described in the reports.
