# Industrial inference deployment

This directory is a staged, advisory-only bridge between the existing
`SLURRY_FREE_ACID` research artifacts and an OPC UA data source.

It does **not** retrain a model and it does **not** replace PLC/DCS control.

## Start here

1. Read `docs/00_architecture_overview.md`.
2. Review unresolved plant information in `docs/07_supervisor_questions.md`.
3. Copy `.env.example` to `.env` if different paths or ports are required.
4. From PowerShell, run `scripts/start_stage1.ps1`.
5. Run `scripts/monitor_stage1.ps1` for a clean live demonstration, or
   `scripts/status_stage1.ps1` for one status check.
6. Inspect the persistent output with `scripts/database_status.ps1`; export it
   with `scripts/export_predictions.ps1`.
7. Stop the laboratory environment with `scripts/stop_stage1.ps1`.

## Directory map

| Path | Purpose |
|---|---|
| `docs/` | One concept or deployment stage per file |
| `config/` | OPC tag mapping and service policy |
| `simulator/` | CSV replay OPC UA server |
| `industrial_inference/` | Read-only subscriber, quality shield and model inference |
| `database/` | Versioned SQLite schema |
| `tests/` | Unit tests that do not need a live plant |
| `runtime/` | Generated health, logs and predictions; ignored by Git |
| `Dockerfile.*` | Reproducible simulator and inference images |
| `compose.yaml` | Stage-1 laboratory topology |

`config/integration_services.yaml` is the machine-readable macro service
inventory. The detailed macro flow, inference internals and database contract
are documented separately in `docs/09_*`, `docs/10_*` and `docs/11_*`.

## Current confirmed assumptions

- Sensor cadence: one minute.
- `SLURRY_FREE_ACID`: manual laboratory measurement, not a confirmed online analyzer.
- Docker host for the first prototype: Windows using Docker Desktop.
- OPC Node IDs: placeholders until the IT/OT team provides the real mapping.
- Plant access: read-only shadow mode; advisory writes are disabled by default.

## Safety boundary

No file in this package contains logic to modify PLC/DCS setpoints, controller
modes, interlocks, permissives, valve commands, pump commands or safety logic.
The only optional write path is a dedicated simulated/advisory namespace, and it
must be explicitly enabled in configuration.

## Output persistence

The Stage-1 audit source is `runtime/predictions.sqlite3`. It records service
runs, predictions and status transitions. JSONL and latest JSON are retained as
human-readable mirrors. SQLite is intentionally embedded in the inference
service; it is not a separate network database service.

## Missing sensor values

The deployed feature pipeline retains Notebook 02's train-fitted imputation and
missingness indicators. Isolated missing process cells therefore remain in the
chronological buffer and inference continues with status `IMPUTED`. More than
25% of required inputs missing at one instant,
bad/uncertain OPC quality, or a real timestamp gap still blocks prediction.
