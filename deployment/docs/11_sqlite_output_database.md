# 11 — SQLite output database

## Why SQLite is used in Stage 1

SQLite provides transactions, a schema, indexes and queries without operating
another database server. It is appropriate for one local inference writer and
supervisor demonstrations. It is not presented as the final plant historian.

Database file: `runtime/predictions.sqlite3`

## Data model

```mermaid
erDiagram
    SERVICE_RUNS ||--o{ PREDICTIONS : produces
    SERVICE_RUNS ||--o{ SERVICE_EVENTS : reports
    SERVICE_RUNS {
      text run_id PK
      text service_name
      text model_version
      text inference_mode
      integer forecast_horizon_minutes
      text started_at
      text configuration_json
    }
    PREDICTIONS {
      integer prediction_id PK
      text run_id FK
      integer sequence
      text source_timestamp
      text prediction_for
      real predicted_slurry_free_acid
      real predicted_change
      text input_quality
      text model_name
    }
    SERVICE_EVENTS {
      integer event_id PK
      text run_id FK
      text event_time
      text status
      integer sequence
      text details_json
    }
```

### `service_runs`

One row per inference-service startup. It records a UUID, mode, horizon, model
version, start time and a configuration snapshot.

### `predictions`

One row per successful prediction. `UNIQUE(run_id, sequence)` prevents duplicate
records within one run, while allowing a historical CSV to be replayed again in
a new run. Timestamps and quality are indexed for review.

### `service_events`

Stores state changes such as `STARTING`, `CONNECTED`, `WARMING_UP`, `RUNNING`,
`DATA_ERROR` and `DISCONNECTED` with explanatory details.

## Durability and access

- Transactions prevent partial prediction rows.
- WAL mode permits a reader while the inference service writes.
- The database is persisted through the Docker bind mount.
- JSONL remains available for simple tools and recovery comparison.
- Back up/copy the database only according to an agreed retention procedure.

## Commands

From `deployment/`:

```powershell
.\scripts\database_status.ps1
.\scripts\export_predictions.ps1
```

The export is written to `runtime/predictions_export.csv`. The database itself
should remain the audit source because CSV does not preserve relationships,
constraints or data types.

