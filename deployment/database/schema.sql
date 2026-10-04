PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS service_runs (
    run_id TEXT PRIMARY KEY,
    service_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    inference_mode TEXT NOT NULL,
    forecast_horizon_minutes INTEGER,
    started_at TEXT NOT NULL,
    stopped_at TEXT,
    final_status TEXT,
    configuration_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    source_timestamp TEXT NOT NULL,
    prediction_for TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    inference_mode TEXT NOT NULL,
    horizon_minutes INTEGER,
    current_slurry_free_acid REAL,
    predicted_change REAL,
    predicted_slurry_free_acid REAL NOT NULL,
    input_quality TEXT NOT NULL,
    imputed_inputs_json TEXT NOT NULL DEFAULT '[]',
    assigned_statistical_cluster INTEGER,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    test_status TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES service_runs(run_id),
    UNIQUE (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS service_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    event_time TEXT NOT NULL,
    status TEXT NOT NULL,
    sequence INTEGER,
    message TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES service_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_source_timestamp
    ON predictions(source_timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_for
    ON predictions(prediction_for);
CREATE INDEX IF NOT EXISTS idx_predictions_quality
    ON predictions(input_quality);
CREATE INDEX IF NOT EXISTS idx_events_time
    ON service_events(event_time);
-- The dashboard reads the newest run and then its latest N predictions.
CREATE INDEX IF NOT EXISTS idx_predictions_run_order
    ON predictions(run_id, prediction_id);
CREATE INDEX IF NOT EXISTS idx_runs_started_at
    ON service_runs(started_at);
