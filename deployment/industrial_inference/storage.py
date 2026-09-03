from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


class PredictionRepository:
    """Durable, auditable SQLite storage for inference runs and outputs."""

    def __init__(self, database_path: str | Path, schema_path: str | Path):
        self.database_path = Path(database_path)
        self.schema_path = Path(schema_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.schema_path.exists():
            raise FileNotFoundError(f"SQLite schema not found: {self.schema_path}")
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(self.schema_path.read_text(encoding="utf-8"))

    def start_run(
        self,
        run_id: str,
        service_name: str,
        model_version: str,
        inference_mode: str,
        horizon_minutes: int | None,
        started_at: str,
        configuration: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO service_runs (
                       run_id, service_name, model_version, inference_mode,
                       forecast_horizon_minutes, started_at, configuration_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, service_name, model_version, inference_mode,
                    horizon_minutes, started_at,
                    json.dumps(configuration, sort_keys=True, default=str),
                ),
            )

    def add_prediction(self, run_id: str, prediction: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO predictions (
                       run_id, sequence, source_timestamp, prediction_for, generated_at,
                       inference_mode, horizon_minutes, current_slurry_free_acid,
                       predicted_change, predicted_slurry_free_acid, input_quality,
                       imputed_inputs_json, assigned_statistical_cluster, model_name,
                       model_version, test_status
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    prediction["sequence"],
                    prediction["source_timestamp"],
                    prediction["prediction_for"],
                    prediction["generated_at"],
                    prediction["mode"],
                    prediction.get("horizon_minutes"),
                    prediction.get("current_slurry_free_acid"),
                    prediction.get("predicted_change"),
                    prediction["predicted_slurry_free_acid"],
                    prediction["input_quality"],
                    json.dumps(prediction.get("imputed_inputs", [])),
                    prediction.get("assigned_statistical_cluster"),
                    prediction["model_name"],
                    prediction["model_version"],
                    prediction.get("test_status"),
                ),
            )

    def add_event(
        self,
        run_id: str,
        event_time: str,
        status: str,
        sequence: int | None,
        message: str | None,
        details: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO service_events (
                       run_id, event_time, status, sequence, message, details_json
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, event_time, status, sequence, message,
                 json.dumps(details, sort_keys=True, default=str)),
            )

    def summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            totals = connection.execute(
                """SELECT COUNT(*) AS predictions,
                          COUNT(DISTINCT run_id) AS runs,
                          MIN(source_timestamp) AS first_source_timestamp,
                          MAX(source_timestamp) AS last_source_timestamp
                   FROM predictions"""
            ).fetchone()
            quality = connection.execute(
                """SELECT input_quality, COUNT(*) AS count
                   FROM predictions GROUP BY input_quality ORDER BY count DESC"""
            ).fetchall()
        return {**dict(totals), "quality_counts": [dict(row) for row in quality]}

    def latest(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT run_id, sequence, source_timestamp, prediction_for,
                          predicted_slurry_free_acid, input_quality, model_name
                   FROM predictions ORDER BY prediction_id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def export_csv(self, output_path: str | Path) -> int:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            cursor = connection.execute("SELECT * FROM predictions ORDER BY prediction_id")
            rows = cursor.fetchall()
            names = [item[0] for item in cursor.description]
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(names)
            writer.writerows([tuple(row) for row in rows])
        return len(rows)
