from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any


def application_root() -> Path:
    """Return deployment/package root in source and PyInstaller modes."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent.parent
    return Path(__file__).resolve().parent.parent


def runtime_directory() -> Path:
    return application_root() / "runtime"


def read_health(path: Path | None = None) -> dict[str, Any]:
    health_path = path or runtime_directory() / "health.json"
    if not health_path.exists():
        return {
            "status": "STARTING",
            "input_quality": "WAITING",
            "message": "Waiting for the inference service",
        }
    # health.json is atomically replaced by the service, but one retry also
    # handles antivirus/network-folder timing on Windows.
    for attempt in range(2):
        try:
            with health_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            payload["status"] = payload.get("status", payload.get("model_status", "STARTING"))
            payload.setdefault("input_quality", "WAITING")
            return payload
        except (OSError, json.JSONDecodeError):
            if attempt:
                raise
    raise RuntimeError("unreachable")


def read_predictions(
    database_path: Path | None = None,
    limit: int = 720,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    path = database_path or runtime_directory() / "predictions.sqlite3"
    if not path.exists():
        return []
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=0.5)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 500")
        if run_id is None:
            latest = connection.execute(
                """SELECT run_id FROM service_runs
                   ORDER BY started_at DESC LIMIT 1"""
            ).fetchone()
            if latest is None:
                return []
            run_id = str(latest["run_id"])
        rows = connection.execute(
            """SELECT run_id, source_timestamp, predicted_slurry_free_acid, input_quality
               FROM (
                   SELECT prediction_id, run_id, source_timestamp,
                          predicted_slurry_free_acid, input_quality
                   FROM predictions
                   WHERE run_id = ?
                   ORDER BY prediction_id DESC
                   LIMIT ?
               )
               ORDER BY prediction_id""",
            (run_id, max(1, int(limit))),
        ).fetchall()
    return [dict(row) for row in rows]
