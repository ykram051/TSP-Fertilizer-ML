import asyncio
import sqlite3
from pathlib import Path

import pytest

from industrial_inference.service import InferenceService
from industrial_inference.storage import PredictionRepository

SCHEMA = Path(__file__).parents[1] / "database" / "schema.sql"


@pytest.fixture
def repository(tmp_path):
    repo = PredictionRepository(tmp_path / "predictions.sqlite3", SCHEMA)
    repo.start_run("run-1", "svc", "0.1.0", "target_free", None,
                   "2026-09-02T10:00:00+00:00", {})
    return repo


def _run_row(repo):
    with sqlite3.connect(repo.database_path) as connection:
        return connection.execute(
            "SELECT stopped_at, final_status FROM service_runs WHERE run_id = 'run-1'"
        ).fetchone()


def test_finish_run_records_end_time_and_status(repository):
    assert _run_row(repository) == (None, None)
    repository.finish_run("run-1", "2026-09-02T11:00:00+00:00", "DISCONNECTED")
    assert _run_row(repository) == ("2026-09-02T11:00:00+00:00", "DISCONNECTED")


def test_finish_run_never_overwrites_the_first_closure(repository):
    repository.finish_run("run-1", "2026-09-02T11:00:00+00:00", "DISCONNECTED")
    repository.finish_run("run-1", "2026-09-02T12:00:00+00:00", "STOPPED")
    assert _run_row(repository) == ("2026-09-02T11:00:00+00:00", "DISCONNECTED")


def test_dashboard_indexes_exist(repository):
    with sqlite3.connect(repository.database_path) as connection:
        names = {row[1] for row in connection.execute("PRAGMA index_list(predictions)")}
        runs = {row[1] for row in connection.execute("PRAGMA index_list(service_runs)")}
    assert "idx_predictions_run_order" in names
    assert "idx_runs_started_at" in runs


def test_schema_can_be_applied_twice_to_an_existing_database(tmp_path):
    PredictionRepository(tmp_path / "db.sqlite3", SCHEMA)
    PredictionRepository(tmp_path / "db.sqlite3", SCHEMA)  # idempotent upgrade


def _bare_service():
    service = InferenceService.__new__(InferenceService)  # no model or OPC needed
    service.queue = asyncio.Queue(maxsize=10)
    return service


def test_drain_queue_discards_stale_notifications():
    service = _bare_service()
    for value in (5, 6, 7):
        service.queue.put_nowait(value)
    assert service.drain_queue() == 3
    assert service.queue.empty()
    assert service.drain_queue() == 0


def test_close_audit_run_survives_database_errors():
    service = _bare_service()
    service.run_id = "run-1"

    class Broken:
        def finish_run(self, *args):
            raise sqlite3.OperationalError("database is locked")

    service.repository = Broken()
    service.close_audit_run("DISCONNECTED")  # must not raise


def test_close_audit_run_without_a_run_is_a_noop():
    service = _bare_service()
    service.run_id = None
    service.close_audit_run("STOPPED")
