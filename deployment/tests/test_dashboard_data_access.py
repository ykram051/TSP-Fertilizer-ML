import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "dashboard"))
from data_access import read_health, read_predictions
from interaction import latest_delta, nearest_point_index, seconds_since


class DashboardDataAccessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_health_reader_accepts_actual_service_status_key(self):
        path = self.root / "health.json"
        path.write_text(json.dumps({"status": "RUNNING", "input_quality": "GOOD"}))
        self.assertEqual(read_health(path)["status"], "RUNNING")

    def test_missing_health_file_returns_starting(self):
        self.assertEqual(read_health(self.root / "missing.json")["status"], "STARTING")

    def test_prediction_reader_returns_latest_rows_in_time_order(self):
        path = self.root / "predictions.sqlite3"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(
                """CREATE TABLE service_runs (
                       run_id TEXT PRIMARY KEY,
                       started_at TEXT)"""
            )
            connection.execute(
                """CREATE TABLE predictions (
                       prediction_id INTEGER PRIMARY KEY,
                       run_id TEXT,
                       source_timestamp TEXT,
                       predicted_slurry_free_acid REAL,
                       input_quality TEXT)"""
            )
            connection.executemany(
                "INSERT INTO service_runs VALUES (?, ?)",
                [("old", "2026-01-01T00:00:00+00:00"),
                 ("current", "2026-01-02T00:00:00+00:00")],
            )
            connection.executemany(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
                [(1, "old", "2026-01-01T23:59:00+00:00", 99.0, "GOOD"),
                 (2, "current", "2026-01-02T00:01:00+00:00", 11.1, "GOOD"),
                 (3, "current", "2026-01-02T00:02:00+00:00", 11.2, "IMPUTED"),
                 (4, "current", "2026-01-02T00:03:00+00:00", 11.3, "GOOD")],
            )
        rows = read_predictions(path, limit=2)
        self.assertEqual([row["predicted_slurry_free_acid"] for row in rows], [11.2, 11.3])
        self.assertEqual({row["run_id"] for row in rows}, {"current"})

    def test_explicit_active_run_returns_no_previous_run_during_warmup(self):
        path = self.root / "predictions.sqlite3"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE service_runs (run_id TEXT PRIMARY KEY, started_at TEXT)")
            connection.execute(
                "CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY, run_id TEXT, "
                "source_timestamp TEXT, predicted_slurry_free_acid REAL, input_quality TEXT)"
            )
            connection.execute("INSERT INTO service_runs VALUES ('old', '2026-01-01T00:00:00Z')")
            connection.execute(
                "INSERT INTO predictions VALUES (1, 'old', '2026-01-01T00:01:00Z', 11.0, 'GOOD')"
            )
        self.assertEqual(read_predictions(path, run_id="new-run"), [])

    def test_nearest_point_lookup_snaps_and_resolves_ties_to_earlier(self):
        values = [10.0, 20.0, 30.0]
        self.assertEqual(nearest_point_index(values, 26.0), 2)
        self.assertEqual(nearest_point_index(values, 25.0), 1)
        self.assertEqual(nearest_point_index(values, -1.0), 0)
        self.assertIsNone(nearest_point_index([], 4.0))

    def test_latest_delta(self):
        self.assertAlmostEqual(latest_delta([11.2, 11.35]), 0.15)
        self.assertIsNone(latest_delta([11.2]))

    def test_freshness_uses_utc_timestamp(self):
        from datetime import datetime, timezone
        now = datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
        self.assertEqual(seconds_since("2026-01-01T00:00:02+00:00", now), 8.0)


if __name__ == "__main__":
    unittest.main()
