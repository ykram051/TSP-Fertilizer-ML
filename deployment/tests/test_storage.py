import tempfile
import unittest
from pathlib import Path

from industrial_inference.storage import PredictionRepository


class PredictionRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        schema = Path(__file__).parents[1] / "database" / "schema.sql"
        self.repository = PredictionRepository(root / "predictions.sqlite3", schema)
        self.repository.start_run(
            "run-1", "test-service", "0.1.0", "target_free", None,
            "2026-09-02T10:00:00+00:00", {"mode": "target_free"},
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_prediction_is_persisted_and_queryable(self):
        self.repository.add_prediction("run-1", {
            "sequence": 31,
            "source_timestamp": "2026-01-01T00:30:00+00:00",
            "prediction_for": "2026-01-01T00:30:00+00:00",
            "generated_at": "2026-09-02T10:00:01+00:00",
            "mode": "target_free",
            "predicted_slurry_free_acid": 11.42,
            "predicted_change": None,
            "input_quality": "GOOD",
            "imputed_inputs": [],
            "assigned_statistical_cluster": 2,
            "model_name": "five_cluster_ridge",
            "model_version": "0.1.0",
            "test_status": "retrospective",
        })
        summary = self.repository.summary()
        self.assertEqual(summary["predictions"], 1)
        self.assertEqual(self.repository.latest(1)[0]["sequence"], 31)

    def test_duplicate_sequence_in_same_run_is_rejected(self):
        prediction = {
            "sequence": 31, "source_timestamp": "2026-01-01T00:30:00+00:00",
            "prediction_for": "2026-01-01T00:30:00+00:00",
            "generated_at": "2026-09-02T10:00:01+00:00",
            "mode": "target_free", "predicted_slurry_free_acid": 11.42,
            "input_quality": "GOOD", "model_name": "model", "model_version": "0.1.0",
        }
        self.repository.add_prediction("run-1", prediction)
        with self.assertRaises(Exception):
            self.repository.add_prediction("run-1", prediction)


if __name__ == "__main__":
    unittest.main()
