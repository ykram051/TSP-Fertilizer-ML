"""Contract test between the saved research artifacts and the runtime."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ARTIFACTS = Path(__file__).resolve().parents[2] / "reports"
pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "target_free_upgrade_artifacts" / "models").exists(),
    reason="research artifacts are not available in this checkout",
)


@pytest.fixture(scope="module")
def runtime():
    from industrial_inference.models import ModelRuntime
    return ModelRuntime(ARTIFACTS)


def _frame(runtime, rows=None):
    rows = rows or runtime.minimum_rows + 4
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({name: rng.normal(50, 5, rows) for name in runtime.builder.process_variables})
    frame["Date"] = pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC")
    frame["SLURRY_FREE_ACID"] = 11.3
    return frame


def test_feature_contract_matches_the_report(runtime):
    assert len(runtime.builder.feature_names) == 178
    assert runtime.minimum_rows == 31


def test_target_free_prediction_is_finite_and_timestamped(runtime):
    frame = _frame(runtime)
    result = runtime.predict_target_free(frame)
    assert np.isfinite(result["predicted_slurry_free_acid"])
    assert result["prediction_for"] == pd.Timestamp(frame["Date"].iloc[-1]).isoformat()
    assert result["predicted_change"] is None


@pytest.mark.parametrize("horizon", [1, 5, 10, 15])
def test_target_anchored_prediction_is_current_plus_delta(runtime, horizon):
    frame = _frame(runtime)
    result = runtime.predict_target_anchored(frame, horizon)
    assert np.isfinite(result["predicted_change"])
    assert result["predicted_slurry_free_acid"] == pytest.approx(
        result["current_slurry_free_acid"] + result["predicted_change"]
    )
    expected = pd.Timestamp(frame["Date"].iloc[-1]) + pd.Timedelta(minutes=horizon)
    assert pd.Timestamp(result["prediction_for"]) == expected


def test_missing_input_cells_are_imputed_not_fatal(runtime):
    frame = _frame(runtime)
    frame.loc[frame.index[-1], runtime.builder.process_variables[0]] = np.nan
    assert np.isfinite(runtime.predict_target_free(frame)["predicted_slurry_free_acid"])


def test_missing_column_is_reported_clearly(runtime):
    frame = _frame(runtime).drop(columns=[runtime.builder.process_variables[0]])
    with pytest.raises(ValueError, match="Missing process variables"):
        runtime.predict_target_free(frame)


def test_service_refuses_a_buffer_too_small_for_the_model(monkeypatch):
    import os
    from industrial_inference.configuration import load_configuration
    from industrial_inference.service import InferenceService

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("SERVICE_CONFIG", str(root / "config" / "service.yaml"))
    monkeypatch.setenv("TAG_CONFIG", str(root / "config" / "opc_tags.yaml"))
    monkeypatch.setenv("ARTIFACT_ROOT", str(ARTIFACTS))
    config, tags = load_configuration()
    config["service"]["maximum_buffer_rows"] = 10
    with pytest.raises(ValueError, match="smaller than"):
        InferenceService(config, tags)
