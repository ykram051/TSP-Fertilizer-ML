from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .feature_engineering import FeatureBuilder, TARGET


def _familiarity(score: float, threshold: float) -> tuple[str, float]:
    ratio = score / max(threshold, 1e-12)
    if ratio <= 0.75:
        return "high", ratio
    if ratio <= 1.0:
        return "moderate", ratio
    return "low", ratio


class ModelRuntime:
    def __init__(self, artifact_root: str | Path):
        root = Path(artifact_root)
        self.builder = FeatureBuilder(root)
        self.virtual_metadata = self.builder.virtual_bundle
        self.target_free = joblib.load(
            root / "target_free_upgrade_artifacts" / "models" / "target_free_upgrade_candidate.joblib"
        )
        self.anchored = joblib.load(
            root / "delta_transition_artifacts" / "models" / "delta_transition_research_candidates.joblib"
        )

    @property
    def minimum_rows(self) -> int:
        # The current row plus the structural lookback.
        return self.builder.maximum_lookback + 1

    def _process_familiarity(self, features: pd.DataFrame) -> tuple[str, float, float]:
        pipeline = self.virtual_metadata["notebook02_pipelines_by_set"]["D_full"]["standard"]
        standardized = pipeline.transform(features)
        score = float(np.sqrt(np.mean(np.square(standardized[-1]))))
        threshold = float(self.virtual_metadata["drift_threshold_99pct"])
        label, ratio = _familiarity(score, threshold)
        return label, score, ratio

    def predict_target_free(self, frame: pd.DataFrame) -> dict[str, Any]:
        aligned, features = self.builder.build(frame)
        latest = features.tail(1)
        tree = self.target_free["imputer"].transform(latest)
        standardized = self.target_free["scaler"].transform(tree)
        clusterer, experts, fallback = self.target_free["model"]
        cluster = int(clusterer.predict(standardized)[0])
        prediction = float(experts.get(cluster, fallback).predict(standardized)[0])
        familiarity, drift_score, ratio = self._process_familiarity(latest)
        return {
            "mode": "target_free",
            "source_timestamp": pd.Timestamp(aligned.iloc[-1]["Date"]).isoformat(),
            "prediction_for": pd.Timestamp(aligned.iloc[-1]["Date"]).isoformat(),
            "predicted_slurry_free_acid": prediction,
            "predicted_change": None,
            "process_familiarity": familiarity,
            "drift_score": drift_score,
            "drift_to_threshold_ratio": ratio,
            "assigned_statistical_cluster": cluster,
            "model_name": self.target_free["selected_candidate"]["name"],
            "model_version": __version__,
            "test_status": self.target_free["test_status"],
        }

    def predict_target_anchored(self, frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
        aligned, features = self.builder.build(frame)
        latest_features = features.tail(1)
        process_pipeline = self.virtual_metadata["notebook02_pipelines_by_set"]["D_full"]["standard"]
        process_scaled = pd.DataFrame(process_pipeline.transform(latest_features), columns=features.columns)

        history = self.builder.target_history(frame).iloc[self.builder.maximum_lookback :].tail(1)
        metadata = self.anchored["experiment_metadata"][(horizon, "F_hybrid_full")]
        history_imputed = pd.DataFrame(
            metadata["history_imputer"].transform(history), columns=history.columns
        )
        history_scaled = pd.DataFrame(
            metadata["history_scaler"].transform(history_imputed), columns=history.columns
        )
        model_input = pd.concat([process_scaled, history_scaled], axis=1)[metadata["features"]]
        delta = float(self.anchored["delta_models"][(horizon, "F_hybrid_full", "ridge")].predict(model_input)[0])
        current = float(pd.to_numeric(aligned.iloc[-1][TARGET], errors="raise"))
        source_time = pd.Timestamp(aligned.iloc[-1]["Date"])
        familiarity, drift_score, ratio = self._process_familiarity(latest_features)
        return {
            "mode": "target_anchored",
            "horizon_minutes": horizon,
            "source_timestamp": source_time.isoformat(),
            "prediction_for": (source_time + pd.Timedelta(minutes=horizon)).isoformat(),
            "current_slurry_free_acid": current,
            "predicted_change": delta,
            "predicted_slurry_free_acid": current + delta,
            "process_familiarity": familiarity,
            "drift_score": drift_score,
            "drift_to_threshold_ratio": ratio,
            "model_name": "ridge_delta_F_hybrid_full",
            "model_version": __version__,
            "test_status": "retrospective; later labeled plant data required",
        }
