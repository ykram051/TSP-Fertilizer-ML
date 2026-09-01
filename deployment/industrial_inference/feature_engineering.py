from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

EPSILON = 1e-6
TARGET = "SLURRY_FREE_ACID"
TIME_COLUMN = "Date"


def rolling_slope(series: pd.Series, window: int = 10) -> pd.Series:
    x = np.arange(window, dtype=float)
    centered = x - x.mean()
    denominator = np.sum(centered**2)
    return series.rolling(window, min_periods=window).apply(
        lambda y: np.sum(centered * (y - y.mean())) / denominator if np.isfinite(y).all() else np.nan,
        raw=True,
    )


def guarded_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe = denominator.where(denominator.abs() > EPSILON)
    return numerator / (safe + np.sign(safe) * EPSILON)


class FeatureBuilder:
    """Reproduce Notebook 02's accepted D_full feature contract."""

    def __init__(self, artifact_root: str | Path):
        root = Path(artifact_root)
        prep = root / "feature_preparation_artifacts"
        virtual_path = root / "exogenous_virtual_sensor_artifacts" / "models" / "target_free_virtual_sensor_bundle.joblib"
        self.virtual_bundle = joblib.load(virtual_path)
        self.registry = pd.read_csv(prep / "variable_registry.csv")
        self.lag_table = pd.read_csv(prep / "lag_selection_summary.csv")
        self.clipping = pd.read_csv(prep / "process_clipping_bounds.csv").set_index("feature")
        self.feature_names = self.virtual_bundle["feature_names_by_set"]["D_full"]
        self.maximum_lookback = int(self.virtual_bundle["maximum_lookback"])
        self.process_variables = [
            value for value in self.registry["variable"].tolist() if value != TARGET
        ]

    def build(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        missing = [name for name in self.process_variables if name not in frame]
        if missing:
            raise ValueError("Missing process variables: " + ", ".join(missing))
        work = frame.copy()
        for name in self.process_variables:
            work[name] = pd.to_numeric(work[name], errors="coerce")

        lag_map: dict[str, list[int]] = {}
        for _, row in self.lag_table.iterrows():
            lag_map[row["variable"]] = [int(v.strip()) for v in str(row["selected_lags_min"]).split(",") if v.strip()]

        values: dict[str, pd.Series] = {}
        for name in self.process_variables:
            series = work[name]
            values[name] = series
            for lag in lag_map[name]:
                values[f"{name}_lag_{lag}"] = series.shift(lag)
            values[f"{name}_roll_mean_10"] = series.rolling(10, min_periods=10).mean()
            values[f"{name}_roll_mean_30"] = series.rolling(30, min_periods=30).mean()
            values[f"{name}_roll_std_10"] = series.rolling(10, min_periods=10).std()
            values[f"{name}_diff_5"] = series.diff(5)
            values[f"{name}_slope_10"] = rolling_slope(series)

        acid_1 = work["FLOW_RATE_PHOSPHORIC_ACID_1"]
        acid_2 = work["FLOW_RATE_PHOSPHORIC_ACID_2"]
        total_acid = acid_1 + acid_2
        total_slurry = work["FLOW_RATE_SLURRY_M3_HR"] + work["FLOW_RATE_SLURRY_M3_HR_2"]
        process = {
            "TOTAL_ACID_FLOW": total_acid,
            "ACID_FLOW_IMBALANCE": (acid_1 - acid_2).abs(),
            "ACID_TO_PHOSPHATE_RATIO": guarded_ratio(total_acid, work["FLOW_RATE_GROUND_PHOSPHATE"]),
            "TOTAL_SLURRY_M3_FLOW": total_slurry,
            "SLURRY_DENSITY_PROXY": guarded_ratio(work["FLOW_RATE_SLURRY_T_PER_HR"], total_slurry),
            "ATTACK_PASSAGE_TEMP_GRADIENT": work["TEMPERATURE_CUVE_ATTAQUE"] - work["TEMPERATURE_CUVE_DE_PASSAGE"],
            "HOT_AIR_DRYER_OUTLET_GRADIENT": work["TEMPERATURE_AIR_CHAUD"] - work["TEMPERATURE_GAZ_SORTIE_SECHEUR"],
            "STEAM_SLURRY_TEMP_GRADIENT": work["TEMPERATURE_VAPEUR"] - work["TEMPERATURE_SLURRY_PULVERISATEUR"],
            "STEAM_ENERGY_PROXY": work["PRESSURE_VAPEUR"] * work["TEMPERATURE_VAPEUR"],
            "DRYING_INTENSITY_PROXY": work["FLOW_RATE_FIOUL"] * (
                work["TEMPERATURE_AIR_CHAUD"] - work["TEMPERATURE_GAZ_SORTIE_SECHEUR"]
            ) / (work["DEPRESSURE_SECHEUR"].abs() + EPSILON),
            "RECYCLE_TO_PRODUCTION_RATIO": guarded_ratio(work["RECYCLAGE"], work["PRODUCTION_TSP_BALANCE"]),
        }
        stability_flows = [
            "FLOW_RATE_PHOSPHORIC_ACID_1", "FLOW_RATE_PHOSPHORIC_ACID_2",
            "FLOW_RATE_GROUND_PHOSPHATE", "FLOW_RATE_FIOUL", "FLOW_RATE_VAPEUR",
        ]
        for name in stability_flows:
            mean = work[name].rolling(10, min_periods=10).mean()
            std = work[name].rolling(10, min_periods=10).std()
            process[f"{name}_CV_10"] = std / (mean.abs() + EPSILON)
        for name, series in process.items():
            if name in self.clipping.index:
                bounds = self.clipping.loc[name]
                series = series.clip(bounds.iloc[0], bounds.iloc[1])
            values[name] = series.replace([np.inf, -np.inf], np.nan)

        features = pd.DataFrame(values, index=work.index)
        for name in self.feature_names:
            if name.endswith("_missing") and name not in features:
                features[name] = work[name[:-8]].isna().astype("int8")
        aligned = work.iloc[self.maximum_lookback :].reset_index(drop=True)
        features = features.iloc[self.maximum_lookback :][self.feature_names].reset_index(drop=True)
        return aligned, features

    @staticmethod
    def target_history(frame: pd.DataFrame) -> pd.DataFrame:
        target = pd.to_numeric(frame[TARGET], errors="coerce")
        values: dict[str, Any] = {f"{TARGET}_current": target}
        for lag in [1, 2, 5, 10, 15, 30]:
            values[f"{TARGET}_lag_{lag}"] = target.shift(lag)
        values[f"{TARGET}_diff_5"] = target - target.shift(5)
        values[f"{TARGET}_roll_mean_10"] = target.rolling(10, min_periods=10).mean()
        values[f"{TARGET}_roll_std_10"] = target.rolling(10, min_periods=10).std()
        return pd.DataFrame(values)
