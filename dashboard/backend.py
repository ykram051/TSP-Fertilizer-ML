from __future__ import annotations

import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response


ROOT = Path(__file__).resolve().parents[1]
TARGET = "SLURRY_FREE_ACID"
TIME_COL = "Date"
MAX_RETURN_ROWS = 2400
EPSILON = 1e-6

app = FastAPI(title="TSP Process Intelligence API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def resources() -> dict[str, Any]:
    artifact = ROOT / "reports" / "exogenous_virtual_sensor_artifacts"
    prep = ROOT / "reports" / "feature_preparation_artifacts"
    virtual = joblib.load(artifact / "models" / "target_free_virtual_sensor_bundle.joblib")
    early = joblib.load(ROOT / "reports" / "delta_transition_artifacts" / "models" / "delta_transition_research_candidates.joblib")
    registry = pd.read_csv(prep / "variable_registry.csv")
    lag_table = pd.read_csv(prep / "lag_selection_summary.csv")
    clip = pd.read_csv(prep / "process_clipping_bounds.csv").set_index("feature")
    benchmark = pd.read_csv(artifact / "full_comparison.csv")
    process_registry = pd.read_csv(prep / "process_feature_registry.csv")
    return {"virtual": virtual, "early": early, "registry": registry, "lags": lag_table,
            "clip": clip, "benchmark": benchmark, "process_registry": process_registry}


def parse_csv(content: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(400, f"CSV could not be read: {exc}") from exc
    if TIME_COL not in df.columns:
        raise HTTPException(400, f"Required timestamp column '{TIME_COL}' is missing.")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    if df[TIME_COL].isna().any():
        raise HTTPException(400, "Some timestamps could not be parsed.")
    return df.sort_values(TIME_COL).drop_duplicates(TIME_COL).reset_index(drop=True)


def downsample(frame: pd.DataFrame, limit: int = MAX_RETURN_ROWS) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    return frame.iloc[np.linspace(0, len(frame) - 1, limit, dtype=int)]


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    safe = frame.copy()
    for col in safe.select_dtypes(include=["datetime", "datetimetz"]).columns:
        safe[col] = safe[col].astype(str)
    safe = safe.replace([np.inf, -np.inf], np.nan)
    return json.loads(safe.to_json(orient="records", date_format="iso"))


def rolling_slope(series: pd.Series, window: int = 10) -> pd.Series:
    x = np.arange(window, dtype=float)
    xc, denom = x - x.mean(), np.sum((x - x.mean()) ** 2)
    return series.rolling(window, min_periods=window).apply(
        lambda y: np.sum(xc * (y - y.mean())) / denom if np.isfinite(y).all() else np.nan, raw=True
    )


def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    safe = den.where(den.abs() > EPSILON)
    return num / (safe + np.sign(safe) * EPSILON)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = resources()
    required = r["registry"]["variable"].tolist()
    required = [c for c in required if c != TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(400, "Missing process columns: " + ", ".join(missing))
    work = df.copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    lag_map: dict[str, list[int]] = {}
    for _, row in r["lags"].iterrows():
        text = str(row["selected_lags_min"])
        lag_map[row["variable"]] = [int(v.strip()) for v in text.split(",") if v.strip()]

    values: dict[str, pd.Series] = {}
    for col in required:
        s = work[col]
        values[col] = s
        for lag in lag_map[col]:
            values[f"{col}_lag_{lag}"] = s.shift(lag)
        values[f"{col}_roll_mean_10"] = s.rolling(10, min_periods=10).mean()
        values[f"{col}_roll_mean_30"] = s.rolling(30, min_periods=30).mean()
        values[f"{col}_roll_std_10"] = s.rolling(10, min_periods=10).std()
        values[f"{col}_diff_5"] = s.diff(5)
        values[f"{col}_slope_10"] = rolling_slope(s)

    a1, a2 = work["FLOW_RATE_PHOSPHORIC_ACID_1"], work["FLOW_RATE_PHOSPHORIC_ACID_2"]
    total_acid = a1 + a2
    total_slurry = work["FLOW_RATE_SLURRY_M3_HR"] + work["FLOW_RATE_SLURRY_M3_HR_2"]
    process = {
        "TOTAL_ACID_FLOW": total_acid,
        "ACID_FLOW_IMBALANCE": (a1 - a2).abs(),
        "ACID_TO_PHOSPHATE_RATIO": ratio(total_acid, work["FLOW_RATE_GROUND_PHOSPHATE"]),
        "TOTAL_SLURRY_M3_FLOW": total_slurry,
        "SLURRY_DENSITY_PROXY": ratio(work["FLOW_RATE_SLURRY_T_PER_HR"], total_slurry),
        "ATTACK_PASSAGE_TEMP_GRADIENT": work["TEMPERATURE_CUVE_ATTAQUE"] - work["TEMPERATURE_CUVE_DE_PASSAGE"],
        "HOT_AIR_DRYER_OUTLET_GRADIENT": work["TEMPERATURE_AIR_CHAUD"] - work["TEMPERATURE_GAZ_SORTIE_SECHEUR"],
        "STEAM_SLURRY_TEMP_GRADIENT": work["TEMPERATURE_VAPEUR"] - work["TEMPERATURE_SLURRY_PULVERISATEUR"],
        "STEAM_ENERGY_PROXY": work["PRESSURE_VAPEUR"] * work["TEMPERATURE_VAPEUR"],
        "DRYING_INTENSITY_PROXY": work["FLOW_RATE_FIOUL"] * (work["TEMPERATURE_AIR_CHAUD"] - work["TEMPERATURE_GAZ_SORTIE_SECHEUR"]) / (work["DEPRESSURE_SECHEUR"].abs() + EPSILON),
        "RECYCLE_TO_PRODUCTION_RATIO": ratio(work["RECYCLAGE"], work["PRODUCTION_TSP_BALANCE"]),
    }
    for col in ["FLOW_RATE_PHOSPHORIC_ACID_1", "FLOW_RATE_PHOSPHORIC_ACID_2", "FLOW_RATE_GROUND_PHOSPHATE", "FLOW_RATE_FIOUL", "FLOW_RATE_VAPEUR"]:
        mean, std = work[col].rolling(10, min_periods=10).mean(), work[col].rolling(10, min_periods=10).std()
        process[f"{col}_CV_10"] = std / (mean.abs() + EPSILON)
    for name, series in process.items():
        if name in r["clip"].index:
            bounds = r["clip"].loc[name]
            series = series.clip(bounds.iloc[0], bounds.iloc[1])
        values[name] = series.replace([np.inf, -np.inf], np.nan)

    all_features = pd.DataFrame(values, index=work.index)
    feature_names = r["virtual"]["feature_names_by_set"]["D_full"]
    for name in feature_names:
        if name.endswith("_missing") and name not in all_features:
            source = name[:-8]
            all_features[name] = work[source].isna().astype("int8")
    warmup = int(r["virtual"]["maximum_lookback"])
    return work.iloc[warmup:].reset_index(drop=True), all_features.iloc[warmup:][feature_names].reset_index(drop=True)


def predict_virtual(df: pd.DataFrame) -> pd.DataFrame:
    r = resources(); bundle = r["virtual"]
    aligned, features = build_features(df)
    member_predictions = []
    transformed: dict[str, pd.DataFrame] = {}
    for key in bundle["blend_members"]:
        feature_set, model_name = key
        variant = bundle["member_input_variants"][key]
        pipeline = bundle["notebook02_pipelines_by_set"][feature_set]["standard" if variant == "X_standard" else "tree"]
        matrix = pd.DataFrame(pipeline.transform(features), columns=features.columns)
        transformed[variant] = matrix
        member_predictions.append(np.asarray(bundle["member_models"][key].predict(matrix)).ravel())
    prediction = bundle["blend_weight_first"] * member_predictions[0] + bundle["blend_weight_second"] * member_predictions[1]
    standard = transformed.get("X_standard")
    if standard is None:
        standard = pd.DataFrame(bundle["notebook02_pipelines_by_set"]["D_full"]["standard"].transform(features), columns=features.columns)
    drift = np.sqrt(np.mean(np.square(standard.to_numpy()), axis=1))
    half = float(bundle["interval_half_width_90"])
    out = pd.DataFrame({TIME_COL: aligned[TIME_COL], "predicted_slurry_free_acid": prediction,
                        "lower_90": prediction - half, "upper_90": prediction + half,
                        "drift_score": drift, "outside_training_envelope": drift > float(bundle["drift_threshold_99pct"])})
    if TARGET in aligned:
        out["actual_slurry_free_acid"] = aligned[TARGET]
        out["absolute_error"] = (out["predicted_slurry_free_acid"] - out["actual_slurry_free_acid"]).abs()
    return out


def target_history(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(df[TARGET], errors="coerce")
    result = {f"{TARGET}_current": y}
    for lag in [1, 2, 5, 10, 15, 30]: result[f"{TARGET}_lag_{lag}"] = y.shift(lag)
    result[f"{TARGET}_diff_5"] = y - y.shift(5)
    result[f"{TARGET}_roll_mean_10"] = y.rolling(10, min_periods=10).mean()
    result[f"{TARGET}_roll_std_10"] = y.rolling(10, min_periods=10).std()
    return pd.DataFrame(result)


def predict_early_warning(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if TARGET not in df.columns:
        raise HTTPException(400, "Early-warning mode requires current and historical SLURRY_FREE_ACID values.")
    r = resources(); aligned, features = build_features(df)
    pipeline = r["virtual"]["notebook02_pipelines_by_set"]["D_full"]["standard"]
    process_scaled = pd.DataFrame(pipeline.transform(features), columns=features.columns)
    history = target_history(df).iloc[int(r["virtual"]["maximum_lookback"]):].reset_index(drop=True)
    meta = r["early"]["experiment_metadata"][(horizon, "F_hybrid_full")]
    history_imputed = pd.DataFrame(meta["history_imputer"].transform(history), columns=history.columns)
    history_scaled = pd.DataFrame(meta["history_scaler"].transform(history_imputed), columns=history.columns)
    X = pd.concat([process_scaled, history_scaled], axis=1)[meta["features"]]
    delta = r["early"]["delta_models"][(horizon, "F_hybrid_full", "ridge")].predict(X)
    current = pd.to_numeric(aligned[TARGET], errors="coerce")
    actual_future = pd.to_numeric(df[TARGET], errors="coerce").shift(-horizon).iloc[
        int(r["virtual"]["maximum_lookback"]):
    ].reset_index(drop=True)
    predicted_future = current + delta
    out = pd.DataFrame({TIME_COL: aligned[TIME_COL], "current_slurry_free_acid": current,
                        "predicted_change": delta, f"predicted_{horizon}min": predicted_future,
                        f"actual_{horizon}min": actual_future})
    out["absolute_error"] = (out[f"predicted_{horizon}min"] - out[f"actual_{horizon}min"]).abs()
    return out[current.notna()].reset_index(drop=True)


def summary(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=np.number)
    missing = df.isna().sum().sort_values(ascending=False)
    target_present = TARGET in df.columns
    correlations = []
    if target_present:
        corr = numeric.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(12)
        correlations = [{"variable": k, "absolute_correlation": float(v)} for k, v in corr.items()]
    process_numeric = numeric.drop(columns=[TARGET], errors="ignore")
    medians = process_numeric.median()
    scales = (process_numeric.quantile(.75) - process_numeric.quantile(.25)).replace(0, np.nan)
    normalized_step = process_numeric.diff().abs().div(scales, axis=1).replace([np.inf, -np.inf], np.nan)
    process_change_score = normalized_step.median(axis=1, skipna=True).fillna(0)
    transition_threshold = float(process_change_score.quantile(.95))
    transition_mask = process_change_score >= transition_threshold

    # Error is attached after a prediction run. Keeping dataset analysis
    # independent avoids executing a full model merely to open Explore.
    virtual_error = pd.Series(np.nan, index=df.index, dtype=float)

    transition_positions = np.flatnonzero(transition_mask.to_numpy())
    episodes: list[tuple[int, int]] = []
    if len(transition_positions):
        starts = np.r_[0, np.flatnonzero(np.diff(transition_positions) > 1) + 1]
        ends = np.r_[starts[1:] - 1, len(transition_positions) - 1]
        episodes = [(int(transition_positions[a]), int(transition_positions[b])) for a, b in zip(starts, ends)]
    episodes = sorted(episodes, key=lambda pair: process_change_score.iloc[pair[0]:pair[1]+1].max(), reverse=True)[:10]

    change_windows = []
    target_series = pd.to_numeric(df[TARGET], errors="coerce") if target_present else None
    for event_id, (start, end) in enumerate(episodes, 1):
        driver_values = normalized_step.iloc[start:end+1].max().dropna().sort_values(ascending=False).head(3)
        driver_names = driver_values.index.tolist()
        pre_error = virtual_error.iloc[max(0, start-30):start].mean()
        event_error = virtual_error.iloc[start:min(len(df), end+31)].mean()
        row: dict[str, Any] = {
            "event_id": event_id, "start_timestamp": str(df.iloc[start][TIME_COL]),
            "end_timestamp": str(df.iloc[end][TIME_COL]), "duration_minutes": int(end-start+1),
            "peak_process_change_score": float(process_change_score.iloc[start:end+1].max()),
            "drivers": ", ".join(driver_names), "driver_list": driver_names,
            "error_before": None if pd.isna(pre_error) else float(pre_error),
            "error_during_after": None if pd.isna(event_error) else float(event_error),
            "error_change_pct": None if pd.isna(pre_error) or pre_error == 0 or pd.isna(event_error) else float(100*(event_error-pre_error)/pre_error),
            "plant_event_label": "Unlabeled",
        }
        if target_present and target_series is not None:
            baseline = target_series.iloc[max(0, start-1)]
            target_at_end = target_series.iloc[end]
            row["target_before_window"] = None if pd.isna(baseline) else float(baseline)
            row["target_at_window_end"] = None if pd.isna(target_at_end) else float(target_at_end)
            row["target_change_during_window"] = None if pd.isna(baseline) or pd.isna(target_at_end) else float(target_at_end-baseline)
            for lag in [1, 5, 10, 15]:
                response_pos = min(len(df)-1, end+lag)
                target_after = target_series.iloc[response_pos]
                total_response = target_after - baseline
                delayed_response = target_after - target_at_end
                row[f"target_after_{lag}min"] = None if pd.isna(target_after) else float(target_after)
                row[f"target_response_{lag}min"] = None if pd.isna(total_response) else float(total_response)
                row[f"target_delayed_response_{lag}min"] = None if pd.isna(delayed_response) else float(delayed_response)
            low, high = target_series.quantile(.05), target_series.quantile(.95)
            after_10 = row.get("target_after_10min")
            row["target_after_10min_empirical_status"] = "unavailable" if after_10 is None else "below dataset central 90%" if after_10 < low else "above dataset central 90%" if after_10 > high else "within dataset central 90%"
        detail_start, detail_end = max(0, start-30), min(len(df), end+31)
        detail = pd.DataFrame({"timestamp": df[TIME_COL].iloc[detail_start:detail_end].astype(str).values,
                               "process_change_score": process_change_score.iloc[detail_start:detail_end].values,
                               "prediction_error": virtual_error.iloc[detail_start:detail_end].values})
        if target_present and target_series is not None:
            detail[TARGET] = target_series.iloc[detail_start:detail_end].values
        for driver in driver_names:
            detail[driver] = process_numeric[driver].iloc[detail_start:detail_end].values
        row["series"] = records(detail)
        change_windows.append(row)

    dynamics = []
    for col in process_numeric.columns:
        series = process_numeric[col]
        dynamics.append({"variable": col, "relative_variability": float(series.std() / (abs(series.mean()) + EPSILON)),
                         "median_normalized_step": float(normalized_step[col].median(skipna=True)),
                         "range": float(series.max() - series.min()), "missing": int(series.isna().sum())})
    sensor_dynamics = sorted(dynamics, key=lambda x: x["median_normalized_step"], reverse=True)

    cadence = df[TIME_COL].diff().dt.total_seconds().median() / 60
    observations = [
        f"The median sampling interval is {cadence:.1f} minute(s).",
        f"{transition_mask.mean()*100:.1f}% of rows are candidate multivariate transitions using the dataset's 95th-percentile process-change threshold.",
    ]
    if sensor_dynamics:
        observations.append(f"{sensor_dynamics[0]['variable']} has the largest typical normalized minute-to-minute movement.")
    if missing.iloc[0] > 0:
        observations.append(f"{missing.index[0]} has the most missing observations ({int(missing.iloc[0])}).")
    if target_present:
        target_step = pd.to_numeric(df[TARGET], errors="coerce").diff().abs()
        observations.append(f"The largest observed one-minute target movement is {target_step.max():.3f}.")

    return {"rows": len(df), "columns": len(df.columns), "start": str(df[TIME_COL].min()), "end": str(df[TIME_COL].max()),
            "target_present": target_present, "missing_cells": int(df.isna().sum().sum()),
            "missing_by_column": [{"variable": k, "missing": int(v)} for k, v in missing.head(12).items()],
            "correlations": correlations, "numeric_columns": list(numeric.columns),
            "observations": observations, "change_windows": change_windows,
            "sensor_dynamics": sensor_dynamics[:15], "transition_threshold": transition_threshold,
            "candidate_transition_rows": int(transition_mask.sum()),
            "stable_rows": int((~transition_mask).sum()), "sampling_interval_minutes": float(cadence)}


RESULT_CACHE: dict[str, pd.DataFrame] = {}


def payload(df: pd.DataFrame, source: str) -> dict[str, Any]:
    # Return every numeric signal in the downsampled chart payload. Limiting this
    # to the first few columns made valid selector choices render an empty chart.
    view_cols = [TIME_COL] + list(df.select_dtypes(include=np.number).columns)
    return {"source": source, "summary": summary(df), "preview": records(downsample(df[view_cols])),
            "benchmark": records(resources()["benchmark"].sort_values("valid_rmse").head(12)),
            "process_features": records(resources()["process_registry"])}


@app.get("/api/health")
def health(): return {"status": "ok", "models": ["target_free_virtual_sensor", "target_anchored_early_warning"]}


@app.get("/api/default")
@lru_cache(maxsize=1)
def default_data():
    return payload(parse_csv((ROOT / "tsp_1min.csv").read_bytes()), "Default January dataset")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    return payload(parse_csv(content), file.filename or "Uploaded CSV")


@app.post("/api/predict/{mode}")
async def predict(mode: str, file: UploadFile = File(...), horizon: int = 10):
    content = await file.read(); df = parse_csv(content)
    if mode == "virtual_sensor": result = predict_virtual(df)
    elif mode == "early_warning" and horizon in [1, 5, 10, 15]: result = predict_early_warning(df, horizon)
    else: raise HTTPException(400, "Unknown mode or unsupported horizon.")
    key = hashlib.sha256(content + mode.encode() + str(horizon).encode()).hexdigest()[:16]
    RESULT_CACHE[key] = result
    metrics = None
    if "actual_slurry_free_acid" in result:
        valid = result.dropna(subset=["actual_slurry_free_acid"])
        metrics = {"mae": float((valid.predicted_slurry_free_acid-valid.actual_slurry_free_acid).abs().mean()),
                   "rmse": float(np.sqrt(np.mean((valid.predicted_slurry_free_acid-valid.actual_slurry_free_acid)**2))),
                   "coverage_90": float(((valid.actual_slurry_free_acid>=valid.lower_90)&(valid.actual_slurry_free_acid<=valid.upper_90)).mean())}
    elif f"actual_{horizon}min" in result:
        actual_col, predicted_col = f"actual_{horizon}min", f"predicted_{horizon}min"
        valid = result.dropna(subset=[actual_col, predicted_col])
        metrics = {"mae": float((valid[predicted_col]-valid[actual_col]).abs().mean()),
                   "rmse": float(np.sqrt(np.mean((valid[predicted_col]-valid[actual_col])**2))),
                   "coverage_90": None,
                   "evaluated_rows": int(len(valid))}
    event_error_metrics = []
    if "absolute_error" in result.columns:
        error_frame = result[[TIME_COL, "absolute_error"]].dropna().copy()
        error_frame[TIME_COL] = pd.to_datetime(error_frame[TIME_COL], utc=True)
        for event in summary(df)["change_windows"]:
            start, end = pd.Timestamp(event["start_timestamp"]), pd.Timestamp(event["end_timestamp"])
            before = error_frame[(error_frame[TIME_COL] >= start-pd.Timedelta(minutes=30)) & (error_frame[TIME_COL] < start)].absolute_error.mean()
            local_error = error_frame[(error_frame[TIME_COL] >= start-pd.Timedelta(minutes=30)) & (error_frame[TIME_COL] <= end+pd.Timedelta(minutes=30))]
            during_after = local_error[local_error[TIME_COL] >= start].absolute_error.mean()
            event_error_metrics.append({"event_id": event["event_id"],
                "error_before": None if pd.isna(before) else float(before),
                "error_during_after": None if pd.isna(during_after) else float(during_after),
                "error_change_pct": None if pd.isna(before) or before == 0 or pd.isna(during_after) else float(100*(during_after-before)/before),
                "error_series": records(local_error.rename(columns={TIME_COL:"timestamp", "absolute_error":"prediction_error"}))})
    return {"result_id": key, "rows": len(result), "metrics": metrics,
            "event_error_metrics": event_error_metrics, "preview": records(downsample(result))}


@app.get("/api/download/{result_id}")
def download(result_id: str):
    if result_id not in RESULT_CACHE: raise HTTPException(404, "Prediction result not found in this session.")
    return Response(RESULT_CACHE[result_id].to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="predictions_{result_id}.csv"'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
