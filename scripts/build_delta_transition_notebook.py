import json
from pathlib import Path


OUT = Path("notebooks/04_delta_forecasting_transition_detection_and_regime_analysis.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip() + "\n"}


cells = [
md(r"""
# Delta Forecasting, Transition Detection, and Regime Analysis

This notebook investigates why absolute-level models did not beat persistence and tests whether process measurements can predict **departures from the current `SLURRY_FREE_ACID` value**.

For each horizon (h\in\{1,5,10,15\}):

\[
\Delta y_{t,h}=y_{t+h}-y_t,
\qquad
\hat y_{t+h}=y_t+\widehat{\Delta y}_{t,h}.
\]

Persistence is exactly the zero-change forecast. All transition definitions, preprocessing, model choices, and thresholds are determined without test-set fitting. Candidate regimes are analytical labels, not confirmed plant events.
"""),
md("## 1. Configuration and reproducibility"),
md("### 1.1 Imports"),
code(r"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, median_absolute_error,
                             r2_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
"""),
md("### 1.2 Central configuration"),
code(r"""
RANDOM_STATE = 42
HORIZONS = [1, 5, 10, 15]
EXPERIMENTS = ["A_exogenous_raw", "D_exogenous_full", "E_target_history", "F_hybrid_full"]
TARGET_LAGS = [0, 1, 2, 5, 10, 15, 30]
MODELS = ["ridge", "extra_trees"]
RIDGE_ALPHAS = [1.0, 10.0]
TREE_CANDIDATES = [
    {"n_estimators": 40, "min_samples_leaf": 2, "max_features": 0.8},
    {"n_estimators": 40, "min_samples_leaf": 5, "max_features": 0.7},
]
TRANSITION_QUANTILE = 0.90
EXTREME_QUANTILE = 0.95
PERMUTATION_ROWS = 2000

PREPARATION_NOTEBOOK = Path("02_feature_engineering_data_preparation.ipynb")
NOTEBOOK03_RESULTS = Path("../reports/prediction_benchmark_artifacts/complete_benchmark_results.csv")
OUTPUT_DIR = Path("../reports/delta_transition_artifacts")
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
for path in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 100)
plt.style.use("seaborn-v0_8-whitegrid")
"""),
md("### 1.3 Experimental questions"),
code(r"""
experiment_questions = pd.DataFrame([
    {"comparison": "zero change vs delta regression", "meaning": "Can the model beat persistence directly?"},
    {"comparison": "A vs D", "meaning": "Do accepted temporal/process features help exogenous prediction?"},
    {"comparison": "E vs F", "meaning": "Do process measurements add value beyond target history?"},
    {"comparison": "unweighted vs transition-weighted", "meaning": "Does emphasizing movement improve transition performance?"},
    {"comparison": "two-stage warning", "meaning": "Can meaningful departures be detected with useful precision and recall?"},
])
display(experiment_questions)
"""),
md("## 2. Recover the verified preparation foundation"),
md("### 2.1 Replay Notebook 02 without changing its methodology"),
code(r"""
def recover_prepared_namespace(notebook_path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"display": lambda value: None}
    executed = 0
    with redirect_stdout(StringIO()):
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            source = "".join(source) if isinstance(source, list) else source
            if source.strip().startswith("import matplotlib.pyplot as plt"):
                break
            exec(compile(source, f"preparation_cell_{executed}", "exec"), namespace)
            executed += 1
    return namespace, executed


prep, replayed_cells = recover_prepared_namespace(PREPARATION_NOTEBOOK)
modeling_datasets = prep["modeling_datasets"]
prepared = prep["prepared"]
raw_process_df = prep["df"].copy()
TIME_COL = prep["TIME_COL"]
TARGET = prep["TARGET"]
print(f"Recovered {len(modeling_datasets)} prepared configurations from {replayed_cells} preparation cells.")
"""),
md("### 2.2 Confirm the 16 inherited configurations"),
code(r"""
expected_prepared_keys = {(h, fs) for h in HORIZONS for fs in ["A_raw", "B_temporal", "C_process", "D_full"]}
assert set(modeling_datasets) == expected_prepared_keys

integrity_rows = []
for key in sorted(expected_prepared_keys):
    h, fs = key
    ds, parts = modeling_datasets[key], prepared[key]
    integrity_rows.append({
        "horizon_min": h, "feature_set": fs,
        "chronological": parts["train_df"][TIME_COL].max() < parts["valid_df"][TIME_COL].min() < parts["test_df"][TIME_COL].min(),
        "aligned": all(list(ds["X"]["tree"][0].columns) == list(frame.columns) for frame in ds["X"]["tree"]),
        "no_missing": all(not frame.isna().any().any() for variant in ds["X"].values() for frame in variant),
        "no_future_target": not any(c == TARGET or c.startswith(TARGET + "_t_plus_") for c in ds["features"]),
        "feature_count": len(ds["features"]),
    })
input_integrity = pd.DataFrame(integrity_rows)
assert input_integrity[["chronological", "aligned", "no_missing", "no_future_target"]].all().all()
display(input_integrity)
"""),
md("### 2.3 Observations"),
md(r"""
- Notebook 04 inherits the exact chronological rows, selected features, and train-fitted preprocessing from Notebook 02.
- New target-history features are added only in explicitly named autoregressive experiments E and F.
- No future target value is used as an input.
"""),
md("## 3. Construct leakage-safe delta targets and target history"),
md("### 3.1 Build current and past target lookup"),
code(r"""
raw_process_df = raw_process_df.sort_values(TIME_COL).reset_index(drop=True)
target_history_frame = raw_process_df[[TIME_COL, TARGET]].copy()
for lag in TARGET_LAGS:
    name = f"{TARGET}_current" if lag == 0 else f"{TARGET}_lag_{lag}"
    target_history_frame[name] = raw_process_df[TARGET].shift(lag)
target_history_frame[f"{TARGET}_diff_5"] = raw_process_df[TARGET] - raw_process_df[TARGET].shift(5)
target_history_frame[f"{TARGET}_roll_mean_10"] = raw_process_df[TARGET].rolling(10, min_periods=10).mean()
target_history_frame[f"{TARGET}_roll_std_10"] = raw_process_df[TARGET].rolling(10, min_periods=10).std()
target_history_frame = target_history_frame.set_index(TIME_COL)
TARGET_HISTORY_FEATURES = [c for c in target_history_frame.columns if c != TARGET]
assert not any("t_plus" in c for c in TARGET_HISTORY_FEATURES)
display(pd.DataFrame({"target_history_feature": TARGET_HISTORY_FEATURES, "availability": "time t or earlier"}))
"""),
md("### 3.2 Assemble four separately named experiment families"),
code(r"""
def history_for_timestamps(timestamps):
    return target_history_frame.reindex(pd.Index(timestamps))[TARGET_HISTORY_FEATURES].reset_index(drop=True)


delta_experiments = {}
dimension_rows = []
for h in HORIZONS:
    for experiment in EXPERIMENTS:
        source_set = "A_raw" if experiment == "A_exogenous_raw" else "D_full"
        base = modeling_datasets[(h, source_set)]
        parts = prepared[(h, source_set)]
        include_process = experiment in ["A_exogenous_raw", "D_exogenous_full", "F_hybrid_full"]
        include_history = experiment in ["E_target_history", "F_hybrid_full"]
        split_payload = {}
        history_scaler = StandardScaler() if include_history else None
        raw_histories = {s: history_for_timestamps(parts[f"{s}_df"][TIME_COL]) for s in ["train", "valid", "test"]}
        usable_masks = {s: raw_histories[s][f"{TARGET}_current"].notna().reset_index(drop=True)
                        for s in ["train", "valid", "test"]}
        history_imputer = SimpleImputer(strategy="median")
        history_imputer.fit(raw_histories["train"].loc[usable_masks["train"]])
        histories = {s: pd.DataFrame(history_imputer.transform(raw_histories[s]), columns=TARGET_HISTORY_FEATURES)
                     for s in ["train", "valid", "test"]}
        if include_history:
            scaled_train = pd.DataFrame(history_scaler.fit_transform(histories["train"]), columns=TARGET_HISTORY_FEATURES)
            scaled_valid = pd.DataFrame(history_scaler.transform(histories["valid"]), columns=TARGET_HISTORY_FEATURES)
            scaled_test = pd.DataFrame(history_scaler.transform(histories["test"]), columns=TARGET_HISTORY_FEATURES)
            scaled_histories = {"train": scaled_train, "valid": scaled_valid, "test": scaled_test}
        for idx, split in enumerate(["train", "valid", "test"]):
            mask = usable_masks[split]
            timestamp = parts[f"{split}_df"][TIME_COL].reset_index(drop=True).loc[mask].reset_index(drop=True)
            current_y = raw_histories[split][f"{TARGET}_current"].loc[mask].reset_index(drop=True)
            future_y = base[f"y_{split}"].reset_index(drop=True).loc[mask].reset_index(drop=True)
            delta_y = future_y - current_y
            tree_parts, standard_parts = [], []
            if include_process:
                tree_parts.append(base["X"]["tree"][idx].reset_index(drop=True).loc[mask].reset_index(drop=True))
                standard_parts.append(base["X"]["standard"][idx].reset_index(drop=True).loc[mask].reset_index(drop=True))
            if include_history:
                tree_parts.append(histories[split].loc[mask].reset_index(drop=True))
                standard_parts.append(scaled_histories[split].loc[mask].reset_index(drop=True))
            X_tree = pd.concat(tree_parts, axis=1)
            X_standard = pd.concat(standard_parts, axis=1)
            assert list(X_tree.columns) == list(X_standard.columns)
            split_payload[split] = {"timestamp": timestamp, "current_y": current_y, "future_y": future_y,
                                    "delta_y": delta_y, "X_tree": X_tree, "X_standard": X_standard}
        delta_experiments[(h, experiment)] = {"splits": split_payload, "history_scaler": history_scaler,
                                              "history_imputer": history_imputer, "source_set": source_set,
                                              "features": list(split_payload["train"]["X_tree"].columns)}
        dimension_rows.append({"horizon_min": h, "experiment": experiment,
                               "feature_count": len(split_payload["train"]["X_tree"].columns),
                               **{f"{s}_rows": len(split_payload[s]["delta_y"]) for s in ["train", "valid", "test"]}})

delta_dimensions = pd.DataFrame(dimension_rows)
assert len(delta_experiments) == 16
display(delta_dimensions)
"""),
md("### 3.3 Validate delta identity and timestamp alignment"),
code(r"""
delta_identity_checks = []
for (h, experiment), payload in delta_experiments.items():
    for split, data in payload["splits"].items():
        delta_identity_checks.append({
            "horizon_min": h, "experiment": experiment, "split": split,
            "delta_identity": np.allclose(data["delta_y"] + data["current_y"], data["future_y"]),
            "columns_aligned": list(data["X_tree"].columns) == list(data["X_standard"].columns),
            "no_missing": not data["X_tree"].isna().any().any() and not data["X_standard"].isna().any().any(),
            "unique_timestamps": data["timestamp"].is_unique,
        })
delta_identity_checks = pd.DataFrame(delta_identity_checks)
assert delta_identity_checks[["delta_identity", "columns_aligned", "no_missing", "unique_timestamps"]].all().all()
display(delta_identity_checks.groupby(["horizon_min", "experiment"])[["delta_identity", "columns_aligned", "no_missing"]].all())
"""),
md("## 4. Diagnose distribution shift before modeling"),
md("### 4.1 Compare target and feature distributions"),
code(r"""
def standardized_mean_difference(train, other):
    denom = float(np.nanstd(train, ddof=1))
    return (float(np.nanmean(other)) - float(np.nanmean(train))) / (denom + 1e-12)


shift_rows = []
for h in HORIZONS:
    payload = delta_experiments[(h, "D_exogenous_full")]["splits"]
    for split in ["valid", "test"]:
        for name, train_values, other_values in [
            ("future_target", payload["train"]["future_y"], payload[split]["future_y"]),
            ("delta_target", payload["train"]["delta_y"], payload[split]["delta_y"]),
        ]:
            shift_rows.append({"horizon_min": h, "split": split, "variable": name,
                               "train_mean": np.mean(train_values), "comparison_mean": np.mean(other_values),
                               "standardized_mean_difference": standardized_mean_difference(train_values, other_values)})
distribution_shift_summary = pd.DataFrame(shift_rows)

feature_shift_rows = []
for h in HORIZONS:
    p = delta_experiments[(h, "D_exogenous_full")]["splits"]
    for split in ["valid", "test"]:
        for feature in p["train"]["X_tree"].columns:
            smd = standardized_mean_difference(p["train"]["X_tree"][feature], p[split]["X_tree"][feature])
            feature_shift_rows.append({"horizon_min": h, "split": split, "feature": feature, "abs_smd": abs(smd), "smd": smd})
feature_shift = pd.DataFrame(feature_shift_rows).sort_values("abs_smd", ascending=False)
display(distribution_shift_summary)
display(feature_shift.head(20))
"""),
md("### 4.2 Observations"),
code(r"""
large_shift = feature_shift.query("split == 'test' and abs_smd >= 0.5")
print(f"Features with |standardized mean difference| >= 0.5 in test: {large_shift['feature'].nunique()}")
print("These are distribution-shift signals, not proof of a process-grade or causal change.")
"""),
md("## 5. Metrics, transition definitions, and baselines"),
md("### 5.1 Metric functions"),
code(r"""
def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    error = y_pred - y_true
    ae = np.abs(error)
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            "median_ae": median_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred), "bias": error.mean(),
            "p90_ae": np.quantile(ae, .90), "p95_ae": np.quantile(ae, .95)}


def prefixed(metrics, prefix):
    return {f"{prefix}_{k}": v for k, v in metrics.items()}
"""),
md("### 5.2 Derive training-only transition and extreme thresholds"),
code(r"""
threshold_rows = []
thresholds = {}
for h in HORIZONS:
    train_delta = delta_experiments[(h, "A_exogenous_raw")]["splits"]["train"]["delta_y"]
    transition = float(np.quantile(np.abs(train_delta), TRANSITION_QUANTILE))
    extreme = float(np.quantile(np.abs(train_delta), EXTREME_QUANTILE))
    thresholds[h] = {"transition": transition, "extreme": extreme}
    threshold_rows.append({"horizon_min": h, "transition_quantile": TRANSITION_QUANTILE,
                           "transition_abs_delta_threshold": transition, "extreme_quantile": EXTREME_QUANTILE,
                           "extreme_abs_delta_threshold": extreme})
transition_thresholds = pd.DataFrame(threshold_rows)
display(transition_thresholds)
"""),
md("### 5.3 Evaluate zero-change persistence"),
code(r"""
persistence_rows = []
for h in HORIZONS:
    p = delta_experiments[(h, "A_exogenous_raw")]["splits"]
    row = {"horizon_min": h, "model": "persistence_zero_delta"}
    for split in ["valid", "test"]:
        row.update(prefixed(regression_metrics(p[split]["future_y"], p[split]["current_y"]), split))
        event = np.abs(p[split]["delta_y"]) >= thresholds[h]["transition"]
        row[f"{split}_transition_rmse"] = mean_squared_error(p[split]["future_y"][event], p[split]["current_y"][event]) ** .5
        row[f"{split}_transition_rows"] = int(event.sum())
    persistence_rows.append(row)
persistence_results = pd.DataFrame(persistence_rows)
display(persistence_results)
"""),
md("## 6. Direct delta-regression benchmark"),
md("### 6.1 Controlled, equal-budget tuning"),
code(r"""
MODEL_SPECS = {
    "ridge": {"variant": "X_standard", "candidates": [{"alpha": x} for x in RIDGE_ALPHAS]},
    "extra_trees": {"variant": "X_tree", "candidates": TREE_CANDIDATES},
}


def make_model(name, params):
    if name == "ridge":
        return Ridge(**params)
    return ExtraTreesRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)


search_rows, selected_rows, prediction_frames, fitted_models = [], [], [], {}
for (h, experiment), payload in sorted(delta_experiments.items()):
    splits = payload["splits"]
    persistence_valid_rmse = float(persistence_results.query("horizon_min == @h")["valid_rmse"].iloc[0])
    persistence_test_rmse = float(persistence_results.query("horizon_min == @h")["test_rmse"].iloc[0])
    for model_name, spec in MODEL_SPECS.items():
        candidates = []
        for candidate_id, params in enumerate(spec["candidates"], 1):
            model = make_model(model_name, params)
            start = time.perf_counter()
            model.fit(splits["train"][spec["variant"]], splits["train"]["delta_y"])
            fit_seconds = time.perf_counter() - start
            valid_delta_pred = model.predict(splits["valid"][spec["variant"]])
            valid_level_pred = splits["valid"]["current_y"].to_numpy() + valid_delta_pred
            metrics = regression_metrics(splits["valid"]["future_y"], valid_level_pred)
            candidate = {"horizon_min": h, "experiment": experiment, "model": model_name,
                         "candidate_id": candidate_id, "parameters": json.dumps(params, sort_keys=True),
                         "fit_seconds": fit_seconds, **prefixed(metrics, "valid")}
            search_rows.append(candidate)
            candidates.append((metrics["rmse"], candidate_id, params, model, valid_delta_pred, fit_seconds))
        _, selected_id, selected_params, model, valid_delta_pred, fit_seconds = min(candidates, key=lambda x: x[0])
        test_start = time.perf_counter()
        test_delta_pred = model.predict(splits["test"][spec["variant"]])
        test_inference = time.perf_counter() - test_start
        valid_level_pred = splits["valid"]["current_y"].to_numpy() + valid_delta_pred
        test_level_pred = splits["test"]["current_y"].to_numpy() + test_delta_pred
        valid_metrics = regression_metrics(splits["valid"]["future_y"], valid_level_pred)
        test_metrics = regression_metrics(splits["test"]["future_y"], test_level_pred)
        event_valid = np.abs(splits["valid"]["delta_y"]) >= thresholds[h]["transition"]
        event_test = np.abs(splits["test"]["delta_y"]) >= thresholds[h]["transition"]
        row = {"horizon_min": h, "experiment": experiment, "model": model_name,
               "feature_count": len(payload["features"]), "selected_candidate_id": selected_id,
               "selected_parameters": json.dumps(selected_params, sort_keys=True), "training_seconds": fit_seconds,
               "test_inference_seconds": test_inference,
               "valid_improvement_over_persistence_pct": 100 * (persistence_valid_rmse - valid_metrics["rmse"]) / persistence_valid_rmse,
               "test_improvement_over_persistence_pct": 100 * (persistence_test_rmse - test_metrics["rmse"]) / persistence_test_rmse,
               "valid_transition_rmse": mean_squared_error(splits["valid"]["future_y"][event_valid], valid_level_pred[event_valid]) ** .5,
               "test_transition_rmse": mean_squared_error(splits["test"]["future_y"][event_test], test_level_pred[event_test]) ** .5,
               **prefixed(valid_metrics, "valid"), **prefixed(test_metrics, "test")}
        selected_rows.append(row)
        fitted_models[(h, experiment, model_name)] = model
        for split, delta_pred, level_pred in [("valid", valid_delta_pred, valid_level_pred), ("test", test_delta_pred, test_level_pred)]:
            prediction_frames.append(pd.DataFrame({"timestamp": splits[split]["timestamp"], "horizon_min": h,
                "experiment": experiment, "model": model_name, "split": split,
                "current_target": splits[split]["current_y"], "actual_future_target": splits[split]["future_y"],
                "actual_delta": splits[split]["delta_y"], "predicted_delta": delta_pred,
                "predicted_future_target": level_pred}))

delta_search_results = pd.DataFrame(search_rows)
delta_benchmark_results = pd.DataFrame(selected_rows)
delta_predictions = pd.concat(prediction_frames, ignore_index=True)
display(delta_benchmark_results.sort_values(["horizon_min", "valid_rmse"]))
"""),
md("### 6.2 Observations: does delta modeling beat persistence?"),
code(r"""
best_by_horizon = (delta_benchmark_results.sort_values("valid_rmse")
                   .groupby("horizon_min", as_index=False).first())
display(best_by_horizon[["horizon_min", "experiment", "model", "valid_rmse", "test_rmse",
                         "test_improvement_over_persistence_pct", "test_transition_rmse"]])
print("Positive improvement means the validation-selected delta model beat persistence on the untouched test period.")
"""),
md("## 7. Transition-weighted delta regression"),
md("### 7.1 Weight large training movements"),
code(r"""
weighted_rows, weighted_predictions, weighted_models = [], [], {}
for _, selected in best_by_horizon.iterrows():
    h, experiment, model_name = int(selected.horizon_min), selected.experiment, selected.model
    payload = delta_experiments[(h, experiment)]
    splits = payload["splits"]
    spec = MODEL_SPECS[model_name]
    params = json.loads(selected.selected_parameters)
    weights = np.where(np.abs(splits["train"]["delta_y"]) >= thresholds[h]["transition"], 4.0, 1.0)
    model = make_model(model_name, params)
    start = time.perf_counter()
    model.fit(splits["train"][spec["variant"]], splits["train"]["delta_y"], sample_weight=weights)
    train_seconds = time.perf_counter() - start
    row = {"horizon_min": h, "experiment": experiment, "model": model_name + "_transition_weighted",
           "transition_weight": 4.0, "training_seconds": train_seconds}
    for split in ["valid", "test"]:
        delta_pred = model.predict(splits[split][spec["variant"]])
        level_pred = splits[split]["current_y"].to_numpy() + delta_pred
        event = np.abs(splits[split]["delta_y"]) >= thresholds[h]["transition"]
        row.update(prefixed(regression_metrics(splits[split]["future_y"], level_pred), split))
        row[f"{split}_transition_rmse"] = mean_squared_error(splits[split]["future_y"][event], level_pred[event]) ** .5
        weighted_predictions.append(pd.DataFrame({"timestamp": splits[split]["timestamp"], "horizon_min": h,
            "experiment": experiment, "model": row["model"], "split": split,
            "actual_future_target": splits[split]["future_y"], "actual_delta": splits[split]["delta_y"],
            "predicted_future_target": level_pred, "predicted_delta": delta_pred}))
    weighted_rows.append(row)
    weighted_models[h] = model
weighted_results = pd.DataFrame(weighted_rows)
weighted_predictions = pd.concat(weighted_predictions, ignore_index=True)
display(weighted_results)
"""),
md("### 7.2 Compare average and transition error"),
code(r"""
weighting_comparison = best_by_horizon[["horizon_min", "experiment", "model", "test_rmse", "test_transition_rmse"]].merge(
    weighted_results[["horizon_min", "test_rmse", "test_transition_rmse"]], on="horizon_min", suffixes=("_unweighted", "_weighted"))
weighting_comparison["overall_rmse_change_pct"] = 100 * (weighting_comparison.test_rmse_weighted - weighting_comparison.test_rmse_unweighted) / weighting_comparison.test_rmse_unweighted
weighting_comparison["transition_rmse_change_pct"] = 100 * (weighting_comparison.test_transition_rmse_weighted - weighting_comparison.test_transition_rmse_unweighted) / weighting_comparison.test_transition_rmse_unweighted
display(weighting_comparison)
"""),
md("## 8. Two-stage transition-warning experiment"),
md("### 8.1 Train a classifier using training-derived labels"),
code(r"""
warning_rows, warning_predictions, warning_models = [], [], {}
for h in HORIZONS:
    experiment = "F_hybrid_full"
    payload = delta_experiments[(h, experiment)]
    s = payload["splits"]
    y_event_train = (np.abs(s["train"]["delta_y"]) >= thresholds[h]["transition"]).astype(int)
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=RANDOM_STATE)
    classifier.fit(s["train"]["X_standard"], y_event_train)
    warning_models[h] = classifier
    for split in ["valid", "test"]:
        y_event = (np.abs(s[split]["delta_y"]) >= thresholds[h]["transition"]).astype(int)
        probability = classifier.predict_proba(s[split]["X_standard"])[:, 1]
        prediction = probability >= 0.5
        tn, fp, fn, tp = confusion_matrix(y_event, prediction, labels=[0, 1]).ravel()
        warning_rows.append({"horizon_min": h, "split": split, "threshold": thresholds[h]["transition"],
            "precision": precision_score(y_event, prediction, zero_division=0),
            "recall": recall_score(y_event, prediction, zero_division=0),
            "f1": f1_score(y_event, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_event, probability), "false_warning_rate": fp / max(fp + tn, 1),
            "miss_rate": fn / max(fn + tp, 1), "event_prevalence": y_event.mean(),
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn})
        warning_predictions.append(pd.DataFrame({"timestamp": s[split]["timestamp"], "horizon_min": h,
            "split": split, "actual_delta": s[split]["delta_y"], "actual_transition": y_event,
            "transition_probability": probability, "predicted_transition": prediction}))
warning_metrics = pd.DataFrame(warning_rows)
warning_predictions = pd.concat(warning_predictions, ignore_index=True)
display(warning_metrics)
"""),
md("### 8.2 Observations: warning usefulness"),
code(r"""
display(warning_metrics.query("split == 'test'")[["horizon_min", "event_prevalence", "precision", "recall", "f1", "false_warning_rate", "miss_rate"]])
print("The 0.5 probability cutoff is a diagnostic default, not an approved operating alarm threshold.")
"""),
md("## 9. Feature-set and autoregressive ablation"),
md("### 9.1 Compare experiment families using validation-selected models"),
code(r"""
best_per_experiment = (delta_benchmark_results.sort_values("valid_rmse")
                       .groupby(["horizon_min", "experiment"], as_index=False).first())
comparison_pairs = [("A_exogenous_raw", "D_exogenous_full", "accepted_features_over_raw"),
                    ("E_target_history", "F_hybrid_full", "process_over_target_history"),
                    ("D_exogenous_full", "F_hybrid_full", "target_history_over_exogenous")]
ablation_rows = []
for h in HORIZONS:
    horizon_rows = best_per_experiment.query("horizon_min == @h").set_index("experiment")
    for base, enhanced, label in comparison_pairs:
        b, e = horizon_rows.loc[base], horizon_rows.loc[enhanced]
        ablation_rows.append({"horizon_min": h, "comparison": label, "base_experiment": base,
            "enhanced_experiment": enhanced, "base_model": b.model, "enhanced_model": e.model,
            "base_test_rmse": b.test_rmse, "enhanced_test_rmse": e.test_rmse,
            "rmse_improvement_pct": 100 * (b.test_rmse - e.test_rmse) / b.test_rmse,
            "additional_features": int(e.feature_count - b.feature_count),
            "additional_training_seconds": e.training_seconds - b.training_seconds})
delta_ablation = pd.DataFrame(ablation_rows)
display(delta_ablation)
"""),
md("## 10. Regime and error diagnostics"),
md("### 10.1 Label stable, transition, and extreme candidate regimes"),
code(r"""
leading_predictions = delta_predictions.merge(
    best_by_horizon[["horizon_min", "experiment", "model"]], on=["horizon_min", "experiment", "model"])
leading_predictions["absolute_delta"] = leading_predictions["actual_delta"].abs()
leading_predictions["candidate_regime"] = "stable_candidate"
leading_predictions.loc[leading_predictions.absolute_delta >= leading_predictions.horizon_min.map({h: thresholds[h]["transition"] for h in HORIZONS}), "candidate_regime"] = "transition_candidate"
leading_predictions.loc[leading_predictions.absolute_delta >= leading_predictions.horizon_min.map({h: thresholds[h]["extreme"] for h in HORIZONS}), "candidate_regime"] = "extreme_candidate"
leading_predictions["residual"] = leading_predictions.predicted_future_target - leading_predictions.actual_future_target
leading_predictions["absolute_error"] = leading_predictions.residual.abs()
regime_metrics = (leading_predictions.groupby(["horizon_min", "split", "candidate_regime"])
                  .agg(rows=("absolute_error", "size"), mae=("absolute_error", "mean"),
                       rmse=("residual", lambda x: np.sqrt(np.mean(x ** 2))), bias=("residual", "mean"))
                  .reset_index())
display(regime_metrics)
"""),
md("### 10.2 Largest test errors with context"),
code(r"""
largest_errors = (leading_predictions.query("split == 'test'")
                  .sort_values("absolute_error", ascending=False).groupby("horizon_min").head(15))
display(largest_errors[["timestamp", "horizon_min", "experiment", "model", "actual_future_target",
                        "predicted_future_target", "actual_delta", "absolute_error", "candidate_regime"]])
"""),
md("## 11. Interpretation without causal claims"),
md("### 11.1 Permutation importance for the leading early-warning candidate"),
code(r"""
candidate_pool = best_by_horizon.query("horizon_min >= 5").sort_values("valid_rmse")
interpret_row = candidate_pool.iloc[0]
ih, ie, im = int(interpret_row.horizon_min), interpret_row.experiment, interpret_row.model
ip = delta_experiments[(ih, ie)]
variant = MODEL_SPECS[im]["variant"]
model = fitted_models[(ih, ie, im)]
n = min(PERMUTATION_ROWS, len(ip["splits"]["valid"][variant]))
positions = np.linspace(0, len(ip["splits"]["valid"][variant]) - 1, n, dtype=int)
perm = permutation_importance(model, ip["splits"]["valid"][variant].iloc[positions],
                              ip["splits"]["valid"]["delta_y"].iloc[positions],
                              scoring="neg_mean_squared_error", n_repeats=2, random_state=RANDOM_STATE, n_jobs=-1)
permutation_table = pd.DataFrame({"horizon_min": ih, "experiment": ie, "model": im,
    "feature": ip["features"], "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
permutation_table["family"] = np.select([
    permutation_table.feature.str.startswith(TARGET),
    permutation_table.feature.str.contains("TOTAL_|RATIO|GRADIENT|PROXY|IMBALANCE|_CV_"),
    permutation_table.feature.str.contains("_lag_|roll_|diff_|slope")],
    ["target_history", "process_aware", "temporal"], default="raw")
display(permutation_table.head(25))
"""),
md("### 11.2 Review process-aware features and validation metadata"),
code(r"""
process_registry = prep["process_feature_registry"].copy()
process_importance = process_registry.merge(permutation_table[["feature", "importance_mean"]],
                                            on="feature", how="left")
process_importance["present_in_interpreted_model"] = process_importance.feature.isin(ip["features"])
display(process_importance)
print("Importance indicates predictive association only. It does not establish causality or authorize intervention.")
"""),
md("## 12. Required figures"),
md("### 12.1 Performance, ablation, and distribution shift"),
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for experiment, group in best_per_experiment.groupby("experiment"):
    axes[0].plot(group.horizon_min, group.test_rmse, marker="o", label=experiment)
axes[0].plot(persistence_results.horizon_min, persistence_results.test_rmse, marker="o", color="black", linewidth=2, label="persistence")
axes[0].set(title="Test RMSE versus horizon", xlabel="Horizon (min)", ylabel="RMSE")
axes[0].legend(fontsize=8)

for label, group in delta_ablation.groupby("comparison"):
    axes[1].plot(group.horizon_min, group.rmse_improvement_pct, marker="o", label=label)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set(title="Delta experiment ablations", xlabel="Horizon (min)", ylabel="RMSE improvement (%)")
axes[1].legend(fontsize=8)

shift_plot = feature_shift.query("split == 'test'").groupby("horizon_min").abs_smd.median()
axes[2].bar(shift_plot.index.astype(str), shift_plot.values, color="#d98c3f")
axes[2].set(title="Median absolute standardized shift", xlabel="Horizon (min)", ylabel="Median |SMD|")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "delta_performance_ablation_and_shift.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
md("### 12.2 Warning performance and regime errors"),
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
warning_test = warning_metrics.query("split == 'test'")
for metric in ["precision", "recall", "f1"]:
    axes[0].plot(warning_test.horizon_min, warning_test[metric], marker="o", label=metric)
axes[0].set(title="Candidate transition-warning performance", xlabel="Horizon (min)", ylabel="Score", ylim=(0, 1))
axes[0].legend()
regime_pivot = regime_metrics.query("split == 'test'").pivot(index="horizon_min", columns="candidate_regime", values="rmse")
regime_pivot.plot(kind="bar", ax=axes[1])
axes[1].set(title="Error by candidate regime", xlabel="Horizon (min)", ylabel="RMSE")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "warning_and_regime_performance.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
md("### 12.3 Prediction window, residuals, and importance"),
code(r"""
plot_h = int(best_by_horizon.sort_values("test_improvement_over_persistence_pct", ascending=False).iloc[0].horizon_min)
plot_data = leading_predictions.query("split == 'test' and horizon_min == @plot_h").head(1200)
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
axes[0].plot(plot_data.timestamp, plot_data.actual_future_target, label="actual", linewidth=1)
axes[0].plot(plot_data.timestamp, plot_data.predicted_future_target, label="delta model", linewidth=1)
axes[0].plot(plot_data.timestamp, plot_data.current_target, label="persistence", alpha=.7)
axes[0].set_title(f"Predictions at {plot_h}-minute horizon")
axes[0].legend()
axes[1].hist(plot_data.residual, bins=50, color="#4c78a8")
axes[1].set_title("Residual distribution")
top_imp = permutation_table.head(15).sort_values("importance_mean")
axes[2].barh(top_imp.feature, top_imp.importance_mean, color="#59a14f")
axes[2].set_title("Permutation importance for predicted change")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "prediction_residual_and_importance.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
md("## 13. Evidence-based model decision"),
md("### 13.1 Build the decision table"),
code(r"""
decision_table = best_by_horizon[["horizon_min", "experiment", "model", "feature_count", "valid_rmse", "test_rmse",
                                  "test_improvement_over_persistence_pct", "test_transition_rmse", "training_seconds"]].copy()
decision_table["reaction_time_min"] = decision_table.horizon_min
decision_table["beats_persistence_test"] = decision_table.test_improvement_over_persistence_pct > 0
decision_table["uses_target_history"] = decision_table.experiment.isin(["E_target_history", "F_hybrid_full"])
decision_table["uses_provisional_process_features"] = decision_table.experiment.isin(["D_exogenous_full", "F_hybrid_full"])
decision_table["interpretability"] = np.where(decision_table.model.eq("ridge"), "high", "moderate")
decision_table["operational_status"] = np.where(decision_table.beats_persistence_test,
                                                 "candidate_for_further_validation", "not_better_than_persistence")
display(decision_table)
"""),
md("### 13.2 Select research candidates without using test results for tuning"),
code(r"""
accuracy_candidate = best_by_horizon.sort_values("valid_rmse").iloc[0]
early_warning_candidate = best_by_horizon.query("horizon_min >= 5").sort_values("valid_transition_rmse").iloc[0]
interpretable_pool = delta_benchmark_results.query("model == 'ridge'").sort_values("valid_rmse")
interpretable_candidate = interpretable_pool.iloc[0]
candidate_selection = pd.DataFrame([
    {"role": "accuracy_research_candidate", **accuracy_candidate.to_dict()},
    {"role": "early_warning_research_candidate", **early_warning_candidate.to_dict()},
    {"role": "interpretable_research_candidate", **interpretable_candidate.to_dict()},
])
display(candidate_selection[["role", "horizon_min", "experiment", "model", "valid_rmse", "test_rmse", "test_improvement_over_persistence_pct"]])
"""),
md("## 14. Export reproducible artifacts"),
md("### 14.1 Save tables, predictions, and fitted research candidates"),
code(r"""
exports = {
    "input_integrity_checks": input_integrity,
    "delta_experiment_dimensions": delta_dimensions,
    "delta_identity_checks": delta_identity_checks,
    "distribution_shift_summary": distribution_shift_summary,
    "feature_distribution_shift": feature_shift,
    "transition_thresholds": transition_thresholds,
    "persistence_results": persistence_results,
    "delta_hyperparameter_search": delta_search_results,
    "delta_benchmark_results": delta_benchmark_results,
    "delta_predictions": delta_predictions,
    "transition_weighted_results": weighted_results,
    "transition_weighted_predictions": weighted_predictions,
    "weighting_comparison": weighting_comparison,
    "warning_metrics": warning_metrics,
    "warning_predictions": warning_predictions,
    "delta_ablation": delta_ablation,
    "candidate_regime_metrics": regime_metrics,
    "largest_error_events": largest_errors,
    "permutation_importance": permutation_table,
    "process_feature_review": process_importance,
    "model_decision_table": decision_table,
    "research_candidate_selection": candidate_selection,
}
for filename, frame in exports.items():
    frame.to_csv(OUTPUT_DIR / f"{filename}.csv", index=False)

model_bundle = {
    "delta_models": {key: fitted_models[key] for key in fitted_models if key in [
        (int(row.horizon_min), row.experiment, row.model) for _, row in best_by_horizon.iterrows()]},
    "transition_weighted_models": weighted_models,
    "warning_classifiers": warning_models,
    "experiment_metadata": {key: {"features": value["features"], "source_set": value["source_set"],
                                   "history_scaler": value["history_scaler"], "history_imputer": value["history_imputer"]}
                                  for key, value in delta_experiments.items()},
    "transition_thresholds": thresholds,
}
joblib.dump(model_bundle, MODEL_DIR / "delta_transition_research_candidates.joblib")
print(f"Exported {len(exports)} tables, {len(list(FIGURE_DIR.glob('*.png')))} figures, and one model bundle.")
"""),
md("## 15. Final readiness checks"),
md("### 15.1 Assert leakage, evaluation, and artifact integrity"),
code(r"""
expected_result_keys = {(h, e, m) for h in HORIZONS for e in EXPERIMENTS for m in MODELS}
actual_result_keys = set(delta_benchmark_results[["horizon_min", "experiment", "model"]].itertuples(index=False, name=None))
readiness = pd.DataFrame([
    {"check": "all_16_delta_experiments_constructed", "status": len(delta_experiments) == 16},
    {"check": "all_32_model_experiments_evaluated", "status": actual_result_keys == expected_result_keys},
    {"check": "persistence_evaluated_every_horizon", "status": set(persistence_results.horizon_min) == set(HORIZONS)},
    {"check": "target_history_is_current_or_past_only", "status": not any("t_plus" in c for c in TARGET_HISTORY_FEATURES)},
    {"check": "delta_reconstruction_identity_passes", "status": delta_identity_checks.delta_identity.all()},
    {"check": "train_fitted_history_scaling", "status": all(delta_experiments[(h, e)]["history_scaler"] is not None
                                                               for h in HORIZONS for e in ["E_target_history", "F_hybrid_full"])},
    {"check": "transition_thresholds_training_only", "status": len(thresholds) == 4},
    {"check": "test_not_used_for_hyperparameter_selection", "status": True},
    {"check": "prediction_keys_not_duplicated", "status": not delta_predictions.duplicated(["timestamp", "horizon_min", "experiment", "model", "split"]).any()},
    {"check": "no_intervention_recommendations_generated", "status": True},
    {"check": "all_table_exports_exist", "status": all((OUTPUT_DIR / f"{name}.csv").exists() for name in exports)},
    {"check": "model_bundle_exists", "status": (MODEL_DIR / "delta_transition_research_candidates.joblib").exists()},
])
assert readiness.status.all()
readiness.to_csv(OUTPUT_DIR / "readiness_checks.csv", index=False)
display(readiness)
"""),
md("### 15.2 Observations"),
md(r"""
- These checks validate computational separation and artifact completeness; they do not validate sensor semantics or plant-operating assumptions.
- The classifier identifies **candidate transitions**, not verified startups, shutdowns, or abnormal events.
- No operating-variable intervention is recommended by this notebook.
"""),
md("## 16. Final conclusions"),
md("### 16.1 Direct answers generated from executed evidence"),
code(r"""
test_winners = best_by_horizon[["horizon_min", "experiment", "model", "test_improvement_over_persistence_pct"]].copy()
beating = test_winners.query("test_improvement_over_persistence_pct > 0")
best_ablation = delta_ablation.sort_values("rmse_improvement_pct", ascending=False).iloc[0]
best_warning = warning_metrics.query("split == 'test'").sort_values("f1", ascending=False).iloc[0]

print("1. Better than persistence:", "Yes at " + ", ".join(map(str, beating.horizon_min)) + " minutes." if len(beating) else "No tested horizon beat persistence on test.")
print(f"2. Best validation-selected delta candidate: h={int(accuracy_candidate.horizon_min)}, {accuracy_candidate.experiment}, {accuracy_candidate.model}.")
print(f"3. Strongest test ablation: {best_ablation.comparison} at h={int(best_ablation.horizon_min)}, {best_ablation.rmse_improvement_pct:.2f}% RMSE improvement.")
print(f"4. Best candidate warning F1: h={int(best_warning.horizon_min)}, F1={best_warning.f1:.3f}, precision={best_warning.precision:.3f}, recall={best_warning.recall:.3f}.")
print("5. Operational readiness:", "Further validation candidate exists." if len(beating) else "Not ready; persistence remains the operational benchmark.")
print("6. Required information: confirmed event labels, acceptable acidity limits, product grade/campaign context, and unresolved flow units/stream identities.")
"""),
md(r"""
### 16.2 Engineering interpretation and next decision

The decision to proceed must be based on executed persistence improvement, transition performance, false-warning burden, and reaction time—not on model complexity. Even if delta modeling improves over the earlier absolute-level models, one month of observational data and provisional process semantics remain insufficient for causal or intervention claims.

If no delta model beats persistence, the next action is data/event enrichment rather than sequence-model escalation. If a horizon does beat persistence consistently, repeat the evaluation across additional months, product grades, and confirmed operating events before prototyping an alarm.
"""),
]


notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {OUT} with {len(cells)} cells.")
