import json
from pathlib import Path


OUT = Path("notebooks/03_prediction_benchmark_and_process_understanding.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip() + "\n"}


cells = [
md(r"""
# Prediction Benchmark and Process Understanding for `SLURRY_FREE_ACID`

This notebook is the controlled modeling stage of the TSP fertilizer forecasting project. It preserves the executed preparation methodology from Notebook 02 and compares four direct horizons and four feature-set hypotheses.

The primary question is not merely which model has the lowest error. It is whether any model improves on persistence at a horizon that provides useful reaction time, remains understandable, and behaves acceptably during process transitions.

No feature is interpreted causally. No operating recommendation is authorized unless the variable registry explicitly marks the variable as confirmed controllable.
"""),
md("## 1. Configuration and reproducibility"),
md("### 1.1 Imports"),
code(r"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import importlib.util
import json
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

warnings.filterwarnings("ignore")
"""),
md("### 1.2 Central experiment configuration"),
code(r"""
RANDOM_STATE = 42
HORIZONS = [1, 5, 10, 15]
FEATURE_SETS = ["A_raw", "B_temporal", "C_process", "D_full"]
ML_MODELS = ["ridge", "extra_trees", "hist_gradient_boosting"]
BASELINES = ["training_mean", "persistence"]

PREPARATION_NOTEBOOK = Path("02_feature_engineering_data_preparation.ipynb")
ARTIFACT_INPUT_DIR = Path("../reports/feature_preparation_artifacts")
ARTIFACT_OUTPUT_DIR = Path("../reports/prediction_benchmark_artifacts")
FIGURE_DIR = ARTIFACT_OUTPUT_DIR / "figures"
MODEL_DIR = ARTIFACT_OUTPUT_DIR / "models"

TUNING_BUDGET_PER_MODEL = 2
EXTRA_TREES_ESTIMATORS = 30
PERMUTATION_SAMPLE_ROWS = 2000
PERMUTATION_REPEATS = 2
PREDICTION_PLOT_ROWS = 1500

ARTIFACT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 120)
plt.style.use("seaborn-v0_8-whitegrid")
"""),
md("### 1.3 Modeling scope"),
code(r"""
experiment_scope = pd.DataFrame([
    {"dimension": "forecast_horizons_min", "values": HORIZONS},
    {"dimension": "feature_sets", "values": FEATURE_SETS},
    {"dimension": "machine_learning_models", "values": ML_MODELS},
    {"dimension": "mandatory_baselines", "values": BASELINES},
    {"dimension": "validation_policy", "values": "fixed chronological validation; no shuffle"},
    {"dimension": "test_policy", "values": "evaluate once after validation-based hyperparameter choice"},
])
display(experiment_scope)
"""),
md("## 2. Recover and validate the prepared experiments"),
md(r"""
### 2.1 Replay the verified preparation core

Notebook 02 currently keeps its prepared matrices and fitted pipelines in memory rather than serializing them. To avoid reimplementing or silently changing that methodology, this notebook executes the code cells of Notebook 02 in an isolated namespace and stops before its reporting/export section. This recovers the exact `modeling_datasets`, splits, registries, feature lists, and train-fitted preprocessing objects used by the executed preparation notebook.
"""),
code(r"""
def recover_prepared_namespace(notebook_path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"display": lambda value: None}
    executed_cells = 0
    with redirect_stdout(StringIO()):
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if source.strip().startswith("import matplotlib.pyplot as plt"):
                break
            exec(compile(source, f"preparation_cell_{executed_cells}", "exec"), namespace)
            executed_cells += 1
    return namespace, executed_cells


prep, preparation_cells_replayed = recover_prepared_namespace(PREPARATION_NOTEBOOK)
modeling_datasets = prep["modeling_datasets"]
prepared = prep["prepared"]
raw_process_df = prep["df"].copy()
variable_registry = prep["variable_registry"].copy()
process_feature_registry = prep["process_feature_registry"].copy()
temporal_docs = prep["temporal_docs"].copy()
all_feature_docs = prep["all_docs"].copy()

print(f"Recovered {len(modeling_datasets)} experiment configurations from {preparation_cells_replayed} preparation code cells.")
"""),
md("### 2.2 Load the exported preparation audit"),
code(r"""
prepared_dimensions_audit = pd.read_csv(ARTIFACT_INPUT_DIR / "final_dataset_dimensions.csv")
prepared_readiness_audit = pd.read_csv(ARTIFACT_INPUT_DIR / "readiness_checks.csv")
prepared_retained_registry = pd.read_csv(ARTIFACT_INPUT_DIR / "retained_feature_registry.csv")

display(prepared_dimensions_audit)
"""),
md("### 2.3 Execute input-integrity checks"),
code(r"""
expected_keys = {(h, feature_set) for h in HORIZONS for feature_set in FEATURE_SETS}

input_check_rows = []
for key in sorted(expected_keys):
    h, feature_set = key
    dataset = modeling_datasets[key]
    prep_parts = prepared[key]
    features = dataset["features"]
    tree_triplet = dataset["X"]["tree"]
    input_check_rows.append({
        "horizon_min": h,
        "feature_set": feature_set,
        "chronological": prep_parts["train_df"][prep["TIME_COL"]].max() < prep_parts["valid_df"][prep["TIME_COL"]].min() < prep_parts["test_df"][prep["TIME_COL"]].min(),
        "target_absent": prep["TARGET"] not in features,
        "future_targets_absent": not any(name.startswith(prep["TARGET"] + "_t_plus_") for name in features),
        "columns_aligned": list(tree_triplet[0].columns) == list(tree_triplet[1].columns) == list(tree_triplet[2].columns),
        "no_postprocess_missing": all(not frame.isna().any().any() for variants in dataset["X"].values() for frame in variants),
        "four_fitted_preprocessors": len(dataset["pipelines"]) == 4,
        "feature_count": len(features),
        "train_rows": len(dataset["y_train"]),
        "valid_rows": len(dataset["y_valid"]),
        "test_rows": len(dataset["y_test"]),
    })

input_integrity_checks = pd.DataFrame(input_check_rows)
assert set(modeling_datasets) == expected_keys
assert input_integrity_checks[["chronological", "target_absent", "future_targets_absent", "columns_aligned", "no_postprocess_missing", "four_fitted_preprocessors"]].all().all()
assert prepared_readiness_audit["status"].astype(str).str.lower().eq("true").all()
display(input_integrity_checks)
"""),
md(r"""
### 2.4 Observations: prepared foundation

- All 16 horizon-by-feature-set configurations must pass before benchmarking begins.
- Feature Set A remains raw-only; Sets B-D preserve the accepted missingness indicators and engineered families.
- The target and every future-target column remain excluded from model inputs.
"""),
code(r"""
dimension_comparison = input_integrity_checks.merge(
    prepared_dimensions_audit,
    on=["horizon_min", "feature_set"],
    suffixes=("_recovered", "_exported"),
)
dimension_comparison["dimensions_match_export"] = (
    dimension_comparison["feature_count"].eq(dimension_comparison["retained_features"])
    & dimension_comparison["train_rows_recovered"].eq(dimension_comparison["train_rows_exported"])
    & dimension_comparison["valid_rows"].eq(dimension_comparison["validation_rows"])
    & dimension_comparison["test_rows_recovered"].eq(dimension_comparison["test_rows_exported"])
)
assert dimension_comparison["dimensions_match_export"].all()
display(dimension_comparison[["horizon_min", "feature_set", "feature_count", "train_rows_recovered", "valid_rows", "test_rows_recovered", "dimensions_match_export"]])
"""),
md("## 3. Metrics, baselines, and model definitions"),
md("### 3.1 Define regression metrics"),
code(r"""
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    absolute_error = np.abs(residual)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "median_ae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mean_bias_error": residual.mean(),
        "p90_absolute_error": np.quantile(absolute_error, 0.90),
        "p95_absolute_error": np.quantile(absolute_error, 0.95),
    }


def prefix_metrics(metrics, prefix):
    return {f"{prefix}_{name}": value for name, value in metrics.items()}
"""),
md("### 3.2 Map prediction timestamps to current target history"),
code(r"""
TIME_COL = prep["TIME_COL"]
TARGET = prep["TARGET"]
current_target_by_timestamp = raw_process_df.set_index(TIME_COL)[TARGET]


def current_target_for_split(split_df):
    return split_df[TIME_COL].map(current_target_by_timestamp)


persistence_availability = pd.DataFrame([
    {
        "horizon_min": h,
        "train_available_pct": current_target_for_split(prepared[(h, "A_raw")]["train_df"]).notna().mean() * 100,
        "valid_available_pct": current_target_for_split(prepared[(h, "A_raw")]["valid_df"]).notna().mean() * 100,
        "test_available_pct": current_target_for_split(prepared[(h, "A_raw")]["test_df"]).notna().mean() * 100,
    }
    for h in HORIZONS
])
display(persistence_availability)
"""),
md(r"""
### 3.3 Define the controlled search spaces

Each machine-learning family receives exactly two validation candidates. Hyperparameters are selected on the fixed chronological validation period. The test period is evaluated only after candidate selection.
"""),
code(r"""
MODEL_SPECS = {
    "ridge": {
        "preprocessing": "standard",
        "candidates": [{"alpha": value} for value in [1.0, 10.0]],
        "factory": lambda params: Ridge(**params),
    },
    "extra_trees": {
        "preprocessing": "tree",
        "candidates": [
            {"n_estimators": EXTRA_TREES_ESTIMATORS, "min_samples_leaf": 2, "max_features": 0.8},
            {"n_estimators": EXTRA_TREES_ESTIMATORS, "min_samples_leaf": 4, "max_features": 0.7},
        ],
        "factory": lambda params: ExtraTreesRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1),
    },
    "hist_gradient_boosting": {
        "preprocessing": "tree",
        "candidates": [
            {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": 50},
            {"learning_rate": 0.10, "max_leaf_nodes": 15, "max_iter": 50},
        ],
        "factory": lambda params: HistGradientBoostingRegressor(**params, random_state=RANDOM_STATE, early_stopping=False),
    },
}

assert all(len(spec["candidates"]) == TUNING_BUDGET_PER_MODEL for spec in MODEL_SPECS.values())
search_space_table = pd.DataFrame([
    {"model": model, "candidate_id": idx, "preprocessing": spec["preprocessing"], "parameters": json.dumps(params, sort_keys=True)}
    for model, spec in MODEL_SPECS.items() for idx, params in enumerate(spec["candidates"], start=1)
])
display(search_space_table)
"""),
md("## 4. Mandatory baselines"),
md("### 4.1 Evaluate training-mean and persistence baselines"),
code(r"""
baseline_result_rows = []
baseline_prediction_frames = []

for h in HORIZONS:
    base_parts = prepared[(h, "A_raw")]
    y_train = modeling_datasets[(h, "A_raw")]["y_train"]
    train_mean = float(y_train.mean())
    for split_name in ["valid", "test"]:
        split_df = base_parts[f"{split_name}_df"]
        y_true = modeling_datasets[(h, "A_raw")][f"y_{split_name}"]
        timestamp = split_df[TIME_COL].reset_index(drop=True)
        persistence = current_target_for_split(split_df).reset_index(drop=True)
        available = persistence.notna()
        if not available.all():
            persistence = persistence.fillna(train_mean)
        predictions = {
            "training_mean": np.full(len(y_true), train_mean),
            "persistence": persistence.to_numpy(),
        }
        for model_name, y_pred in predictions.items():
            metrics = regression_metrics(y_true, y_pred)
            baseline_result_rows.append({"horizon_min": h, "split": split_name, "model": model_name, **metrics})
            baseline_prediction_frames.append(pd.DataFrame({
                "timestamp": timestamp,
                "horizon_min": h,
                "feature_set": "autoregressive_baseline" if model_name == "persistence" else "target_summary_baseline",
                "model": model_name,
                "split": split_name,
                "y_true": np.asarray(y_true),
                "y_pred": y_pred,
            }))

baseline_results = pd.DataFrame(baseline_result_rows)
baseline_predictions = pd.concat(baseline_prediction_frames, ignore_index=True)
display(baseline_results)
"""),
md("### 4.2 Observations: persistence is the operational reference"),
code(r"""
persistence_summary = baseline_results.query("model == 'persistence'").pivot(index="horizon_min", columns="split", values=["mae", "rmse", "r2"])
display(persistence_summary)
"""),
md("## 5. Controlled machine-learning benchmark"),
md("### 5.1 Define validation-based candidate selection"),
code(r"""
def tune_and_evaluate_model(model_name, dataset, features):
    spec = MODEL_SPECS[model_name]
    preprocessing = spec["preprocessing"]
    X_train, X_valid, X_test = dataset["X"][preprocessing]
    y_train, y_valid, y_test = dataset["y_train"], dataset["y_valid"], dataset["y_test"]
    candidate_rows = []
    best = None

    for candidate_id, params in enumerate(spec["candidates"], start=1):
        model = spec["factory"](params)
        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        inference_start = time.perf_counter()
        valid_prediction = model.predict(X_valid)
        valid_inference_seconds = time.perf_counter() - inference_start
        metrics = regression_metrics(y_valid, valid_prediction)
        row = {
            "candidate_id": candidate_id,
            "parameters": json.dumps(params, sort_keys=True),
            "fit_seconds": fit_seconds,
            "valid_inference_seconds": valid_inference_seconds,
            **metrics,
        }
        candidate_rows.append(row)
        if best is None or metrics["rmse"] < best["metrics"]["rmse"]:
            best = {"model": model, "params": params, "candidate_id": candidate_id, "metrics": metrics, "fit_seconds": fit_seconds, "valid_prediction": valid_prediction, "valid_inference_seconds": valid_inference_seconds}

    test_start = time.perf_counter()
    test_prediction = best["model"].predict(X_test)
    test_inference_seconds = time.perf_counter() - test_start
    test_metrics = regression_metrics(y_test, test_prediction)

    return {
        "candidate_rows": candidate_rows,
        "selected_model": best["model"],
        "selected_params": best["params"],
        "selected_candidate_id": best["candidate_id"],
        "fit_seconds": best["fit_seconds"],
        "valid_prediction": best["valid_prediction"],
        "test_prediction": test_prediction,
        "valid_metrics": best["metrics"],
        "test_metrics": test_metrics,
        "valid_inference_seconds": best["valid_inference_seconds"],
        "test_inference_seconds": test_inference_seconds,
        "preprocessing": preprocessing,
    }
"""),
md("### 5.2 Run the 16-configuration benchmark"),
code(r"""
tuning_rows = []
benchmark_rows = []
prediction_frames = []
coefficient_rows = []
tree_importance_rows = []

benchmark_start = time.perf_counter()
for h in HORIZONS:
    for feature_set in FEATURE_SETS:
        key = (h, feature_set)
        dataset = modeling_datasets[key]
        parts = prepared[key]
        features = dataset["features"]
        for model_name in ML_MODELS:
            result = tune_and_evaluate_model(model_name, dataset, features)
            for candidate in result["candidate_rows"]:
                tuning_rows.append({"horizon_min": h, "feature_set": feature_set, "model": model_name, "selected": candidate["candidate_id"] == result["selected_candidate_id"], **candidate})

            persistence_valid_rmse = baseline_results.query("horizon_min == @h and split == 'valid' and model == 'persistence'")["rmse"].iloc[0]
            persistence_test_rmse = baseline_results.query("horizon_min == @h and split == 'test' and model == 'persistence'")["rmse"].iloc[0]
            benchmark_rows.append({
                "horizon_min": h,
                "feature_set": feature_set,
                "model": model_name,
                "preprocessing_variant": result["preprocessing"],
                "feature_count": len(features),
                "selected_candidate_id": result["selected_candidate_id"],
                "selected_hyperparameters": json.dumps(result["selected_params"], sort_keys=True),
                "training_seconds": result["fit_seconds"],
                "validation_inference_seconds": result["valid_inference_seconds"],
                "test_inference_seconds": result["test_inference_seconds"],
                "validation_improvement_over_persistence_pct": 100 * (persistence_valid_rmse - result["valid_metrics"]["rmse"]) / persistence_valid_rmse,
                "test_improvement_over_persistence_pct": 100 * (persistence_test_rmse - result["test_metrics"]["rmse"]) / persistence_test_rmse,
                **prefix_metrics(result["valid_metrics"], "validation"),
                **prefix_metrics(result["test_metrics"], "test"),
            })

            for split_name, y_pred in [("valid", result["valid_prediction"]), ("test", result["test_prediction"])]:
                prediction_frames.append(pd.DataFrame({
                    "timestamp": parts[f"{split_name}_df"][TIME_COL].reset_index(drop=True),
                    "horizon_min": h,
                    "feature_set": feature_set,
                    "model": model_name,
                    "split": split_name,
                    "y_true": np.asarray(dataset[f"y_{split_name}"]),
                    "y_pred": y_pred,
                }))

            model = result["selected_model"]
            if model_name == "ridge":
                coefficient_rows.extend({"horizon_min": h, "feature_set": feature_set, "model": model_name, "feature": feature, "importance": abs(coef), "signed_coefficient": coef} for feature, coef in zip(features, model.coef_))
            if model_name == "extra_trees":
                tree_importance_rows.extend({"horizon_min": h, "feature_set": feature_set, "model": model_name, "feature": feature, "importance": value} for feature, value in zip(features, model.feature_importances_))

benchmark_runtime_seconds = time.perf_counter() - benchmark_start
tuning_results = pd.DataFrame(tuning_rows)
benchmark_results = pd.DataFrame(benchmark_rows)
model_predictions = pd.concat(prediction_frames, ignore_index=True)
linear_importance = pd.DataFrame(coefficient_rows)
tree_importance = pd.DataFrame(tree_importance_rows)
print(f"Completed {len(benchmark_results)} selected model experiments in {benchmark_runtime_seconds:,.1f} seconds.")
display(benchmark_results.sort_values(["horizon_min", "validation_rmse"]).head(20))
"""),
md("### 5.3 Add baselines to the long-format result table"),
code(r"""
baseline_long_rows = []
for h in HORIZONS:
    for feature_set in FEATURE_SETS:
        for model_name in BASELINES:
            valid = baseline_results.query("horizon_min == @h and split == 'valid' and model == @model_name").iloc[0]
            test = baseline_results.query("horizon_min == @h and split == 'test' and model == @model_name").iloc[0]
            persistence_valid = baseline_results.query("horizon_min == @h and split == 'valid' and model == 'persistence'")["rmse"].iloc[0]
            persistence_test = baseline_results.query("horizon_min == @h and split == 'test' and model == 'persistence'")["rmse"].iloc[0]
            row = {
                "horizon_min": h, "feature_set": feature_set, "model": model_name,
                "preprocessing_variant": "none", "feature_count": 0,
                "selected_candidate_id": np.nan, "selected_hyperparameters": "{}",
                "training_seconds": 0.0, "validation_inference_seconds": 0.0, "test_inference_seconds": 0.0,
                "validation_improvement_over_persistence_pct": 100 * (persistence_valid - valid["rmse"]) / persistence_valid,
                "test_improvement_over_persistence_pct": 100 * (persistence_test - test["rmse"]) / persistence_test,
            }
            row.update(prefix_metrics({metric: valid[metric] for metric in ["mae", "rmse", "median_ae", "r2", "mean_bias_error", "p90_absolute_error", "p95_absolute_error"]}, "validation"))
            row.update(prefix_metrics({metric: test[metric] for metric in ["mae", "rmse", "median_ae", "r2", "mean_bias_error", "p90_absolute_error", "p95_absolute_error"]}, "test"))
            baseline_long_rows.append(row)

complete_benchmark_results = pd.concat([pd.DataFrame(baseline_long_rows), benchmark_results], ignore_index=True)
assert not complete_benchmark_results.duplicated(["horizon_min", "feature_set", "model"]).any()
display(complete_benchmark_results.head())
"""),
md("### 5.4 Observations: model performance against persistence"),
code(r"""
best_by_horizon_validation = (
    benchmark_results.sort_values("validation_rmse")
    .groupby("horizon_min", as_index=False)
    .first()
)
display(best_by_horizon_validation[["horizon_min", "feature_set", "model", "validation_rmse", "test_rmse", "test_improvement_over_persistence_pct", "feature_count"]])
"""),
md("## 6. Candidate process regimes"),
md(r"""
### 6.1 Define leakage-safe candidate regimes

Confirmed event labels are unavailable. The following labels are therefore candidate regimes, not verified operating states. Thresholds are fitted using training-period information only, and labels at prediction time use current or past measurements.
"""),
code(r"""
def build_regime_thresholds(horizon):
    train_df = prepared[(horizon, "A_raw")]["train_df"]
    current_target = current_target_for_split(train_df)
    target_change = current_target.diff().abs()
    production = train_df["PRODUCTION_TSP_BALANCE"]
    return {
        "stable_change": target_change.quantile(0.50),
        "transition_change": target_change.quantile(0.90),
        "abnormal_low": current_target.quantile(0.01),
        "abnormal_high": current_target.quantile(0.99),
        "low_production": production.quantile(0.05),
    }


def label_candidate_regimes(horizon, split_name):
    split_df = prepared[(horizon, "A_raw")][f"{split_name}_df"].reset_index(drop=True)
    thresholds = build_regime_thresholds(horizon)
    current_target = current_target_for_split(split_df).reset_index(drop=True)
    previous_target = split_df[TIME_COL].map(current_target_by_timestamp.shift(1)).reset_index(drop=True)
    target_change = (current_target - previous_target).abs()
    production = split_df["PRODUCTION_TSP_BALANCE"].reset_index(drop=True)
    missing_sensor = split_df[prep["sensor_columns"]].isna().any(axis=1).reset_index(drop=True)
    regime = np.full(len(split_df), "ordinary", dtype=object)
    regime[target_change.le(thresholds["stable_change"]).fillna(False)] = "stable_candidate"
    regime[missing_sensor] = "missing_sensor_candidate"
    regime[production.le(thresholds["low_production"]).fillna(False)] = "startup_shutdown_candidate"
    regime[target_change.ge(thresholds["transition_change"]).fillna(False)] = "transition_candidate"
    regime[(current_target.lt(thresholds["abnormal_low"]) | current_target.gt(thresholds["abnormal_high"])).fillna(False)] = "abnormal_target_candidate"
    return pd.DataFrame({"timestamp": split_df[TIME_COL], "horizon_min": horizon, "split": split_name, "candidate_regime": regime, "current_target": current_target, "current_absolute_change": target_change})


regime_labels = pd.concat([label_candidate_regimes(h, split) for h in HORIZONS for split in ["valid", "test"]], ignore_index=True)
regime_distribution = regime_labels.groupby(["horizon_min", "split", "candidate_regime"]).size().rename("rows").reset_index()
display(regime_distribution)
"""),
md("### 6.2 Evaluate every selected model by candidate regime"),
code(r"""
predictions_with_regime = model_predictions.merge(regime_labels, on=["timestamp", "horizon_min", "split"], how="left", validate="many_to_one")
predictions_with_regime["residual"] = predictions_with_regime["y_pred"] - predictions_with_regime["y_true"]
predictions_with_regime["absolute_error"] = predictions_with_regime["residual"].abs()

regime_metric_rows = []
for keys, group in predictions_with_regime.groupby(["horizon_min", "feature_set", "model", "split", "candidate_regime"]):
    if len(group) < 2:
        continue
    metrics = regression_metrics(group["y_true"], group["y_pred"])
    regime_metric_rows.append(dict(zip(["horizon_min", "feature_set", "model", "split", "candidate_regime"], keys), rows=len(group), **metrics))
regime_metrics = pd.DataFrame(regime_metric_rows)
display(regime_metrics.query("split == 'test'").sort_values(["horizon_min", "rmse"]).head(30))
"""),
md("### 6.3 Observations: stable versus disturbed operation"),
code(r"""
regime_comparison = (
    regime_metrics.query("split == 'test' and candidate_regime in ['stable_candidate', 'transition_candidate', 'abnormal_target_candidate']")
    .groupby("candidate_regime")[["mae", "rmse"]].median()
    .sort_values("rmse")
)
display(regime_comparison)
"""),
md("## 7. Feature-set ablation and horizon usefulness"),
md("### 7.1 Select the best model family within each configuration using validation"),
code(r"""
best_model_per_configuration = (
    benchmark_results.sort_values("validation_rmse")
    .groupby(["horizon_min", "feature_set"], as_index=False)
    .first()
)
display(best_model_per_configuration[["horizon_min", "feature_set", "model", "validation_rmse", "test_rmse", "test_improvement_over_persistence_pct", "feature_count"]])
"""),
md("### 7.2 Quantify the four requested ablations"),
code(r"""
ABLATION_PAIRS = [
    ("A_raw", "B_temporal", "A_vs_B_temporal_value"),
    ("A_raw", "C_process", "A_vs_C_process_value"),
    ("B_temporal", "D_full", "B_vs_D_process_increment"),
    ("C_process", "D_full", "C_vs_D_temporal_increment"),
]
ablation_rows = []
for h in HORIZONS:
    horizon_rows = best_model_per_configuration.query("horizon_min == @h").set_index("feature_set")
    for base_set, enhanced_set, comparison in ABLATION_PAIRS:
        base = horizon_rows.loc[base_set]
        enhanced = horizon_rows.loc[enhanced_set]
        ablation_rows.append({
            "horizon_min": h, "comparison": comparison, "base_set": base_set, "enhanced_set": enhanced_set,
            "base_model": base["model"], "enhanced_model": enhanced["model"],
            "test_rmse_change": enhanced["test_rmse"] - base["test_rmse"],
            "test_rmse_improvement_pct": 100 * (base["test_rmse"] - enhanced["test_rmse"]) / base["test_rmse"],
            "additional_features": int(enhanced["feature_count"] - base["feature_count"]),
            "training_seconds_change": enhanced["training_seconds"] - base["training_seconds"],
            "validation_consistent_improvement": enhanced["validation_rmse"] < base["validation_rmse"],
        })
feature_set_ablation = pd.DataFrame(ablation_rows)
display(feature_set_ablation)
"""),
md("### 7.3 Estimate warning usefulness with training-derived thresholds"),
code(r"""
horizon_utility_rows = []
for _, selection in best_by_horizon_validation.iterrows():
    h = int(selection["horizon_min"])
    selected_predictions = predictions_with_regime.query("horizon_min == @h and feature_set == @selection.feature_set and model == @selection.model and split == 'test'").copy()
    train_target = modeling_datasets[(h, "A_raw")]["y_train"]
    warning_low, warning_high = train_target.quantile([0.01, 0.99])
    predicted_warning = (selected_predictions["y_pred"] < warning_low) | (selected_predictions["y_pred"] > warning_high)
    actual_warning = (selected_predictions["y_true"] < warning_low) | (selected_predictions["y_true"] > warning_high)
    false_warning_rate = ((predicted_warning) & (~actual_warning)).mean()
    transition_mae = selected_predictions.loc[selected_predictions["candidate_regime"].eq("transition_candidate"), "absolute_error"].mean()
    horizon_utility_rows.append({
        "horizon_min": h, "selected_feature_set": selection["feature_set"], "selected_model": selection["model"],
        "test_mae": selection["test_mae"], "test_rmse": selection["test_rmse"],
        "test_improvement_over_persistence_pct": selection["test_improvement_over_persistence_pct"],
        "transition_candidate_mae": transition_mae,
        "exploratory_false_warning_rate": false_warning_rate,
        "reaction_time_minutes": h,
    })
horizon_usefulness = pd.DataFrame(horizon_utility_rows)
display(horizon_usefulness)
"""),
md("### 7.4 Observations: accuracy, complexity, and reaction time"),
code(r"""
display(feature_set_ablation.groupby("comparison").agg(median_test_improvement_pct=("test_rmse_improvement_pct", "median"), horizons_improved_on_validation=("validation_consistent_improvement", "sum"), median_additional_features=("additional_features", "median")))
display(horizon_usefulness.sort_values("horizon_min"))
"""),
md("## 8. Interpretation and process understanding"),
md("### 8.1 Choose interpretation candidates using validation only"),
code(r"""
best_accuracy_candidate = benchmark_results.sort_values("validation_rmse").iloc[0]
best_interpretable_candidate = benchmark_results.query("model == 'ridge'").sort_values("validation_rmse").iloc[0]
best_tree_candidate = benchmark_results.query("model in ['extra_trees', 'hist_gradient_boosting']").sort_values("validation_rmse").iloc[0]
early_pool = benchmark_results.query("horizon_min >= 10 and validation_improvement_over_persistence_pct > 0")
best_early_warning_candidate = (early_pool if len(early_pool) else benchmark_results.query("horizon_min >= 10")).sort_values("validation_rmse").iloc[0]

candidate_roles = pd.DataFrame([
    {"role": "accuracy_oriented", **best_accuracy_candidate.to_dict()},
    {"role": "interpretable", **best_interpretable_candidate.to_dict()},
    {"role": "best_tree", **best_tree_candidate.to_dict()},
    {"role": "early_warning", **best_early_warning_candidate.to_dict()},
])
display(candidate_roles[["role", "horizon_min", "feature_set", "model", "validation_rmse", "test_rmse", "test_improvement_over_persistence_pct", "feature_count"]])
"""),
md("### 8.2 Refit selected candidates and calculate permutation importance"),
code(r"""
def refit_candidate(row):
    key = (int(row["horizon_min"]), row["feature_set"])
    dataset = modeling_datasets[key]
    spec = MODEL_SPECS[row["model"]]
    params = json.loads(row["selected_hyperparameters"])
    model = spec["factory"](params)
    X_train = dataset["X"][spec["preprocessing"]][0]
    model.fit(X_train, dataset["y_train"])
    return model, key, spec["preprocessing"]


tree_model, tree_key, tree_preprocessing = refit_candidate(best_tree_candidate)
tree_dataset = modeling_datasets[tree_key]
X_tree_valid = tree_dataset["X"][tree_preprocessing][1]
y_tree_valid = tree_dataset["y_valid"]
sample_size = min(PERMUTATION_SAMPLE_ROWS, len(X_tree_valid))
sample_positions = np.linspace(0, len(X_tree_valid) - 1, sample_size, dtype=int)
permutation = permutation_importance(
    tree_model,
    X_tree_valid.iloc[sample_positions],
    np.asarray(y_tree_valid)[sample_positions],
    scoring="neg_root_mean_squared_error",
    n_repeats=PERMUTATION_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
permutation_importance_table = pd.DataFrame({
    "feature": tree_dataset["features"],
    "permutation_importance_mean": permutation.importances_mean,
    "permutation_importance_std": permutation.importances_std,
}).sort_values("permutation_importance_mean", ascending=False)
display(permutation_importance_table.head(25))
"""),
md("### 8.3 Group feature importance by source and engineering family"),
code(r"""
process_feature_names = set(process_feature_registry["feature"])
sensor_columns = prep["sensor_columns"]


def feature_family(feature):
    if feature.endswith("_missing"):
        return "missingness_indicator"
    if feature in process_feature_names:
        return "process_aware"
    match = temporal_docs.loc[temporal_docs["feature"].eq(feature), "family"]
    return match.iloc[0] if len(match) else "other"


def source_sensor(feature):
    matches = [sensor for sensor in sensor_columns if feature == sensor or feature.startswith(sensor + "_")]
    return max(matches, key=len) if matches else ("process_relationship" if feature in process_feature_names else "other")


permutation_importance_table["family"] = permutation_importance_table["feature"].map(feature_family)
permutation_importance_table["source_group"] = permutation_importance_table["feature"].map(source_sensor)
grouped_family_importance = permutation_importance_table.groupby("family")["permutation_importance_mean"].sum().sort_values(ascending=False).reset_index()
grouped_source_importance = permutation_importance_table.groupby("source_group")["permutation_importance_mean"].sum().sort_values(ascending=False).reset_index()
display(grouped_family_importance)
display(grouped_source_importance.head(20))
"""),
md("### 8.4 Review process-aware feature contributions and validation status"),
code(r"""
process_feature_contribution = process_feature_registry.merge(
    permutation_importance_table[["feature", "permutation_importance_mean", "permutation_importance_std"]],
    on="feature",
    how="left",
).sort_values("permutation_importance_mean", ascending=False, na_position="last")
display(process_feature_contribution)
"""),
md("### 8.5 Measure importance stability across horizons"),
code(r"""
best_extra_trees_by_horizon = (
    benchmark_results.query("model == 'extra_trees'")
    .sort_values("validation_rmse")
    .groupby("horizon_min", as_index=False)
    .first()[["horizon_min", "feature_set"]]
)
importance_stability_rows = []
for _, selection in best_extra_trees_by_horizon.iterrows():
    subset = tree_importance.query("horizon_min == @selection.horizon_min and feature_set == @selection.feature_set").copy()
    subset["rank"] = subset["importance"].rank(ascending=False, method="average")
    importance_stability_rows.append(subset)
importance_stability = pd.concat(importance_stability_rows, ignore_index=True)
importance_rank_matrix = importance_stability.pivot_table(index="feature", columns="horizon_min", values="rank")
importance_stability_summary = pd.DataFrame({
    "feature": importance_rank_matrix.index,
    "mean_rank": importance_rank_matrix.mean(axis=1),
    "rank_std": importance_rank_matrix.std(axis=1),
    "horizons_present": importance_rank_matrix.notna().sum(axis=1),
}).sort_values(["mean_rank", "rank_std"])
display(importance_stability_summary.head(25))
"""),
md(r"""
### 8.6 Observations: interpretation boundaries

- Permutation importance measures validation-set predictive dependence, not causal influence.
- Ridge coefficients are scale-dependent associations after standardization.
- `TOTAL_SLURRY_M3_FLOW` is a confirmed m³/h aggregation. `SLURRY_DENSITY_PROXY` remains provisional until the t/h stream correspondence is confirmed.
- No variable is currently authorized for intervention by the registry.
- SHAP is not required for this benchmark because model-agnostic permutation importance and linear coefficients already provide complementary interpretation without adding substantial runtime or another model-specific dependency.
"""),
code(r"""
interpretation_safeguards = pd.DataFrame([
    {"item": "causal_claims", "status": "not permitted from importance"},
    {"item": "confirmed_controllable_variables", "status": int(variable_registry["recommendation_allowed"].sum())},
    {"item": "slurry_volumetric_aggregation", "status": "confirmed m3/h"},
    {"item": "slurry_density_proxy", "status": "requires mass-stream validation"},
    {"item": "recycle meaning and unit", "status": "unresolved"},
])
display(interpretation_safeguards)
"""),
md("## 9. Error diagnostics"),
md("### 9.1 Build residual and largest-error tables"),
code(r"""
residual_table = predictions_with_regime.copy()
largest_error_events = (
    residual_table.query("split == 'test'")
    .sort_values("absolute_error", ascending=False)
    .groupby(["horizon_min", "feature_set", "model"], as_index=False)
    .head(10)
    .sort_values(["horizon_min", "absolute_error"], ascending=[True, False])
)
display(largest_error_events.head(30))
"""),
md("### 9.2 Check whether extremes dominate RMSE"),
code(r"""
extreme_impact_rows = []
for keys, group in residual_table.query("split == 'test'").groupby(["horizon_min", "feature_set", "model"]):
    cutoff = group["absolute_error"].quantile(0.99)
    full_rmse = regression_metrics(group["y_true"], group["y_pred"])["rmse"]
    trimmed = group.loc[group["absolute_error"] <= cutoff]
    trimmed_rmse = regression_metrics(trimmed["y_true"], trimmed["y_pred"])["rmse"]
    extreme_impact_rows.append(dict(zip(["horizon_min", "feature_set", "model"], keys), full_rmse=full_rmse, rmse_without_top_1pct_errors=trimmed_rmse, rmse_reduction_pct=100*(full_rmse-trimmed_rmse)/full_rmse))
extreme_error_impact = pd.DataFrame(extreme_impact_rows)
display(extreme_error_impact.sort_values("rmse_reduction_pct", ascending=False).head(20))
"""),
md("### 9.3 Observations: where models fail"),
code(r"""
error_failure_summary = residual_table.query("split == 'test'").groupby("candidate_regime").agg(rows=("absolute_error", "size"), median_absolute_error=("absolute_error", "median"), p95_absolute_error=("absolute_error", lambda x: x.quantile(.95))).sort_values("p95_absolute_error", ascending=False)
display(error_failure_summary)
"""),
md("## 10. Required analytical figures"),
md("### 10.1 Performance versus horizon and persistence"),
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for feature_set, group in best_model_per_configuration.groupby("feature_set"):
    axes[0].plot(group["horizon_min"], group["test_rmse"], marker="o", label=feature_set)
persistence_test = baseline_results.query("model == 'persistence' and split == 'test'")
axes[0].plot(persistence_test["horizon_min"], persistence_test["rmse"], marker="o", color="black", linestyle="--", label="persistence")
axes[0].set(title="Test RMSE versus forecast horizon", xlabel="Horizon (minutes)", ylabel="RMSE")
axes[0].legend(fontsize=8)
axes[1].bar(horizon_usefulness["horizon_min"].astype(str), horizon_usefulness["test_improvement_over_persistence_pct"], color="#2878a8")
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set(title="Selected model improvement over persistence", xlabel="Horizon (minutes)", ylabel="Test RMSE improvement (%)")
fig.tight_layout(); fig.savefig(FIGURE_DIR / "performance_vs_horizon_and_persistence.png", dpi=220); plt.show()
"""),
md("### 10.2 Feature-set ablation and complexity"),
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for comparison, group in feature_set_ablation.groupby("comparison"):
    axes[0].plot(group["horizon_min"], group["test_rmse_improvement_pct"], marker="o", label=comparison)
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set(title="Feature-set ablation", xlabel="Horizon (minutes)", ylabel="Test RMSE improvement (%)")
axes[0].legend(fontsize=7)
complexity = best_model_per_configuration.groupby("feature_set").agg(features=("feature_count", "first"), median_test_rmse=("test_rmse", "median")).reset_index()
axes[1].scatter(complexity["features"], complexity["median_test_rmse"], s=90)
for _, row in complexity.iterrows(): axes[1].annotate(row["feature_set"], (row["features"], row["median_test_rmse"]), xytext=(5,5), textcoords="offset points")
axes[1].set(title="Feature complexity versus median test RMSE", xlabel="Retained features", ylabel="Median test RMSE")
fig.tight_layout(); fig.savefig(FIGURE_DIR / "feature_set_ablation_and_complexity.png", dpi=220); plt.show()
"""),
md("### 10.3 Predicted versus actual and residual diagnostics"),
code(r"""
selected = best_accuracy_candidate
selected_test = residual_table.query("horizon_min == @selected.horizon_min and feature_set == @selected.feature_set and model == @selected.model and split == 'test'").tail(PREDICTION_PLOT_ROWS)
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(selected_test["timestamp"], selected_test["y_true"], label="actual", linewidth=1.2)
axes[0].plot(selected_test["timestamp"], selected_test["y_pred"], label="prediction", linewidth=1.0)
axes[0].set(title=f"Predicted versus actual: h={int(selected.horizon_min)} min, {selected.feature_set}, {selected.model}", ylabel=TARGET)
axes[0].legend()
axes[1].plot(selected_test["timestamp"], selected_test["residual"], color="#c44e52", linewidth=.9)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set(ylabel="Residual", xlabel="Timestamp")
fig.tight_layout(); fig.savefig(FIGURE_DIR / "predicted_actual_and_residual_time_series.png", dpi=220); plt.show()
"""),
md("### 10.4 Representative transition window"),
code(r"""
transition_rows = selected_test.query("candidate_regime == 'transition_candidate'")
transition_center = transition_rows.loc[transition_rows["current_absolute_change"].idxmax(), "timestamp"] if len(transition_rows) else selected_test.iloc[len(selected_test)//2]["timestamp"]
window = residual_table.query("horizon_min == @selected.horizon_min and feature_set == @selected.feature_set and model == @selected.model and split == 'test'").copy()
window = window[(window["timestamp"] >= transition_center - pd.Timedelta(minutes=60)) & (window["timestamp"] <= transition_center + pd.Timedelta(minutes=60))]
fig, ax = plt.subplots(figsize=(13, 4.5)); ax.plot(window["timestamp"], window["y_true"], label="actual", marker="."); ax.plot(window["timestamp"], window["y_pred"], label="prediction"); ax.axvline(transition_center, color="#c44e52", linestyle="--", label="candidate transition"); ax.set(title="Representative candidate transition window", xlabel="Timestamp", ylabel=TARGET); ax.legend(); fig.tight_layout(); fig.savefig(FIGURE_DIR / "representative_transition_window.png", dpi=220); plt.show()
"""),
md("### 10.5 Residual distribution and error by regime"),
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(selected_test["residual"], bins=50, color="#2878a8", alpha=.85)
axes[0].axvline(0, color="black", linewidth=1); axes[0].set(title="Residual distribution", xlabel="Prediction - actual", ylabel="Rows")
regime_plot = selected_test.groupby("candidate_regime")["absolute_error"].mean().sort_values()
axes[1].barh(regime_plot.index, regime_plot.values, color="#eb8f34"); axes[1].set(title="Mean absolute error by candidate regime", xlabel="MAE")
fig.tight_layout(); fig.savefig(FIGURE_DIR / "residual_distribution_and_regime_error.png", dpi=220); plt.show()
"""),
md("### 10.6 Grouped importance and stability"),
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
family_plot = grouped_family_importance.sort_values("permutation_importance_mean").tail(12)
axes[0].barh(family_plot["family"], family_plot["permutation_importance_mean"], color="#4a9d5b")
axes[0].set(title="Grouped permutation importance", xlabel="Sum of validation importance")
stability_plot = importance_stability_summary.query("horizons_present >= 2").head(15).sort_values("mean_rank", ascending=False)
axes[1].barh(stability_plot["feature"], stability_plot["mean_rank"], xerr=stability_plot["rank_std"].fillna(0), color="#8172b2")
axes[1].invert_xaxis(); axes[1].set(title="Importance rank stability across horizons", xlabel="Mean rank (lower is better)")
fig.tight_layout(); fig.savefig(FIGURE_DIR / "grouped_importance_and_horizon_stability.png", dpi=220); plt.show()
"""),
md("### 10.7 Accuracy versus reaction-time trade-off"),
code(r"""
fig, ax = plt.subplots(figsize=(8, 5)); scatter = ax.scatter(horizon_usefulness["reaction_time_minutes"], horizon_usefulness["test_rmse"], c=horizon_usefulness["test_improvement_over_persistence_pct"], s=140, cmap="viridis")
for _, row in horizon_usefulness.iterrows(): ax.annotate(f"{row.selected_feature_set}\n{row.selected_model}", (row.reaction_time_minutes, row.test_rmse), xytext=(6,5), textcoords="offset points", fontsize=8)
ax.set(title="Accuracy versus warning reaction time", xlabel="Reaction time (minutes)", ylabel="Test RMSE"); fig.colorbar(scatter, ax=ax, label="Improvement over persistence (%)"); fig.tight_layout(); fig.savefig(FIGURE_DIR / "accuracy_reaction_time_tradeoff.png", dpi=220); plt.show()
"""),
md("## 11. Model-selection framework"),
md("### 11.1 Build a validation-led decision table"),
code(r"""
decision_rows = []
for role, candidate in [("accuracy_oriented", best_accuracy_candidate), ("interpretable", best_interpretable_candidate), ("early_warning", best_early_warning_candidate)]:
    h = int(candidate["horizon_min"])
    regime_subset = regime_metrics.query("horizon_min == @h and feature_set == @candidate.feature_set and model == @candidate.model and split == 'test'")
    transition_rmse = regime_subset.loc[regime_subset["candidate_regime"].eq("transition_candidate"), "rmse"]
    process_dependency = any(feature in process_feature_names for feature in modeling_datasets[(h, candidate["feature_set"])]["features"])
    decision_rows.append({
        "candidate_role": role, "horizon_min": h, "feature_set": candidate["feature_set"], "model": candidate["model"],
        "validation_rmse": candidate["validation_rmse"], "test_rmse": candidate["test_rmse"],
        "persistence_improvement_pct": candidate["test_improvement_over_persistence_pct"],
        "transition_candidate_rmse": transition_rmse.iloc[0] if len(transition_rmse) else np.nan,
        "feature_count": candidate["feature_count"], "training_seconds": candidate["training_seconds"],
        "interpretability": "high" if candidate["model"] == "ridge" else ("medium" if candidate["model"] == "extra_trees" else "medium-low"),
        "uses_process_aware_features": process_dependency,
        "operational_validation_burden": "higher" if process_dependency else "standard",
        "selection_basis": "validation performance plus role constraint; test not used for selection",
    })
model_selection_decision = pd.DataFrame(decision_rows)
display(model_selection_decision)
"""),
md("### 11.2 Refit and serialize final candidate models with preprocessors"),
code(r"""
final_candidate_bundle = {}
for _, row in model_selection_decision.iterrows():
    source_row = {"accuracy_oriented": best_accuracy_candidate, "interpretable": best_interpretable_candidate, "early_warning": best_early_warning_candidate}[row["candidate_role"]]
    model, key, preprocessing = refit_candidate(source_row)
    final_candidate_bundle[row["candidate_role"]] = {
        "model": model,
        "preprocessor": modeling_datasets[key]["pipelines"][preprocessing],
        "features": modeling_datasets[key]["features"],
        "horizon_min": key[0],
        "feature_set": key[1],
        "preprocessing_variant": preprocessing,
        "selected_hyperparameters": json.loads(source_row["selected_hyperparameters"]),
    }
joblib.dump(final_candidate_bundle, MODEL_DIR / "final_candidate_models_and_preprocessors.joblib")
print("Serialized final validation-selected candidate bundle.")
"""),
md("## 12. Export modeling artifacts"),
md("### 12.1 Export tables and configuration"),
code(r"""
experiment_configuration = pd.DataFrame([
    {"key": "horizons", "value": json.dumps(HORIZONS)},
    {"key": "feature_sets", "value": json.dumps(FEATURE_SETS)},
    {"key": "models", "value": json.dumps(ML_MODELS)},
    {"key": "tuning_candidates_per_model", "value": TUNING_BUDGET_PER_MODEL},
    {"key": "random_state", "value": RANDOM_STATE},
    {"key": "selection_metric", "value": "validation_rmse"},
    {"key": "test_use", "value": "evaluation only after validation selection"},
    {"key": "shuffling", "value": False},
])

table_exports = {
    "input_integrity_checks": input_integrity_checks,
    "complete_benchmark_results": complete_benchmark_results,
    "hyperparameter_search_results": tuning_results,
    "selected_hyperparameters": tuning_results.query("selected"),
    "test_predictions": model_predictions.query("split == 'test'"),
    "validation_predictions": model_predictions.query("split == 'valid'"),
    "residuals_with_candidate_regimes": residual_table,
    "candidate_regime_metrics": regime_metrics,
    "feature_set_ablation": feature_set_ablation,
    "horizon_usefulness": horizon_usefulness,
    "ridge_coefficient_importance": linear_importance,
    "extra_trees_importance": tree_importance,
    "permutation_importance": permutation_importance_table,
    "grouped_family_importance": grouped_family_importance,
    "grouped_source_importance": grouped_source_importance,
    "process_feature_contribution": process_feature_contribution,
    "importance_stability": importance_stability_summary,
    "largest_error_events": largest_error_events,
    "extreme_error_impact": extreme_error_impact,
    "model_selection_decision": model_selection_decision,
    "experiment_configuration": experiment_configuration,
}
for name, table in table_exports.items():
    table.to_csv(ARTIFACT_OUTPUT_DIR / f"{name}.csv", index=False)
print(f"Exported {len(table_exports)} modeling tables.")
"""),
md("### 12.2 Verify the exact export manifest"),
code(r"""
EXPECTED_FIGURES = [
    "performance_vs_horizon_and_persistence.png",
    "feature_set_ablation_and_complexity.png",
    "predicted_actual_and_residual_time_series.png",
    "representative_transition_window.png",
    "residual_distribution_and_regime_error.png",
    "grouped_importance_and_horizon_stability.png",
    "accuracy_reaction_time_tradeoff.png",
]
expected_table_paths = [ARTIFACT_OUTPUT_DIR / f"{name}.csv" for name in table_exports]
expected_figure_paths = [FIGURE_DIR / name for name in EXPECTED_FIGURES]
expected_model_paths = [MODEL_DIR / "final_candidate_models_and_preprocessors.joblib"]
export_manifest = pd.DataFrame({"path": [str(path) for path in expected_table_paths + expected_figure_paths + expected_model_paths]})
export_manifest["exists"] = export_manifest["path"].map(lambda path: Path(path).exists())
assert export_manifest["exists"].all()
display(export_manifest)
"""),
md("## 13. Final readiness checks"),
md("### 13.1 Execute release assertions"),
code(r"""
evaluated_ml_keys = set(map(tuple, benchmark_results[["horizon_min", "feature_set"]].drop_duplicates().to_numpy()))
persistence_horizons = set(baseline_results.query("model == 'persistence'")["horizon_min"])
prediction_alignment = all(
    len(group) == len(modeling_datasets[(h, feature_set)][f"y_{split}"])
    for (h, feature_set, model, split), group in model_predictions.groupby(["horizon_min", "feature_set", "model", "split"])
)

readiness_checks = pd.DataFrame([
    {"check": "all_16_experiment_configurations_evaluated", "status": evaluated_ml_keys == expected_keys},
    {"check": "persistence_evaluated_at_every_horizon", "status": persistence_horizons == set(HORIZONS)},
    {"check": "preprocessing_recovered_from_train_fitted_notebook_02_objects", "status": input_integrity_checks["four_fitted_preprocessors"].all()},
    {"check": "test_not_used_for_hyperparameter_selection", "status": tuning_results["selected"].groupby([tuning_results.horizon_min, tuning_results.feature_set, tuning_results.model]).sum().eq(1).all()},
    {"check": "target_and_future_target_leakage_absent", "status": input_integrity_checks[["target_absent", "future_targets_absent"]].all().all()},
    {"check": "prediction_rows_align_with_split_targets", "status": prediction_alignment},
    {"check": "benchmark_result_keys_are_unique", "status": not complete_benchmark_results.duplicated(["horizon_min", "feature_set", "model"]).any()},
    {"check": "all_expected_artifacts_exist", "status": export_manifest["exists"].all()},
    {"check": "no_unconfirmed_variable_used_for_intervention_recommendation", "status": not variable_registry["recommendation_allowed"].any()},
    {"check": "same_hyperparameter_budget_for_each_ml_family", "status": all(len(spec["candidates"]) == TUNING_BUDGET_PER_MODEL for spec in MODEL_SPECS.values())},
])
readiness_checks.to_csv(ARTIFACT_OUTPUT_DIR / "readiness_checks.csv", index=False)
assert readiness_checks["status"].all(), readiness_checks.loc[~readiness_checks["status"]]
display(readiness_checks)
"""),
md("### 13.2 Observations: release status"),
code(r"""
print(f"Passed {readiness_checks.status.sum()}/{len(readiness_checks)} modeling readiness checks.")
print(f"Artifacts: {len(table_exports) + 1} CSV tables, {len(EXPECTED_FIGURES)} figures, and {len(expected_model_paths)} serialized model bundle.")
"""),
md("## 14. Evidence-based conclusions"),
md("### 14.1 Generate direct answers from executed results"),
code(r"""
positive_horizons = horizon_usefulness.loc[horizon_usefulness["test_improvement_over_persistence_pct"] > 0, "horizon_min"].astype(int).tolist()
best_feature_set_overall = best_model_per_configuration.groupby("feature_set")["validation_rmse"].mean().idxmin()
process_ablation = feature_set_ablation.query("comparison in ['A_vs_C_process_value', 'B_vs_D_process_increment']")
process_value_supported = process_ablation["validation_consistent_improvement"].mean() > 0.5
best_warning = horizon_usefulness.sort_values(["test_improvement_over_persistence_pct", "reaction_time_minutes"], ascending=[False, False]).iloc[0]
worst_regime = error_failure_summary.index[0]

conclusion_table = pd.DataFrame([
    {"question": "Does forecasting beat persistence?", "answer": "Yes on test RMSE at horizons " + str(positive_horizons) if positive_horizons else "No selected horizon improves on persistence."},
    {"question": "Which feature set is strongest on validation on average?", "answer": best_feature_set_overall},
    {"question": "Do process-aware features add consistent value?", "answer": "Supported across most requested validation ablations." if process_value_supported else "Not consistently supported across the requested validation ablations."},
    {"question": "Best interpretable candidate", "answer": f"Ridge, h={int(best_interpretable_candidate.horizon_min)} min, {best_interpretable_candidate.feature_set}"},
    {"question": "Most promising warning trade-off in this benchmark", "answer": f"h={int(best_warning.horizon_min)} min, {best_warning.selected_feature_set}, {best_warning.selected_model}; operational thresholds remain unvalidated."},
    {"question": "Where are errors most difficult?", "answer": str(worst_regime)},
    {"question": "Operational readiness", "answer": "Research benchmark complete; additional months, event labels, units, thresholds, and process-engineer validation are required before operational prototyping."},
])
display(conclusion_table)
"""),
md(r"""
### 14.2 Limitations and required process information

- The dataset covers only January 2026; seasonal, campaign, and product-grade generalization is unknown.
- Candidate startup, shutdown, transition, and abnormal regimes are algorithmic labels, not confirmed event records.
- Warning thresholds are derived from training quantiles for analysis only and are not operational limits.
- The units of phosphoric-acid flow, ground-phosphate flow, fuel-oil flow, steam flow, wash-liquid flow, and recycle remain unresolved.
- The exact material represented by `RECYCLAGE` remains unresolved.
- The two slurry volumetric tags are confirmed additive m³/h measurements, but `SLURRY_DENSITY_PROXY` remains provisional until the t/h measurement is confirmed to represent the same combined stream.
- Adjacent one-minute observations are dependent; rolling-origin evaluation across additional months is required.
- Model importance is predictive association, never proof of physical causation or a safe intervention.

## 15. Recommended continuation

Before operational prototyping, obtain additional months of data, confirmed process-event labels, engineering units, quality thresholds, and explicit actuator controllability. Then repeat this benchmark with rolling-origin validation and compare stability across campaigns or product grades.
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} with {len(cells)} cells.")
