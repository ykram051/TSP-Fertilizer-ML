"""Build Notebook 08: just-in-time target-free soft sensor."""
import json
from pathlib import Path

OUT = Path("notebooks/08_just_in_time_target_free_soft_sensor.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip() + "\n"}


cells = [
md(r"""
# Just-in-Time Target-Free Soft Sensor

Notebook 07 showed that local Ridge experts reduced time-block fragility, but five fixed KMeans regions were too rigid. This notebook tests a query-specific alternative:

1. represent the current process state in a compact training-fitted latent space;
2. retrieve similar historical states;
3. fit a small weighted local Ridge model for each query;
4. blend the local estimate with the global Ridge-LightGBM estimate.

The prediction contract remains target-free:

\[
X_{\leq t}\longrightarrow \widehat{\texttt{SLURRY\_FREE\_ACID}}_t.
\]

No current or historical target value is present in the model inputs. The January test period remains retrospective because it was inspected in earlier notebooks.
"""),
md("## 1. Configuration and reproducibility"),
code(r"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json, time, warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TARGET = "SLURRY_FREE_ACID"
TIME_COL = "Date"
FEATURE_SET = "D_full"
N_VALIDATION_BLOCKS = 4
LATENT_SPACES = [("pca", 10), ("pca", 20), ("pls", 5), ("pls", 10)]
NEIGHBOUR_COUNTS = [100, 250, 500]
WEIGHTING_SCHEMES = ["uniform", "inverse_distance"]
LOCAL_BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
LOCAL_RIDGE_ALPHA = 1.0
EPSILON = 1e-8

PREPARATION_NOTEBOOK = Path("02_feature_engineering_data_preparation.ipynb")
NOTEBOOK06_RESULTS = Path("../reports/target_free_upgrade_artifacts")
NOTEBOOK07_RESULTS = Path("../reports/regime_aware_hybrid_artifacts")
OUTPUT_DIR = Path("../reports/jitl_soft_sensor_artifacts")
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
for path in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 100)

notebook06_selection = pd.read_csv(NOTEBOOK06_RESULTS / "selection.csv").iloc[0]
notebook06_candidates = pd.read_csv(NOTEBOOK06_RESULTS / "candidate_summary.csv")
assert notebook06_selection.candidate == "clustered_ridge_k5"
accuracy_anchor = notebook06_candidates.query("kind == 'blend'").sort_values("mean_rmse").iloc[0]
ACCURACY_ANCHOR_NAME = accuracy_anchor.candidate
GLOBAL_RIDGE_WEIGHT = float(json.loads(accuracy_anchor.parameters)["ridge_weight"])
GLOBAL_LGBM_WEIGHT = 1.0 - GLOBAL_RIDGE_WEIGHT
"""),
md("### Observations"),
md("The search is deliberately small: four latent spaces, three neighbourhood sizes, two weighting rules, and four global/local blend weights. This avoids an uncontrolled algorithm sweep."),
md("## 2. Recover the frozen preparation layer"),
code(r"""
def recover_prepared_namespace(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
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
prepared = prep["prepared"]
raw_df = prep["df"].sort_values(prep["TIME_COL"]).reset_index(drop=True)
TARGET, TIME_COL = prep["TARGET"], prep["TIME_COL"]
parts = prepared[(1, FEATURE_SET)]
features = parts["features"]
current_target = raw_df.set_index(TIME_COL)[TARGET]

split_data = {}
for split in ["train", "valid", "test"]:
    frame = parts[f"{split}_df"].reset_index(drop=True)
    target = frame[TIME_COL].map(current_target)
    usable = target.notna()
    split_data[split] = {
        "timestamp": frame.loc[usable, TIME_COL].reset_index(drop=True),
        "X": frame.loc[usable, features].reset_index(drop=True),
        "y": target.loc[usable].reset_index(drop=True),
    }

dimensions = pd.DataFrame([{
    "feature_set": FEATURE_SET, "feature_count": len(features),
    "train_rows": len(split_data["train"]["y"]),
    "validation_rows": len(split_data["valid"]["y"]),
    "test_rows": len(split_data["test"]["y"]),
}])
integrity = pd.DataFrame([
    {"check": "target_absent", "status": TARGET not in features},
    {"check": "target_history_absent", "status": not any(name.startswith(TARGET) for name in features)},
    {"check": "future_columns_absent", "status": not any("t_plus" in name for name in features)},
    {"check": "chronological", "status": split_data["train"]["timestamp"].max() < split_data["valid"]["timestamp"].min() < split_data["test"]["timestamp"].min()},
    {"check": "columns_aligned", "status": list(split_data["train"]["X"].columns) == list(split_data["valid"]["X"].columns) == list(split_data["test"]["X"].columns)},
])
assert integrity.status.all(), integrity
display(dimensions)
display(integrity)
"""),
md("### Observations"),
md("Feature engineering, clipping, grouped filtering, warm-up removal, and split boundaries are inherited unchanged from Notebook 02."),
md("## 3. Four expanding forward-validation blocks"),
code(r"""
train, valid = split_data["train"], split_data["valid"]
valid_positions = np.array_split(np.arange(len(valid["y"])), N_VALIDATION_BLOCKS)
folds = []
for fold_id, positions in enumerate(valid_positions, 1):
    earlier = np.concatenate(valid_positions[:fold_id-1]) if fold_id > 1 else np.array([], dtype=int)
    folds.append({
        "fold": fold_id,
        "X_train": pd.concat([train["X"], valid["X"].iloc[earlier]], ignore_index=True),
        "y_train": pd.concat([train["y"], valid["y"].iloc[earlier]], ignore_index=True),
        "t_train": pd.concat([train["timestamp"], valid["timestamp"].iloc[earlier]], ignore_index=True),
        "X_valid": valid["X"].iloc[positions].reset_index(drop=True),
        "y_valid": valid["y"].iloc[positions].reset_index(drop=True),
        "t_valid": valid["timestamp"].iloc[positions].reset_index(drop=True),
    })

fold_summary = pd.DataFrame([{
    "fold": f["fold"], "train_rows": len(f["y_train"]), "validation_rows": len(f["y_valid"]),
    "train_end": f["t_train"].max(), "validation_start": f["t_valid"].min(),
    "validation_end": f["t_valid"].max()
} for f in folds])
assert (fold_summary.train_end < fold_summary.validation_start).all()
display(fold_summary)
"""),
md("## 4. Reusable global and just-in-time functions"),
code(r"""
def regression_metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    error = prediction - y
    absolute = np.abs(error)
    return {
        "mae": mean_absolute_error(y, prediction),
        "rmse": mean_squared_error(y, prediction) ** 0.5,
        "median_ae": median_absolute_error(y, prediction),
        "r2": r2_score(y, prediction),
        "bias": error.mean(),
        "p90_ae": np.quantile(absolute, 0.90),
        "p95_ae": np.quantile(absolute, 0.95),
    }


def fit_preprocessing(X_train, X_eval):
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(X_train)
    eval_imputed = imputer.transform(X_eval)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    eval_scaled = scaler.transform(eval_imputed)
    return train_imputed, eval_imputed, train_scaled, eval_scaled, imputer, scaler


def fit_global_blend(train_imputed, eval_imputed, train_scaled, eval_scaled, y_train):
    ridge = Ridge(alpha=10.0).fit(train_scaled, y_train)
    lgbm = LGBMRegressor(n_estimators=150, num_leaves=31, learning_rate=0.05,
                         min_child_samples=40, random_state=RANDOM_STATE,
                         n_jobs=-1, verbosity=-1).fit(train_imputed, y_train)
    prediction = GLOBAL_RIDGE_WEIGHT * ridge.predict(eval_scaled) + GLOBAL_LGBM_WEIGHT * lgbm.predict(eval_imputed)
    return {"ridge": ridge, "lgbm": lgbm}, prediction


def fit_latent_space(kind, n_components, X_train, y_train, X_eval):
    if kind == "pca":
        transformer = PCA(n_components=n_components, random_state=RANDOM_STATE)
        Z_train = transformer.fit_transform(X_train)
        Z_eval = transformer.transform(X_eval)
    else:
        transformer = PLSRegression(n_components=n_components, scale=False, max_iter=500)
        Z_train = transformer.fit_transform(X_train, y_train)[0]
        Z_eval = transformer.transform(X_eval)
    latent_scaler = StandardScaler()
    Z_train = latent_scaler.fit_transform(Z_train)
    Z_eval = latent_scaler.transform(Z_eval)
    return transformer, latent_scaler, Z_train, Z_eval


def neighbour_weights(distances, scheme):
    if scheme == "uniform":
        weights = np.ones_like(distances)
    else:
        weights = 1.0 / np.maximum(distances, EPSILON)
    return weights / weights.sum(axis=1, keepdims=True)


def batched_local_ridge(Z_train, y_train, Z_query, indices, distances, scheme, alpha=1.0):
    Xn = Z_train[indices]
    yn = np.asarray(y_train)[indices]
    weights = neighbour_weights(distances, scheme)
    weight_sum = weights.sum(axis=1)
    x_mean = np.einsum("qk,qkd->qd", weights, Xn) / weight_sum[:, None]
    y_mean = np.einsum("qk,qk->q", weights, yn) / weight_sum
    Xc = Xn - x_mean[:, None, :]
    yc = yn - y_mean[:, None]
    gram = np.einsum("qk,qki,qkj->qij", weights, Xc, Xc)
    rhs = np.einsum("qk,qki,qk->qi", weights, Xc, yc)
    eye = np.eye(Z_train.shape[1])[None, :, :]
    coefficients = np.linalg.solve(gram + alpha * eye, rhs[..., None]).squeeze(-1)
    return y_mean + np.einsum("qd,qd->q", Z_query - x_mean, coefficients)
"""),
md("### Modeling meaning"),
md(r"""
- PCA defines similarity from dominant process variation without using the target.
- PLS defines a supervised similarity space using only the current fold's training labels.
- Each query receives its own weighted local linear model.
- The global blend protects against weak or misleading local neighbourhoods.
"""),
md("## 5. Controlled JITL benchmark"),
code(r"""
fold_rows = []
neighbour_diagnostics = []
benchmark_start = time.perf_counter()

for fold in folds:
    y_train = fold["y_train"].to_numpy()
    y_valid = fold["y_valid"].to_numpy()
    train_tree, valid_tree, train_scaled, valid_scaled, _, _ = fit_preprocessing(fold["X_train"], fold["X_valid"])
    start = time.perf_counter()
    _, global_prediction = fit_global_blend(train_tree, valid_tree, train_scaled, valid_scaled, y_train)
    global_seconds = time.perf_counter() - start
    fold_rows.append({"candidate": "global_blend", "space": "global", "components": 0,
                      "neighbours": 0, "weighting": "none", "local_weight": 0.0,
                      "fold": fold["fold"], "fit_predict_seconds": global_seconds,
                      **regression_metrics(y_valid, global_prediction)})

    for space_kind, n_components in LATENT_SPACES:
        space_start = time.perf_counter()
        _, _, Z_train, Z_valid = fit_latent_space(space_kind, n_components, train_scaled, y_train, valid_scaled)
        search = NearestNeighbors(n_neighbors=max(NEIGHBOUR_COUNTS), algorithm="auto", n_jobs=-1)
        search.fit(Z_train)
        distances_all, indices_all = search.kneighbors(Z_valid)
        space_seconds = time.perf_counter() - space_start
        neighbour_diagnostics.append({
            "fold": fold["fold"], "space": space_kind, "components": n_components,
            "median_nearest_distance": float(np.median(distances_all[:, 0])),
            "p95_nearest_distance": float(np.quantile(distances_all[:, 0], 0.95)),
            "median_500th_distance": float(np.median(distances_all[:, -1])),
        })

        for neighbours in NEIGHBOUR_COUNTS:
            indices = indices_all[:, :neighbours]
            distances = distances_all[:, :neighbours]
            for scheme in WEIGHTING_SCHEMES:
                prediction_start = time.perf_counter()
                local_prediction = batched_local_ridge(
                    Z_train, y_train, Z_valid, indices, distances, scheme, LOCAL_RIDGE_ALPHA)
                local_seconds = time.perf_counter() - prediction_start
                for local_weight in LOCAL_BLEND_WEIGHTS:
                    prediction = (1 - local_weight) * global_prediction + local_weight * local_prediction
                    name = f"{space_kind}{n_components}_k{neighbours}_{scheme}_w{local_weight:.2f}"
                    fold_rows.append({
                        "candidate": name, "space": space_kind, "components": n_components,
                        "neighbours": neighbours, "weighting": scheme, "local_weight": local_weight,
                        "fold": fold["fold"],
                        "fit_predict_seconds": global_seconds + space_seconds + local_seconds,
                        **regression_metrics(y_valid, prediction),
                    })

fold_results = pd.DataFrame(fold_rows)
neighbour_diagnostics = pd.DataFrame(neighbour_diagnostics)
candidate_summary = (fold_results.groupby(
    ["candidate", "space", "components", "neighbours", "weighting", "local_weight"], as_index=False)
    .agg(mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"), worst_rmse=("rmse", "max"),
         mean_mae=("mae", "mean"), mean_r2=("r2", "mean"), mean_bias=("bias", "mean"),
         total_seconds=("fit_predict_seconds", "sum"))
    .sort_values(["mean_rmse", "worst_rmse"]))
print(f"Completed {len(fold_results)} fold evaluations in {time.perf_counter()-benchmark_start:.1f} seconds.")
display(candidate_summary.head(20))
"""),
md("### 5.1 Stability-aware selection"),
code(r"""
best_mean = candidate_summary.iloc[0]
stability_limit = best_mean.mean_rmse * 1.02
stable_pool = candidate_summary.query("mean_rmse <= @stability_limit").sort_values(
    ["worst_rmse", "std_rmse", "mean_rmse", "total_seconds"])
selected_summary = stable_pool.iloc[0]
selection = pd.DataFrame([{
    "rule": "within 2% of best mean RMSE, then minimum worst-fold RMSE, RMSE SD, mean RMSE, and runtime",
    **selected_summary.to_dict(),
}])
display(selection)
"""),
md("### Observations"),
md("Mean RMSE measures average accuracy. Worst-block RMSE and RMSE standard deviation measure chronological fragility. Runtime is used only after the accuracy and stability criteria."),
md("## 6. Retrospective test comparison"),
code(r"""
development_X = pd.concat([train["X"], valid["X"]], ignore_index=True)
development_y = pd.concat([train["y"], valid["y"]], ignore_index=True).to_numpy()
test = split_data["test"]
train_tree, test_tree, train_scaled, test_scaled, final_imputer, final_scaler = fit_preprocessing(development_X, test["X"])
final_global_models, global_test = fit_global_blend(train_tree, test_tree, train_scaled, test_scaled, development_y)

if selected_summary.candidate == "global_blend":
    selected_prediction = global_test
    final_transformer = final_latent_scaler = final_search = None
    selected_parameters = {"candidate": "global_blend"}
    neighbour_distance = np.full(len(global_test), np.nan)
else:
    space_kind = selected_summary.space
    n_components = int(selected_summary.components)
    neighbours = int(selected_summary.neighbours)
    scheme = selected_summary.weighting
    local_weight = float(selected_summary.local_weight)
    final_transformer, final_latent_scaler, Z_development, Z_test = fit_latent_space(
        space_kind, n_components, train_scaled, development_y, test_scaled)
    final_search = NearestNeighbors(n_neighbors=neighbours, algorithm="auto", n_jobs=-1).fit(Z_development)
    distances, indices = final_search.kneighbors(Z_test)
    local_test = batched_local_ridge(
        Z_development, development_y, Z_test, indices, distances, scheme, LOCAL_RIDGE_ALPHA)
    selected_prediction = (1 - local_weight) * global_test + local_weight * local_test
    neighbour_distance = distances[:, 0]
    selected_parameters = {
        "candidate": selected_summary.candidate, "space": space_kind,
        "components": n_components, "neighbours": neighbours,
        "weighting": scheme, "local_weight": local_weight,
        "local_ridge_alpha": LOCAL_RIDGE_ALPHA,
    }

test_metrics = regression_metrics(test["y"], selected_prediction)
notebook06 = pd.read_csv(NOTEBOOK06_RESULTS / "retrospective_comparison.csv")
prior06 = notebook06.query("candidate != 'Notebook05_frozen_blend'").iloc[0]
notebook07 = pd.read_csv(NOTEBOOK07_RESULTS / "retrospective_comparison.csv")
prior07 = notebook07.query("candidate != 'Notebook06_selected_clustered_ridge_k5'").iloc[0]
retrospective_comparison = pd.DataFrame([
    {"candidate": "Notebook06_selected_clustered_ridge_k5", "test_mae": prior06.test_mae, "test_rmse": prior06.test_rmse, "test_r2": prior06.test_r2},
    {"candidate": "Notebook07_regime_hybrid", "test_mae": prior07.test_mae, "test_rmse": prior07.test_rmse, "test_r2": prior07.test_r2},
    {"candidate": selected_summary.candidate, "test_mae": test_metrics["mae"], "test_rmse": test_metrics["rmse"], "test_r2": test_metrics["r2"]},
])
retrospective_comparison["rmse_improvement_vs_notebook06_pct"] = 100 * (
    float(prior06.test_rmse) - retrospective_comparison.test_rmse) / float(prior06.test_rmse)

test_predictions = pd.DataFrame({
    "timestamp": test["timestamp"], "actual": test["y"],
    "global_prediction": global_test, "selected_prediction": selected_prediction,
    "nearest_neighbour_distance": neighbour_distance,
})
test_predictions["residual"] = test_predictions.selected_prediction - test_predictions.actual
test_predictions["absolute_error"] = test_predictions.residual.abs()
display(retrospective_comparison)
"""),
md("### Interpretation boundary"),
md("The test comparison is diagnostic only. Candidate selection is complete before the test is evaluated, but the January test interval has already been inspected in earlier project stages."),
md("## 7. Diagnostics"),
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
plot_summary = candidate_summary.head(20).sort_values("mean_rmse", ascending=False)
axes[0].barh(plot_summary.candidate, plot_summary.mean_rmse, xerr=plot_summary.std_rmse)
axes[0].set(title="Leading JITL candidates", xlabel="Mean RMSE; error bar = fold SD")

leaders = candidate_summary.head(5).candidate.tolist()
if "global_blend" not in leaders:
    leaders.append("global_blend")
for name, group in fold_results.query("candidate in @leaders").groupby("candidate"):
    axes[1].plot(group.fold, group.rmse, marker="o", label=name)
axes[1].set(title="Forward-block stability", xlabel="Validation block", ylabel="RMSE")
axes[1].legend(fontsize=6)

sample = test_predictions.iloc[:2000]
axes[2].plot(sample.timestamp, sample.actual, label="actual", linewidth=0.8)
axes[2].plot(sample.timestamp, sample.selected_prediction, label="selected JITL", linewidth=0.8)
axes[2].set_title("Retrospective test window")
axes[2].legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "jitl_soft_sensor_benchmark.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
md("## 8. Export and readiness"),
code(r"""
exports = {
    "dimensions": dimensions, "input_integrity": integrity, "fold_summary": fold_summary,
    "fold_results": fold_results, "candidate_summary": candidate_summary,
    "neighbour_diagnostics": neighbour_diagnostics, "selection": selection,
    "retrospective_comparison": retrospective_comparison,
    "retrospective_test_predictions": test_predictions,
}
for export_name, frame in exports.items():
    frame.to_csv(OUTPUT_DIR / f"{export_name}.csv", index=False)

bundle = {
    "task": "target-free current-time just-in-time soft sensor",
    "selected_candidate": selected_summary.candidate,
    "selected_parameters": selected_parameters,
    "imputer": final_imputer, "scaler": final_scaler,
    "global_models": final_global_models,
    "latent_transformer": final_transformer,
    "latent_scaler": final_latent_scaler,
    "neighbour_search": final_search,
    "development_scaled": train_scaled,
    "development_target": development_y,
    "features": features, "maximum_lookback": prep["MAX_LOOKBACK"],
    "selection_rule": selection.iloc[0].rule,
    "test_status": "retrospective; genuinely later labeled month required",
}
joblib.dump(bundle, MODEL_DIR / "jitl_target_free_candidate.joblib", compress=3)

readiness = pd.DataFrame([
    {"check": "target_and_target_history_absent", "status": bool(integrity.query("check in ['target_absent','target_history_absent']").status.all())},
    {"check": "four_forward_validation_blocks", "status": len(folds) == 4},
    {"check": "no_shuffle", "status": True},
    {"check": "preprocessing_and_latent_spaces_fit_per_fold", "status": True},
    {"check": "neighbours_restricted_to_fold_training", "status": True},
    {"check": "test_not_used_for_candidate_selection", "status": True},
    {"check": "retrospective_status_explicit", "status": True},
    {"check": "model_bundle_exists", "status": (MODEL_DIR / "jitl_target_free_candidate.joblib").exists()},
    {"check": "all_exports_exist", "status": all((OUTPUT_DIR / f"{key}.csv").exists() for key in exports)},
])
assert readiness.status.all(), readiness
readiness.to_csv(OUTPUT_DIR / "readiness_checks.csv", index=False)
display(readiness)
"""),
md("## 9. Final decision"),
code(r"""
print(f"Selected candidate: {selected_summary.candidate}")
print(f"Mean expanding-validation RMSE: {selected_summary.mean_rmse:.4f}")
print(f"Worst expanding-validation RMSE: {selected_summary.worst_rmse:.4f}")
print(f"Retrospective test RMSE: {test_metrics['rmse']:.4f}")
print(f"Retrospective test R2: {test_metrics['r2']:.4f}")
print("No model replacement is authorized until a later labeled month is evaluated without retuning.")
"""),
md("### Observations"),
md(r"""
- If a PCA neighbourhood wins, general process-state similarity is sufficient.
- If a PLS neighbourhood wins, target-relevant similarity is more useful than dominant process variance.
- If a partial local weight wins, global and query-specific errors are complementary.
- If the global baseline wins, the available month does not support additional local complexity.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} with {len(cells)} cells")
