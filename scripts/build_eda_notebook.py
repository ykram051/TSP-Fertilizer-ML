import json
from pathlib import Path


OUT = Path("notebooks/01_data_exploration.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip() + "\n"}


cells = [
    md("""
# Exploratory Data Analysis for SLURRY_FREE_ACID Prediction

## Project Overview

This notebook is a professional exploratory data analysis report for the TSP fertilizer production dataset sampled at one-minute resolution.

The confirmed target variable is `SLURRY_FREE_ACID`. This is treated as the process quality response for future machine learning work and is never included in any feature matrix.

The analysis uses only `tsp_1min.csv`. The other CSV file present in the repository is intentionally ignored.

The goal is to evaluate data quality, time-series integrity, target behavior, sensor health, delayed process effects, feature engineering opportunities, and machine learning readiness for predicting `SLURRY_FREE_ACID`.

The notebook is organized as an engineering report. Each section answers a specific question and avoids duplicate analyses, duplicate PCA blocks, duplicate readiness reports, and repeated plotting code.
"""),
    md("""
## 1. Data Loading and Configuration

This section centralizes imports, constants, display options, plotting style, and reusable helpers. Central configuration keeps the notebook reproducible and prevents target-name drift.
"""),
    code("""
from pathlib import Path
import importlib.util
import warnings

required_packages = ["numpy", "pandas", "matplotlib", "seaborn", "scipy", "sklearn"]
missing_packages = [package for package in required_packages if importlib.util.find_spec(package) is None]
if missing_packages:
    raise ImportError(
        "Missing required package(s): "
        + ", ".join(missing_packages)
        + ". Install them in the active Jupyter kernel before running this EDA notebook."
    )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH_CANDIDATES = [Path("../tsp_1min.csv"), Path("tsp_1min.csv")]
DATA_PATH = next((path for path in DATA_PATH_CANDIDATES if path.exists()), DATA_PATH_CANDIDATES[0])
TIME_COL = "Date"
TARGET = "SLURRY_FREE_ACID"
SAMPLING_FREQUENCY = "1min"
ROLLING_WINDOWS = {"1h": 60, "6h": 360, "24h": 1440}
LAGS_TO_TEST = [1, 2, 3, 5, 10, 15, 30, 60, 120]
MAX_PAIRPLOT_ROWS = 2000
MAX_MODEL_ROWS = 20000
MAX_CLUSTER_ROWS = 8000

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 80)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (12, 5),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

PALETTE = {
    "target": "#0F766E",
    "accent": "#B45309",
    "positive": "#2563EB",
    "negative": "#DC2626",
    "neutral": "#475569",
}


def clean_axis(axis, title=None, xlabel=None, ylabel=None):
    if title:
        axis.set_title(title, pad=10)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    return axis


def display_section_table(title, dataframe, rows=20):
    print(f"\\n{title}")
    display(dataframe.head(rows) if rows is not None else dataframe)


def coefficient_of_variation(series):
    series = pd.Series(series).dropna()
    mean = series.mean()
    if series.empty or np.isclose(mean, 0):
        return np.nan
    return series.std(ddof=1) / abs(mean)


def iqr_bounds(series, multiplier=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr, iqr


def outlier_share_iqr(series):
    series = pd.Series(series).dropna()
    if series.empty:
        return np.nan
    lower, upper, _ = iqr_bounds(series)
    return ((series < lower) | (series > upper)).mean() * 100


def summarize_numeric_frame(dataframe, columns):
    rows = []
    for column in columns:
        series = dataframe[column]
        clean = series.dropna()
        lower, upper, iqr = iqr_bounds(clean) if not clean.empty else (np.nan, np.nan, np.nan)
        rows.append({
            "variable": column,
            "count": int(clean.size),
            "missing_count": int(series.isna().sum()),
            "missing_pct": series.isna().mean() * 100,
            "mean": clean.mean(),
            "median": clean.median(),
            "std": clean.std(ddof=1),
            "variance": clean.var(ddof=1),
            "min": clean.min(),
            "q1": clean.quantile(0.25),
            "q3": clean.quantile(0.75),
            "max": clean.max(),
            "iqr": iqr,
            "cv": coefficient_of_variation(clean),
            "skewness": clean.skew(),
            "kurtosis": clean.kurtosis(),
            "iqr_outlier_pct": outlier_share_iqr(clean),
            "unique_values": int(clean.nunique()),
        })
    return pd.DataFrame(rows)


def plot_distribution_panel(dataframe, column, bins=60):
    series = dataframe[column].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sns.histplot(series, bins=bins, kde=False, ax=axes[0, 0], color=PALETTE["target"])
    clean_axis(axes[0, 0], f"Histogram of {column}", column, "Count")
    sns.kdeplot(series, ax=axes[0, 1], color=PALETTE["positive"], fill=True)
    clean_axis(axes[0, 1], f"KDE of {column}", column, "Density")
    sns.boxplot(x=series, ax=axes[1, 0], color="#94A3B8")
    clean_axis(axes[1, 0], f"Boxplot of {column}", column, None)
    sns.violinplot(x=series, ax=axes[1, 1], color="#99F6E4", inner="quartile")
    clean_axis(axes[1, 1], f"Violin plot of {column}", column, None)
    plt.tight_layout()
    plt.show()


def safe_sample(dataframe, max_rows, random_state=RANDOM_STATE):
    if len(dataframe) <= max_rows:
        return dataframe.copy()
    return dataframe.sample(max_rows, random_state=random_state).sort_index()


def make_time_features(dataframe):
    result = dataframe.copy()
    result["hour"] = result[TIME_COL].dt.hour
    result["day"] = result[TIME_COL].dt.day
    result["day_of_week"] = result[TIME_COL].dt.dayofweek
    result["week"] = result[TIME_COL].dt.isocalendar().week.astype(int)
    return result


def plot_correlation_heatmap(corr_matrix, title, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap="vlag", center=0, square=False, linewidths=0.2)
    plt.title(title)
    plt.tight_layout()
    plt.show()
"""),
    code("""
raw_df = pd.read_csv(DATA_PATH)

if TARGET not in raw_df.columns:
    raise ValueError(f"Confirmed target {TARGET!r} was not found in {DATA_PATH}.")
if TIME_COL not in raw_df.columns:
    raise ValueError(f"Timestamp column {TIME_COL!r} was not found in {DATA_PATH}.")

df = raw_df.copy()
df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
df = df.sort_values(TIME_COL).reset_index(drop=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [column for column in numeric_cols if column != TARGET]
major_process_vars = [
    "FLOW_RATE_PHOSPHORIC_ACID_1",
    "FLOW_RATE_PHOSPHORIC_ACID_2",
    "FLOW_RATE_GROUND_PHOSPHATE",
    "FLOW_RATE_SLURRY_T_PER_HR",
    "FLOW_RATE_SLURRY_M3_HR",
    "RECYCLAGE",
    "PRESSURE_VAPEUR",
    "TEMPERATURE_VAPEUR",
    "TEMPERATURE_CUVE_ATTAQUE",
    "TEMPERATURE_CUVE_DE_PASSAGE",
    "TEMPERATURE_GAZ_SORTIE_SECHEUR",
    "FLOW_RATE_FIOUL",
    "TEMPERATURE_AIR_CHAUD",
    "FLOW_RATE_VAPEUR",
    "TEMPERATURE_BRIQUE",
]
major_process_vars = [column for column in major_process_vars if column in feature_cols]
non_negative_cols = [column for column in numeric_cols if column not in ["DEPRESSURE_SECHEUR"]]

print(f"Loaded {len(df):,} rows and {df.shape[1]:,} columns from {DATA_PATH}.")
print(f"Time span: {df[TIME_COL].min()} to {df[TIME_COL].max()}")
print(f"Confirmed target: {TARGET}")
print(f"Feature count excluding target: {len(feature_cols)}")
assert TARGET not in feature_cols, "Target leakage: target is present in feature_cols."
"""),
    md("""
## 2. Dataset Overview

This section establishes the structure of the file before any interpretation: shape, schema, data types, first records, descriptive statistics, memory use, duplicate rows, and duplicate timestamps.
"""),
    code("""
overview_table = pd.DataFrame({
    "metric": [
        "rows", "columns", "numeric_columns", "feature_columns_excluding_target",
        "memory_mb", "duplicate_rows", "duplicate_timestamps", "timestamp_start", "timestamp_end"
    ],
    "value": [
        len(df), df.shape[1], len(numeric_cols), len(feature_cols),
        round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        int(df.duplicated().sum()), int(df[TIME_COL].duplicated().sum()),
        df[TIME_COL].min(), df[TIME_COL].max(),
    ],
})

display_section_table("Dataset overview", overview_table, rows=None)
display_section_table("Column names", pd.DataFrame({"column": df.columns}), rows=None)
display_section_table("Data types", df.dtypes.rename("dtype").reset_index().rename(columns={"index": "column"}), rows=None)
display_section_table("First rows", df.head(), rows=None)
display_section_table("Descriptive statistics", df[numeric_cols].describe().T, rows=None)
"""),
    md("""
## 3. Time Series Validation

The dataset must be interpreted as a chronological process sequence. This section checks timestamp parsing, ordering, sampling frequency, missing timestamps, irregular intervals, and target continuity over time.
"""),
    code("""
time_diffs = df[TIME_COL].diff().dt.total_seconds().div(60)
expected_index = pd.date_range(df[TIME_COL].min(), df[TIME_COL].max(), freq=SAMPLING_FREQUENCY, tz="UTC")
missing_timestamps = expected_index.difference(df[TIME_COL])
interval_counts = time_diffs.value_counts(dropna=True).sort_index().rename_axis("interval_minutes").reset_index(name="count")
irregular_intervals = interval_counts[interval_counts["interval_minutes"] != 1.0]

series_validation = pd.DataFrame({
    "check": [
        "timestamp_parse_failures", "chronological_order", "expected_frequency",
        "expected_timestamp_count", "observed_timestamp_count", "missing_timestamps",
        "duplicate_timestamps", "irregular_interval_types",
    ],
    "result": [
        int(df[TIME_COL].isna().sum()), bool(df[TIME_COL].is_monotonic_increasing), SAMPLING_FREQUENCY,
        len(expected_index), df[TIME_COL].nunique(), len(missing_timestamps),
        int(df[TIME_COL].duplicated().sum()), len(irregular_intervals),
    ],
})

display_section_table("Time-series validation", series_validation, rows=None)
display_section_table("Sampling interval distribution", interval_counts, rows=20)
if len(missing_timestamps) > 0:
    display_section_table("First missing timestamps", pd.DataFrame({TIME_COL: missing_timestamps[:20]}), rows=None)

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=False)
axes[0].plot(df[TIME_COL], np.ones(len(df)), marker="|", linestyle="None", color=PALETTE["neutral"], alpha=0.35)
clean_axis(axes[0], "Timestamp coverage", "Timestamp", "Observed")
axes[0].set_yticks([])
axes[1].plot(df[TIME_COL], time_diffs, linewidth=0.8, color=PALETTE["accent"])
clean_axis(axes[1], "Sampling interval over time", "Timestamp", "Minutes since previous row")
axes[2].plot(df[TIME_COL], df[TARGET], linewidth=0.7, color=PALETTE["target"])
clean_axis(axes[2], f"{TARGET} timeline", "Timestamp", TARGET)
plt.tight_layout()
plt.show()
"""),
    md("""
## 4. Data Quality Assessment

This section evaluates missing values, duplicate records, constant and near-constant columns, low variance variables, invalid values, infinite values, physically suspicious negatives, sensor availability, and completeness.
"""),
    code("""
missing_summary = (
    df.isna().sum().rename("missing_count").to_frame()
    .assign(missing_pct=lambda x: x["missing_count"] / len(df) * 100)
    .sort_values("missing_pct", ascending=False)
)

constant_columns = [column for column in numeric_cols if df[column].nunique(dropna=True) <= 1]
near_constant_summary = pd.DataFrame({
    "variable": numeric_cols,
    "unique_values": [df[column].nunique(dropna=True) for column in numeric_cols],
    "dominant_value_pct": [df[column].value_counts(dropna=True, normalize=True).max() * 100 if df[column].notna().any() else np.nan for column in numeric_cols],
    "variance": [df[column].var(ddof=1) for column in numeric_cols],
}).sort_values(["dominant_value_pct", "variance"], ascending=[False, True])

invalid_summary = pd.DataFrame({
    "variable": numeric_cols,
    "infinite_count": [int(np.isinf(df[column]).sum()) for column in numeric_cols],
    "negative_count": [int((df[column] < 0).sum()) if column in non_negative_cols else np.nan for column in numeric_cols],
    "negative_pct": [float((df[column] < 0).mean() * 100) if column in non_negative_cols else np.nan for column in numeric_cols],
})

sensor_availability = pd.DataFrame({
    "variable": numeric_cols,
    "availability_pct": [df[column].notna().mean() * 100 for column in numeric_cols],
    "missing_pct": [df[column].isna().mean() * 100 for column in numeric_cols],
    "first_valid_timestamp": [df.loc[df[column].first_valid_index(), TIME_COL] if df[column].first_valid_index() is not None else pd.NaT for column in numeric_cols],
    "last_valid_timestamp": [df.loc[df[column].last_valid_index(), TIME_COL] if df[column].last_valid_index() is not None else pd.NaT for column in numeric_cols],
}).sort_values("availability_pct")

display_section_table("Missing values", missing_summary, rows=None)
display_section_table("Constant columns", pd.DataFrame({"constant_column": constant_columns}), rows=None)
display_section_table("Near-constant and dominant-value summary", near_constant_summary, rows=15)
display_section_table("Lowest-variance features", near_constant_summary.sort_values("variance").head(10), rows=None)
display_section_table("Invalid, infinite, and negative-value checks", invalid_summary, rows=None)
display_section_table("Sensor availability", sensor_availability, rows=None)
"""),
    md("""
## 5. Target Variable Analysis: SLURRY_FREE_ACID

This section studies the confirmed target through distribution, KDE, boxplot, violin plot, percentiles, outliers, skewness, kurtosis, normality, time evolution, rolling mean, rolling standard deviation, and target stability.
"""),
    code("""
target_series = df[TARGET]
target_clean = target_series.dropna()
target_lower, target_upper, target_iqr = iqr_bounds(target_clean)
target_outlier_mask = (target_series < target_lower) | (target_series > target_upper)

target_summary = pd.DataFrame({
    "metric": [
        "count", "missing_count", "missing_pct", "mean", "median", "std", "variance",
        "min", "q1", "q3", "max", "iqr", "cv", "skewness", "kurtosis",
        "iqr_lower_bound", "iqr_upper_bound", "iqr_outlier_count", "iqr_outlier_pct",
    ],
    "value": [
        target_clean.size, target_series.isna().sum(), target_series.isna().mean() * 100,
        target_clean.mean(), target_clean.median(), target_clean.std(ddof=1), target_clean.var(ddof=1),
        target_clean.min(), target_clean.quantile(0.25), target_clean.quantile(0.75), target_clean.max(),
        target_iqr, coefficient_of_variation(target_clean), target_clean.skew(), target_clean.kurtosis(),
        target_lower, target_upper, int(target_outlier_mask.sum()), target_outlier_mask.mean() * 100,
    ],
})
percentiles = target_clean.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).rename("value").to_frame()

display_section_table(f"{TARGET} statistical summary", target_summary, rows=None)
display_section_table(f"{TARGET} percentiles", percentiles, rows=None)
plot_distribution_panel(df, TARGET)

rolling_mean_1h = target_series.rolling(ROLLING_WINDOWS["1h"], min_periods=15).mean()
rolling_std_1h = target_series.rolling(ROLLING_WINDOWS["1h"], min_periods=15).std()
rolling_mean_24h = target_series.rolling(ROLLING_WINDOWS["24h"], min_periods=120).mean()

fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
axes[0].plot(df[TIME_COL], target_series, linewidth=0.7, color=PALETTE["target"], label=TARGET)
axes[0].plot(df[TIME_COL], rolling_mean_1h, linewidth=1.3, color=PALETTE["accent"], label="1-hour rolling mean")
axes[0].legend()
clean_axis(axes[0], f"{TARGET} time evolution", "Timestamp", TARGET)
axes[1].plot(df[TIME_COL], rolling_std_1h, linewidth=1.0, color=PALETTE["positive"], label="1-hour rolling std")
axes[1].legend()
clean_axis(axes[1], f"{TARGET} short-term volatility", "Timestamp", "Rolling std")
axes[2].plot(df[TIME_COL], rolling_mean_24h, linewidth=1.1, color=PALETTE["negative"], label="24-hour rolling mean")
axes[2].scatter(df.loc[target_outlier_mask, TIME_COL], df.loc[target_outlier_mask, TARGET], s=12, color="black", alpha=0.5, label="IQR outliers")
axes[2].legend()
clean_axis(axes[2], f"{TARGET} long-term stability and outliers", "Timestamp", TARGET)
plt.tight_layout()
plt.show()

sampled_target = target_clean.sample(min(5000, len(target_clean)), random_state=RANDOM_STATE)
normality_stat, normality_p = stats.shapiro(sampled_target)
print(f"Normality check on up to 5,000 observations: statistic={normality_stat:.4f}, p-value={normality_p:.4g}")
print("Normality tests are sensitive for large historian datasets; interpret them with distribution shape, tails, and time stability.")
"""),
    md("""
## 6. Univariate Analysis

Every numerical variable is summarized with distribution statistics, missing percentage, variance, coefficient of variation, skewness, kurtosis, and IQR outlier percentage. Compact histogram and boxplot grids provide visual screening without repeating one-off plotting code.
"""),
    code("""
univariate_summary = summarize_numeric_frame(df, numeric_cols).sort_values("variable")
display_section_table("Univariate numerical summary", univariate_summary, rows=None)

plot_columns = [TARGET] + major_process_vars[:11]
fig, axes = plt.subplots(4, 3, figsize=(17, 14))
axes = axes.ravel()
for axis, column in zip(axes, plot_columns):
    sns.histplot(df[column], bins=50, kde=True, ax=axis, color=PALETTE["target"] if column == TARGET else PALETTE["neutral"])
    clean_axis(axis, title=f"Distribution: {column}", xlabel=column, ylabel="Count")
for axis in axes[len(plot_columns):]:
    axis.axis("off")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 3, figsize=(17, 14))
axes = axes.ravel()
for axis, column in zip(axes, plot_columns):
    sns.boxplot(x=df[column], ax=axis, color="#CBD5E1")
    clean_axis(axis, title=f"Boxplot: {column}", xlabel=column, ylabel=None)
for axis in axes[len(plot_columns):]:
    axis.axis("off")
plt.tight_layout()
plt.show()
"""),
    md("""
## 7. Outlier Analysis

Outliers are flagged, not removed. In industrial production they may represent startup, shutdown, maintenance, process disturbances, abnormal but valid regimes, or sensor faults. The purpose here is triage for engineering review.
"""),
    code("""
iqr_flags = pd.DataFrame(index=df.index)
zscore_flags = pd.DataFrame(index=df.index)
for column in numeric_cols:
    series = df[column]
    lower, upper, _ = iqr_bounds(series.dropna())
    iqr_flags[column] = (series < lower) | (series > upper)
    zscore_flags[column] = ((series - series.mean()) / series.std(ddof=0)).abs() > 3

model_input = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
model_input = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(model_input), columns=numeric_cols, index=df.index)
model_input_scaled = StandardScaler().fit_transform(model_input)
iso_flag = pd.Series(
    IsolationForest(n_estimators=200, contamination=0.02, random_state=RANDOM_STATE, n_jobs=-1).fit_predict(model_input_scaled) == -1,
    index=df.index,
    name="isolation_forest_flag",
)

outlier_summary = pd.DataFrame({
    "variable": numeric_cols,
    "iqr_outlier_count": iqr_flags.sum().values,
    "iqr_outlier_pct": iqr_flags.mean().values * 100,
    "zscore_outlier_count": zscore_flags.sum().values,
    "zscore_outlier_pct": zscore_flags.mean().values * 100,
}).sort_values("iqr_outlier_pct", ascending=False)

row_level_outliers = pd.DataFrame({
    TIME_COL: df[TIME_COL],
    "iqr_flag_count": iqr_flags.sum(axis=1),
    "zscore_flag_count": zscore_flags.sum(axis=1),
    "isolation_forest_flag": iso_flag,
    TARGET: df[TARGET],
})

display_section_table("Outlier summary by variable", outlier_summary, rows=None)
display_section_table("Rows with most simultaneous outlier flags", row_level_outliers.sort_values(["iqr_flag_count", "zscore_flag_count"], ascending=False), rows=20)

fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
axes[0].plot(df[TIME_COL], row_level_outliers["iqr_flag_count"], linewidth=0.8, color=PALETTE["accent"])
clean_axis(axes[0], "Number of IQR outlier flags per timestamp", "Timestamp", "Flag count")
axes[1].scatter(df[TIME_COL], df[TARGET], c=iso_flag.map({True: 1, False: 0}), cmap="coolwarm", s=8, alpha=0.7)
clean_axis(axes[1], f"Isolation Forest flags over {TARGET}", "Timestamp", TARGET)
plt.tight_layout()
plt.show()
"""),
    md("""
## 8. Correlation Analysis

Pearson, Spearman, and Kendall correlations are compared to identify linear and monotonic relationships with `SLURRY_FREE_ACID`. Correlation is useful for screening, but it does not imply physical causation.
"""),
    code("""
corr_methods = {method: df[numeric_cols].corr(method=method) for method in ["pearson", "spearman", "kendall"]}
for method, corr_matrix in corr_methods.items():
    plot_correlation_heatmap(corr_matrix, f"{method.title()} correlation heatmap", figsize=(13, 10))

target_corr = pd.concat(
    [corr_methods[method][TARGET].drop(TARGET).rename(method) for method in corr_methods],
    axis=1,
)
target_corr["abs_pearson"] = target_corr["pearson"].abs()
target_corr_sorted = target_corr.sort_values("abs_pearson", ascending=False)

display_section_table(f"Correlations with {TARGET}", target_corr_sorted, rows=None)
display_section_table("Strongest positive Pearson relationships with target", target_corr.sort_values("pearson", ascending=False).head(8), rows=None)
display_section_table("Strongest negative Pearson relationships with target", target_corr.sort_values("pearson", ascending=True).head(8), rows=None)

target_corr_sorted.drop(columns="abs_pearson").head(15).iloc[::-1].plot(kind="barh", figsize=(12, 8))
plt.title(f"Top absolute correlations with {TARGET}")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()
"""),
    md("""
## 9. Time-Lag Analysis

Industrial responses are rarely instantaneous. Acid feed, phosphate feed, slurry flow, recycle, steam, fuel, and dryer temperatures may influence `SLURRY_FREE_ACID` after residence time and control-loop delays. A positive lag shifts a feature into the past relative to the current target.
"""),
    code("""
lag_rows = []
lag_curves = {}
for feature in major_process_vars:
    curve = []
    for lag in LAGS_TO_TEST:
        corr = df[feature].shift(lag).corr(df[TARGET])
        curve.append(corr)
        lag_rows.append({"feature": feature, "lag_minutes": lag, "correlation_with_target": corr})
    lag_curves[feature] = pd.Series(curve, index=LAGS_TO_TEST)

lag_summary = pd.DataFrame(lag_rows)
best_lag_summary = (
    lag_summary.assign(abs_corr=lambda x: x["correlation_with_target"].abs())
    .sort_values(["feature", "abs_corr"], ascending=[True, False])
    .groupby("feature", as_index=False)
    .first()
    .sort_values("abs_corr", ascending=False)
)
display_section_table(f"Best lagged correlations with {TARGET}", best_lag_summary, rows=None)

n_cols = 3
n_rows = int(np.ceil(len(major_process_vars) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, max(4 * n_rows, 5)), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
for axis, feature in zip(axes, major_process_vars):
    curve = lag_curves[feature]
    best_lag = curve.abs().idxmax()
    axis.plot(curve.index, curve.values, marker="o", linewidth=1.4, color=PALETTE["target"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(best_lag, color=PALETTE["accent"], linestyle="--", linewidth=1.0)
    clean_axis(axis, title=f"Lag correlation: {feature}", xlabel="Lag minutes", ylabel="Correlation")
for axis in axes[len(major_process_vars):]:
    axis.axis("off")
plt.tight_layout()
plt.show()
"""),
    md("""
## 10. Rolling Statistics

Rolling mean, standard deviation, minimum, and maximum describe operating stability over trailing windows. Future modeling should only use trailing rolling features because centered windows leak future information.
"""),
    code("""
rolling_vars = [TARGET] + major_process_vars[:5]
fig, axes = plt.subplots(len(rolling_vars), 1, figsize=(15, max(3 * len(rolling_vars), 6)), sharex=True)
axes = np.atleast_1d(axes)
for axis, column in zip(axes, rolling_vars):
    axis.plot(df[TIME_COL], df[column], color="#CBD5E1", linewidth=0.5, alpha=0.7, label="raw")
    axis.plot(df[TIME_COL], df[column].rolling(60, min_periods=15).mean(), color=PALETTE["target"], linewidth=1.2, label="1h mean")
    axis.plot(df[TIME_COL], df[column].rolling(360, min_periods=60).mean(), color=PALETTE["accent"], linewidth=1.2, label="6h mean")
    axis.fill_between(df[TIME_COL], df[column].rolling(60, min_periods=15).min(), df[column].rolling(60, min_periods=15).max(), color="#E2E8F0", alpha=0.35, label="1h min-max")
    axis.legend(loc="upper right")
    clean_axis(axis, title=f"Rolling stability: {column}", xlabel="Timestamp", ylabel=column)
plt.tight_layout()
plt.show()
"""),
    md("""
## 11. Multivariate Analysis

Pairplots, scatterplots, hexbin plots, and joint distributions help identify nonlinear relationships, saturation zones, and dense operating regions that univariate summaries cannot show.
"""),
    code("""
top_corr_features = target_corr_sorted.head(5).index.tolist()
pairplot_cols = [TARGET] + top_corr_features
pairplot_sample = safe_sample(df[pairplot_cols].dropna(), MAX_PAIRPLOT_ROWS)
sns.pairplot(pairplot_sample, corner=True, diag_kind="kde", plot_kws={"alpha": 0.25, "s": 14})
plt.suptitle(f"Sampled pairplot for {TARGET} and top correlated features", y=1.02)
plt.show()

hexbin_features = top_corr_features[:6]
n_cols = 3
n_rows = int(np.ceil(len(hexbin_features) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, max(4.5 * n_rows, 5)))
axes = np.atleast_1d(axes).ravel()
for axis, feature in zip(axes, hexbin_features):
    axis.hexbin(df[feature], df[TARGET], gridsize=45, cmap="viridis", mincnt=1)
    clean_axis(axis, title=f"Hexbin: {feature} vs {TARGET}", xlabel=feature, ylabel=TARGET)
for axis in axes[len(hexbin_features):]:
    axis.axis("off")
plt.tight_layout()
plt.show()
"""),
    md("""
## 12. Process Stability Analysis

This section identifies stable and high-variability periods for `SLURRY_FREE_ACID` and highlights possible operating regime changes. These periods should be reviewed with production context before deciding how to train the final model.
"""),
    code("""
stability_window = ROLLING_WINDOWS["1h"]
stability = pd.DataFrame({
    TIME_COL: df[TIME_COL],
    f"{TARGET}_rolling_mean_1h": df[TARGET].rolling(stability_window, min_periods=15).mean(),
    f"{TARGET}_rolling_std_1h": df[TARGET].rolling(stability_window, min_periods=15).std(),
})
std_threshold = stability[f"{TARGET}_rolling_std_1h"].quantile(0.90)
stability["high_variability_period"] = stability[f"{TARGET}_rolling_std_1h"] >= std_threshold

regime_change_rows = []
for column in [TARGET] + major_process_vars:
    rolling_mean = df[column].rolling(ROLLING_WINDOWS["6h"], min_periods=60).mean()
    diff = rolling_mean.diff().abs()
    regime_change_rows.append({
        "variable": column,
        "large_6h_mean_change_threshold": diff.quantile(0.95),
        "large_change_count": int((diff >= diff.quantile(0.95)).sum()),
    })
regime_change_summary = pd.DataFrame(regime_change_rows).sort_values("large_change_count", ascending=False)

display_section_table("High target variability periods", stability[stability["high_variability_period"]].head(30), rows=None)
display_section_table("Potential operating regime change indicators", regime_change_summary, rows=None)

fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
axes[0].plot(df[TIME_COL], df[TARGET], linewidth=0.7, color=PALETTE["target"])
axes[0].scatter(stability.loc[stability["high_variability_period"], TIME_COL], df.loc[stability["high_variability_period"], TARGET], s=10, color=PALETTE["negative"], alpha=0.6)
clean_axis(axes[0], f"High variability periods on {TARGET}", "Timestamp", TARGET)
axes[1].plot(stability[TIME_COL], stability[f"{TARGET}_rolling_std_1h"], linewidth=1.0, color=PALETTE["accent"])
axes[1].axhline(std_threshold, color=PALETTE["negative"], linestyle="--", label="90th percentile volatility")
axes[1].legend()
clean_axis(axes[1], f"1-hour rolling standard deviation of {TARGET}", "Timestamp", "Rolling std")
plt.tight_layout()
plt.show()
"""),
    md("""
## 13. Seasonality Analysis

The file covers January 1 to January 31, 2026. Hourly, daily, and weekly patterns can be explored, but broader seasonal conclusions are limited by the one-month duration and should not be overinterpreted.
"""),
    code("""
time_df = make_time_features(df)
duration_days = (df[TIME_COL].max() - df[TIME_COL].min()).total_seconds() / 86400
print(f"Dataset duration: {duration_days:.1f} days. Interpret weekly patterns cautiously and do not infer annual seasonality.")

hourly_profile = time_df.groupby("hour")[TARGET].agg(["count", "mean", "median", "std"]).reset_index()
daily_profile = time_df.groupby("day")[TARGET].agg(["count", "mean", "median", "std"]).reset_index()
weekday_profile = time_df.groupby("day_of_week")[TARGET].agg(["count", "mean", "median", "std"]).reset_index()

display_section_table("Hourly target profile", hourly_profile, rows=None)
display_section_table("Daily target profile", daily_profile, rows=None)
display_section_table("Day-of-week target profile", weekday_profile, rows=None)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.lineplot(data=hourly_profile, x="hour", y="mean", marker="o", ax=axes[0], color=PALETTE["target"])
clean_axis(axes[0], f"Hourly mean {TARGET}", "Hour", TARGET)
sns.lineplot(data=daily_profile, x="day", y="mean", marker="o", ax=axes[1], color=PALETTE["accent"])
clean_axis(axes[1], f"Daily mean {TARGET}", "Day of month", TARGET)
sns.lineplot(data=weekday_profile, x="day_of_week", y="mean", marker="o", ax=axes[2], color=PALETTE["positive"])
clean_axis(axes[2], f"Day-of-week mean {TARGET}", "Day of week (0=Monday)", TARGET)
plt.tight_layout()
plt.show()

hourly_series = df.set_index(TIME_COL)[TARGET].resample("H").mean().interpolate(limit_direction="both")
if len(hourly_series) >= 48:
    centered = hourly_series - hourly_series.mean()
    frequencies = np.fft.rfftfreq(len(centered), d=1)
    power = np.abs(np.fft.rfft(centered.values)) ** 2
    plt.figure(figsize=(13, 4))
    plt.plot(frequencies[1:], power[1:], color=PALETTE["neutral"])
    plt.title(f"FFT power spectrum of hourly {TARGET}")
    plt.xlabel("Cycles per hour")
    plt.ylabel("Power")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
"""),
    md("""
## 14. Sensor Health Assessment

Sensor health is assessed separately from process quality. The section flags constant sensors, possible stuck sensors, drifting sensors, noisy sensors, and missing sensor periods.
"""),
    code("""
sensor_rows = []
for column in numeric_cols:
    series = df[column]
    diffs = series.diff()
    rolling_std = series.rolling(ROLLING_WINDOWS["1h"], min_periods=15).std()
    first_half_mean = series.iloc[: len(series) // 2].mean()
    second_half_mean = series.iloc[len(series) // 2 :].mean()
    global_std = series.std(ddof=1)
    longest_missing_run = series.isna().astype(int).groupby(series.notna().astype(int).cumsum()).sum().max()
    sensor_rows.append({
        "variable": column,
        "availability_pct": series.notna().mean() * 100,
        "unique_values": series.nunique(dropna=True),
        "zero_diff_pct": (diffs == 0).mean() * 100,
        "median_1h_rolling_std": rolling_std.median(),
        "global_std": global_std,
        "noise_ratio_median_rolling_to_global": rolling_std.median() / global_std if global_std and not np.isclose(global_std, 0) else np.nan,
        "mean_drift_second_minus_first": second_half_mean - first_half_mean,
        "longest_missing_run_minutes": int(longest_missing_run) if pd.notna(longest_missing_run) else 0,
    })

sensor_health = pd.DataFrame(sensor_rows)
sensor_health["constant_sensor_flag"] = sensor_health["unique_values"] <= 1
sensor_health["possible_stuck_sensor_flag"] = sensor_health["zero_diff_pct"] >= 95
sensor_health["high_noise_flag"] = sensor_health["noise_ratio_median_rolling_to_global"] >= 0.75
sensor_health["drift_flag"] = sensor_health["mean_drift_second_minus_first"].abs() >= sensor_health["global_std"]

display_section_table("Sensor health summary", sensor_health.sort_values(["constant_sensor_flag", "possible_stuck_sensor_flag", "availability_pct"], ascending=[False, False, True]), rows=None)
display_section_table("Sensors with possible health concerns", sensor_health.query("constant_sensor_flag or possible_stuck_sensor_flag or high_noise_flag or drift_flag"), rows=None)
"""),
    md("""
## 15. Feature Engineering Exploration

The engineered variables below are exploratory and are stored separately from the original variables. They include lag features, trailing rolling averages, rolling standard deviations, differences, percentage changes, interactions, and meaningful process ratios.

Confirmed process information: `FLOW_RATE_SLURRY_M3_HR` and `FLOW_RATE_SLURRY_M3_HR_2` are both measured in m3/h and must be added to obtain total slurry volumetric flow. The density proxy therefore uses their sum as its volumetric denominator. The two phosphoric-acid flow lines are also added to obtain total acid flow; their engineering unit remains a process-validation question.
"""),
    code("""
engineered = pd.DataFrame(index=df.index)
engineered[TIME_COL] = df[TIME_COL]
engineered[TARGET] = df[TARGET]

for column in major_process_vars:
    for lag in [1, 5, 10, 30, 60]:
        engineered[f"{column}_lag_{lag}min"] = df[column].shift(lag)
    engineered[f"{column}_rolling_mean_1h"] = df[column].rolling(60, min_periods=15).mean()
    engineered[f"{column}_rolling_std_1h"] = df[column].rolling(60, min_periods=15).std()
    engineered[f"{column}_diff_1min"] = df[column].diff(1)
    engineered[f"{column}_pct_change_5min"] = df[column].pct_change(5).replace([np.inf, -np.inf], np.nan)

if {"FLOW_RATE_PHOSPHORIC_ACID_1", "FLOW_RATE_PHOSPHORIC_ACID_2"}.issubset(df.columns):
    engineered["TOTAL_PHOSPHORIC_ACID_FLOW"] = df["FLOW_RATE_PHOSPHORIC_ACID_1"] + df["FLOW_RATE_PHOSPHORIC_ACID_2"]
if {"FLOW_RATE_PHOSPHORIC_ACID_1", "FLOW_RATE_PHOSPHORIC_ACID_2", "FLOW_RATE_GROUND_PHOSPHATE"}.issubset(df.columns):
    total_acid = df["FLOW_RATE_PHOSPHORIC_ACID_1"] + df["FLOW_RATE_PHOSPHORIC_ACID_2"]
    engineered["ACID_TO_PHOSPHATE_RATIO"] = total_acid / df["FLOW_RATE_GROUND_PHOSPHATE"].replace(0, np.nan)
if {"FLOW_RATE_SLURRY_M3_HR", "FLOW_RATE_SLURRY_M3_HR_2"}.issubset(df.columns):
    engineered["TOTAL_SLURRY_M3_FLOW"] = df["FLOW_RATE_SLURRY_M3_HR"] + df["FLOW_RATE_SLURRY_M3_HR_2"]
if {"FLOW_RATE_SLURRY_T_PER_HR", "FLOW_RATE_SLURRY_M3_HR", "FLOW_RATE_SLURRY_M3_HR_2"}.issubset(df.columns):
    total_slurry_m3 = df["FLOW_RATE_SLURRY_M3_HR"] + df["FLOW_RATE_SLURRY_M3_HR_2"]
    engineered["SLURRY_DENSITY_PROXY_T_PER_M3"] = df["FLOW_RATE_SLURRY_T_PER_HR"] / total_slurry_m3.replace(0, np.nan)
if {"FLOW_RATE_FIOUL", "TEMPERATURE_AIR_CHAUD"}.issubset(df.columns):
    engineered["FUEL_TO_HOT_AIR_TEMP_RATIO"] = df["FLOW_RATE_FIOUL"] / df["TEMPERATURE_AIR_CHAUD"].replace(0, np.nan)
if {"TEMPERATURE_CUVE_ATTAQUE", "TEMPERATURE_CUVE_DE_PASSAGE"}.issubset(df.columns):
    engineered["ATTACK_TO_PASSAGE_TEMP_DELTA"] = df["TEMPERATURE_CUVE_ATTAQUE"] - df["TEMPERATURE_CUVE_DE_PASSAGE"]
if {"RECYCLAGE", "FLOW_RATE_GROUND_PHOSPHATE"}.issubset(df.columns):
    engineered["RECYCLE_TO_PHOSPHATE_RATIO"] = df["RECYCLAGE"] / df["FLOW_RATE_GROUND_PHOSPHATE"].replace(0, np.nan)
if {"RECYCLAGE", "FLOW_RATE_SLURRY_T_PER_HR"}.issubset(df.columns):
    engineered["RECYCLE_X_SLURRY_FLOW"] = df["RECYCLAGE"] * df["FLOW_RATE_SLURRY_T_PER_HR"]

engineered_feature_cols = [column for column in engineered.columns if column not in [TIME_COL, TARGET]]
engineered_summary = summarize_numeric_frame(engineered, engineered_feature_cols).sort_values("missing_pct")
display_section_table("Engineered feature inventory", pd.DataFrame({"engineered_feature": engineered_feature_cols}), rows=80)
display_section_table("Engineered feature summary", engineered_summary, rows=30)
print(f"Original feature count: {len(feature_cols)}")
print(f"Exploratory engineered feature count: {len(engineered_feature_cols)}")
assert TARGET not in engineered_feature_cols, "Target leakage: target appears in engineered feature list."
"""),
    md("""
## 16. Feature Importance Exploration

Random Forest, Extra Trees, and permutation importance are used only as exploratory tools. Feature importance does not equal physical causation. The split is chronological and `SLURRY_FREE_ACID` is excluded from the feature matrix.
"""),
    code("""
model_df = df[[TIME_COL, TARGET] + feature_cols].dropna(subset=[TARGET]).replace([np.inf, -np.inf], np.nan).copy()
if len(model_df) > MAX_MODEL_ROWS:
    model_df = model_df.iloc[-MAX_MODEL_ROWS:].copy()

split_idx = int(len(model_df) * 0.80)
train_df = model_df.iloc[:split_idx]
test_df = model_df.iloc[split_idx:]
X_train, y_train = train_df[feature_cols], train_df[TARGET]
X_test, y_test = test_df[feature_cols], test_df[TARGET]
assert TARGET not in X_train.columns and TARGET not in X_test.columns

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=250, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
et_model = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X_train_imp, y_train)
et_model.fit(X_train_imp, y_train)

perm_result = permutation_importance(rf_model, X_test_imp, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1, scoring="neg_mean_absolute_error")

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "random_forest_importance": rf_model.feature_importances_,
    "extra_trees_importance": et_model.feature_importances_,
    "permutation_importance_mean": perm_result.importances_mean,
    "permutation_importance_std": perm_result.importances_std,
})
importance_df["mean_rank"] = importance_df[["random_forest_importance", "extra_trees_importance", "permutation_importance_mean"]].rank(ascending=False).mean(axis=1)
importance_df = importance_df.sort_values("mean_rank")
display_section_table(f"Exploratory feature importance for predicting {TARGET}", importance_df, rows=None)

plot_importance = importance_df.head(15).sort_values("random_forest_importance")
plt.figure(figsize=(11, 8))
plt.barh(plot_importance["feature"], plot_importance["random_forest_importance"], color=PALETTE["target"])
plt.title(f"Top Random Forest feature importances for {TARGET}")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid(True, axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
"""),
    md("""
## 17. Dimensionality Reduction

PCA is applied only for exploration. It summarizes variance in the feature space and helps visualize operating states, but it should not be recommended for the final predictive model unless later validation proves value.
"""),
    code("""
pca_df = df[feature_cols + [TARGET]].replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET]).copy()
pca_sample = safe_sample(pca_df, MAX_CLUSTER_ROWS)
X_pca_imp = SimpleImputer(strategy="median").fit_transform(pca_sample[feature_cols])
X_pca_scaled = StandardScaler().fit_transform(X_pca_imp)

pca = PCA(n_components=min(10, len(feature_cols)), random_state=RANDOM_STATE)
pca_scores = pca.fit_transform(X_pca_scaled)
explained_variance = pd.DataFrame({
    "component": [f"PC{i}" for i in range(1, pca.n_components_ + 1)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
})
loadings = pd.DataFrame(pca.components_.T[:, :3], index=feature_cols, columns=["PC1", "PC2", "PC3"])
display_section_table("PCA explained variance", explained_variance, rows=None)
display_section_table("Largest absolute PCA loadings", loadings.abs().sort_values("PC1", ascending=False).head(12), rows=None)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker="o", color=PALETTE["target"], label="Individual")
axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker="s", color=PALETTE["accent"], label="Cumulative")
axes[0].legend()
clean_axis(axes[0], "PCA scree plot", "Component", "Explained variance ratio")
scatter = axes[1].scatter(pca_scores[:, 0], pca_scores[:, 1], c=pca_sample[TARGET], cmap="viridis", s=10, alpha=0.7)
plt.colorbar(scatter, ax=axes[1], label=TARGET)
clean_axis(axes[1], f"PCA projection colored by {TARGET}", "PC1", "PC2")
plt.tight_layout()
plt.show()
"""),
    md("""
## 18. Clustering Exploration

KMeans and DBSCAN are used to explore whether operating conditions form regimes. Any apparent cluster should be validated with plant events, production context, and engineering knowledge before it is used downstream.
"""),
    code("""
cluster_df = safe_sample(df[[TIME_COL, TARGET] + feature_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET]), MAX_CLUSTER_ROWS)
cluster_X = SimpleImputer(strategy="median").fit_transform(cluster_df[feature_cols])
cluster_X_scaled = StandardScaler().fit_transform(cluster_X)
cluster_pca = PCA(n_components=min(5, len(feature_cols)), random_state=RANDOM_STATE).fit_transform(cluster_X_scaled)

silhouette_rows = []
for k in range(2, 8):
    labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(cluster_pca[:, :3])
    silhouette_rows.append({"k": k, "silhouette_score": silhouette_score(cluster_pca[:, :3], labels)})
silhouette_df = pd.DataFrame(silhouette_rows)
best_k = int(silhouette_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])

cluster_df["kmeans_cluster"] = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit_predict(cluster_pca[:, :3])
cluster_df["dbscan_cluster"] = DBSCAN(eps=1.25, min_samples=30).fit_predict(cluster_pca[:, :3])
cluster_df["PC1"] = cluster_pca[:, 0]
cluster_df["PC2"] = cluster_pca[:, 1]
cluster_profile = cluster_df.groupby("kmeans_cluster")[[TARGET] + feature_cols].agg(["count", "mean", "std"])

display_section_table("KMeans silhouette scores", silhouette_df, rows=None)
display_section_table("KMeans cluster profile", cluster_profile, rows=None)
display_section_table("DBSCAN cluster counts", cluster_df["dbscan_cluster"].value_counts().sort_index().rename("count").to_frame(), rows=None)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="kmeans_cluster", palette="tab10", s=14, alpha=0.7, ax=axes[0])
clean_axis(axes[0], "KMeans clusters in PCA space", "PC1", "PC2")
sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="dbscan_cluster", palette="tab10", s=14, alpha=0.7, ax=axes[1])
clean_axis(axes[1], "DBSCAN clusters in PCA space", "PC1", "PC2")
plt.tight_layout()
plt.show()
"""),
    md("""
## 19. Machine Learning Readiness

This checklist consolidates data risks for future modeling. The essential rule is that `SLURRY_FREE_ACID` must never appear in the feature matrix. Future model validation must be time-aware and must never shuffle the time series.
"""),
    code("""
readiness_items = [
    {"aspect": "Target definition", "status": "confirmed", "finding": TARGET},
    {"aspect": "Target leakage", "status": "pass" if TARGET not in feature_cols else "fail", "finding": f"feature_cols excludes {TARGET}"},
    {"aspect": "Missing values", "status": "requires preprocessing" if df[numeric_cols].isna().any().any() else "pass", "finding": f"Maximum missing pct: {missing_summary['missing_pct'].max():.2f}%"},
    {"aspect": "Feature scaling", "status": "model dependent", "finding": "Required for PCA, clustering, linear models, and distance-based models; not required for tree ensembles."},
    {"aspect": "Categorical variables", "status": "pass", "finding": "No categorical process variables are present after timestamp parsing."},
    {"aspect": "Duplicate rows", "status": "pass" if df.duplicated().sum() == 0 else "review", "finding": f"Duplicate rows: {df.duplicated().sum()}"},
    {"aspect": "Duplicate timestamps", "status": "pass" if df[TIME_COL].duplicated().sum() == 0 else "review", "finding": f"Duplicate timestamps: {df[TIME_COL].duplicated().sum()}"},
    {"aspect": "Multicollinearity", "status": "review", "finding": "Strongly correlated process sensors should be reviewed before linear or interpretation-heavy models."},
    {"aspect": "Stationarity", "status": "review", "finding": "Rolling statistics should guide time-aware validation and drift checks."},
    {"aspect": "Feature engineering", "status": "recommended", "finding": "Lag, rolling, ratio, difference, interaction, and operating-regime features are promising."},
    {"aspect": "Validation strategy", "status": "mandatory", "finding": "Use latest-period holdout and expanding-window or rolling-origin cross-validation. Never shuffle rows."},
]
readiness = pd.DataFrame(readiness_items)
score_map = {"pass": 1.0, "confirmed": 1.0, "model dependent": 0.75, "recommended": 0.75, "requires preprocessing": 0.5, "review": 0.5, "mandatory": 0.5, "fail": 0.0}
readiness["score_component"] = readiness["status"].map(score_map).fillna(0.5)
overall_readiness_score = readiness["score_component"].mean() * 100
display_section_table("Machine learning readiness checklist", readiness.drop(columns="score_component"), rows=None)
print(f"Overall readiness score: {overall_readiness_score:.1f}/100")
assert TARGET not in feature_cols
"""),
    md("""
## 20. Engineering Insights

This section summarizes the most influential variables, stable variables, potential control variables, delayed effects, process insights, and uncertainties. The results should be reviewed with process engineers before any causal interpretation.
"""),
    code("""
insight_tables = {
    "Most correlated variables with target": target_corr_sorted.head(10),
    "Most stable variables by coefficient of variation": univariate_summary.sort_values("cv").head(10),
    "Highest missing-rate sensors": missing_summary.head(10),
    "Most delayed candidate relationships": best_lag_summary.head(10),
    "Top exploratory model features": importance_df.head(10),
}
for title, table in insight_tables.items():
    display_section_table(title, table, rows=None)

potential_control_variables = [
    column for column in [
        "FLOW_RATE_PHOSPHORIC_ACID_1", "FLOW_RATE_PHOSPHORIC_ACID_2", "FLOW_RATE_GROUND_PHOSPHATE",
        "RECYCLAGE", "PRESSURE_VAPEUR", "FLOW_RATE_FIOUL", "FLOW_RATE_VAPEUR", "FLOW_RATE_LIQUIDE_LAVAGE",
    ] if column in feature_cols
]
print("Potential control or manipulated variables to review with process engineers:")
for column in potential_control_variables:
    print(f"- {column}")
print("\\nUncertainties to resolve: sensor calibration, true controllability, startup/shutdown periods, maintenance events, and residence-time expectations.")
"""),
    md("""
## 21. Final Conclusions

The dataset is a coherent one-minute time-series foundation for predicting `SLURRY_FREE_ACID`, provided that future modeling respects chronological order and prevents target leakage.

### Dataset Quality

The notebook checks timestamp integrity, duplicate rows, duplicate timestamps, missingness, sensor availability, invalid values, low-variance signals, and sensor-health indicators. Missing values and outliers should be handled deliberately rather than removed automatically.

### Key Findings to Carry Forward

- `SLURRY_FREE_ACID` is the confirmed target and must remain excluded from all feature matrices.
- Industrial outliers may correspond to startup, shutdown, maintenance, disturbances, abnormal but valid regimes, or sensor faults.
- Correlation and feature importance identify candidate predictive variables, but neither proves physical causation.
- Lag analysis is essential because material residence time and control response can delay the target response.
- Rolling statistics and clustering may reveal stable operation, high-variability periods, and potential operating regimes.

### Recommended Preprocessing

- Parse and sort timestamps before every modeling step.
- Use time-aware train/test splits and rolling-origin validation.
- Impute missing process values using train-only fit operations.
- Use robust scaling where required by the model family.
- Treat outliers with process context; consider flags instead of deletion.
- Review constant, stuck, duplicate, or physically invalid signals.

### Recommended Feature Engineering

- Lag features for acid feed, phosphate feed, slurry flow, recycle, steam, fuel, and dryer temperatures.
- Trailing rolling means, rolling standard deviations, rolling minima, and rolling maxima.
- Differences and percentage changes to capture disturbances.
- Process ratios such as total acid to phosphate, slurry density proxy, recycle to phosphate, and thermal efficiency proxies.
- Operating-regime labels only if supported by clustering and process-engineering validation.

### Recommended Model Families

- Regularized linear models for transparent baselines.
- Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, or CatBoost if available.
- Time-aware forecasting regressors using lagged and rolling features.
- Regime-specific models only if operating regimes are stable and interpretable.

The next notebook should convert these EDA findings into a leakage-safe feature pipeline and a chronological validation framework for predicting `SLURRY_FREE_ACID`.
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
