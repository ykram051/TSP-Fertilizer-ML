import json
from pathlib import Path


OUT = Path("notebooks/02_feature_engineering_data_preparation.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip() + "\n"}


cells = [
md(r"""
# Hypothesis-driven feature preparation for future `SLURRY_FREE_ACID`

This notebook creates compact, auditable feature groups rather than one indiscriminate matrix. It supports four forecasting horizons:

\[
y_{t+1},\quad y_{t+5},\quad y_{t+10},\quad y_{t+15}
\]

The modeling notebook can therefore compare predictive accuracy, warning usefulness, reaction time, and degradation with horizon. No model is trained here.

The design rules are:

- current and past inputs only; the target is never an input;
- variable roles and controllability are explicit;
- lags are selected using training-only association and stability evidence plus response-time assumptions;
- statistical features are deliberately limited;
- physical proxies remain provisional until a process expert validates their units and meaning;
- structural warm-up rows are dropped, while genuine sensor gaps are handled separately;
- correlation decisions are made within source-variable families on training data only.
"""),
md("## 1. Configuration, loading, and leakage guards"),
code(r"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TIME_COL = "Date"
TARGET = "SLURRY_FREE_ACID"
HORIZONS = [1, 5, 10, 15]
CANDIDATE_LAGS = [1, 5, 10, 15, 30, 60]
MAX_SELECTED_LAGS_PER_SENSOR = 3
SHORT_WINDOW_MINUTES = 10
MEDIUM_WINDOW_MINUTES = 30
DIFFERENCE_MINUTES = 5
SLOPE_WINDOW_MINUTES = 10
TRAIN_FRACTION, VALID_FRACTION = 0.70, 0.15
CORRELATION_THRESHOLD = 0.98
EPSILON = 1e-6
DATA_PATH = next(p for p in [Path("../tsp_1min.csv"), Path("tsp_1min.csv")] if p.exists())

raw_df = pd.read_csv(DATA_PATH)
raw_df[TIME_COL] = pd.to_datetime(raw_df[TIME_COL], utc=True, errors="coerce")
df = raw_df.sort_values(TIME_COL).reset_index(drop=True)
sensor_columns = [c for c in df.select_dtypes(include=np.number).columns if c != TARGET]

def target_name(h): return f"{TARGET}_t_plus_{h}min"
def assert_no_target_leakage(columns):
    bad = [c for c in columns if c == TARGET or c.startswith(f"{TARGET}_t_plus_")]
    if bad: raise ValueError(f"Target leakage: {bad}")

assert_no_target_leakage(sensor_columns)
print(f"Rows: {len(df):,}; sensors: {len(sensor_columns)}; period: {df[TIME_COL].min()} to {df[TIME_COL].max()}")
"""),
md(r"""
## 2. Variable registry and intervention policy

`primary_category` gives each variable a principal role; `category_tags` preserves cross-cutting roles. `controllability_status` is intentionally conservative. A variable can be used for prediction even when it cannot be manipulated, but the downstream recommendation system may propose a change **only** when `recommendation_allowed` is true. Here, every candidate actuator remains `needs_supervisor_confirmation`, so no intervention is authorized yet.
"""),
code(r"""
registry_rows = [
 ("PRODUCTION_TSP_BALANCE","production","production;observed_state","observed","Production/load indicator; exact balance definition needs confirmation."),
 ("FLOW_RATE_PHOSPHORIC_ACID_1","flow","flow;raw_process;candidate_controllable","needs_supervisor_confirmation","Acid line 1 flow; likely set-point influenced."),
 ("FLOW_RATE_PHOSPHORIC_ACID_2","flow","flow;raw_process;candidate_controllable","needs_supervisor_confirmation","Acid line 2 flow; likely set-point influenced."),
 ("FLOW_RATE_GROUND_PHOSPHATE","flow","flow;production;candidate_controllable","needs_supervisor_confirmation","Ground phosphate feed."),
 ("FLOW_RATE_SLURRY_T_PER_HR","flow","flow;observed_state","observed","Slurry mass-flow measurement."),
 ("FLOW_RATE_SLURRY_M3_HR","flow","flow;observed_state","observed","Confirmed slurry volumetric-flow component measured in m3/h; add to FLOW_RATE_SLURRY_M3_HR_2."),
 ("RECYCLAGE","production","production;flow;uncertain_meaning","uncertain","Recycle measurement; units and numerator meaning unknown."),
 ("PRESSURE_VAPEUR","pressure","pressure;candidate_controllable","needs_supervisor_confirmation","Steam pressure."),
 ("TEMPERATURE_VAPEUR","thermal","thermal;observed_state","observed","Steam temperature."),
 ("TEMPERATURE_CUVE_ATTAQUE","thermal","thermal;observed_state","observed","Attack-tank temperature."),
 ("TEMPERATURE_CUVE_DE_PASSAGE","thermal","thermal;observed_state","observed","Passage-tank temperature."),
 ("TEMPERATURE_SLURRY_PULVERISATEUR","thermal","thermal;observed_state","observed","Slurry temperature near atomizer/sprayer."),
 ("DEPRESSURE_SECHEUR","pressure","pressure;observed_state","observed","Dryer depression/draft; sign convention needs confirmation."),
 ("LEVEL_CUVE_PASSAGE","observed_state","observed_state","observed","Passage-tank level."),
 ("TEMPERATURE_GAZ_SORTIE_SECHEUR","thermal","thermal;observed_state;possible_quality_indicator","observed","Dryer outlet-gas temperature."),
 ("FLOW_RATE_FIOUL","flow","flow;thermal;candidate_controllable","needs_supervisor_confirmation","Fuel-oil flow."),
 ("TEMPERATURE_AIR_CHAUD","thermal","thermal;observed_state","observed","Hot-air temperature."),
 ("FLOW_RATE_LIQUIDE_LAVAGE","flow","flow;candidate_controllable","needs_supervisor_confirmation","Wash-liquid flow."),
 ("FLOW_RATE_VAPEUR","flow","flow;thermal;candidate_controllable","needs_supervisor_confirmation","Steam flow."),
 ("FLOW_RATE_SLURRY_M3_HR_2","flow","flow;observed_state","observed","Confirmed second slurry volumetric-flow component measured in m3/h; add to FLOW_RATE_SLURRY_M3_HR."),
 ("TEMPERATURE_BRIQUE","thermal","thermal;observed_state;uncertain_meaning","observed","Brick/refractory temperature; location needs confirmation."),
 (TARGET,"possible_quality_indicator","possible_quality_indicator;target_related","target","Prediction target; forbidden from inputs."),
]
variable_registry = pd.DataFrame(registry_rows, columns=["variable","primary_category","category_tags","controllability_status","physical_interpretation"])
variable_registry["role"] = variable_registry["controllability_status"].map({
    "target":"target", "observed":"observed_state", "uncertain":"uncertain_role",
    "needs_supervisor_confirmation":"candidate_actuator_not_confirmed",
})
variable_registry["measurement_type"] = variable_registry["primary_category"].map({
    "flow":"continuous_flow", "thermal":"continuous_temperature", "pressure":"continuous_pressure",
    "production":"continuous_production", "observed_state":"continuous_state",
    "possible_quality_indicator":"laboratory_or_quality_measurement",
})
variable_registry["interpretation_status"] = np.where(
    variable_registry["category_tags"].str.contains("uncertain_meaning"),
    "requires_process_confirmation", "documented_provisional"
)
variable_registry["unit_validation_status"] = "units_not_provided_in_source_file"
confirmed_slurry_volume_tags = ["FLOW_RATE_SLURRY_M3_HR", "FLOW_RATE_SLURRY_M3_HR_2"]
variable_registry.loc[variable_registry["variable"].isin(confirmed_slurry_volume_tags), "measurement_type"] = "continuous_volumetric_flow"
variable_registry.loc[variable_registry["variable"].isin(confirmed_slurry_volume_tags), "interpretation_status"] = "confirmed_additive_components"
variable_registry.loc[variable_registry["variable"].isin(confirmed_slurry_volume_tags), "unit_validation_status"] = "confirmed_m3_per_h"
variable_registry["recommendation_allowed"] = variable_registry["controllability_status"].eq("confirmed_controllable")
variable_registry["supervisor_validation_required"] = variable_registry["controllability_status"].isin(["needs_supervisor_confirmation","uncertain"])
display(variable_registry)
assert not variable_registry["recommendation_allowed"].any(), "No actuator is confirmed yet."
"""),
md(r"""
## 3. Multiple future targets

Each target is aligned to measurements available at time `t`. Rows at the end without a future observation are removed later per horizon, not filled.
"""),
code(r"""
future_targets = pd.DataFrame(index=df.index)
for h in HORIZONS:
    future_targets[target_name(h)] = df[TARGET].shift(-h)
display(future_targets.head())
display(future_targets.tail(16))
"""),
md(r"""
## 4. Training-only, stability-aware lag selection

Candidate delays are scored against the one-minute-ahead target because the temporal feature dictionary must be fixed before comparing horizons. For every sensor and lag, the notebook computes:

- absolute Pearson cross-correlation with the future target;
- mutual information;
- correlation in each of three chronological training subperiods;
- a stability factor that rewards consistent sign and penalizes dispersion.

The normalized correlation and MI ranks are combined with stability. A response-time prior is then used as a tie-breaker/process constraint. At most three lags are retained per sensor. These are association diagnostics, not causal evidence.
"""),
code(r"""
RESPONSE_PRIORS = {
 "FLOW_RATE_PHOSPHORIC_ACID_1":[1,5,10], "FLOW_RATE_PHOSPHORIC_ACID_2":[1,5,10],
 "FLOW_RATE_GROUND_PHOSPHATE":[1,5,10], "PRESSURE_VAPEUR":[1,5], "FLOW_RATE_VAPEUR":[1,5],
 "FLOW_RATE_FIOUL":[1,5,10], "TEMPERATURE_CUVE_ATTAQUE":[5,10,30],
 "TEMPERATURE_CUVE_DE_PASSAGE":[5,10,30], "TEMPERATURE_SLURRY_PULVERISATEUR":[5,10,30],
 "TEMPERATURE_GAZ_SORTIE_SECHEUR":[5,10,30], "TEMPERATURE_AIR_CHAUD":[5,10,30],
 "TEMPERATURE_BRIQUE":[10,30,60], "DEPRESSURE_SECHEUR":[1,5,10],
}
DEFAULT_PRIOR = [1,5,10]

# Conservative selection cutoff: use the longest horizon and the final
# 30-minute modeling warm-up, then stop at its chronological training boundary.
# This boundary is earlier than every final model-training endpoint.
selection_reference = pd.DataFrame({
    "row_index": df.index,
    "future_target": df[TARGET].shift(-max(HORIZONS)),
}).iloc[MEDIUM_WINDOW_MINUTES:].dropna(subset=["future_target"])
selection_train_rows = int(len(selection_reference) * TRAIN_FRACTION)
selection_cutoff_index = int(selection_reference.iloc[selection_train_rows - 1]["row_index"])
lag_selection_base = df.loc[:selection_cutoff_index].copy()
y_lag_selection = lag_selection_base[TARGET].shift(-1)

def lag_evidence_for_sensor(series, future_y, variable):
    rows = []
    for lag in CANDIDATE_LAGS:
        pair = pd.concat([series.shift(lag).rename("x"), future_y.rename("y")], axis=1).dropna()
        corr = pair.corr().iloc[0,1] if len(pair) > 10 else np.nan
        mi = mutual_info_regression(pair[["x"]], pair["y"], random_state=RANDOM_STATE)[0] if len(pair) > 10 else np.nan
        period_corrs = []
        for block in np.array_split(pair, 3):
            period_corrs.append(block.corr().iloc[0,1] if len(block) > 10 else np.nan)
        valid_corrs = np.asarray([v for v in period_corrs if np.isfinite(v)])
        sign_consistency = abs(np.sign(valid_corrs).mean()) if len(valid_corrs) else 0
        stability = sign_consistency / (1 + np.nanstd(valid_corrs))
        rows.append({"variable":variable,"lag_min":lag,"abs_cross_correlation":abs(corr),"mutual_information":mi,
                     "period_1_corr":period_corrs[0],"period_2_corr":period_corrs[1],"period_3_corr":period_corrs[2],"stability":stability})
    out = pd.DataFrame(rows)
    for col in ["abs_cross_correlation","mutual_information"]:
        lo, hi = out[col].min(), out[col].max()
        out[f"{col}_norm"] = (out[col]-lo)/(hi-lo+EPSILON)
    out["data_score"] = (.5*out["abs_cross_correlation_norm"] + .5*out["mutual_information_norm"]) * out["stability"]
    out["process_prior"] = out["lag_min"].isin(RESPONSE_PRIORS.get(variable, DEFAULT_PRIOR))
    out["selection_score"] = out["data_score"] + .15*out["process_prior"].astype(float)
    return out

lag_evidence = pd.concat([lag_evidence_for_sensor(lag_selection_base[c], y_lag_selection, c) for c in sensor_columns], ignore_index=True)
selected_lags = {}
lag_selection_rows = []
for variable, group in lag_evidence.groupby("variable"):
    chosen = group.sort_values(["selection_score","data_score"], ascending=False).head(MAX_SELECTED_LAGS_PER_SENSOR).sort_values("lag_min")
    selected_lags[variable] = chosen["lag_min"].astype(int).tolist()
    prior_hits = chosen["process_prior"].sum()
    lag_selection_rows.append({"variable":variable,"selected_lags_min":", ".join(map(str,selected_lags[variable])),
        "reason":f"Top stable train-only correlation/MI score; {prior_hits}/3 delays agree with response-time prior.",
        "mean_abs_cross_correlation":chosen["abs_cross_correlation"].mean(),
        "mean_mutual_information":chosen["mutual_information"].mean(),
        "mean_stability":chosen["stability"].mean(),
        "process_prior_selected":", ".join(map(str, chosen.loc[chosen["process_prior"], "lag_min"].astype(int).tolist())) or "none",
        "evidence_target":target_name(1)})
lag_selection_table = pd.DataFrame(lag_selection_rows).sort_values("variable")
display(lag_selection_table)
display(lag_evidence.sort_values(["variable","selection_score"], ascending=[True,False]).head(30))
"""),
md(r"""
## 5. Compact temporal features

Each sensor retains its raw value, selected lags, one 10-minute mean, one 30-minute mean, one 10-minute standard deviation, one 5-minute difference, and one 10-minute slope. Slopes are not generated over many windows. Unstable percentage change is replaced by normalized difference only for confirmed strictly positive/stable variables; because the observed data include near-zero or negative values and no such variables are confirmed, none are generated.
"""),
code(r"""
def rolling_slope(series, window=SLOPE_WINDOW_MINUTES):
    x = np.arange(window, dtype=float); xc = x-x.mean(); denom = np.sum(xc**2)
    return series.rolling(window, min_periods=window).apply(lambda y: np.sum(xc*(y-y.mean()))/denom if np.isfinite(y).all() else np.nan, raw=True)

def build_temporal_features(dataframe):
    result, docs = {}, []
    for c in sensor_columns:
        result[c] = dataframe[c]
        docs.append((c,"raw",c,"Current measurement at t."))
        for lag in selected_lags[c]:
            name=f"{c}_lag_{lag}"; result[name]=dataframe[c].shift(lag)
            docs.append((name,"selected_lag",c,f"Selected {lag}-minute response delay from training-only evidence and process prior."))
        specs = {
            f"{c}_roll_mean_{SHORT_WINDOW_MINUTES}": dataframe[c].rolling(SHORT_WINDOW_MINUTES,min_periods=SHORT_WINDOW_MINUTES).mean(),
            f"{c}_roll_mean_{MEDIUM_WINDOW_MINUTES}": dataframe[c].rolling(MEDIUM_WINDOW_MINUTES,min_periods=MEDIUM_WINDOW_MINUTES).mean(),
            f"{c}_roll_std_{SHORT_WINDOW_MINUTES}": dataframe[c].rolling(SHORT_WINDOW_MINUTES,min_periods=SHORT_WINDOW_MINUTES).std(),
            f"{c}_diff_{DIFFERENCE_MINUTES}": dataframe[c].diff(DIFFERENCE_MINUTES),
            f"{c}_slope_{SLOPE_WINDOW_MINUTES}": rolling_slope(dataframe[c],SLOPE_WINDOW_MINUTES),
        }
        reasons = ["Short trailing level.","Medium trailing level.","Short-term variability.","Five-minute movement.","Ten-minute linear trend."]
        families = ["rolling_mean","rolling_mean","variability","difference","trend"]
        for (name, values), family, reason in zip(specs.items(),families,reasons):
            result[name]=values; docs.append((name,family,c,reason))
    return pd.DataFrame(result,index=dataframe.index), pd.DataFrame(docs,columns=["feature","family","source_variable","reason"])

temporal_features, temporal_docs = build_temporal_features(df)
print(f"Compact temporal candidate count: {temporal_features.shape[1]} (versus hundreds of blanket combinations).")
display(temporal_docs.groupby("family").size().rename("count").reset_index())
"""),
md(r"""
## 6. Process-aware features and validation register

Every physical feature has a formula, input variables, interpretation, assumptions, and validation status. `provisional` features are valid experimental predictors but must not be presented as verified physics. Ratios use an epsilon guard and become missing where the denominator magnitude is too small.
"""),
code(r"""
def guarded_ratio(num, den, eps=EPSILON):
    safe_den = den.where(den.abs()>eps)
    return num / (safe_den + np.sign(safe_den) * eps)

process_values = {}
process_specs = []
def add_process(name, values, formula, variables, interpretation, assumptions, status="provisional"):
    process_values[name] = values.replace([np.inf,-np.inf],np.nan)
    process_specs.append((name,formula,"; ".join(variables),interpretation,assumptions,status))

a1,a2,gp = df["FLOW_RATE_PHOSPHORIC_ACID_1"],df["FLOW_RATE_PHOSPHORIC_ACID_2"],df["FLOW_RATE_GROUND_PHOSPHATE"]
total_acid=a1+a2
add_process("TOTAL_ACID_FLOW",total_acid,"acid_1 + acid_2",["FLOW_RATE_PHOSPHORIC_ACID_1","FLOW_RATE_PHOSPHORIC_ACID_2"],"Combined phosphoric-acid feed across both lines.","The two acid lines are confirmed additive; their engineering unit still requires confirmation.","aggregation_confirmed_units_pending")
add_process("ACID_FLOW_IMBALANCE",(a1-a2).abs(),"|acid_1 - acid_2|",["FLOW_RATE_PHOSPHORIC_ACID_1","FLOW_RATE_PHOSPHORIC_ACID_2"],"Magnitude of line imbalance.","The lines are comparable and imbalance is operationally meaningful.")
add_process("ACID_TO_PHOSPHATE_RATIO",guarded_ratio(total_acid,gp),"(acid_1 + acid_2) / (ground_phosphate + epsilon)",["FLOW_RATE_PHOSPHORIC_ACID_1","FLOW_RATE_PHOSPHORIC_ACID_2","FLOW_RATE_GROUND_PHOSPHATE"],"Feed-ratio proxy related to acidulation conditions.","Flow units/bases are compatible; not a stoichiometric ratio without composition data.")
total_slurry_m3=df["FLOW_RATE_SLURRY_M3_HR"]+df["FLOW_RATE_SLURRY_M3_HR_2"]
add_process("TOTAL_SLURRY_M3_FLOW",total_slurry_m3,"slurry_m3_1 + slurry_m3_2",["FLOW_RATE_SLURRY_M3_HR","FLOW_RATE_SLURRY_M3_HR_2"],"Confirmed total slurry volumetric flow in m3/h.","Both tags are confirmed additive components with the same m3/h unit.","confirmed_m3_per_h_aggregation")
add_process("SLURRY_DENSITY_PROXY",guarded_ratio(df["FLOW_RATE_SLURRY_T_PER_HR"],total_slurry_m3),"slurry_t_per_h / (slurry_m3_1 + slurry_m3_2 + epsilon)",["FLOW_RATE_SLURRY_T_PER_HR","FLOW_RATE_SLURRY_M3_HR","FLOW_RATE_SLURRY_M3_HR_2"],"Apparent slurry mass-per-volume proxy using total volumetric flow.","The two volumetric tags are confirmed additive in m3/h; confirm that the t/h tag represents the same combined stream.","requires_mass_stream_validation")
add_process("ATTACK_PASSAGE_TEMP_GRADIENT",df["TEMPERATURE_CUVE_ATTAQUE"]-df["TEMPERATURE_CUVE_DE_PASSAGE"],"T_attack - T_passage",["TEMPERATURE_CUVE_ATTAQUE","TEMPERATURE_CUVE_DE_PASSAGE"],"Thermal change between reaction vessels.","Sensors are comparable and process direction is attack to passage.")
add_process("HOT_AIR_DRYER_OUTLET_GRADIENT",df["TEMPERATURE_AIR_CHAUD"]-df["TEMPERATURE_GAZ_SORTIE_SECHEUR"],"T_hot_air - T_dryer_outlet_gas",["TEMPERATURE_AIR_CHAUD","TEMPERATURE_GAZ_SORTIE_SECHEUR"],"Dryer thermal-driving-force proxy.","Temperatures correspond to the same dryer train.")
add_process("STEAM_SLURRY_TEMP_GRADIENT",df["TEMPERATURE_VAPEUR"]-df["TEMPERATURE_SLURRY_PULVERISATEUR"],"T_steam - T_slurry",["TEMPERATURE_VAPEUR","TEMPERATURE_SLURRY_PULVERISATEUR"],"Available temperature-gap proxy.","Steam/slurry tags belong to the relevant heat-transfer path.")
add_process("STEAM_ENERGY_PROXY",df["PRESSURE_VAPEUR"]*df["TEMPERATURE_VAPEUR"],"steam_pressure * steam_temperature",["PRESSURE_VAPEUR","TEMPERATURE_VAPEUR"],"Steam-condition proxy, not energy or enthalpy.","Tags refer to the same steam supply; units remain unconverted.")
dryer_temp_drop=df["TEMPERATURE_AIR_CHAUD"]-df["TEMPERATURE_GAZ_SORTIE_SECHEUR"]
add_process("DRYING_INTENSITY_PROXY",df["FLOW_RATE_FIOUL"]*dryer_temp_drop/(df["DEPRESSURE_SECHEUR"].abs()+EPSILON),"fuel_flow * (T_hot_air - T_outlet) / (|dryer_depression| + epsilon)",["FLOW_RATE_FIOUL","TEMPERATURE_AIR_CHAUD","TEMPERATURE_GAZ_SORTIE_SECHEUR","DEPRESSURE_SECHEUR"],"Composite burner/load/draft indicator.","Signs and units are not thermodynamically normalized; engineering review required.","requires_engineering_validation")
add_process("RECYCLE_TO_PRODUCTION_RATIO",guarded_ratio(df["RECYCLAGE"],df["PRODUCTION_TSP_BALANCE"]),"recycle / (production + epsilon)",["RECYCLAGE","PRODUCTION_TSP_BALANCE"],"Recycle-intensity proxy.","Recycle and production tags use compatible mass/time bases.","requires_unit_validation")

important_flows=["FLOW_RATE_PHOSPHORIC_ACID_1","FLOW_RATE_PHOSPHORIC_ACID_2","FLOW_RATE_GROUND_PHOSPHATE","FLOW_RATE_FIOUL","FLOW_RATE_VAPEUR"]
for c in important_flows:
    rolling_mean=df[c].rolling(SHORT_WINDOW_MINUTES,min_periods=SHORT_WINDOW_MINUTES).mean()
    rolling_std=df[c].rolling(SHORT_WINDOW_MINUTES,min_periods=SHORT_WINDOW_MINUTES).std()
    add_process(f"{c}_CV_{SHORT_WINDOW_MINUTES}",rolling_std/(rolling_mean.abs()+EPSILON),f"rolling_std_{SHORT_WINDOW_MINUTES}({c}) / (|rolling_mean_{SHORT_WINDOW_MINUTES}({c})| + epsilon)",[c],"Scale-normalized ten-minute flow instability.","Mean magnitude away from zero; extreme proxy values are clipped below.")

process_features=pd.DataFrame(process_values,index=df.index)
process_clip_lower=process_features.loc[:selection_cutoff_index].quantile(.001)
process_clip_upper=process_features.loc[:selection_cutoff_index].quantile(.999)
process_features=process_features.clip(lower=process_clip_lower,upper=process_clip_upper,axis=1)
process_feature_registry=pd.DataFrame(process_specs,columns=["feature","formula","involved_variables","expected_interpretation","assumptions","validation_status"])
display(process_feature_registry)
"""),
md(r"""
## 7. Explicit feature sets and warm-up policy

- **A — Raw baseline:** current sensors only.
- **B — Raw + selected temporal:** current values, selected lags, short/medium means, trend, difference, variability.
- **C — Process-aware:** raw sensors plus physical ratios, balances, gradients, and stability indicators.
- **D — Full candidate:** all accepted temporal and process features.

The maximum required history is calculated from accepted temporal definitions. The initial structural warm-up is dropped before splitting. Missingness indicators represent genuine sensor gaps after that boundary; median imputation is fitted on training data only.
"""),
code(r"""
raw_feature_names=sensor_columns.copy()
temporal_feature_names=temporal_features.columns.tolist()
process_feature_names=process_features.columns.tolist()
feature_sets_unfiltered={
 "A_raw":raw_feature_names,
 "B_temporal":temporal_feature_names,
 "C_process":raw_feature_names+process_feature_names,
 "D_full":list(dict.fromkeys(temporal_feature_names+process_feature_names)),
}
all_features=pd.concat([temporal_features,process_features],axis=1)
MAX_LOOKBACK=max(MEDIUM_WINDOW_MINUTES,max(max(v) for v in selected_lags.values()))
warmup_log=pd.DataFrame([{"policy":"drop_structural_warmup","rows_dropped":MAX_LOOKBACK,"reason":"History does not exist for accepted lag/rolling features."},
                         {"policy":"preserve_genuine_sensor_missingness","rows_dropped":0,"reason":"Post-warm-up gaps receive indicators and train-fitted median imputation."}])
modeling_base=pd.concat([df[[TIME_COL]],all_features,future_targets],axis=1).iloc[MAX_LOOKBACK:].reset_index(drop=True)

missing_indicator_sources=[c for c in sensor_columns if modeling_base[c].isna().any()]
for c in missing_indicator_sources:
    modeling_base[f"{c}_missing"] = modeling_base[c].isna().astype("int8")
missing_indicator_names=[f"{c}_missing" for c in missing_indicator_sources]
for key in ["B_temporal", "C_process", "D_full"]:
    feature_sets_unfiltered[key]=feature_sets_unfiltered[key]+missing_indicator_names
display(warmup_log)
display(pd.DataFrame([{"feature_set":k,"candidate_count":len(v),"hypothesis":{"A_raw":"Current state alone is sufficient.","B_temporal":"Selected response histories improve forecasts.","C_process":"Physical relationships improve generalization and interpretation.","D_full":"Temporal and physical evidence are complementary."}[k]} for k,v in feature_sets_unfiltered.items()]))
"""),
md(r"""
## 8. Chronological splits and grouped correlation filtering

Filtering is fitted separately for each horizon and feature set using only that training split. Correlation comparisons are limited to features derived from the same original sensor. Process-aware features and missingness indicators are protected. Within a family, interpretability priority is raw value, selected lag, rolling mean, difference, slope, then variability. A removal log records every decision.
"""),
code(r"""
def chronological_indices(n):
    tr=int(n*TRAIN_FRACTION); va=int(n*(TRAIN_FRACTION+VALID_FRACTION))
    return slice(0,tr),slice(tr,va),slice(va,n)

def source_family(feature):
    matches=[c for c in sensor_columns if feature==c or feature.startswith(c+"_")]
    return max(matches,key=len) if matches else None

def interpretability_priority(feature,source):
    if feature==source:return 0
    if "_lag_" in feature:return 1
    if "_roll_mean_" in feature:return 2
    if "_diff_" in feature:return 3
    if "_slope_" in feature:return 4
    if "_roll_std_" in feature:return 5
    return 9

def grouped_correlation_filter(X_train, columns, threshold=CORRELATION_THRESHOLD):
    keep=list(columns); logs=[]
    imputed=pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_train[columns]),columns=columns,index=X_train.index)
    for source in sensor_columns:
        family=[c for c in columns if source_family(c)==source and not c.endswith("_missing")]
        family=sorted(family,key=lambda c:(interpretability_priority(c,source),c))
        retained=[]
        for candidate in family:
            conflicts=[r for r in retained if abs(imputed[candidate].corr(imputed[r]))>threshold]
            if conflicts:
                retained_feature=conflicts[0]; corr=abs(imputed[candidate].corr(imputed[retained_feature]))
                if candidate in keep: keep.remove(candidate)
                logs.append({"removed_feature":candidate,"retained_feature":retained_feature,"correlation":corr,
                             "reason":f"Same {source} family; retained more interpretable representative. Transition semantics should be reviewed."})
            else: retained.append(candidate)
    return keep,pd.DataFrame(logs,columns=["removed_feature","retained_feature","correlation","reason"])

prepared = {}
correlation_logs=[]
split_summaries=[]
for h in HORIZONS:
    ycol=target_name(h)
    horizon_df=modeling_base.dropna(subset=[ycol]).reset_index(drop=True)
    tr,va,te=chronological_indices(len(horizon_df))
    for set_name,candidates in feature_sets_unfiltered.items():
        train=horizon_df.iloc[tr]; valid=horizon_df.iloc[va]; test=horizon_df.iloc[te]
        non_null=[c for c in candidates if not train[c].isna().all()]
        selected,log=grouped_correlation_filter(train,non_null)
        if len(log): log=log.assign(horizon_min=h,feature_set=set_name); correlation_logs.append(log)
        prepared[(h,set_name)]={"train_df":train,"valid_df":valid,"test_df":test,"features":selected,"target":ycol}
        split_summaries.append({"horizon_min":h,"feature_set":set_name,"train_rows":len(train),"valid_rows":len(valid),"test_rows":len(test),"features_before":len(candidates),"features_after":len(selected)})
correlation_removal_log=pd.concat(correlation_logs,ignore_index=True) if correlation_logs else pd.DataFrame()
display(pd.DataFrame(split_summaries))
display(correlation_removal_log.head(50))
"""),
md(r"""
## 9. Train-fitted preprocessing and modeling-ready dictionaries

Outputs are keyed by `(horizon_minutes, feature_set)`. This prevents accidental comparison of mismatched targets or matrices. Tree, robust, standard, and MinMax variants are available; all imputers/scalers are fitted on training data only.
"""),
code(r"""
def make_pipeline(scaler=None):
    steps=[("imputer",SimpleImputer(strategy="median",add_indicator=False))]
    if scaler is not None: steps.append(("scaler",scaler))
    return Pipeline(steps)

modeling_datasets={}
for key,parts in prepared.items():
    cols=parts["features"]; target_col=parts["target"]
    Xtr,Xva,Xte=[parts[n][cols] for n in ["train_df","valid_df","test_df"]]
    ytr,yva,yte=[parts[n][target_col] for n in ["train_df","valid_df","test_df"]]
    variants={"tree":make_pipeline(),"robust":make_pipeline(RobustScaler()),"standard":make_pipeline(StandardScaler()),"minmax":make_pipeline(MinMaxScaler())}
    transformed={}
    for name,pipe in variants.items():
        pipe.fit(Xtr)
        transformed[name]=tuple(pd.DataFrame(pipe.transform(X),columns=cols,index=X.index) for X in [Xtr,Xva,Xte])
    modeling_datasets[key]={"features":cols,"y_train":ytr,"y_valid":yva,"y_test":yte,"pipelines":variants,"X":transformed}

# Backward-compatible default: one-minute, full accepted candidate set.
default=modeling_datasets[(1,"D_full")]
X_train_tree,X_valid_tree,X_test_tree=default["X"]["tree"]
X_train_robust,X_valid_robust,X_test_robust=default["X"]["robust"]
y_train,y_valid,y_test=default["y_train"],default["y_valid"],default["y_test"]
selected_feature_columns=default["features"]
print("Prepared keys:",list(modeling_datasets))
print("Default shapes:",X_train_tree.shape,X_valid_tree.shape,X_test_tree.shape)
"""),
md("## 10. Audit tables and readiness checks"),
code(r"""
all_docs=pd.concat([temporal_docs,process_feature_registry.assign(family="process_aware",source_variable="multiple")[['feature','family','source_variable','expected_interpretation']].rename(columns={'expected_interpretation':'reason'})],ignore_index=True)
feature_set_registry=pd.DataFrame([{"feature_set":k,"feature":c} for k,v in feature_sets_unfiltered.items() for c in v])
retained_feature_registry=pd.DataFrame([
    {"horizon_min":h,"feature_set":set_name,"feature":feature}
    for (h,set_name),parts in prepared.items() for feature in parts["features"]
])
display(all_docs.groupby("family").size().rename("documented_features").reset_index())
print(f"Candidate feature memberships: {len(feature_set_registry):,}")
print(f"Retained feature memberships across all configurations: {len(retained_feature_registry):,}")
"""),
md("## 11. Exported audit artifacts and figures"),
code(r"""
import matplotlib.pyplot as plt

dataset_summary=pd.DataFrame([{"source_file":DATA_PATH.name,"raw_rows":len(df),"numeric_sensors_excluding_target":len(sensor_columns),"missing_target_rows":int(df[TARGET].isna().sum()),"start_timestamp":df[TIME_COL].min(),"end_timestamp":df[TIME_COL].max(),"sampling_interval":"1 minute","coverage":"31 days"}])
variable_registry_summary=variable_registry.groupby(["primary_category","role"],dropna=False).size().rename("variable_count").reset_index()
target_horizon_summary=pd.DataFrame([{"horizon_min":h,"target_column":target_name(h),"usable_rows_after_warmup_and_target_alignment":len(modeling_base.dropna(subset=[target_name(h)])),"unavailable_tail_rows":h,"target_missing_rows_in_usable_period":int(modeling_base[target_name(h)].isna().sum())} for h in HORIZONS])
temporal_family_counts=temporal_docs.groupby("family").size().rename("candidate_count").reset_index()
process_clipping_bounds=pd.DataFrame({"feature":process_features.columns,"lower_0_1pct_training_quantile":process_clip_lower.values,"upper_99_9pct_training_quantile":process_clip_upper.values,"epsilon":EPSILON})

split_boundary_rows=[]; feature_result_rows=[]
for h in HORIZONS:
    hdf=modeling_base.dropna(subset=[target_name(h)]).reset_index(drop=True); tr,va,te=chronological_indices(len(hdf))
    for split_name,sl in [("train",tr),("validation",va),("test",te)]:
        part=hdf.iloc[sl]; split_boundary_rows.append({"horizon_min":h,"split":split_name,"rows":len(part),"start_timestamp":part[TIME_COL].min(),"end_timestamp":part[TIME_COL].max(),"fraction_policy":"70% / 15% / 15%"})
    for set_name,candidates in feature_sets_unfiltered.items():
        final_count=len(prepared[(h,set_name)]["features"])
        feature_result_rows.append({"horizon_min":h,"feature_set":set_name,"candidate_features":len(candidates),"retained_features":final_count,"removed_features":len(candidates)-final_count,"missingness_indicators":sum(c.endswith("_missing") for c in candidates),"train_rows":len(prepared[(h,set_name)]["train_df"]),"validation_rows":len(prepared[(h,set_name)]["valid_df"]),"test_rows":len(prepared[(h,set_name)]["test_df"]),"train_columns":final_count,"validation_columns":final_count,"test_columns":final_count})
split_boundaries=pd.DataFrame(split_boundary_rows); feature_set_results=pd.DataFrame(feature_result_rows)
filtering_summary=feature_set_results[["horizon_min","feature_set","candidate_features","retained_features","removed_features"]].copy()
correlation_removal_log["source_sensor"]=correlation_removal_log["removed_feature"].map(source_family)
feature_set_definitions=pd.DataFrame([
 {"feature_set":"A_raw","contents":"Current sensor measurements only","hypothesis":"The current process state alone contains sufficient predictive information."},
 {"feature_set":"B_temporal","contents":"Raw plus selected lags, rolling levels, variability, difference, slope, and genuine missingness indicators","hypothesis":"Selected process histories and recent dynamics improve forecasting."},
 {"feature_set":"C_process","contents":"Raw plus balances, ratios, gradients, stability proxies, and genuine missingness indicators","hypothesis":"Physically interpretable relationships improve prediction and generalization."},
 {"feature_set":"D_full","contents":"Accepted temporal and process-aware features plus genuine missingness indicators","hypothesis":"Temporal and process-relationship evidence are complementary."}])
preprocessing_variants=pd.DataFrame([
 {"variant":"tree","steps":"training-median imputation; no scaling","intended_models":"tree ensembles and gradient boosting"},
 {"variant":"robust","steps":"training-median imputation; RobustScaler","intended_models":"outlier-aware linear and neural baselines"},
 {"variant":"standard","steps":"training-median imputation; StandardScaler","intended_models":"regularized linear models, PCA, diagnostics"},
 {"variant":"minmax","steps":"training-median imputation; MinMaxScaler","intended_models":"bounded-input neural baselines"}])
limitations=pd.DataFrame([
 {"limitation":"Only one month of one-minute data","validation_requirement":"Test on additional months, seasons, campaigns, and product grades."},
 {"limitation":"Units are absent from the source CSV","validation_requirement":"Confirm every tag unit and calculation basis with process engineering."},
 {"limitation":"Process-aware features are proxies","validation_requirement":"Validate stream pairing, sign conventions, and physical interpretation before operational use."},
 {"limitation":"No confirmed operational thresholds","validation_requirement":"Define warning and intervention thresholds with operations and quality teams."},
 {"limitation":"No variable is confirmed controllable","validation_requirement":"Authorize actuators explicitly before scenario or recommendation use."},
 {"limitation":"Adjacent observations are strongly dependent","validation_requirement":"Use chronological evaluation, persistence baselines, and transition-specific diagnostics."},
 {"limitation":"No predictive benchmark is trained here","validation_requirement":"Compare against persistence and simple baselines at every horizon."}])

def series_match(left, right):
    return np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True)

lag_transformations_verified=all(
    series_match(temporal_features[f"{sensor}_lag_{lag}"],df[sensor].shift(lag))
    for sensor,lags in selected_lags.items() for lag in lags
)
pipeline_training_fit_verified=all(
    np.allclose(
        pipeline.named_steps["imputer"].statistics_,
        parts["train_df"][parts["features"]].median().to_numpy(),
        equal_nan=True,
    )
    for key,parts in prepared.items()
    for pipeline in modeling_datasets[key]["pipelines"].values()
)
selection_cutoff_precedes_validation=all(
    lag_selection_base[TIME_COL].max() < parts["valid_df"][TIME_COL].min()
    for parts in prepared.values()
)

readiness_checks=pd.DataFrame([
 {"check":"chronological_order_preserved_for_every_horizon","status":all(p["train_df"][TIME_COL].max()<p["valid_df"][TIME_COL].min()<p["test_df"][TIME_COL].min() for p in prepared.values())},
 {"check":"four_future_targets_created","status":set(future_targets.columns)==set(target_name(h) for h in HORIZONS)},
 {"check":"target_and_future_targets_absent_from_inputs","status":all(TARGET not in v and not any(c.startswith(TARGET+"_t_plus_") for c in v) for v in feature_sets_unfiltered.values())},
 {"check":"lag_transformations_match_positive_historical_shifts","status":lag_transformations_verified},
 {"check":"all_four_feature_sets_prepared_for_every_horizon","status":len(modeling_datasets)==16},
 {"check":"preprocessor_imputation_statistics_match_training_data","status":pipeline_training_fit_verified},
 {"check":"train_validation_test_columns_aligned","status":all(all(list(t[0].columns)==list(t[1].columns)==list(t[2].columns) for t in d["X"].values()) for d in modeling_datasets.values())},
 {"check":"all_prepared_matrices_have_no_missing_values","status":all(not x.isna().any().any() for d in modeling_datasets.values() for triplet in d["X"].values() for x in triplet)},
 {"check":"selection_cutoff_precedes_every_validation_period","status":selection_cutoff_precedes_validation},
 {"check":"process_clipping_bounds_use_selection_training_period_only","status":process_clip_lower.equals(pd.DataFrame(process_values).loc[:selection_cutoff_index].quantile(.001))},
 {"check":"structural_warmup_removed_before_splitting","status":modeling_base[TIME_COL].min()==df[TIME_COL].iloc[MAX_LOOKBACK]},
 {"check":"registries_and_filtering_log_populated","status":len(variable_registry)>0 and len(process_feature_registry)>0 and len(correlation_removal_log)>0},
 {"check":"no_change_recommendations_authorized","status":not variable_registry.recommendation_allowed.any()},
 {"check":"no_predictive_model_trained_in_notebook","status":True}])

ARTIFACT_DIR=DATA_PATH.resolve().parent/"reports"/"feature_preparation_artifacts"; FIGURE_DIR=ARTIFACT_DIR/"figures"
ARTIFACT_DIR.mkdir(parents=True,exist_ok=True); FIGURE_DIR.mkdir(parents=True,exist_ok=True)
tables={"dataset_summary":dataset_summary,"variable_registry":variable_registry,"variable_registry_summary":variable_registry_summary,"target_horizon_summary":target_horizon_summary,"lag_selection_summary":lag_selection_table,"lag_evidence":lag_evidence,"temporal_family_counts":temporal_family_counts,"feature_documentation":all_docs,"process_feature_registry":process_feature_registry,"process_clipping_bounds":process_clipping_bounds,"warmup_policy":warmup_log,"feature_set_definitions":feature_set_definitions,"feature_set_registry":feature_set_registry,"retained_feature_registry":retained_feature_registry,"split_boundaries":split_boundaries,"filtering_summary":filtering_summary,"correlation_removal_log":correlation_removal_log,"final_dataset_dimensions":feature_set_results,"preprocessing_variants":preprocessing_variants,"limitations":limitations,"readiness_checks":readiness_checks}
for name,table in tables.items(): table.to_csv(ARTIFACT_DIR/f"{name}.csv",index=False)

plt.style.use("seaborn-v0_8-whitegrid")
fig,ax=plt.subplots(figsize=(12,2.8)); ax.axis("off"); boxes=[("Raw data","44,640 rows"),("Registry + targets","4 horizons"),("Lag evidence","train only"),("Feature sets","A / B / C / D"),("Grouped filter","per horizon/set"),("Preprocessing","4 variants")]
for i,(a,b) in enumerate(boxes):
    x=.02+i*.162; ax.add_patch(plt.Rectangle((x,.25),.135,.5,fc="#e8f1f8",ec="#174a6e",lw=1.5)); ax.text(x+.0675,.55,a,ha="center",va="center",fontsize=9,weight="bold"); ax.text(x+.0675,.40,b,ha="center",va="center",fontsize=8)
    if i<len(boxes)-1: ax.annotate("",xy=(x+.158,.5),xytext=(x+.137,.5),arrowprops=dict(arrowstyle="->",color="#174a6e"))
fig.tight_layout(); fig.savefig(FIGURE_DIR/"pipeline_overview.png",dpi=220,bbox_inches="tight"); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4)); ax.bar(temporal_family_counts.family,temporal_family_counts.candidate_count,color="#2878a8"); ax.set_ylabel("Candidate features"); ax.set_title("Compact temporal feature-family composition"); ax.tick_params(axis="x",rotation=30); fig.tight_layout(); fig.savefig(FIGURE_DIR/"feature_family_composition.png",dpi=220); plt.close(fig)
lag_counts=lag_evidence[lag_evidence.apply(lambda r:int(r.lag_min) in selected_lags[r.variable],axis=1)].groupby("lag_min").size(); fig,ax=plt.subplots(figsize=(7,4)); ax.bar(lag_counts.index.astype(str),lag_counts.values,color="#eb8f34"); ax.set(xlabel="Selected lag (minutes)",ylabel="Selections across sensors",title="Selected-lag distribution"); fig.tight_layout(); fig.savefig(FIGURE_DIR/"selected_lag_distribution.png",dpi=220); plt.close(fig)
plot_counts=filtering_summary.copy(); plot_counts["configuration"]=plot_counts.horizon_min.astype(str)+"m "+plot_counts.feature_set.str.replace("_"," "); fig,ax=plt.subplots(figsize=(12,5)); x=np.arange(len(plot_counts)); ax.bar(x-.2,plot_counts.candidate_features,.4,label="candidate"); ax.bar(x+.2,plot_counts.retained_features,.4,label="retained"); ax.set_xticks(x,plot_counts.configuration,rotation=55,ha="right"); ax.set_ylabel("Features"); ax.legend(); ax.set_title("Features before and after grouped filtering"); fig.tight_layout(); fig.savefig(FIGURE_DIR/"feature_counts_before_after.png",dpi=220); plt.close(fig)
set_plot=feature_set_results.groupby("feature_set").agg(candidate=("candidate_features","first"),retained=("retained_features","mean")).reset_index(); fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(4); ax.bar(x-.2,set_plot.candidate,.4,label="candidate"); ax.bar(x+.2,set_plot.retained,.4,label="mean retained"); ax.set_xticks(x,set_plot.feature_set); ax.set_ylabel("Features"); ax.legend(); ax.set_title("Feature-set complexity comparison"); fig.tight_layout(); fig.savefig(FIGURE_DIR/"feature_set_comparison.png",dpi=220); plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4)); colors={"train":"#2878a8","validation":"#eb8f34","test":"#4a9d5b"}
for yi,h in enumerate(HORIZONS):
    for _,r in split_boundaries[split_boundaries.horizon_min==h].iterrows():
        start=pd.Timestamp(r.start_timestamp); end=pd.Timestamp(r.end_timestamp); ax.barh(yi,(end-start).total_seconds()/86400,left=(start-df[TIME_COL].min()).total_seconds()/86400,color=colors[r.split],label=r.split if yi==0 else None)
ax.set_yticks(range(4),[f"h={h} min" for h in HORIZONS]); ax.set_xlabel("Days since dataset start"); ax.set_title("Chronological split timeline by horizon"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURE_DIR/"split_timeline.png",dpi=220); plt.close(fig)
sources=sorted(set(v for s in process_feature_registry.involved_variables.str.split("; ") for v in s)); incidence=np.zeros((len(process_feature_registry),len(sources)))
for i,s in enumerate(process_feature_registry.involved_variables.str.split("; ")):
    for v in s: incidence[i,sources.index(v)]=1
fig,ax=plt.subplots(figsize=(14,8)); ax.imshow(incidence,aspect="auto",cmap="Blues",vmin=0,vmax=1); ax.set_yticks(range(len(process_feature_registry)),process_feature_registry.feature,fontsize=7); ax.set_xticks(range(len(sources)),sources,rotation=65,ha="right",fontsize=7); ax.set_title("Process-aware feature-to-source-sensor map"); fig.tight_layout(); fig.savefig(FIGURE_DIR/"process_feature_map.png",dpi=220); plt.close(fig)

expected_figure_names=["pipeline_overview.png","feature_family_composition.png","selected_lag_distribution.png","feature_counts_before_after.png","feature_set_comparison.png","split_timeline.png","process_feature_map.png"]
expected_files=[ARTIFACT_DIR/f"{name}.csv" for name in tables]+[FIGURE_DIR/name for name in expected_figure_names]
readiness_checks.loc[len(readiness_checks)]={"check":"all_expected_exported_artifacts_exist","status":all(p.exists() for p in expected_files)}
readiness_checks.to_csv(ARTIFACT_DIR/"readiness_checks.csv",index=False)
display(readiness_checks); assert readiness_checks.status.all(),readiness_checks.loc[~readiness_checks.status]
print(f"Exported {len(tables)} tables and 7 figures to {ARTIFACT_DIR}")
"""),
md(r"""
## 12. Handoff to modeling

Use `modeling_datasets[(horizon, feature_set)]`, where horizons are `1`, `5`, `10`, and `15`, and feature sets are `A_raw`, `B_temporal`, `C_process`, and `D_full`. Compare the same chronological folds and metrics at every horizon. Report predictive accuracy alongside warning lead time and operational usefulness.

Interpretation safeguards:

- lag evidence is predictive association, not proof of response or causality;
- process proxies must retain their validation labels in reports;
- no what-if recommendation may alter a variable until the supervisor changes its registry status to `confirmed_controllable`;
- correlations discarded inside a source family remain logged and should be revisited if transition behavior is a research focus.
"""),
]


def split_into_subsections(source, subsection_markers):
    """Split one generated code cell into ordered, single-purpose cells."""
    positions = []
    for title, marker in subsection_markers:
        position = source.find(marker)
        if position < 0:
            raise ValueError(f"Notebook organization marker not found: {marker!r}")
        positions.append((position, title))
    positions.sort()

    expanded = []
    for index, (start, title) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(source)
        fragment = source[start:end].strip()
        guidance = OBSERVATION_GUIDANCE.get(title, "")
        heading = f"### {title}" + (f"\n\n{guidance}" if guidance else "")
        expanded.extend([md(heading), code(fragment)])
    return expanded


OBSERVATION_GUIDANCE = {
    "2.3 Observations: registry coverage and intervention safety": "Review category coverage and confirm that `recommendation_allowed` remains false until a process supervisor explicitly validates an actuator.",
    "3.2 Observations: target alignment at the beginning and end": "The head confirms direct alignment at `t+h`; the tail shows why each longer horizon loses additional rows rather than imputing unavailable future targets.",
    "4.5 Observations: selected delays and supporting evidence": "Interpret the selected delays as stable predictive associations, not causal response times. Compare statistical evidence with the recorded process prior.",
    "5.3 Observations: temporal feature composition": "Confirm that the compact generator creates only raw state, selected delays, two operating levels, one variability measure, one movement measure, and one trend per sensor.",
    "6.5 Observations: process-feature validation registry": "Treat every proxy according to its validation status. Unit-, stream-, and engineering-validation requirements must remain visible in downstream interpretation.",
    "7.4 Observations: feature-set hypotheses and candidate sizes": "Use these four sets as an ablation design. Increasing feature count is not itself evidence of improved forecasting.",
    "8.5 Observations: filtering results and removal examples": "Check both the summary and removal log. Highly correlated transition-sensitive features may merit restoration in a later targeted experiment.",
    "9.4 Observations: prepared experiment dictionary": "Verify that all 16 horizon-by-feature-set keys exist and that the backward-compatible default is only a convenience alias.",
    "10.3 Observations: documentation coverage": "Candidate and retained registries make every feature membership auditable across experimental configurations.",
    "11.6 Observations: verify the exact artifact manifest": "All executable checks must pass and every expected CSV or figure must exist before results are used by the report or modeling stage.",
}


SUBSECTION_LAYOUT = {
    "from pathlib import Path": [
        ("1.1 Imports and reproducibility", "from pathlib import Path"),
        ("1.2 Central configuration and helper functions", "warnings.filterwarnings"),
        ("1.3 Load, sort, and validate the source data", "raw_df = pd.read_csv"),
    ],
    "registry_rows = [": [
        ("2.1 Declare the process-variable registry", "registry_rows = ["),
        ("2.2 Derive governance and interpretation fields", "variable_registry = pd.DataFrame"),
        ("2.3 Observations: registry coverage and intervention safety", "display(variable_registry)"),
    ],
    "future_targets = pd.DataFrame": [
        ("3.1 Construct the four direct future targets", "future_targets = pd.DataFrame"),
        ("3.2 Observations: target alignment at the beginning and end", "display(future_targets.head())"),
    ],
    "RESPONSE_PRIORS = {": [
        ("4.1 Define candidate delays and process-response priors", "RESPONSE_PRIORS = {"),
        ("4.2 Establish a conservative training-only selection boundary", "selection_reference = pd.DataFrame"),
        ("4.3 Score one sensor across candidate lags", "def lag_evidence_for_sensor"),
        ("4.4 Select at most three stable lags per sensor", "lag_evidence = pd.concat"),
        ("4.5 Observations: selected delays and supporting evidence", "display(lag_selection_table)"),
    ],
    "def rolling_slope": [
        ("5.1 Define the rolling-slope calculation", "def rolling_slope"),
        ("5.2 Build the compact temporal feature family", "def build_temporal_features"),
        ("5.3 Observations: temporal feature composition", "temporal_features, temporal_docs"),
    ],
    "def guarded_ratio": [
        ("6.1 Define guarded numerical operations", "def guarded_ratio"),
        ("6.2 Create balances, ratios, gradients, and operating proxies", "process_values = {}"),
        ("6.3 Add normalized flow-stability indicators", "important_flows=["),
        ("6.4 Apply training-only clipping and document assumptions", "process_features=pd.DataFrame"),
        ("6.5 Observations: process-feature validation registry", "display(process_feature_registry)"),
    ],
    "raw_feature_names=sensor_columns.copy()": [
        ("7.1 Define the four experimental feature sets", "raw_feature_names=sensor_columns.copy()"),
        ("7.2 Remove structurally unavailable history", "all_features=pd.concat"),
        ("7.3 Add indicators for genuine post-warm-up sensor gaps", "missing_indicator_sources="),
        ("7.4 Observations: feature-set hypotheses and candidate sizes", "display(warmup_log)"),
    ],
    "def chronological_indices": [
        ("8.1 Define chronological split boundaries", "def chronological_indices"),
        ("8.2 Define source families and interpretability priority", "def source_family"),
        ("8.3 Filter correlation within source-variable families", "def grouped_correlation_filter"),
        ("8.4 Apply filtering independently to every experiment", "prepared = {}"),
        ("8.5 Observations: filtering results and removal examples", "display(pd.DataFrame(split_summaries))"),
    ],
    "def make_pipeline": [
        ("9.1 Define reusable preprocessing pipelines", "def make_pipeline"),
        ("9.2 Fit four preprocessing variants per experiment", "modeling_datasets={}"),
        ("9.3 Expose the backward-compatible default experiment", "# Backward-compatible default"),
        ("9.4 Observations: prepared experiment dictionary", "print(\"Prepared keys:\""),
    ],
    "all_docs=pd.concat": [
        ("10.1 Combine statistical and process-feature documentation", "all_docs=pd.concat"),
        ("10.2 Register retained features by horizon and feature set", "retained_feature_registry=pd.DataFrame"),
        ("10.3 Observations: documentation coverage", "display(all_docs.groupby"),
    ],
    "import matplotlib.pyplot as plt": [
        ("11.1 Import reporting dependencies", "import matplotlib.pyplot as plt"),
        ("11.2 Assemble executed audit tables", "dataset_summary=pd.DataFrame"),
        ("11.3 Build executable integrity checks", "def series_match"),
        ("11.4 Export machine-readable tables", "ARTIFACT_DIR="),
        ("11.5 Generate analytical figures", "plt.style.use"),
        ("11.6 Observations: verify the exact artifact manifest", "expected_figure_names="),
    ],
}


organized_cells = []
for cell in cells:
    if cell["cell_type"] != "code":
        organized_cells.append(cell)
        continue
    source = cell["source"].strip()
    layout = next((value for prefix, value in SUBSECTION_LAYOUT.items() if source.startswith(prefix)), None)
    if layout is None:
        organized_cells.append(cell)
    else:
        organized_cells.extend(split_into_subsections(source, layout))

cells = organized_cells

notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","pygments_lexer":"ipython3"}},"nbformat":4,"nbformat_minor":5}
OUT.write_text(json.dumps(notebook,indent=1,ensure_ascii=False),encoding="utf-8")
print(f"Wrote {OUT} with {len(cells)} cells.")
