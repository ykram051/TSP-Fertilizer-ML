import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "reports" / "feature_preparation_artifacts"
OUT = ROOT / "reports" / "feature_preparation_report.tex"


def rows(name):
    with (ART / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def esc(value):
    s = str(value)
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                     ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}")]:
        s = s.replace(old, new)
    return s


def code(value):
    return r"\nolinkurl{" + str(value).replace("%", r"\%") + "}"


def fnum(value, digits=3):
    return f"{float(value):.{digits}f}"


dataset = rows("dataset_summary")[0]
registry_summary = rows("variable_registry_summary")
targets = rows("target_horizon_summary")
lags = rows("lag_selection_summary")
temporal = rows("temporal_family_counts")
process = rows("process_feature_registry")
warmup = rows("warmup_policy")
feature_defs = rows("feature_set_definitions")
splits = rows("split_boundaries")
filtering = rows("filtering_summary")
dimensions = rows("final_dataset_dimensions")
preprocessing = rows("preprocessing_variants")
checks = rows("readiness_checks")
limitations = rows("limitations")
correlations = rows("correlation_removal_log")[:8]


def tabular(headers, body, spec, caption, label=None, size=r"\small"):
    label_text = f"\\label{{{label}}}" if label else ""
    lines = [r"\begin{table}[H]", r"\centering", size, f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}", f"\\caption{{{caption}}}{label_text}", r"\end{table}"]
    return "\n".join(lines)


parts = [r"""
\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{booktabs,longtable,array,tabularx,makecell}
\usepackage{amsmath,amssymb}
\usepackage{graphicx,float,placeins}
\usepackage{xcolor,hyperref,url}
\usepackage{enumitem,pdflscape}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\definecolor{navy}{HTML}{174A6E}
\definecolor{lightblue}{HTML}{E8F1F8}
\definecolor{passgreen}{HTML}{237A3B}
\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=navy}
\setlist{nosep,leftmargin=1.5em}
\renewcommand{\arraystretch}{1.15}
\newcommand{\decision}{\paragraph{Decision.}}
\newcommand{\implementation}{\paragraph{Implementation.}}
\newcommand{\actual}{\paragraph{Actual output.}}
\newcommand{\modelmeaning}{\paragraph{Modeling meaning.}}
\newcommand{\limitation}{\paragraph{Limitation.}}
\title{\textbf{Feature Preparation and Experimental Design Report\\for \texttt{SLURRY\_FREE\_ACID} Forecasting}}
\author{TSP Fertilizer Process Analytics Project}
\date{Executed notebook basis: 16 July 2026}
\begin{document}
\maketitle
\begin{abstract}
This report documents the fully executed \texttt{02\_feature\_engineering\_data\_preparation.ipynb}. The preparation layer now supports four direct forecasting horizons and four hypothesis-driven feature sets. It does not train a predictive model. All numerical claims, tables, and figures in this report are derived from the notebook's exported audit artifacts after a clean, top-to-bottom execution.
\end{abstract}
\tableofcontents
\clearpage

\section{Executive decision and scientific purpose}
The preparation stage is no longer intended to maximize matrix width. Its purpose is to establish a controlled experiment that can test: (1) whether current measurements suffice; (2) whether selected histories add value; (3) whether process relationships improve prediction and interpretation; (4) whether temporal and physical evidence are complementary; and (5) how predictability degrades as warning time increases.

The direct targets are
\[
y_{t+h}=\mathrm{SLURRY\_FREE\_ACID}_{t+h},\qquad h\in\{1,5,10,15\}\ \text{minutes}.
\]
Every input attached to timestamp $t$ uses only measurements available at $t$ or earlier. Feature-set ablation and horizon degradation are therefore first-class experimental questions for the next stage.

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{feature_preparation_artifacts/figures/pipeline_overview.png}
\caption{Executed feature-preparation pipeline.}\end{figure}

\section{Dataset and execution basis}
\decision The notebook uses one chronologically ordered one-minute dataset and excludes the target from the 21 sensor inputs.
\implementation Timestamps are parsed as UTC and sorted before target or feature construction. The notebook was regenerated, executed in a fresh Python process, and saved with execution counts for every code cell.
\actual The execution completed all 11 code cells, exported 21 CSV audit tables and seven figures, and passed all readiness assertions.
\modelmeaning The report is traceable to machine-readable artifacts rather than copied source-code intentions.
\limitation A complete month is available, but one month cannot establish seasonal, campaign, or product-grade robustness.
"""]

parts.append(tabular(["Item", "Executed value"], [
    ["Source", code(dataset["source_file"])], ["Raw rows", f"{int(dataset['raw_rows']):,}"],
    ["Input sensors", dataset["numeric_sensors_excluding_target"]], ["Missing target observations", dataset["missing_target_rows"]],
    ["Timestamp range", esc(dataset["start_timestamp"] + " to " + dataset["end_timestamp"])],
    ["Sampling / coverage", esc(dataset["sampling_interval"] + "; " + dataset["coverage"])],
], r"p{0.31\textwidth}p{0.62\textwidth}", "Dataset and timestamp summary.", "tab:dataset"))

parts.append(r"""
\section{Process-variable registry and intervention safety}
\decision Every source variable is registered before engineering, with a primary category, category tags, role, controllability status, measurement type, physical interpretation, interpretation status, unit-validation status, and recommendation flag.
\implementation Candidate actuator tags are labelled \texttt{needs\_supervisor\_confirmation}; they are not silently promoted to controllable. \texttt{recommendation\_allowed} becomes true only for \texttt{confirmed\_controllable} variables.
\actual No variable is currently confirmed controllable and no change recommendation is authorized. Units are not supplied by the source CSV. Two tags have uncertain roles and two flow-related interpretations require process confirmation.
\modelmeaning The registry supports grouping, explanations, scenario design, and a hard safety boundary between observed state and authorized intervention.
\limitation Registry entries are engineering hypotheses until tag dictionaries, units, locations, and actuator authority are reviewed by the supervisor.
""")
parts.append(tabular(["Primary category", "Role", "Variables"], [[code(r["primary_category"]), code(r["role"]), r["variable_count"]] for r in registry_summary], r"p{0.25\textwidth}p{0.48\textwidth}r", "Variable-registry summary by category and role."))

parts.append(r"""
\section{Multi-horizon target construction}
\decision Four direct future targets are retained to compare accuracy, warning usefulness, reaction time, and horizon degradation.
\implementation For horizon $h$, the target is created with a negative $h$-row shift. Future target values are never features. Rows without a valid future target are removed independently for each horizon.
\actual Longer horizons have fewer usable rows because more observations at the end have no future label; ordinary missing target readings also remain excluded.
\modelmeaning A highly accurate one-minute model can now be compared with less accurate but operationally earlier warnings.
\limitation These are direct targets sharing the same input definitions; horizon-specific feature selection may still be worth testing later under nested validation.
""")
parts.append(tabular(["Horizon", "Target column", "Usable rows", "Unavailable tail", "Missing targets"], [[r["horizon_min"]+" min", code(r["target_column"]), f"{int(r['usable_rows_after_warmup_and_target_alignment']):,}", r["unavailable_tail_rows"], r["target_missing_rows_in_usable_period"]] for r in targets], r"rp{0.38\textwidth}rrr", "Target-horizon summary after structural warm-up removal."))

parts.append(r"""
\section{Training-only selected-lag methodology}
\decision Uniform lag generation was replaced with evidence-based selection capped at three lags per sensor.
\implementation Candidate lags are 1, 5, 10, 15, 30, and 60 minutes. Against the one-minute future target, the raw training period only is used to calculate absolute lagged Pearson correlation, mutual information, and correlation stability over three chronological training segments. Correlation and mutual-information scores are min-max normalized, averaged, multiplied by stability, and augmented by a 0.15 response-prior bonus. The top three scores are retained.
\actual Sixty-three lags were selected. Twenty sensors retained 1, 5, and 10 minutes; \texttt{TEMPERATURE\_CUVE\_ATTAQUE} retained 5, 10, and 15 minutes.
\modelmeaning Selected lags represent plausible response delays supported by both association and response-time assumptions.
\limitation This evidence is predictive, not causal. The common lag dictionary was selected against $t+1$ for controlled feature-set comparisons and may not be optimal for every horizon.
""")
lag_body=[]
for r in lags:
    evidence=f"|r|={fnum(r['mean_abs_cross_correlation'])}; MI={fnum(r['mean_mutual_information'])}; stability={fnum(r['mean_stability'])}"
    lag_body.append([code(r["variable"]),esc(r["selected_lags_min"]),esc(evidence),esc(r["process_prior_selected"]),esc(r["reason"])])
parts += [r"\begin{landscape}\scriptsize\begin{longtable}{p{0.24\linewidth}p{0.09\linewidth}p{0.24\linewidth}p{0.09\linewidth}p{0.27\linewidth}}\caption{Selected-lag evidence and rationale.}\\\toprule Source variable & Selected lags & Statistical evidence (selected-lag means) & Prior hits & Final reason \\\midrule\endfirsthead\toprule Source variable & Selected lags & Statistical evidence & Prior hits & Final reason \\\midrule\endhead" + "\n" + "\n".join(" & ".join(row)+r" \\" for row in lag_body) + r"\bottomrule\end{longtable}\end{landscape}"]
parts.append(r"\begin{figure}[H]\centering\includegraphics[width=.72\textwidth]{feature_preparation_artifacts/figures/selected_lag_distribution.png}\caption{Distribution of the 63 selected lags.}\end{figure}")

parts.append(r"""
\section{Compact temporal engineering}
\decision Each sensor receives a small, consistent temporal description instead of every statistic at every window.
\implementation The executed families are current value; three selected lags; 10- and 30-minute trailing means; 10-minute trailing standard deviation; five-minute difference; and 10-minute rolling slope. Windows are trailing and require complete history.
\actual The temporal candidate block contains 189 features before missingness indicators: 21 raw, 63 selected lags, 42 means, and 21 each for variability, difference, and trend.
\modelmeaning Raw values encode present state; lags encode delayed response; means encode recent level; standard deviation encodes instability; differences encode movement; slopes encode direction.
\limitation The same summary windows are used across sensors after lag selection. Later ablation should test whether every retained family earns its complexity.
""")
parts.append(tabular(["Family", "Candidates", "Modeling hypothesis"], [[code(r["family"]),r["candidate_count"],esc({"raw":"current process state","selected_lag":"delayed process response","rolling_mean":"recent operating level","variability":"short-term instability","difference":"recent movement","trend":"short-term direction"}[r["family"]])] for r in temporal], r"p{0.25\textwidth}r p{0.53\textwidth}", "Temporal-feature family counts from the executed notebook."))
parts.append(r"\begin{figure}[H]\centering\includegraphics[width=.72\textwidth]{feature_preparation_artifacts/figures/feature_family_composition.png}\caption{Compact temporal candidate composition.}\end{figure}")

parts.append(r"""
\section{Process-aware features and validation metadata}
\decision Physical relationships are explicit experimental features, never undocumented interactions.
\implementation Each derived feature stores its exact formula, source tags, expected interpretation, assumptions, and validation status. Five coefficient-of-variation indicators use a 10-minute standard deviation divided by the magnitude of the 10-minute mean.
\actual Sixteen process-aware features were created: eleven balances, totals, ratios, gradients, or operating proxies, plus five flow-stability indicators. The total slurry volumetric flow is a confirmed m$^3$/h aggregation; the total acid aggregation is confirmed while its unit remains pending; eleven features remain provisional and three require mass-stream, unit, or engineering validation.
\modelmeaning These features test whether engineering relationships generalize better and explain more clearly than sensor-by-sensor statistics.
\limitation None is a verified physical quantity solely because it appears in this registry. In particular, pressure times temperature is a steam-condition model proxy, not energy or enthalpy.
""")
process_body=[[code(r["feature"]),code(r["formula"]),code(r["involved_variables"]),esc(r["expected_interpretation"]),esc(r["assumptions"]),code(r["validation_status"])] for r in process]
parts += [r"\begin{landscape}\tiny\begin{longtable}{p{0.17\linewidth}p{0.19\linewidth}p{0.20\linewidth}p{0.14\linewidth}p{0.18\linewidth}p{0.10\linewidth}}\caption{Complete process-aware feature registry.}\\\toprule Feature & Formula & Inputs & Expected interpretation & Assumptions & Status \\\midrule\endfirsthead\toprule Feature & Formula & Inputs & Interpretation & Assumptions & Status \\\midrule\endhead" + "\n" + "\n".join(" & ".join(row)+r" \\" for row in process_body) + r"\bottomrule\end{longtable}\end{landscape}"]
parts.append(r"\begin{figure}[H]\centering\includegraphics[width=.98\textwidth]{feature_preparation_artifacts/figures/process_feature_map.png}\caption{Process-aware features linked to their source sensors.}\end{figure}")

parts.append(r"""
\section{Guarded ratios, clipping, and numerical stability}
\decision Unbounded percentage changes are not generated because several sensors approach or cross zero.
\implementation The configured epsilon is $10^{-6}$. Guarded ratios are missing when $|d|\leq\epsilon$; otherwise the signed denominator receives a signed epsilon adjustment. Infinite values are replaced by missing values. Each process feature is clipped to its own 0.1st and 99.9th percentile bounds learned from the raw training period only, then applied unchanged to later periods.
\actual Sixteen feature-specific clipping pairs are exported in \texttt{process\_clipping\_bounds.csv}. The readiness audit confirms that these bounds use training data only.
\modelmeaning The controls limit near-zero explosions and extreme leverage without using validation/test distributions.
\limitation Quantile clipping is statistical protection, not a substitute for engineering plausibility limits. Negative source values and very large drying-proxy bounds indicate that sensor-quality and shutdown regimes need explicit review.

\section{Structural warm-up versus genuine missingness}
\decision Structurally unavailable history is removed, not median-imputed.
\implementation The maximum accepted lookback is 30 minutes (the largest selected lag is 15 minutes, but the medium rolling mean requires 30). Rows 0--29 are dropped before splitting. After this boundary, genuine gaps in all 21 sensor tags receive binary indicators in Sets B--D and median imputation fitted on training only. Set A remains strictly raw and contains no indicators.
\actual Exactly 30 rows were removed. The usable feature timeline begins at 2026-01-01 00:30 UTC. Twenty-one genuine-missingness indicators are available in Sets B, C, and D.
\modelmeaning Structural absence and sensor failure now have different treatments and interpretations.
\limitation A binary indicator identifies missingness but does not distinguish maintenance, communications failure, or process shutdown.
""")
parts.append(tabular(["Policy", "Rows", "Reason"], [[code(r["policy"]),r["rows_dropped"],esc(r["reason"])] for r in warmup], r"p{0.30\textwidth}r p{0.55\textwidth}", "Warm-up and genuine-missingness policy."))

parts.append(r"""
\section{Four feature-set hypotheses}
\decision Four named input configurations replace one universal matrix.
\implementation Every configuration is independently filtered and preprocessed for every horizon, producing 16 experimental keys.
\actual Candidate widths are 21 (A), 210 (B), 58 (C), and 226 (D). Set A has no missing indicators; Sets B--D each include 21.
\modelmeaning Direct ablation can quantify predictive gain against complexity and physical interpretability.
\limitation Set C includes raw measurements as specified; it tests incremental process relationships rather than process features in isolation.
""")
parts.append(tabular(["Set", "Contents", "Hypothesis"], [[code(r["feature_set"]),esc(r["contents"]),esc(r["hypothesis"])] for r in feature_defs], r"p{0.12\textwidth}p{0.38\textwidth}p{0.42\textwidth}", "Feature-set definitions and hypotheses.", size=r"\footnotesize"))
parts.append(r"\begin{figure}[H]\centering\includegraphics[width=.72\textwidth]{feature_preparation_artifacts/figures/feature_set_comparison.png}\caption{Candidate and mean retained feature counts by set.}\end{figure}")

parts.append(r"""
\section{Chronological splitting by horizon}
\decision All configurations use a 70/15/15 chronological policy without shuffling.
\implementation Target-missing rows are removed separately by horizon before integer split boundaries are calculated.
\actual Split endpoints differ by minutes as the horizon increases. There is no overlap: training precedes validation, which precedes test.
\modelmeaning Evaluation mimics forward deployment and prevents adjacent future observations from leaking backward through random allocation.
\limitation One holdout month and highly dependent one-minute samples do not quantify seasonal uncertainty; rolling-origin validation should follow.
""")
split_body=[[r["horizon_min"],esc(r["split"]),f"{int(r['rows']):,}",esc(r["start_timestamp"]),esc(r["end_timestamp"])] for r in splits]
parts.append(tabular(["Horizon", "Split", "Rows", "Start (UTC)", "End (UTC)"], split_body, r"rrr p{0.27\textwidth}p{0.27\textwidth}", "Exact chronological boundaries by target horizon.", size=r"\footnotesize"))
parts.append(r"\begin{figure}[H]\centering\includegraphics[width=.88\textwidth]{feature_preparation_artifacts/figures/split_timeline.png}\caption{Chronological split timeline by horizon.}\end{figure}")

parts.append(r"""
\section{Grouped, interpretability-aware correlation filtering}
\decision Correlation filtering is limited to features from the same original sensor; process-aware features and missingness indicators are protected.
\implementation The filter is fitted separately on the training matrix for each horizon and set at $|r|>0.98$. Priority is: (1) raw value, (2) selected lag, (3) rolling mean, (4) difference, (5) slope, and (6) variability. Each removal records both features, absolute correlation, source sensor, and reason.
\actual Sets A and C lose no features. Sets B and D each remove 48 and retain 162 and 178 respectively at every horizon.
\modelmeaning High correlation within a source family can be reduced while semantically distinct balances, ratios, gradients, and stability proxies remain available for ablation.
\limitation During transitions, a raw value and lag may behave differently despite high full-period correlation. The log should be revisited for transition-focused models.
""")
parts.append(tabular(["Horizon", "Set", "Candidate", "Retained", "Removed"], [[r["horizon_min"],code(r["feature_set"]),r["candidate_features"],r["retained_features"],r["removed_features"]] for r in filtering], "rrrrr", "Grouped filtering results by horizon and feature set."))
corr_body=[[code(r["removed_feature"]),code(r["retained_feature"]),fnum(r["correlation"],4),code(r["source_sensor"]),"Same family; retained more interpretable representative"] for r in correlations]
parts += [r"\begin{landscape}\scriptsize\begin{longtable}{p{0.25\linewidth}p{0.25\linewidth}r p{0.23\linewidth}p{0.20\linewidth}}\caption{Representative correlation-removal decisions (one-minute temporal set).}\\\toprule Removed & Retained & $|r|$ & Source sensor & Reason \\\midrule" + "\n" + "\n".join(" & ".join(row)+r" \\" for row in corr_body) + r"\bottomrule\end{longtable}\end{landscape}"]
parts.append(r"\begin{figure}[H]\centering\includegraphics[width=.98\textwidth]{feature_preparation_artifacts/figures/feature_counts_before_after.png}\caption{Features before and after grouped filtering for all 16 configurations.}\end{figure}")

parts.append(r"""
\section{Final dimensions and preprocessing variants}
\decision Each horizon--set pair has four preprocessing variants.
\implementation Median imputers and any scaler are fitted on training only. Validation and test are transformed without refitting and retain identical ordered columns.
\actual The artifact dictionary contains 16 experiment keys and four variants per key: unscaled tree, robust, standard, and MinMax. No post-transform missing values remain.
\modelmeaning Algorithm comparisons can use appropriate scaling without changing data partitions or feature definitions.
\limitation MinMax values can leave the training range in future operation; this must be monitored rather than silently clipped.
""")
dim_body=[[r["horizon_min"],code(r["feature_set"]),r["candidate_features"],r["retained_features"],r["missingness_indicators"],f"{int(r['train_rows']):,} x {r['train_columns']}",f"{int(r['validation_rows']):,} x {r['validation_columns']}",f"{int(r['test_rows']):,} x {r['test_columns']}"] for r in dimensions]
parts += [r"\begin{landscape}\footnotesize\begin{longtable}{rrr r r rrr}\caption{Final matrix dimensions; cells show rows $\times$ columns.}\\\toprule Horizon & Set & Candidate & Retained & Indicators & Train & Validation & Test \\\midrule\endfirsthead\toprule Horizon & Set & Candidate & Retained & Indicators & Train & Validation & Test \\\midrule\endhead" + "\n" + "\n".join(" & ".join(row)+r" \\" for row in dim_body) + r"\bottomrule\end{longtable}\end{landscape}"]
parts.append(tabular(["Variant", "Transformation", "Intended model families"], [[code(r["variant"]),esc(r["steps"]),esc(r["intended_models"])] for r in preprocessing], r"p{0.13\textwidth}p{0.37\textwidth}p{0.42\textwidth}", "Preprocessing variants created for every horizon and feature set."))

parts.append(r"""
\section{Readiness audit}
The notebook's final assertions verify execution integrity, time order, leakage boundaries, configuration completeness, fitted preprocessing, column alignment, missing-value removal, populated registries/logs, intervention safety, artifact exports, and the absence of model training. ``Current or past only'' is an implementation invariant: target shifts are confined to target creation; sensor lag operations use positive shifts and trailing windows.
""")
parts.append(tabular(["Validation check", "Result"], [[code(r["check"]),r"\textcolor{passgreen}{\textbf{PASS}}" if r["status"]=="True" else r"\textcolor{red}{FAIL}"] for r in checks], r"p{0.78\textwidth}p{0.12\textwidth}", "Executed readiness checks.", size=r"\footnotesize"))

parts.append(r"""
\section{Limitations and engineering validation requirements}
The following limitations are explicit release conditions, not footnotes.
""")
parts.append(tabular(["Limitation", "Required validation"], [[esc(r["limitation"]),esc(r["validation_requirement"])] for r in limitations], r"p{0.38\textwidth}p{0.54\textwidth}", "Limitations and required engineering validation.", size=r"\footnotesize"))
parts.append(r"""
Additional constraints are that operational warning thresholds are not defined; adjacent samples are dependent; persistence has not yet been benchmarked; tag units and stream locations remain uncertain; and product-grade or seasonal coverage cannot be inferred from January alone. Autoregressive target-history features are intentionally absent from these exogenous sets. If target history is available at deployment, it should be tested as a separate configuration to avoid obscuring the value of process inputs.

\section{Implications for the next modeling stage}
The next stage should not begin with the most complex algorithm. It should use the fixed experiment keys to test:
\begin{enumerate}
\item every model against naive persistence and simple mean/linear baselines;
\item degradation from 1 to 5, 10, and 15 minutes alongside operational warning value;
\item A versus B versus C versus D under identical chronological folds;
\item exogenous-only versus separately defined autoregressive configurations when target history is available;
\item stable operation versus transitions, shutdowns, restarts, and abnormal excursions;
\item predictive gain versus feature count, engineering validation burden, latency, and explanation quality.
\end{enumerate}
Performance reporting should include average error, tail/transition error, calibration or interval coverage if uncertainty is modeled, and stability across rolling-origin folds.

\section{Change log from the previous report}
\begin{itemize}
\item Replaced the one-minute-only framing with direct 1-, 5-, 10-, and 15-minute targets.
\item Replaced blanket lag and rolling generation with training-only selected lags and compact temporal families.
\item Removed the obsolete 799-to-346 feature narrative; current candidates are 21, 210, 58, and 226 by feature set, with 21, 162, 58, and 178 retained.
\item Replaced one universal feature matrix with 16 horizon--feature-set configurations and four preprocessing variants each.
\item Replaced global correlation filtering with grouped, training-only, interpretability-prioritized filtering and a removal log.
\item Replaced structural warm-up median imputation with removal of the first 30 rows, followed by indicators and train-fitted imputation for genuine gaps.
\item Added variable and process-feature registries, explicit controllability safeguards, process validation statuses, training-only clipping bounds, and exported artifacts.
\item Removed the recommendation to proceed immediately to sequence-model training. The correct next step is controlled benchmarking and process understanding.
\end{itemize}

\section{Recommended next deliverable}
The next stage should be implemented as:
\begin{center}
\Large\textbf{\texttt{03\_prediction\_benchmark\_and\_process\_understanding.ipynb}}
\end{center}
It should preserve the exported experiment definitions, begin with persistence and simple baselines, and add complexity only when out-of-time evidence justifies it.

\appendix
\section{Executed artifact inventory}
The report is backed by CSV files in \texttt{reports/feature\_preparation\_artifacts/}: dataset summary, complete variable registry and summary, horizon summary, lag evidence and selection, temporal counts, process registry and clipping bounds, warm-up policy, feature-set definitions and membership, split boundaries, filtering summary and removal log, final dimensions, preprocessing variants, limitations, and readiness checks. Seven figures are stored in the corresponding \texttt{figures/} directory.

\end{document}
""")

OUT.write_text("\n\n".join(parts), encoding="utf-8")
print(f"Wrote {OUT}")
