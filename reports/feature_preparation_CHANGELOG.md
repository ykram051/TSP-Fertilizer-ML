# Feature-preparation report change log

- Replaced the one-minute-only framing with direct 1-, 5-, 10-, and 15-minute targets.
- Replaced blanket lag and rolling generation with training-only selected lags and compact temporal families.
- Removed the obsolete 799-to-346 feature narrative and reported actual counts for Sets A-D.
- Replaced one universal feature matrix with 16 horizon/feature-set configurations and four preprocessing variants.
- Replaced global correlation filtering with grouped, training-only, interpretability-aware filtering and a removal log.
- Replaced structural warm-up median imputation with removal of the first 30 rows, followed by indicators and train-fitted imputation for genuine gaps.
- Added variable and process-feature registries, validation statuses, controllability safeguards, guarded ratios, and training-only clipping bounds.
- Replaced immediate sequence-model training with controlled persistence benchmarking, feature-set ablation, horizon comparison, and process understanding.
- Set the recommended next stage to `03_prediction_benchmark_and_process_understanding.ipynb`.
