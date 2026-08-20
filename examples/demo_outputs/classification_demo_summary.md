# Classification Demo Summary

## Dataset

- File: `examples/sample_data/breast_cancer_wisconsin.csv`
- Rows: 569
- Target column: `diagnosis`
- Task type: classification

Run it with `python scripts/run_demo.py --offline`. The offline fallback keeps
the run fully local while exercising the same validation gate, deterministic
cleaning, EDA, modeling, and reporting stages used by an API-backed run.

## Workflow stages

1. Profile the CSV and record the independent and deterministic plans.
2. Complete the pre-training validation gate.
3. Apply safe structural cleaning.
4. Compute EDA summaries and plots.
5. Fit the approved classification model with training-only preprocessing.
6. Report cross-validation, untouched holdout, and baseline metrics.
7. Persist the report and replay script.

## Run artifacts

Each run is written to `runs/<run_id>/` and includes:

- `profile.json`, `decision.json`, `cleaning.json`, `eda.json`, and `modeling.json`
- `data/cleaned.csv`
- `plots/target_distribution.png` and, when applicable, `plots/correlation_heatmap.png`
- `model/selected_model.joblib`
- `report.md`, `reproduce_analysis.py`, and `run.json`

The checked-in figures in [`docs/demo.md`](../../docs/demo.md) are a compact
visual walkthrough. Metrics can vary when dependency versions or the seed
change; the run artifacts are the source of truth for a particular execution.
