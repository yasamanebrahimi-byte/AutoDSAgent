# Classification Demo Summary

## Dataset

- File: `examples/sample_data/breast_cancer_wisconsin.csv`
- Rows: 569
- Target column: `diagnosis`
- Expected task type: classification

## Expected Workflow Stages

1. Create a run folder.
2. Preserve raw CSV as `input/raw_data.csv`.
3. Save metadata.
4. Generate profile.
5. Generate cleaning plan.
6. Apply safe cleaning.
7. Generate EDA summary, findings, and plots.
8. Train baseline and candidate classification models.
9. Evaluate models with accuracy, precision, recall, F1, and optional ROC-AUC.
10. Generate final reports.

## Expected Report Outputs

- `reports/final_report.md`
- `reports/executive_summary.md`
- `reports/technical_summary.md`
- `reports/limitations.md`
- `reports/report_index.json`

## Expected Model Outputs

- `intermediate/modeling_summary.json`
- `intermediate/evaluation_summary.json`
- `models/baseline_model.pkl`
- `models/selected_model.pkl`
- `models/best_model.pkl` (legacy selected-model alias)
- `models/model_results.json`
- `plots/evaluation/model_comparison.png`
- `plots/evaluation/confusion_matrix.png`

The primary metric is F1, where higher is better.

## Checked-In Reference Run

The gallery was regenerated from a deterministic reference run with random
state 42:

- 455 training rows and 114 held-out rows
- 5-fold stratified cross-validation on the training partition
- selected model: logistic regression
- CV macro F1: 0.9716 ± 0.0162
- holdout macro F1: 0.9619
- holdout accuracy: 0.9649

The exact values may change when dependencies, modeling candidates, or workflow
configuration change; regenerate the gallery and record the new provenance when
that happens.

## Where Artifacts Are Saved

Artifacts are saved under:

```text
runs/<run_id>/
```

The script prints the exact run ID and paths after completion.

## What To Look For

- The final report should summarize diagnostic feature relationships, cleaning actions, EDA findings, selected classifier, and limitations.
- Metrics should be interpreted with the documented split and uncertainty rather than as clinical performance.
- The evaluation summary should identify the selected model, best candidate, and most-frequent baseline classifier comparison.
- If MLflow is enabled, look for run tags, classification metrics, evaluation plots, and model artifacts.
