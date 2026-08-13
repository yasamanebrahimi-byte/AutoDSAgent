# Regression Demo Summary

## Dataset

- File: `examples/sample_data/regression_housing.csv`
- Target column: `sale_price`
- Expected task type: regression

## Expected Workflow Stages

1. Create a run folder.
2. Preserve raw CSV as `input/raw_data.csv`.
3. Save metadata.
4. Generate profile.
5. Generate cleaning plan.
6. Apply safe cleaning.
7. Generate EDA summary, findings, and plots.
8. Train baseline and candidate regression models.
9. Evaluate models with MAE, RMSE, and R2.
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
- `models/best_model.pkl`
- `models/model_results.json`
- `plots/evaluation/model_comparison.png`
- `plots/evaluation/predicted_vs_actual.png`
- `plots/evaluation/residuals.png`

The primary metric is RMSE, where lower is better.

## Where Artifacts Are Saved

Artifacts are saved under:

```text
runs/<run_id>/
```

The script prints the exact run ID and paths after completion.

## What To Look For

- The final report should summarize housing features, cleaning actions, EDA findings, selected regression model, and limitations.
- The evaluation summary should identify the best model and compare it to the baseline median regressor.
- If MLflow is enabled, look for run tags, regression metrics, evaluation plots, and model artifacts.
