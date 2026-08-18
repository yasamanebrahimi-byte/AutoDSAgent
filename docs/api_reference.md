# API Reference

This is a practical summary of the AutoDS Agent API. Full interactive docs are available from FastAPI at `/docs` when the backend is running.

## Health And Config

- `GET /health`: service health check.
- `GET /config/status`: non-secret runtime settings, including environment, runs directory, and MLflow status.

## Uploads And Runs

- `POST /upload`: upload a CSV, create a run folder, preserve `input/raw_data.csv`, save `intermediate/metadata.json`, and return dataset metadata.
- `GET /runs`: list available runs.
- `GET /runs/{run_id}`: return saved run metadata.

## Profiling

- `POST /runs/{run_id}/profile`: generate `intermediate/profile.json`.
- `GET /runs/{run_id}/profile`: return an existing profile.

## Cleaning

- `POST /runs/{run_id}/cleaning-plan`: generate `intermediate/cleaning_plan.json`.
- `GET /runs/{run_id}/cleaning-plan`: return an existing cleaning plan.
- `POST /runs/{run_id}/clean`: apply structural cleaning and save `intermediate/cleaned_data.csv` plus `intermediate/cleaning_summary.json`. Learned imputation is deferred to modeling pipelines.
- `GET /runs/{run_id}/cleaning-summary`: return an existing cleaning summary.

## EDA And Plots

- `POST /runs/{run_id}/eda`: generate EDA summaries, findings, Markdown EDA summary, and plots under `plots/eda/`.
- `GET /runs/{run_id}/eda`: return existing EDA summary and findings.
- `GET /runs/{run_id}/plots`: list generated plot files.
- `GET /runs/{run_id}/plots/{plot_path}`: return one generated plot image.

## Modeling And Evaluation

- `POST /runs/{run_id}/model`: train baseline and candidate models against `intermediate/cleaned_data.csv`, select the overall model with training-only CV including the baseline, evaluate the selected model once on the holdout, and save model artifacts, evaluation summaries, and plots under `plots/evaluation/`.
- `GET /runs/{run_id}/modeling-summary`: return `intermediate/modeling_summary.json`.
- `GET /runs/{run_id}/evaluation-summary`: return `intermediate/evaluation_summary.json`.
- `GET /runs/{run_id}/models`: list saved model and result artifacts.

Modeling and evaluation summaries expose `best_candidate_name`, `best_candidate_metrics`, `selected_model_name`, `selected_model_role`, `baseline_model_name`, `baseline_metrics`, `candidate_beats_baseline`, `selection_metric`, `selection_direction`, and `selection_outcome`. New runs save `models/selected_model.pkl`; `models/best_model.pkl` remains a legacy alias for the same selected estimator.

When MLflow is enabled, modeling also logs run parameters, metrics, tags, and artifacts to the configured tracking server.

## Automated Workflow

- `POST /runs/{run_id}/workflow/start`: queue a deterministic workflow and return `202 Accepted` with a pollable job URL.
- `GET /runs/{run_id}/workflow/jobs/{job_id}`: poll background workflow execution status.
- `GET /runs/{run_id}/workflow/state`: return `logs/workflow_state.json`.
- `GET /runs/{run_id}/workflow/trace`: return `logs/agent_trace.json`.
- `POST /runs/{run_id}/workflow/approve`: approve or reject a waiting cleaning or modeling gate.
- `POST /runs/{run_id}/workflow/retry`: retry a failed step when attempts remain.
- `POST /runs/{run_id}/workflow/reset`: reset workflow logs without deleting data artifacts.

Mutating endpoints are serialized per run. A concurrent mutation returns `409 Conflict` with `Retry-After: 1`; clients should poll the active workflow or retry after the current mutation finishes.

Workflow order:

```text
profile -> cleaning_plan -> cleaning -> eda -> modeling -> report
```

## Reports

- `POST /runs/{run_id}/reports/generate`: generate final Markdown reports from available artifacts.
- `GET /runs/{run_id}/reports`: return report metadata and report index.
- `GET /runs/{run_id}/reports/{report_name}`: return Markdown content for `final_report`, `executive_summary`, `technical_summary`, or `limitations`.
- `GET /runs/{run_id}/reports/download/{report_name}`: download the selected Markdown report.

Reports are deterministic and may be partial when optional artifacts are missing.
