# AutoDS Agent: Automated Data Science Workflow

## Short Description

AutoDS Agent is a portfolio-ready automated workflow for tabular CSV datasets. It creates reproducible analysis runs with profiling, conservative cleaning, EDA, modeling, evaluation, workflow trace logs, and final reports.

## Problem Statement

Data science work often starts in notebooks, where cleaning choices, model settings, generated plots, and final conclusions can become hard to reproduce. Recruiters and engineers also need a fast way to see whether a project has real structure beyond a demo notebook.

## Solution

AutoDS Agent packages the tabular analysis workflow into a local web application and API. Each dataset becomes a run folder with preserved raw data, structured JSON artifacts, saved plots, model files, workflow state, trace logs, and Markdown reports.

## Key Features

- CSV upload with raw input preservation.
- Dataset metadata and schema inference.
- Data quality profiling.
- Conservative cleaning plan and safe cleaning execution.
- Deterministic EDA summaries, findings, and plots.
- Regression and classification task inference.
- Baseline and candidate model training.
- Task-specific model evaluation.
- Optional MLflow tracking.
- Deterministic workflow orchestration with approval gates.
- Final analyst-style Markdown reports.
- Full demo and smoke-test scripts.

## Architecture Summary

The backend is organized around FastAPI routes and deterministic services. Agents wrap the services behind consistent boundaries, and the workflow layer coordinates the full analysis sequence. The frontend is a Streamlit app that makes the workflow visible for demos.

```text
CSV -> Run Folder -> Profile -> Cleaning Plan -> Cleaned Data -> EDA -> Modeling -> Reports
```

## Tech Stack

- FastAPI
- Streamlit
- pandas
- scikit-learn
- Matplotlib
- Pydantic
- joblib
- Optional MLflow
- Docker Compose
- Pytest
- GitHub Actions

## Example Workflow

1. Load `classification_churn.csv`.
2. Use target `churn`.
3. Run the automated workflow.
4. Review approval gates and trace events.
5. Inspect EDA plots and model metrics.
6. Open `reports/final_report.md`.

## Demo Outputs

A successful run saves:

- `input/raw_data.csv`
- `intermediate/profile.json`
- `intermediate/cleaning_plan.json`
- `intermediate/cleaned_data.csv`
- `intermediate/eda_summary.json`
- `intermediate/modeling_summary.json`
- `models/best_model.pkl`
- `plots/**/*.png`
- `reports/final_report.md`
- `logs/workflow_state.json`
- `logs/agent_trace.json`

## What I Learned

- How to turn a notebook-style workflow into a service-oriented application.
- How to design transparent agent boundaries without relying on paid LLM calls.
- How to preserve reproducibility through structured artifacts.
- How to balance automation with human approval gates.
- How to package a project for recruiters, engineers, and interviews.

## Future Work

- Add richer leakage detection.
- Add user-facing model selection controls and lightweight hyperparameter options.
- Add optional LLM-assisted narrative generation.
- Add production authentication and artifact storage.
- Add monitoring dashboards for workflow health and model quality.

## Links

- GitHub: https://github.com/yasamanebrahimi-byte/AutoDSAgent
