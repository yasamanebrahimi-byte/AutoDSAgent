# AutoDS Agent: Automated Data Science Workflow

AutoDS Agent is a deterministic, service-backed workflow for tabular CSV datasets. It preserves raw data, profiles the dataset, creates a conservative cleaning plan, performs EDA, trains baseline and candidate machine learning models, evaluates results, tracks the workflow, and generates analyst-style Markdown reports.

The project is deterministic by default and does not require paid LLM API calls.

## Overview

AutoDS Agent turns a raw CSV into an auditable run folder:

```text
CSV upload
  -> metadata
  -> dataset profile
  -> cleaning plan
  -> cleaned dataset
  -> EDA summary and plots
  -> modeling and evaluation
  -> workflow trace
  -> final reports
```

Each run is saved under `runs/<run_id>/`, with raw input kept separate from derived artifacts.

## Demo

Run a complete demo without starting the web app:

```bash
python scripts/run_full_demo.py --dataset classification
python scripts/run_full_demo.py --dataset regression
```

If `python` points to an older global install on Windows, run the project
environment directly:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_demo.py --dataset classification
```

Then open the printed final report path, usually:

```text
runs/<run_id>/reports/final_report.md
```

## Why This Project

Many data science projects live as notebooks that are difficult to reproduce, inspect, test, or demo. AutoDS Agent packages that workflow as a small application with APIs, services, saved artifacts, orchestration state, a demo UI, tests, Docker, and clear documentation.

It is designed to show engineering judgment as well as data science ability.

## Key Features

- CSV upload with raw data preservation.
- Unique run folders with structured artifacts.
- Metadata generation and dataset preview.
- Schema inference and data quality profiling.
- Conservative structural cleaning plans and safe cleaning execution.
- EDA summaries, findings, Markdown reports, and PNG plots.
- Target-aware EDA for regression or classification demos.
- Regression and classification task inference with saved reasons.
- Baseline and candidate sklearn model training with training-only preprocessing.
- Cross-validated model selection and one final holdout evaluation for the selected model.
- Task-specific CV and holdout metrics, including richer classification metrics.
- Saved baseline, selected model, and legacy selected-model alias artifacts.
- Optional MLflow experiment tracking.
- Deterministic workflow orchestration with approval gates.
- Workflow state, retries, trace logs, and final report generation.
- Cross-process per-run locking for workflow and manual artifact mutations.
- Streamlit demo UI with bundled sample datasets.
- Docker Compose and GitHub Actions test workflow.

## Deterministic Agentic Workflow

The project uses agent boundaries around deterministic services. These agents are orchestration wrappers, not open-ended autonomous reasoning systems:

- `ProfilerAgent`
- `CleaningAgent`
- `EDAAgent`
- `ModelingAgent`
- `EvaluationAgent`
- `ReportAgent`
- `OrchestratorAgent`

The canonical workflow is:

```text
profile -> cleaning_plan -> cleaning -> eda -> modeling -> report
```

The workflow can run from start to finish, pause for human approval, retry failed steps, skip optional modeling when no target is provided, and persist state to `logs/workflow_state.json`. Workflow starts run as background jobs so API clients receive a poll URL immediately instead of holding a long HTTP request open. Trace events are saved to `logs/agent_trace.json`.

## Modeling Methodology

The supervised workflow is target-sensitive and keeps learned preprocessing inside sklearn pipelines:

```text
target selection
  -> structural cleaning
  -> train/test split
  -> training-only preprocessing
  -> cross-validated candidate selection on training data
  -> fit selected model on the complete training partition
  -> one final evaluation on the untouched test data
```

Cleaning removes or normalizes structural issues such as exact duplicates, constant columns, and deterministic datetime formats. It does not fill missing feature values with whole-dataset medians, modes, or category labels. Numeric/categorical/boolean imputers, scaling, and one-hot encoders are fit inside sklearn pipelines during CV and final training.

Task inference is conservative: boolean targets, low-cardinality categorical strings, and low-cardinality discrete numeric targets are classification; continuous numeric or numeric-string targets are regression. High-cardinality text, identifier-like targets, constant targets, and rare classification classes fail with actionable errors. Classification selection uses macro F1 by default; final classification reporting includes weighted F1, macro F1, balanced accuracy, per-class precision/recall/F1, a confusion matrix, and binary ROC-AUC/average precision when probabilities are available.

## Architecture

AutoDS Agent separates API routing, deterministic services, agent boundaries, workflow state, reusable tools, and frontend presentation.

```text
app/backend/routes/      FastAPI endpoints
app/backend/services/    Core deterministic workflow services
app/backend/schemas/     Pydantic request and response models
app/agents/              Specialist agent boundaries
app/workflows/           Workflow state and orchestration logic
app/tools/               Data, modeling, plotting, reporting, and I/O helpers
app/frontend/            Streamlit demo application
```

Generated artifacts live under `runs/<run_id>/` and are ignored by Git.

## Tech Stack

- FastAPI
- Streamlit
- pandas
- scikit-learn
- Matplotlib
- Pydantic
- joblib
- Optional MLflow
- Docker and Docker Compose
- Pytest
- GitHub Actions

## Quickstart

Use Python 3.11 or newer.

```bash
python -m venv .venv
```

On Windows:

```bash
.\.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -c constraints-dev.txt
```

`pyproject.toml` keeps compatible dependency ranges for normal project installs. `constraints-dev.txt` pins the development and CI test environment for reproducible validation.

Start the backend:

```bash
uvicorn app.backend.main:app --reload
```

Start the frontend in a second terminal:

```bash
streamlit run app/frontend/streamlit_app.py
```

Open:

- Streamlit app: `http://localhost:8501`
- Backend health: `http://localhost:8000/health`
- Backend docs: `http://localhost:8000/docs`

## Run With Docker

Start backend, frontend, and MLflow:

```bash
docker compose up --build
```

Open:

- Frontend: `http://localhost:8501`
- Backend docs: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`

Docker Compose mounts `runs/` and `mlruns/` so local artifacts persist between restarts.
The image install uses `constraints-dev.txt` so the local/demo Docker environment follows the same pinned dependency strategy as tests.

## Try The Example Datasets

Bundled synthetic datasets live in `examples/sample_data/`.

| Dataset                  | File                       | Target       | Task           |
| ------------------------ | -------------------------- | ------------ | -------------- |
| Synthetic Customer Churn | `classification_churn.csv` | `churn`      | Classification |
| Synthetic Housing        | `regression_housing.csv`   | `sale_price` | Regression     |

In Streamlit, use `Try A Sample Dataset` to load either dataset. The UI sets the recommended target and task type for a smoother demo.

## Run The Full Demo

The full demo runner uses internal services and the deterministic workflow. It does not require the backend server, Docker, MLflow, or paid API keys.

```bash
python scripts/run_full_demo.py --dataset classification
python scripts/run_full_demo.py --dataset regression
```

If `python` points to an older global install on Windows, run the project
environment directly:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_demo.py --dataset classification
```

Optional:

```bash
python scripts/run_full_demo.py --dataset classification --include-html
python scripts/run_full_demo.py --dataset regression --runs-dir .demo_runs
```

The script prints the run ID, workflow status, target, inferred task type, selected model, primary metric, final report path, and key artifact paths.

## Generated Artifacts

A successful run can contain:

```text
runs/<run_id>/
  input/
    raw_data.csv
  intermediate/
    metadata.json
    profile.json
    cleaning_plan.json
    cleaned_data.csv
    cleaning_summary.json
    eda_summary.json
    eda_findings.json
    modeling_summary.json
    evaluation_summary.json
    report_metadata.json
  models/
    baseline_model.pkl
    selected_model.pkl
    best_model.pkl        # legacy alias for the selected model
    model_results.json
  plots/
    eda/
      missing_values.png
      numeric_distributions/
      categorical_distributions/
      target_relationships/
    evaluation/
  reports/
    eda_summary.md
    final_report.md
    executive_summary.md
    technical_summary.md
    limitations.md
    report_index.json
  logs/
    workflow_state.json
    agent_trace.json
```

The raw uploaded file is never overwritten.

## MLflow Tracking

MLflow is optional. When enabled, modeling logs parameters, metrics, tags, evaluation plots, and selected artifacts.

Install optional MLflow support:

```bash
python -m pip install -e ".[dev,mlflow]"
```

Run MLflow locally:

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri mlruns --default-artifact-root mlruns
```

Enable tracking:

```bash
AUTODS_ENABLE_MLFLOW=true
AUTODS_MLFLOW_TRACKING_URI=http://localhost:5000
```

If MLflow is disabled or unavailable, local model training and artifact generation still continue.

## API Overview

Core endpoints:

- `GET /health`
- `GET /config/status`
- `POST /upload`
- `GET /runs`
- `GET /runs/{run_id}`
- `POST /runs/{run_id}/profile`
- `POST /runs/{run_id}/cleaning-plan`
- `POST /runs/{run_id}/clean`
- `POST /runs/{run_id}/eda`
- `GET /runs/{run_id}/plots`
- `POST /runs/{run_id}/model`
- `POST /runs/{run_id}/reports/generate`
- `POST /runs/{run_id}/workflow/start`
- `GET /runs/{run_id}/workflow/jobs/{job_id}`
- `GET /runs/{run_id}/workflow/state`
- `GET /runs/{run_id}/workflow/trace`
- `POST /runs/{run_id}/workflow/approve`
- `POST /runs/{run_id}/workflow/retry`

See [docs/api_reference.md](docs/api_reference.md) for more detail.

## Testing

Install the pinned development toolchain:

```bash
python -m pip install -e ".[dev]" -c constraints-dev.txt
```

The suite includes Streamlit AppTest coverage for uploads, automated workflow submission and polling, advanced manual controls, report downloads, and recovery from backend failures.

Run lightweight local checks:

Run all tests with the same 75% coverage floor enforced by CI:

```bash
python -m pytest --cov=app --cov-report=term-missing --cov-fail-under=75
```

Run linting, core-contract type checking, and the dependency audit:

```bash
python -m ruff check app tests scripts
python -m mypy app/backend/schemas app/backend/config.py app/backend/services/run_manager.py app/workflows/workflow_state.py app/workflows/workflow_steps.py
python -m pip_audit --progress-spinner off
```

Run the lightweight packaging and application checks:

```bash
python scripts/smoke_test.py
python scripts/validate_project.py
```

GitHub Actions runs smoke checks and the full test suite on Python 3.11, 3.12, and 3.13. It also enforces linting, typed core contracts, the coverage floor, a dependency vulnerability audit, MLflow installation and import checks, and a Docker build plus container smoke test.

## Project Structure

```text
autods-agent/
  app/
    backend/
      routes/
      schemas/
      services/
    frontend/
    agents/
    tools/
    workflows/
  docs/
  examples/
    sample_data/
    demo_outputs/
  scripts/
  tests/
  runs/
  mlruns/
  .github/
```

## Documentation

- [Architecture](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Demo Walkthrough](docs/demo_walkthrough.md)
- [Final Demo Script](docs/final_demo_script.md)
- [Recruiter Walkthrough](docs/recruiter_walkthrough.md)
- [Interview Talking Points](docs/interview_talking_points.md)
- [Resume Bullets](docs/resume_bullets.md)
- [Project Showcase](docs/project_showcase.md)
- [Portfolio Summary](docs/portfolio_summary.md)
- [Demo Output Notes](examples/demo_outputs/README.md)
- [Project Status](PROJECT_STATUS.md)
- [Changelog](CHANGELOG.md)

## Current Limitations

- The current agent workflow is deterministic and does not use paid LLM reasoning.
- Text, time-series, geospatial, deep learning, and causal inference workflows are out of scope.
- Temporal and domain-specific leakage checks should be expanded before production use.
- Model training is intentionally lightweight for local demo speed.
- Reports summarize saved artifacts and do not invent unavailable findings.
- Docker Compose is intended for local demos, not hardened production deployment.

## Future Work

- Stronger temporal leakage and target-contamination checks.
- User-facing model selection controls and lightweight hyperparameter options.
- Optional LLM-assisted narrative generation behind explicit configuration.
- Background workers for long-running workflow steps.
- Database-backed run metadata and object storage for artifacts.
- Production auth, access control, and monitoring.

## Resume Highlights

- Built an automated data science analysis platform with FastAPI, Streamlit, Docker Compose, Pydantic, and Pytest, turning raw CSV uploads into reproducible run folders with saved artifacts and reports.
- Designed modular services for upload, profiling, cleaning, EDA, modeling, evaluation, reporting, and workflow orchestration with clear API boundaries.
- Implemented persistent workflow state, retry handling, human approval gates, and trace logs to make automated analysis runs auditable and recoverable.
- Packaged the project for portfolio use with sample datasets, full-demo scripts, smoke tests, CI, Docker Compose, technical documentation, recruiter walkthroughs, and interview talking points.
