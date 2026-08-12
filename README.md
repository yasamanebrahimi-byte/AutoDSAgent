# AutoDS Agent

**AutoDS Agent: An Autonomous Multi-Agent Data Science Analyst** is designed to grow into a system that can ingest raw tabular datasets, profile them, clean them, run EDA, train baseline models, evaluate results, generate visualizations, and write final analysis reports.

AutoDS Agent is built as a portfolio-ready engineering project: deterministic by default, artifact-driven, Docker-ready, and designed so every analysis run can be inspected after the fact. Through Week 7, it does not require paid LLM API credits.

## What It Does

- Upload a CSV and preserve the raw dataset.
- Generate profile, cleaning, EDA, modeling, evaluation, and report artifacts.
- Run the analysis manually through API/UI controls or through an autonomous workflow.
- Track model experiments in MLflow when enabled.
- Produce Markdown reports that clearly show available results, missing artifacts, limitations, and next steps.

## Tech Stack

- FastAPI backend
- Streamlit frontend
- pandas, scikit-learn, Matplotlib, joblib
- Pydantic schemas
- Optional MLflow experiment tracking
- Docker and Docker Compose
- Pytest

## Demo Materials

Example datasets live in `examples/sample_data/`.

- Classification demo: `classification_churn.csv`, target `churn`
- Regression demo: `regression_housing.csv`, target `sale_price`

Suggested screenshot placeholders for a portfolio README:

- Streamlit upload and workflow screen
- Generated EDA plots
- Final report preview
- MLflow experiment comparison

Detailed docs:

- [Architecture](docs/architecture.md)
- [Demo Walkthrough](docs/demo_walkthrough.md)
- [API Reference](docs/api_reference.md)
- [Portfolio Summary](docs/portfolio_summary.md)

## Long-Term Vision

The eventual system will use a multi-agent architecture with specialist agents for:

- Dataset profiling
- Cleaning recommendations and execution
- Exploratory data analysis
- Hypothesis generation
- Modeling
- Evaluation
- Report generation

The current implementation keeps those boundaries visible while using deterministic Python services.

## Features

### Week 1 Foundation

- FastAPI backend with health, upload, and run metadata endpoints
- Streamlit UI for CSV uploads and dataset preview
- Unique `run_id` generation for every analysis run
- Run folder structure under `runs/<run_id>/`
- Raw CSV preservation at `runs/<run_id>/input/raw_data.csv`
- Metadata JSON saved at `runs/<run_id>/intermediate/metadata.json`
- Pytest coverage for run management and dataset metadata

### Week 2 Profiling And Cleaning

- Rich dataset profile saved at `runs/<run_id>/intermediate/profile.json`
- Semantic schema inference for `numeric`, `categorical`, `boolean`, `datetime`, `text`, `id`, and `unknown`
- Dataset-level metrics:
  - Row and column counts
  - Total missing values
  - Duplicate row count
  - Memory usage
  - Column count by inferred type
  - Empty, duplicate, and missing-value flags
- Column-level metrics:
  - Pandas dtype and inferred semantic type
  - Missing and unique counts/percentages
  - Sample values
  - Constant, high-cardinality, ID-like, datetime-like, numeric, categorical, boolean, and text-like flags
  - Numeric statistics and simple IQR outlier counts
  - Top values and average string length for text-like columns
- Data quality warnings saved inside `profile.json`
- Conservative cleaning plan saved at `runs/<run_id>/intermediate/cleaning_plan.json`
- Safe cleaned dataset saved at `runs/<run_id>/intermediate/cleaned_data.csv`
- Cleaning summary saved at `runs/<run_id>/intermediate/cleaning_summary.json`
- Streamlit buttons and views for profiling, cleaning planning, and safe cleaning
- Tests for schema inference, profiling, and cleaning services

### Week 3 Exploratory Data Analysis And Visualization

- Deterministic EDA service that prefers `intermediate/cleaned_data.csv`
- Raw dataset fallback when cleaned data is missing, with a clear warning
- Structured EDA summary saved at `runs/<run_id>/intermediate/eda_summary.json`
- Structured EDA findings saved at `runs/<run_id>/intermediate/eda_findings.json`
- Markdown EDA report saved at `runs/<run_id>/reports/eda_summary.md`
- Matplotlib plot generation under `runs/<run_id>/plots/`
- Missing-values chart when missing values remain
- Numeric histograms for useful numeric columns, limited by request options
- Categorical top-value bar charts, limited by request options
- Numeric correlation heatmap when at least two useful numeric columns exist
- Optional target-column analysis with target distribution and relationship plots
- Streamlit EDA section for target selection, summaries, findings, next steps, and plots
- EDA agent boundary that wraps deterministic service logic
- Tests for EDA analysis helpers, visualization helpers, and EDA service artifacts
- No model training in Week 3; modeling begins in Week 4

### Week 4 Modeling And Evaluation

- Modeling service that requires `intermediate/cleaned_data.csv`
- User-selected target column with validation for missing, constant, too-small, and ID-like targets
- Task-type inference for regression versus classification, with an API/UI override
- Reusable sklearn preprocessing for numeric, categorical, boolean, and simple datetime features
- ID-like, unsupported, and free-text columns excluded from modeling by default
- Baseline models trained before candidates:
  - Regression: `DummyRegressor(strategy="median")`
  - Classification: `DummyClassifier(strategy="most_frequent")`
- Lightweight candidate models:
  - Regression: linear regression, ridge, random forest, hist gradient boosting
  - Classification: logistic regression, random forest, hist gradient boosting
- Regression metrics: MAE, RMSE, and R2, with RMSE used for model selection
- Classification metrics: accuracy, precision, recall, F1, and optional ROC-AUC, with F1 used for model selection
- Best-model selection, baseline comparison, and structured summaries
- Saved model artifacts under `runs/<run_id>/models/`
- Evaluation plots under `runs/<run_id>/plots/evaluation/`
- Streamlit modeling tab for target selection, task selection, metrics, failures, and plots
- Modeling and evaluation agent boundaries that wrap deterministic services
- Tests for task inference, preprocessing, modeling service artifacts, metrics, plots, and failed-model handling
- No LLM API calls, paid credits, MLflow, LangGraph, deep learning, or heavy tuning in Week 4

### Week 5 Agent Orchestration Layer

- Deterministic autonomous workflow that connects profiling, cleaning planning, safe cleaning, EDA, and optional modeling
- Structured workflow state saved at `runs/<run_id>/logs/workflow_state.json`
- Ordered agent trace log saved at `runs/<run_id>/logs/agent_trace.json`
- Canonical step statuses: `pending`, `running`, `completed`, `failed`, `skipped`, and `waiting_for_approval`
- Agent wrappers with a consistent `run(state)` boundary around deterministic services
- Human approval gates before risky cleaning actions and before modeling when enabled
- Approval actions for `approve` and `reject`, with rejection recorded in state and trace
- Retry support for failed steps while attempts remain
- FastAPI workflow endpoints for start, state, trace, approval, retry, and reset
- Streamlit `Autonomous Workflow` section with target selection, task-type override, approval toggles, status table, approval prompts, retry controls, artifacts, and trace events
- Advanced manual controls remain available for direct Week 1-4 service calls
- Tests for workflow state, orchestration, approval gates, retries, trace logging, and workflow API routes
- No LLM API calls, paid credits, MLflow, LangGraph dependency, or final full analyst report generation in Week 5

### Week 6 Final Report Generation Layer

- Deterministic report service that reads saved run artifacts and writes analyst-quality Markdown reports
- Full final analysis report saved at `runs/<run_id>/reports/final_report.md`
- Concise executive summary saved at `runs/<run_id>/reports/executive_summary.md`
- Technical methodology summary saved at `runs/<run_id>/reports/technical_summary.md`
- Limitations and next steps report saved at `runs/<run_id>/reports/limitations.md`
- Report metadata saved at `runs/<run_id>/intermediate/report_metadata.json`
- Report index saved at `runs/<run_id>/reports/report_index.json`
- Optional lightweight HTML export when requested
- Partial-report support when EDA, modeling, workflow, or trace artifacts are missing
- Report API endpoints for generation, metadata/index retrieval, content preview, and download
- Report agent integrated as the final autonomous workflow step after optional modeling
- Streamlit `Final Reports` section with generation controls, metadata, previews, index, and download buttons
- Tests for report builders, report service, report agent, API routes, and workflow integration
- No LLM API calls, paid credits, MLflow, or cloud deployment in Week 6

### Week 7 Portfolio Polish, MLflow, And Docker

- Optional MLflow tracking for modeling runs
- Parent AutoDS MLflow run plus nested model runs when tracking is enabled
- MLflow tags for run ID, target column, task type, dataset path, project, and modeling stage
- MLflow parameters for target, task type, split settings, feature counts, row counts, and selected models
- MLflow metrics for regression and classification models
- MLflow artifact logging for modeling/evaluation summaries, model results, evaluation plots, selected model artifact, and final report when present
- Non-fatal MLflow behavior: if MLflow is disabled, unavailable, or fails, local modeling still completes
- Expanded environment configuration in `app/backend/config.py`
- Structured application logging in `app/tools/app_logging.py`
- Non-secret config status endpoint at `GET /config/status`
- Dockerfile and Docker Compose setup for backend, frontend, and MLflow
- Small synthetic regression and classification example datasets
- `scripts/create_demo_run.py` for generating complete demo runs from bundled datasets
- `scripts/validate_project.py` for lightweight project validation
- Architecture, demo walkthrough, API reference, and portfolio summary docs
- Streamlit runtime notes showing MLflow status and demo dataset guidance

## Project Architecture

```text
autods-agent/
  README.md
  pyproject.toml
  .env.example
  .gitignore
  Dockerfile
  docker-compose.yml
  .dockerignore

  app/
    backend/
      main.py
      config.py
      routes/
        config.py
        upload.py
        runs.py
        profile.py
        cleaning.py
        eda.py
        modeling.py
        reports.py
        workflow.py
      services/
        run_manager.py
        dataset_service.py
        profiling_service.py
        cleaning_service.py
        eda_service.py
        modeling_service.py
        evaluation_service.py
        report_service.py
        workflow_service.py
      schemas/
        run.py
        dataset.py
        profile.py
        cleaning.py
        eda.py
        modeling.py
        reports.py
        workflow.py

    frontend/
      streamlit_app.py

    agents/
      base_agent.py
      orchestrator.py
      profiler_agent.py
      cleaning_agent.py
      eda_agent.py
      hypothesis_agent.py
      modeling_agent.py
      evaluation_agent.py
      report_agent.py

    tools/
      data_loader.py
      schema_inference.py
      data_quality.py
      cleaning.py
      eda_analysis.py
      preprocessing.py
      modeling.py
      evaluation.py
      report_sections.py
      report_builder.py
      report_export.py
      app_logging.py
      mlflow_logger.py
      model_persistence.py
      approval.py
      trace_logger.py
      file_utils.py
      statistics_utils.py
      visualization.py

    workflows/
      analysis_graph.py
      workflow_state.py
      workflow_steps.py

  runs/
    .gitkeep

  mlruns/
    .gitkeep

  examples/
    sample_data/
      regression_housing.csv
      classification_churn.csv
      README.md

  scripts/
    create_demo_run.py
    validate_project.py

  docs/
    architecture.md
    demo_walkthrough.md
    api_reference.md
    portfolio_summary.md

  tests/
    test_run_manager.py
    test_dataset_service.py
    test_schema_inference.py
    test_profiling_service.py
    test_cleaning_service.py
    test_eda_analysis.py
    test_visualization.py
    test_eda_service.py
    test_preprocessing.py
    test_modeling_tools.py
    test_modeling_service.py
    test_evaluation_service.py
    test_report_builder.py
    test_report_service.py
    test_report_agent.py
    test_report_workflow_integration.py
    test_mlflow_logger.py
    test_config.py
    test_demo_scripts.py
    test_workflow_state.py
    test_workflow_service.py
    test_orchestrator.py
    test_approval_gates.py
    test_retry_logic.py
```

## Installation

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

Install the project with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install optional MLflow support when you want experiment tracking:

```bash
python -m pip install -e ".[dev,mlflow]"
```

Optional environment settings:

```bash
AUTODS_BACKEND_URL=http://localhost:8000
AUTODS_RUNS_DIR=runs
AUTODS_ENV=local
AUTODS_LOG_LEVEL=INFO
AUTODS_ENABLE_MLFLOW=false
AUTODS_MLFLOW_TRACKING_URI=http://localhost:5000
AUTODS_MLFLOW_EXPERIMENT_NAME=AutoDS-Agent
```

## Run The Backend

From the project root:

```bash
uvicorn app.backend.main:app --reload
```

Health check:

```text
http://localhost:8000/health
```

## Run The Frontend

In a second terminal, from the project root:

```bash
streamlit run app/frontend/streamlit_app.py
```

The frontend expects the backend at `http://localhost:8000` unless `AUTODS_BACKEND_URL` is set.

## Run With Docker Compose

Docker Compose starts the backend, frontend, and MLflow UI together:

```bash
docker compose up --build
```

Open:

- Frontend: `http://localhost:8501`
- Backend docs: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`

Compose mounts `runs/` and `mlruns/` so local artifacts and MLflow tracking data persist between container restarts.

To disable MLflow tracking while still running the MLflow service:

```bash
AUTODS_ENABLE_MLFLOW=false docker compose up --build
```

## Run A Demo Analysis

Use the built-in demo script to generate a full analysis run without clicking through the UI:

```bash
python scripts/create_demo_run.py --dataset classification --target churn
python scripts/create_demo_run.py --dataset regression --target sale_price
```

The script creates a run, profiles data, plans and applies cleaning, runs EDA, trains models, generates reports, and prints key artifact paths.

## Example Workflow

1. Start the FastAPI backend.
2. Start the Streamlit frontend.
3. Upload a CSV file.
4. The backend creates `runs/<run_id>/` and saves the raw file as `input/raw_data.csv`.
5. The UI displays Week 1 metadata and a preview.
6. In `Autonomous Workflow`, optionally select a target column.
7. Keep task type on `Auto-detect` or choose `Regression` or `Classification`.
8. Choose whether cleaning and modeling require approval.
9. Click `Start Autonomous Workflow`.
10. The workflow runs profiling and cleaning-plan generation automatically.
11. If cleaning approval is required, review the reason and choose `Approve and Continue` or `Reject Step`.
12. After cleaning is approved or skipped, the workflow generates EDA.
13. If a target column is provided and modeling approval is required, review the modeling gate and approve or reject it.
14. The workflow generates final Markdown reports from whichever artifacts are available.
15. Review the final workflow status, step table, generated artifacts, report previews, retry controls, and agent trace.

The original manual profiling, cleaning, EDA, and modeling controls remain available under `Advanced Manual Controls`.
The `Final Reports` section can also generate reports manually for a partially completed run.

## Example Datasets

Bundled datasets are intentionally small and synthetic:

- `examples/sample_data/classification_churn.csv`: binary classification demo with target `churn`
- `examples/sample_data/regression_housing.csv`: regression demo with target `sale_price`

Both include numeric, categorical, boolean, missing-value, duplicate-row, and ID-like-column examples.

## View Generated Reports

Final reports are saved under each run:

```text
runs/<run_id>/reports/final_report.md
runs/<run_id>/reports/executive_summary.md
runs/<run_id>/reports/technical_summary.md
runs/<run_id>/reports/limitations.md
```

The Streamlit `Final Reports` section previews and downloads these Markdown files.

## View MLflow Experiments

MLflow is optional. When enabled, modeling runs log parameters, metrics, tags, and artifacts.

For Docker Compose, open:

```text
http://localhost:5000
```

For local Python usage, install the optional extra and start MLflow:

```bash
python -m pip install -e ".[mlflow]"
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri mlruns --default-artifact-root mlruns
```

Then set:

```bash
AUTODS_ENABLE_MLFLOW=true
AUTODS_MLFLOW_TRACKING_URI=http://localhost:5000
```

If MLflow is unavailable, AutoDS Agent logs a warning and keeps the modeling workflow running.

## Run Artifacts

After a successful Week 6 workflow, a run can contain:

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
    best_model.pkl
    model_results.json
  plots/
    missing_values.png
    numeric_distributions/
      <column_name>_histogram.png
    categorical_distributions/
      <column_name>_bar.png
    correlation_heatmap.png
    target_relationships/
      ...
    evaluation/
      model_comparison.png
      confusion_matrix.png
      predicted_vs_actual.png
      residuals.png
      feature_importance.png
  reports/
    eda_summary.md
    final_report.md
    executive_summary.md
    technical_summary.md
    limitations.md
    report_index.json
    final_report.html
  logs/
    workflow_state.json
    agent_trace.json
```

The raw dataset is never overwritten.
Some evaluation plots are task-specific, so a run will only contain the plots that apply.

## Autonomous Workflow Orchestration

Week 7 uses a local deterministic state machine in `app/workflows/analysis_graph.py`. It deliberately avoids paid LLM calls and keeps future extension points open for LangGraph, OpenAI Agents SDK, or tool-calling agents.

Canonical workflow order:

```text
profile -> cleaning_plan -> cleaning -> eda -> modeling -> report
```

The workflow saves state after each transition. Each step records attempts, timestamps, outputs, errors, and approval metadata. Required steps fail the workflow when they error; optional modeling is skipped clearly when no target column is provided.
The report step runs after EDA and optional modeling. If modeling is skipped, the report still generates and clearly marks modeling and evaluation sections as unavailable.

Approval gates:

- Cleaning pauses when enabled and the plan recommends column drops, duplicate-row removal, review warnings, or multi-column imputation.
- Modeling pauses when enabled and a target column is present.
- Approval continues from the paused step.
- Rejection skips the step, records a warning, and appends a trace event.

Retry behavior:

- Step attempts increment when the step starts.
- Failed steps record their error in `workflow_state.json`.
- `POST /runs/{run_id}/workflow/retry` retries a failed step while attempts remain.
- Retry events are appended to `agent_trace.json`.

Trace events include workflow start/completion, step start/completion/failure, approval required/granted/rejected, retries, and skipped steps.
Report generation adds `ReportAgent` events and saves final report paths into workflow state.

## Conservative Cleaning Rules

Safe cleaning intentionally avoids aggressive decisions:

- Removes exact duplicate rows.
- Imputes low or moderate numeric missing values with the median.
- Fills categorical and text missing values with `Unknown`.
- Fills boolean missing values with the mode when available.
- Parses datetime-like columns to ISO-style strings.
- Keeps ID columns for traceability and marks them for future modeling exclusion.
- Recommends dropping constant columns and drops them only within the configured automatic drop limit.
- Recommends review for very high missingness instead of automatically dropping those columns.
- Preserves all actions and warnings in JSON artifacts.

## Exploratory Data Analysis Rules

EDA uses a production-minded artifact flow rather than notebook state:

- Uses `runs/<run_id>/intermediate/cleaned_data.csv` when it exists.
- Falls back to `runs/<run_id>/input/raw_data.csv` when cleaned data is missing.
- Adds this warning on fallback: `Cleaned dataset was not found. EDA was generated from the raw uploaded dataset.`
- Saves machine-readable summaries and findings for later agents.
- Saves static Matplotlib PNG plots for the UI and reports.
- Limits numeric, categorical, and target relationship plots to avoid chart sprawl.
- Skips ID-like columns for automatic distribution plots.
- Does not train models or infer causality.

Optional target-column support is available through the API and Streamlit UI. When a target is selected, the service validates that it exists, generates target distribution plots, adds simple target findings, and saves relationship plots under `plots/target_relationships/`. If no target is selected, general EDA still runs and target-specific analysis is skipped.

Generated plot types:

- `missing_values.png`: bar chart of columns with remaining missing values.
- `numeric_distributions/*.png`: histograms for useful numeric columns.
- `categorical_distributions/*.png`: top-category bar charts for categorical and boolean columns.
- `correlation_heatmap.png`: numeric correlation heatmap when enough numeric columns exist.
- `target_relationships/*.png`: target distribution and simple feature-target relationship plots when a target is provided.

## Modeling And Evaluation Rules

Modeling uses a production-minded artifact flow rather than notebook state:

- Uses `runs/<run_id>/intermediate/cleaned_data.csv`.
- Returns a clear error if cleaned data is missing. It does not silently train on `input/raw_data.csv`.
- Requires a target column selected by the user.
- Excludes the target column from features.
- Excludes likely ID columns from features by default.
- Excludes free-text and unsupported columns for Week 4 and records the reason.
- Builds sklearn pipelines so imputers, scalers, and encoders are fit only on the training split.
- Always uses a train/test split.
- Always trains a baseline model before candidate models.
- Records failed model attempts without crashing the whole modeling run.
- Selects the best model by the primary metric: lower RMSE for regression, higher F1 for classification.
- Saves the baseline model as `models/baseline_model.pkl`.
- Saves the selected best model as `models/best_model.pkl`.
- Saves model comparison details as `models/model_results.json`.

Task inference follows simple, deterministic rules:

- Numeric targets with many unique values are treated as regression.
- Numeric targets with few unique values are treated as classification.
- Boolean and low-cardinality categorical targets are treated as classification.
- High-cardinality text targets are rejected for Week 4 because text modeling is future work.
- The API and UI can override task type with `regression` or `classification`.

Baseline models set a plain reference point. Candidate models must beat the baseline to show useful predictive lift, but the baseline can remain the best model if candidates do not improve the primary metric.

When MLflow is enabled, modeling logs run metadata, parameters, metrics, nested model runs, evaluation plots, and selected artifacts. MLflow logging is best-effort: failures are recorded as warnings and do not stop local artifact generation.

## Final Report Generation

Week 6 adds a deterministic reporting layer in `app/backend/services/report_service.py`.
It loads the JSON artifacts already saved under a run folder, records which source artifacts were used or missing, and writes polished Markdown reports under `runs/<run_id>/reports/`.

Generated report artifacts:

- `reports/final_report.md`: full end-to-end analysis report
- `reports/executive_summary.md`: short nontechnical summary
- `reports/technical_summary.md`: preprocessing, EDA, modeling, evaluation, and artifact methodology
- `reports/limitations.md`: limitations, skipped steps, warnings, and next steps
- `intermediate/report_metadata.json`: report status, generated sections, skipped sections, warnings, and source artifact audit
- `reports/report_index.json`: list of generated report files and descriptions
- `reports/final_report.html`: optional simple HTML export when requested

Reports can be `completed` or `partial`.
Partial reports are expected when optional or downstream artifacts are missing, such as when modeling is skipped because no target column was selected.
Unavailable sections are included with explicit explanations instead of invented analysis.

The report text follows deterministic interpretation rules:

- EDA findings are described as associations, not causal claims.
- Modeling metrics are included only when saved modeling and evaluation artifacts exist.
- Failed models, skipped steps, missing artifacts, and warnings are surfaced.
- Limitations are always included.
- Report generation does not use LLM API calls, paid credits, or cloud deployment.

## API Endpoints

### `GET /health`

Returns a simple service health check.

### `GET /config/status`

Returns non-secret runtime configuration status, including environment, runs directory, backend URL, default modeling settings, logging level, and MLflow status.

### `POST /upload`

Accepts a CSV file upload, creates a run, saves the raw dataset, generates metadata, saves it, and returns the metadata.

### `GET /runs`

Returns available run IDs with lightweight metadata when available.

### `GET /runs/{run_id}`

Returns saved Week 1 metadata for one run.

### `POST /runs/{run_id}/profile`

Loads `input/raw_data.csv`, generates `profile.json`, and returns the profile.

### `GET /runs/{run_id}/profile`

Returns an existing profile or `404` if it has not been generated.

### `POST /runs/{run_id}/cleaning-plan`

Ensures a profile exists, generates `cleaning_plan.json`, and returns the plan.

### `GET /runs/{run_id}/cleaning-plan`

Returns an existing cleaning plan or `404`.

### `POST /runs/{run_id}/clean`

Ensures a cleaning plan exists, applies safe cleaning, saves `cleaned_data.csv` and `cleaning_summary.json`, and returns the summary.

### `GET /runs/{run_id}/cleaning-summary`

Returns an existing cleaning summary or `404`.

### `POST /runs/{run_id}/eda`

Generates EDA artifacts and returns the summary plus findings. The request body is optional.

Example request body:

```json
{
  "target_column": "SalePrice",
  "max_numeric_plots": 10,
  "max_categorical_plots": 10,
  "max_target_relationship_plots": 5
}
```

Saved artifacts:

- `intermediate/eda_summary.json`
- `intermediate/eda_findings.json`
- `reports/eda_summary.md`
- `plots/**/*.png`

### `GET /runs/{run_id}/eda`

Returns existing EDA summary and findings or `404` if EDA has not been generated.

### `GET /runs/{run_id}/plots`

Returns generated plot files with relative paths, labels, and categories.

### `GET /runs/{run_id}/plots/{plot_path}`

Returns one generated PNG plot. This is used by the Streamlit UI to display backend-generated images.

### `POST /runs/{run_id}/model`

Requires `intermediate/cleaned_data.csv`, trains the baseline and candidate models, evaluates them, saves model artifacts, saves evaluation plots, and returns modeling and evaluation summaries.

Example request body:

```json
{
  "target_column": "SalePrice",
  "task_type": null,
  "test_size": 0.2,
  "random_state": 42
}
```

Use `null` for auto-detection, or set `task_type` to `regression` or `classification`.

Saved artifacts:

- `intermediate/modeling_summary.json`
- `intermediate/evaluation_summary.json`
- `models/baseline_model.pkl`
- `models/best_model.pkl`
- `models/model_results.json`
- `plots/evaluation/*.png`

### `GET /runs/{run_id}/modeling-summary`

Returns an existing modeling summary or `404` if models have not been trained.

### `GET /runs/{run_id}/evaluation-summary`

Returns an existing evaluation summary or `404` if models have not been trained.

### `GET /runs/{run_id}/models`

Returns saved model artifacts and model result files for one run.

### `POST /runs/{run_id}/reports/generate`

Generates final reports from available run artifacts.

Example request body:

```json
{
  "include_html": false,
  "force_regenerate": true
}
```

Saved artifacts:

- `reports/final_report.md`
- `reports/executive_summary.md`
- `reports/technical_summary.md`
- `reports/limitations.md`
- `intermediate/report_metadata.json`
- `reports/report_index.json`
- `reports/final_report.html` when `include_html` is `true`

The response returns report metadata plus the report index.

### `GET /runs/{run_id}/reports`

Returns `report_metadata.json` and `report_index.json`.
If reports have not been generated yet, returns `404`.

### `GET /runs/{run_id}/reports/{report_name}`

Returns Markdown content for one report.
Supported report names are `final_report`, `executive_summary`, `technical_summary`, and `limitations`.

### `GET /runs/{run_id}/reports/download/{report_name}`

Returns the selected Markdown report as a downloadable file.

### `POST /runs/{run_id}/workflow/start`

Starts or restarts the autonomous deterministic workflow for an existing run.

Example request body:

```json
{
  "target_column": "SalePrice",
  "task_type": null,
  "require_cleaning_approval": true,
  "require_modeling_approval": true
}
```

The workflow runs until it completes, fails, or reaches a human approval gate. It returns `workflow_state.json` as structured JSON. Successful workflows include a final report step.

### `GET /runs/{run_id}/workflow/state`

Returns the current workflow state saved under `logs/workflow_state.json`.

### `GET /runs/{run_id}/workflow/trace`

Returns ordered agent trace events saved under `logs/agent_trace.json`.

### `POST /runs/{run_id}/workflow/approve`

Applies approval or rejection to a waiting step.

Example request body:

```json
{
  "step": "cleaning",
  "action": "approve"
}
```

Supported steps are `cleaning` and `modeling`. Supported actions are `approve` and `reject`.

### `POST /runs/{run_id}/workflow/retry`

Retries a failed step when attempts remain.

Example request body:

```json
{
  "step": "eda"
}
```

### `POST /runs/{run_id}/workflow/reset`

Resets workflow state and trace logs for a run without deleting raw data or generated analysis artifacts.

## Run Tests

```bash
pytest
```

If the local default temp directory has permission issues on Windows, this equivalent command keeps Pytest scratch files inside the workspace:

```powershell
$env:TEMP=(Resolve-Path ".pytest_tmp").Path
$env:TMP=$env:TEMP
pytest --basetemp=.pytest_tmp/run -o cache_dir=.pytest_tmp_cache
```

## Current Limitations

- Analysis is deterministic and does not use LLM reasoning or paid API calls through Week 7.
- Text, time-series, geospatial, and deep learning workflows are intentionally out of scope.
- Leakage detection is conservative and should be reviewed before trusting model results.
- Hyperparameter tuning is lightweight.
- Reports summarize saved artifacts and do not invent unavailable findings.
- Docker Compose is intended for local development and demos, not hardened production deployment.

## Future Work

Recommended Week 8 work:

- Add final demo screenshots and a portfolio walkthrough video.
- Polish resume-ready packaging and release notes.
- Add stronger leakage checks and richer validation strategies.
- Add optional LLM-assisted narrative generation only behind explicit configuration.
- Add model registry or richer MLflow experiment comparison workflows.

## Resume Bullet Suggestions

- Built an autonomous tabular data science analyst with FastAPI, Streamlit, sklearn, MLflow, Docker Compose, and Pytest.
- Designed an artifact-first workflow for profiling, cleaning, EDA, modeling, evaluation, and deterministic report generation.
- Implemented optional MLflow experiment tracking that logs model metrics, parameters, tags, and artifacts without breaking local runs when unavailable.
- Added workflow state management, approval gates, retry logic, trace logs, and partial-report handling for reliable autonomous analysis.
