# AutoDS Agent

**AutoDS Agent: An Autonomous Multi-Agent Data Science Analyst** is a resume-quality AI side project designed to grow into a system that can ingest raw tabular datasets, profile them, clean them, run EDA, train baseline models, evaluate results, generate visualizations, and write final analysis reports.

Week 2 adds deterministic dataset profiling and conservative cleaning. The project still does not use LLM API calls, paid credits, MLflow, LangGraph, databases, EDA plots, or model training.

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

## Project Architecture

```text
autods-agent/
  README.md
  pyproject.toml
  .env.example
  .gitignore

  app/
    backend/
      main.py
      config.py
      routes/
        upload.py
        runs.py
        profile.py
        cleaning.py
      services/
        run_manager.py
        dataset_service.py
        profiling_service.py
        cleaning_service.py
      schemas/
        run.py
        dataset.py
        profile.py
        cleaning.py

    frontend/
      streamlit_app.py

    agents/
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
      file_utils.py

    workflows/
      analysis_graph.py

  runs/
    .gitkeep

  examples/
    sample_data/
      .gitkeep

  tests/
    test_run_manager.py
    test_dataset_service.py
    test_schema_inference.py
    test_profiling_service.py
    test_cleaning_service.py
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

Optional environment settings:

```bash
AUTODS_BACKEND_URL=http://localhost:8000
AUTODS_RUNS_DIR=runs
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

## Example Workflow

1. Start the FastAPI backend.
2. Start the Streamlit frontend.
3. Upload a CSV file.
4. The backend creates `runs/<run_id>/` and saves the raw file as `input/raw_data.csv`.
5. The UI displays Week 1 metadata and a preview.
6. Click `Generate Dataset Profile`.
7. Review dataset metrics, column profiles, and data quality warnings.
8. Click `Generate Cleaning Plan`.
9. Review duplicate handling, missing-value strategies, recommended drops, type conversions, and warnings.
10. Click `Apply Safe Cleaning`.
11. Review the cleaning summary and saved artifacts.

## Run Artifacts

After a successful Week 2 flow, a run can contain:

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
  models/
  plots/
  reports/
  logs/
```

The raw dataset is never overwritten.

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

## API Endpoints

### `GET /health`

Returns a simple service health check.

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

## Run Tests

```bash
pytest
```

## Week 3 Direction

Recommended Week 3 work:

- Add deterministic EDA summaries and visualizations.
- Use `cleaned_data.csv` as the default input for EDA when available.
- Save plots under `runs/<run_id>/plots/`.
- Save EDA artifacts under `runs/<run_id>/intermediate/`.
- Keep model training, MLflow, LangGraph, and LLM calls out until later weeks.
