# AutoDS Agent

**AutoDS Agent: An Autonomous Multi-Agent Data Science Analyst** is designed to grow into a system that can ingest raw tabular datasets, profile them, clean them, run EDA, train baseline models, evaluate results, generate visualizations, and write final analysis reports.

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
- No LLM API calls, paid credits, MLflow, LangGraph, or model training in Week 3

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
        eda.py
      services/
        run_manager.py
        dataset_service.py
        profiling_service.py
        cleaning_service.py
        eda_service.py
      schemas/
        run.py
        dataset.py
        profile.py
        cleaning.py
        eda.py

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
      eda_analysis.py
      file_utils.py
      statistics_utils.py
      visualization.py

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
    test_eda_analysis.py
    test_visualization.py
    test_eda_service.py
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
12. Open `Exploratory Data Analysis`.
13. Optionally select a target column.
14. Click `Generate EDA`.
15. Review dataset choice, column type summaries, remaining data quality notes, findings, recommended next steps, and generated plots.

## Run Artifacts

After a successful Week 3 flow, a run can contain:

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
  models/
  plots/
    missing_values.png
    numeric_distributions/
      <column_name>_histogram.png
    categorical_distributions/
      <column_name>_bar.png
    correlation_heatmap.png
    target_relationships/
      ...
  reports/
    eda_summary.md
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

## Run Tests

```bash
pytest
```

If the local default temp directory has permission issues on Windows, this equivalent command keeps Pytest scratch files inside the workspace:

```bash
pytest --basetemp=.pytest_tmp -o cache_dir=.pytest_tmp_cache
```

## Week 4 Direction

Recommended Week 4 work:

- Generate deterministic hypothesis candidates from `profile.json`, `cleaning_summary.json`, `eda_summary.json`, and `eda_findings.json`.
- Prepare target-aware feature selection guidance.
- Add baseline modeling only after the Week 3 EDA artifacts are stable.
- Keep correlation findings framed as relationships, not causal claims.
- Continue avoiding LLM API calls and paid credits until the project intentionally adds them.
