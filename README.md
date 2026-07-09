# AutoDS Agent

**AutoDS Agent: An Autonomous Multi-Agent Data Science Analyst** is a resume-quality AI side project designed to grow into a system that can ingest raw tabular datasets, profile them, clean them, run EDA, train baseline models, evaluate results, generate visualizations, and write final analysis reports.

Week 1 focuses on the foundation only: clean project structure, CSV upload, run folder management, basic dataset metadata, a FastAPI backend, a Streamlit frontend, and tests. There are no LLM calls, paid API dependencies, MLflow tracking, or model training yet.

## Long-Term Vision

The eventual system will use a multi-agent architecture with specialist agents for:

- Dataset profiling
- Cleaning recommendations and execution
- Exploratory data analysis
- Hypothesis generation
- Modeling
- Evaluation
- Report generation

The Week 1 code intentionally keeps those pieces as placeholders so the foundation stays simple and easy to extend.

## Week 1 Features

- FastAPI backend with health, upload, and run metadata endpoints
- Streamlit UI for CSV uploads and dataset preview
- Unique `run_id` generation for every analysis run
- Run folder structure under `runs/<run_id>/`
- Raw CSV preservation at `runs/<run_id>/input/raw_data.csv`
- Metadata JSON saved at `runs/<run_id>/intermediate/metadata.json`
- Basic metadata:
  - Row count
  - Column count
  - Column names
  - Column data types
  - Missing value counts
  - Duplicate row count
  - First five rows as preview
- Placeholder agent and workflow modules for future weeks
- Pytest coverage for run management and dataset metadata

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
      services/
        run_manager.py
        dataset_service.py
      schemas/
        run.py
        dataset.py

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

Optional: copy `.env.example` to `.env` if you want to customize local settings.

```bash
AUTODS_BACKEND_URL=http://localhost:8000
AUTODS_RUNS_DIR=runs
```

## Run the Backend

From the project root:

```bash
uvicorn app.backend.main:app --reload
```

Health check:

```text
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "autods-agent-backend"
}
```

## Run the Frontend

In a second terminal, from the project root:

```bash
streamlit run app/frontend/streamlit_app.py
```

The frontend expects the backend at:

```text
http://localhost:8000
```

Set `AUTODS_BACKEND_URL` if your backend runs somewhere else.

## Example Usage Flow

1. Start the FastAPI backend.
2. Start the Streamlit frontend.
3. Upload a CSV file in the Streamlit app.
4. The backend creates a unique run folder under `runs/`.
5. The raw dataset is saved as `runs/<run_id>/input/raw_data.csv`.
6. Metadata is saved as `runs/<run_id>/intermediate/metadata.json`.
7. The UI displays the run ID, dataset shape, data types, missing values, duplicate rows, and a preview.

## API Endpoints

### `GET /health`

Returns a simple service health check.

### `POST /upload`

Accepts a CSV file upload, creates a run, saves the raw dataset, generates metadata, saves it, and returns the metadata.

### `GET /runs/{run_id}`

Returns saved metadata for one run.

### `GET /runs`

Returns available run IDs with lightweight metadata when available.

## Run Tests

```bash
pytest
```

## Future Weeks

Recommended Week 2 work:

- Add richer dataset profiling
- Add schema inference beyond basic pandas dtypes
- Detect high-cardinality columns, constants, likely identifiers, and target candidates
- Produce a cleaning plan without modifying the raw input
- Add tests for malformed CSVs and upload endpoint behavior
- Start defining a shared analysis state object for future agents

Later weeks can add EDA plots, modeling, evaluation, MLflow, LangGraph or OpenAI Agents SDK orchestration, and report generation.
