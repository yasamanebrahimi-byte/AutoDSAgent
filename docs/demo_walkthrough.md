# Demo Walkthrough

This guide shows the quickest way to demonstrate AutoDS Agent locally or with Docker Compose.

## Local Run

Install the project:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Start the backend:

```bash
uvicorn app.backend.main:app --reload
```

Start the frontend:

```bash
streamlit run app/frontend/streamlit_app.py
```

Open Streamlit at `http://localhost:8501`.

## Docker Compose Run

Start backend, frontend, and MLflow:

```bash
docker compose up --build
```

Open:

- Frontend: `http://localhost:8501`
- Backend docs: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`

## Example Datasets

Use the bundled sample CSV files:

- `examples/sample_data/breast_cancer_wisconsin.csv`, target `diagnosis`
- `examples/sample_data/diabetes_progression.csv`, target `disease_progression`

## Suggested Recruiter Demo Path

1. Open the Streamlit app.
2. Upload `breast_cancer_wisconsin.csv`.
3. Create an analysis run.
4. Start the automated workflow.
5. Use target column `diagnosis`.
6. Leave approval gates off for the fastest demo.
7. Review workflow state, artifacts, and trace events.
8. Open `Final Reports`.
9. Preview the executive summary and final report.
10. Download `final_report.md`.
11. If Docker Compose is running, open MLflow and inspect model metrics.

## Demo Script

You can create a full run without using the UI:

```bash
python scripts/create_demo_run.py --dataset classification --target diagnosis
python scripts/create_demo_run.py --dataset regression --target disease_progression
```

The script uses internal services directly. It creates a run, preserves the raw CSV, profiles data, plans and applies cleaning, runs EDA, trains models, and generates final reports.

## MLflow

For local Python runs, install the optional MLflow dependency:

```bash
python -m pip install -e ".[mlflow]"
```

Start MLflow:

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri mlruns --default-artifact-root mlruns
```

Enable tracking:

```bash
AUTODS_ENABLE_MLFLOW=true
AUTODS_MLFLOW_TRACKING_URI=http://localhost:5000
AUTODS_MLFLOW_EXPERIMENT_NAME=AutoDS-Agent
```

If MLflow is unavailable, modeling still completes and saves local artifacts.
