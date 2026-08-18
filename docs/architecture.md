# AutoDS Agent Architecture

AutoDS Agent is a deterministic, agent-structured data science workflow for tabular CSV datasets. The project is organized around a persistent run folder so every step produces auditable files that can be inspected, downloaded, reported on, or tracked in MLflow.

## System Overview

The system has two user-facing surfaces:

- FastAPI backend: upload, analysis, modeling, workflow, report, and config endpoints.
- Streamlit frontend: upload flow, workflow controls, artifact previews, final reports, and download buttons.

The backend is intentionally service-oriented. Agents wrap deterministic services rather than calling paid LLM APIs. This keeps the project reproducible while leaving clear extension points for future LLM-assisted planning and narrative generation.

## Backend Structure

```text
app/backend/
  main.py
  config.py
  routes/
  services/
  schemas/
```

Routes expose HTTP endpoints. Services implement the actual data work. Schemas keep API responses and saved artifacts structured.

## Artifact Lifecycle

Each upload creates a unique run folder:

```text
runs/<run_id>/
  input/
  intermediate/
  models/
  plots/
  reports/
  logs/
```

The raw CSV is preserved at `input/raw_data.csv`. Later steps save machine-readable JSON, plots, model files, Markdown reports, and workflow logs. Report generation reads available artifacts and produces partial reports when optional upstream artifacts are missing.

EDA and evaluation plots have separate ownership under `plots/eda/` and `plots/evaluation/`. Regenerating EDA clears only EDA plots and does not delete model-evaluation artifacts.

## Agent Workflow

The automated workflow uses a deterministic state machine:

```text
profile -> cleaning_plan -> cleaning -> eda -> modeling -> report
```

Each step updates `logs/workflow_state.json`. Human approval gates can pause cleaning and modeling. `logs/agent_trace.json` records workflow events separately from application logs.

## Manual Endpoints Versus Workflow

Manual endpoints let users run profiling, cleaning, EDA, modeling, and reports directly. The automated workflow calls the same deterministic services through agent wrappers, so manual and workflow behavior stay aligned.

Workflow starts are submitted to a bounded background worker pool. The API returns a pollable job immediately, while the Streamlit frontend refreshes the persisted workflow state every two seconds until execution completes, fails, or pauses for approval.

Every workflow execution and manual artifact mutation acquires a per-run file lock under `logs/.mutation.lock`. The lock coordinates threads and backend processes; conflicting API mutations fail fast with `409 Conflict` and a `Retry-After` header instead of racing to overwrite state or artifacts. Read-only endpoints remain available while a run is active because JSON artifacts are saved atomically.

The frontend is exercised headlessly with Streamlit AppTest. Tests interact with the real widget tree and session state while a deterministic fake backend covers upload, background polling, manual controls, report downloads, and error recovery without network or browser flakiness.

## Supervised Modeling Flow

Modeling uses one train/test split per run. The test partition is not used for candidate selection:

```text
target selection
  -> structural cleaning
  -> train/test split
  -> training-only preprocessing
  -> cross-validated candidate selection on training data
  -> fit selected model on the complete training partition
  -> one final evaluation on untouched test data
```

Structural cleaning may remove exact duplicates, drop safe constant columns, or normalize deterministic datatypes. Learned operations such as numeric imputation, categorical filling, scaling, and one-hot encoding belong to sklearn pipelines and are fit only within CV folds or on the final training partition. Classification uses stratified splitting and stratified CV when class counts make that mathematically reliable; otherwise the workflow fails with a clear rare-class or split-feasibility message.

## MLflow Integration

MLflow is optional and controlled by environment variables:

```text
AUTODS_ENABLE_MLFLOW=true
AUTODS_MLFLOW_TRACKING_URI=http://localhost:5000
AUTODS_MLFLOW_EXPERIMENT_NAME=AutoDS-Agent
```

When enabled, modeling logs a parent run for the AutoDS run and nested runs for each attempted model. Parameters, metrics, tags, model summaries, evaluation summaries, model results, evaluation plots, and selected model artifacts are logged when available. If MLflow is disabled or unreachable, modeling continues and local artifacts remain the source of truth.

## Future LLM Integration Points

Future LLM-assisted work can plug into these boundaries:

- Agent planning before workflow steps
- Hypothesis generation from profile and EDA summaries
- Narrative polishing for reports
- Interactive Q&A over saved run artifacts

The current project does not require paid LLM API calls.
