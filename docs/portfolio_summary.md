# Portfolio Summary

## Project Pitch

AutoDS Agent is a deterministic, agent-structured data science workflow for tabular CSV datasets. It turns an uploaded dataset into a reproducible analysis run with profiling, conservative cleaning, exploratory analysis, baseline modeling, evaluation, final reports, and optional MLflow experiment tracking.

The project is designed as a serious engineering portfolio piece: deterministic, auditable, locally runnable, Docker-ready, and organized around durable artifacts rather than notebook state.

## Key Technical Features

- FastAPI backend with structured routes, services, and schemas
- Streamlit frontend for upload, workflow control, previews, and downloads
- Persistent run folders with raw data preservation
- Rich dataset profiling and data quality warnings
- Conservative cleaning plan and safe cleaning execution
- Deterministic EDA findings and Matplotlib plot artifacts
- Regression and classification modeling with sklearn pipelines
- Baseline comparison and evaluation metrics
- Final Markdown reports with partial-report handling
- Deterministic workflow orchestration with retry logic and approval gates
- Optional MLflow experiment tracking
- Docker Compose setup for backend, frontend, and MLflow
- Example datasets and demo scripts

## Tech Stack

- Python
- FastAPI
- Streamlit
- pandas
- scikit-learn
- Matplotlib
- Pydantic
- MLflow
- Docker Compose
- Pytest

## Engineering Highlights

- Separation of concerns across tools, services, routes, schemas, agents, and workflows
- Artifact-first design that makes analysis reproducible and inspectable
- Optional MLflow integration that never blocks local modeling
- Clear safety boundaries around cleaning and modeling approval gates
- Reports that avoid invented claims and surface missing artifacts
- Test coverage for core analysis, workflow, reports, configuration, and MLflow behavior

## Deterministic Agentic Workflow

The automated workflow can run the end-to-end analysis sequence without manually calling each API endpoint. It manages state, tracks attempts, pauses for human approval when configured, retries failed steps, skips optional modeling when no target is selected, and writes final reports from the artifacts that exist.

## Resume Bullet Ideas

- Built an automated tabular data science platform with FastAPI, Streamlit, sklearn, MLflow, and Docker Compose.
- Implemented artifact-driven workflows for profiling, cleaning, EDA, modeling, evaluation, and deterministic report generation.
- Designed optional experiment tracking that logs metrics, parameters, tags, and artifacts without making MLflow a hard runtime dependency.
- Added approval gates, retry logic, trace logs, and partial-report handling for robust automated analysis workflows.

## Future Work

- Record a short walkthrough video using the checked-in real-run screenshot flow
- Add stronger leakage detection and validation strategies
- Add optional LLM-assisted narrative generation behind explicit configuration
- Add model registry and richer experiment comparison
- Package a polished portfolio release
