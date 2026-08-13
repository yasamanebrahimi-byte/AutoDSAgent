# Resume Bullets

Use these as source material and tailor the wording to the role.

## Software Engineering Version

- Built an automated data science analysis platform with FastAPI, Streamlit, Docker Compose, Pydantic, and Pytest, turning raw CSV uploads into reproducible run folders with saved artifacts and reports.
- Designed modular backend services for dataset upload, profiling, cleaning, EDA, modeling, evaluation, reporting, and workflow orchestration with clear API boundaries.
- Implemented persistent workflow state, retry handling, human approval gates, and agent trace logs to make automated analysis runs auditable and recoverable.
- Added Docker and GitHub Actions support to simplify local demos, validation, and CI testing.

## Data Science Version

- Developed a deterministic tabular ML workflow that profiles datasets, detects data quality issues, applies conservative cleaning, performs EDA, trains regression and classification models, and evaluates results with task-specific metrics.
- Built sklearn preprocessing pipelines that handle numeric, categorical, boolean, and datetime-like features while excluding ID-like and unsupported free-text columns.
- Compared baseline and candidate models using RMSE for regression and F1 for classification, saving evaluation summaries, plots, model artifacts, and reproducible metadata.
- Integrated optional MLflow experiment tracking for model parameters, metrics, tags, and artifacts without requiring external services for local execution.

## AI And Agent Version

- Designed a deterministic multi-agent orchestration layer with specialist agent boundaries for profiling, cleaning, EDA, modeling, evaluation, and reporting.
- Implemented human-in-the-loop approval gates before higher-risk cleaning and modeling steps, with persisted decisions and traceable workflow transitions.
- Built an artifact-first reporting agent that generates executive, technical, final, and limitations reports from saved analysis outputs without paid LLM API calls.
- Added retry logic, structured workflow state, and agent trace logs to improve transparency across end-to-end automated analysis runs.

## Combined Version

- Built AutoDS Agent, a deterministic agent-structured data science workflow that profiles raw datasets, generates conservative cleaning plans, performs EDA, trains baseline and candidate ML models, evaluates results, and produces analyst-ready reports through a FastAPI, Streamlit, and Dockerized workflow.
- Designed a structured orchestration layer with persistent workflow state, human approval gates, retry logic, and agent trace logs, improving transparency and reliability across end-to-end automated analysis runs.
- Integrated optional MLflow experiment tracking to log model parameters, metrics, artifacts, and evaluation outputs across regression and classification workflows.
- Packaged the project for portfolio use with sample datasets, full-demo scripts, smoke tests, CI, Docker Compose, technical documentation, recruiter walkthroughs, and interview talking points.
