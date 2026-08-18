# Final Demo Script

Use this script when recording a portfolio demo, presenting in an interview, or walking someone through the project live.

## 1. Opening

"This is AutoDS Agent, a deterministic automated data science workflow for tabular CSV datasets. It takes a raw dataset, preserves the original file, profiles the data, creates a conservative cleaning plan, runs EDA, trains baseline and candidate models, evaluates them, and generates final analyst-style reports."

## 2. Why The Project Exists

"Many data science projects start in notebooks and become hard to reproduce. I built AutoDS Agent to show how the same workflow can be packaged as an auditable application with services, APIs, a UI, persistent artifacts, tests, Docker, and documentation."

## 3. Deterministic Agentic Workflow

"The workflow is not just a button that runs one function. It has specialist agent boundaries for profiling, cleaning, EDA, modeling, evaluation, and reporting. A workflow state file tracks each step, approval gates can pause risky actions, retries are recorded, and an agent trace log explains what happened."

## 4. Tech Stack

- FastAPI backend for APIs and services.
- Streamlit frontend for demo and inspection.
- pandas and scikit-learn for deterministic data science work.
- Matplotlib for saved plots.
- Pydantic for structured request and response schemas.
- Optional MLflow for experiment tracking.
- Docker Compose for local backend, frontend, and MLflow.
- Pytest and GitHub Actions for validation.

## 5. Demo Setup

Open two terminals:

```bash
uvicorn app.backend.main:app --reload
```

```bash
streamlit run app/frontend/streamlit_app.py
```

Then open the Streamlit URL and keep the backend docs available at:

```text
http://localhost:8000/docs
```

## 6. Load A Sample Dataset

In the UI, use `Try A Sample Dataset`.

For classification:

- Select `Breast Cancer Wisconsin Classification`.
- Target: `diagnosis`.
- Task type: `Classification`.

For regression:

- Select `Diabetes Progression Regression`.
- Target: `disease_progression`.
- Task type: `Regression`.

Explain: "The sample is uploaded through the same backend path as a real CSV, so the raw input is preserved under the run folder."

## 7. Run The Automated Workflow

Go to `Automated Workflow`.

- Confirm the target column.
- Leave approval gates on if you want to show human review.
- Click `Start Automated Workflow`.

Explain: "The workflow runs until it completes or reaches an approval gate. State is persisted to `logs/workflow_state.json`."

## 8. Show Approval Gates

If the workflow pauses:

- Open the approval section.
- Explain why the step is gated.
- Approve and continue.

Say: "The point is not to hide risk. Cleaning and modeling are visible, reviewable steps."

## 9. Show EDA Outputs

Open the generated artifact summaries.

Point out:

- Column type summary.
- Missing values and duplicate handling.
- Target distribution.
- Correlation heatmap.
- Target relationship plots.

Say: "EDA is deterministic and saved as JSON, Markdown, and PNG artifacts."

## 10. Show Model Results

Open modeling and evaluation.

Point out:

- Inferred or selected task type.
- Baseline model.
- Candidate models.
- Best candidate model.
- Selected model.
- Primary metric.
- Evaluation plots.

Say: "The preprocessing pipeline is fit on the training split, which helps avoid leakage from fitting imputers or encoders on the test data."

## 11. Show Final Report

Open `Final Reports`.

Preview:

- Executive summary.
- Final report.
- Technical summary.
- Limitations.

Say: "The report is generated from saved artifacts. It marks missing sections clearly instead of inventing unavailable analysis."

## 12. Show MLflow If Enabled

If using Docker Compose or local MLflow:

```text
http://localhost:5000
```

Show:

- Experiment name.
- Run ID tags.
- Parameters.
- Metrics.
- Artifacts.

Say: "MLflow is optional. If it is unavailable, local model training and reporting still complete."

## 13. Show Repo Architecture

Walk through:

- `app/backend/routes`: API endpoints.
- `app/backend/services`: deterministic business logic.
- `app/agents`: agent boundaries.
- `app/workflows`: workflow state and step orchestration.
- `app/tools`: reusable data science utilities.
- `runs/<run_id>`: generated artifacts.
- `tests`: service, workflow, and demo coverage.
- `docs`: portfolio and technical documentation.

## 14. Closing

"This project demonstrates full-stack engineering, data science workflow design, reproducibility, testing, artifact management, and agent-style orchestration. Future work would add stronger leakage checks, richer model validation, production authentication, and optional LLM-assisted narrative generation behind explicit configuration."
