# Recruiter Walkthrough

## What It Does

AutoDS Agent is an automated data science workflow for CSV datasets. It profiles raw data, creates a conservative cleaning plan, runs exploratory analysis, trains baseline and candidate machine learning models, evaluates results, and generates final Markdown reports.

## Why It Is Useful

It turns a common notebook workflow into a repeatable application. A recruiter or hiring manager can run a sample dataset and inspect the saved outputs without needing to understand every implementation detail.

## Technologies Used

- FastAPI
- Streamlit
- pandas
- scikit-learn
- Matplotlib
- Pydantic
- Optional MLflow
- Docker Compose
- Pytest
- GitHub Actions

## Skills Demonstrated

- Backend API design.
- Frontend demo design.
- Data profiling and quality checks.
- Conservative data cleaning.
- EDA and visualization.
- Regression and classification modeling.
- Evaluation metrics and model comparison.
- Workflow orchestration.
- Human approval gates.
- Artifact management.
- Testing, CI, Docker, and documentation.

## What Makes It Different From A Basic Chatbot

AutoDS Agent does not simply generate text. It executes a structured local workflow, saves artifacts, trains models, records trace logs, and produces reproducible reports. The agent boundaries are deterministic and inspectable, with no paid LLM calls required.

## Artifacts It Produces

- Raw preserved CSV.
- Dataset metadata.
- Profile JSON.
- Cleaning plan and cleaning summary.
- Cleaned CSV.
- EDA summary, findings, Markdown report, and plots.
- Modeling and evaluation summaries.
- Saved baseline and best models.
- Final report, executive summary, technical summary, and limitations report.
- Workflow state and agent trace logs.

## How To Evaluate Quickly

1. Read the README overview and quickstart.
2. Run `python scripts/smoke_test.py`.
3. Run `python scripts/run_full_demo.py --dataset classification`.
4. Open the generated `runs/<run_id>/reports/final_report.md`.
5. Skim `docs/project_showcase.md` and `docs/resume_bullets.md`.
