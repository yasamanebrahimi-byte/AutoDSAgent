# Changelog

## Methodology Hardening

- Changed model selection to use cross-validation on the training partition, followed by one final holdout evaluation for the selected model.
- Moved learned missing-value handling out of structural cleaning and into sklearn preprocessing pipelines.
- Added stricter target validation, rare-class handling, target inference reasons, sparse one-hot defaults, and richer classification metrics.
- Isolated EDA plots under `plots/eda/` so EDA reruns do not delete `plots/evaluation/` artifacts.
- Updated Docker builds to install with the repository constraints file and added local Compose healthchecks.

## Week 8 - Portfolio Readiness

- Added full demo runner for regression and classification sample datasets.
- Added smoke test script and expanded project validation.
- Polished Streamlit demo flow with sample dataset loading and clearer stage guidance.
- Added demo output documentation, screenshot checklist, final demo script, recruiter walkthrough, interview talking points, resume bullets, and project showcase copy.
- Added license, contributing notes, project status, issue templates, and GitHub Actions test workflow.
- Added end-to-end demo and smoke workflow tests.

## Week 7 - Engineering Polish

- Added optional MLflow tracking.
- Added Docker and Docker Compose support.
- Expanded configuration management and logging.
- Added example datasets, demo scripts, architecture docs, API docs, and portfolio README polish.

## Week 6 - Final Reports

- Added deterministic final report generation.
- Added executive summary, technical summary, limitations report, report metadata, and report index artifacts.
- Integrated report generation into the autonomous workflow and Streamlit UI.

## Week 5 - Agent Orchestration

- Added persistent workflow state.
- Added structured agent boundaries, workflow step tracking, retry logic, human approval gates, and agent trace logs.
- Added Streamlit workflow controls.

## Week 4 - Modeling And Evaluation

- Added task-type inference, preprocessing pipelines, baseline and candidate models, regression and classification metrics, evaluation plots, and saved model artifacts.

## Week 3 - EDA

- Added deterministic exploratory data analysis, EDA summaries, target-specific analysis, visualization utilities, EDA findings, saved plots, and Markdown EDA reports.

## Week 2 - Profiling And Cleaning

- Added richer schema inference, data quality warnings, cleaning plan generation, safe cleaning execution, and cleaning artifacts.

## Week 1 - Foundation

- Added FastAPI backend, Streamlit frontend, CSV upload flow, unique run folders, raw dataset preservation, metadata generation, placeholder agents, tests, and documentation.
