# Interview Talking Points

## 30-Second Explanation

AutoDS Agent is a portfolio-ready automated data science workflow. It accepts a CSV, preserves the raw input, profiles and cleans it conservatively, runs EDA, trains baseline and candidate models, evaluates them, and generates final reports. It is built with FastAPI, Streamlit, pandas, scikit-learn, optional MLflow, Docker, and Pytest.

## 2-Minute Explanation

The project packages a typical tabular data science workflow into an auditable application. Each run gets a unique folder with raw data, intermediate JSON, plots, models, reports, workflow state, and trace logs. The backend is split into services for profiling, cleaning, EDA, modeling, evaluation, workflow orchestration, and reporting. The Streamlit frontend makes the workflow easy to demo, including bundled regression and classification datasets. The agent layer is deterministic for now, which keeps computation reproducible and avoids paid LLM calls.

## Architecture Explanation

- FastAPI routes expose upload, run lookup, profiling, cleaning, EDA, modeling, reporting, and workflow endpoints.
- Services contain the core deterministic behavior.
- Tools hold reusable functions for schema inference, data quality checks, preprocessing, modeling, evaluation, visualization, report building, and artifact I/O.
- Agents wrap services behind consistent boundaries.
- The workflow layer tracks step status, retries, approval state, outputs, warnings, errors, and artifacts.
- Streamlit calls the backend and visualizes outputs.

## Agent Orchestration Explanation

The workflow order is:

```text
profile -> cleaning_plan -> cleaning -> eda -> modeling -> report
```

Each step has a specialist agent wrapper. The orchestrator runs steps until completion, failure, or an approval gate. State is saved after transitions to `logs/workflow_state.json`, and trace events are saved to `logs/agent_trace.json`.

## Data Cleaning And Safety Explanation

Cleaning is intentionally conservative. It removes exact duplicate rows, handles structural issues such as safe constant-column drops and deterministic datetime normalization, and records warnings for risky cases. It does not learn medians, modes, or categorical fill values across the full dataset; those preprocessing steps live inside sklearn pipelines. Raw uploads are never overwritten; cleaned data is saved separately as `intermediate/cleaned_data.csv`.

## Modeling And Evaluation Explanation

Modeling requires an explicit target column and cleaned data. The workflow creates one train/test split, keeps the test split untouched during model selection, compares candidates with cross-validation on the training split, then fits the selected model on the full training partition and evaluates it once on the holdout. Regression uses MAE, RMSE, and R2, with RMSE as the primary CV metric. Classification uses macro F1 for selection and reports weighted F1, macro F1, balanced accuracy, per-class precision/recall/F1, a confusion matrix, and binary ROC-AUC/average precision when feasible.

## MLflow And Reproducibility Explanation

MLflow is optional. When enabled, modeling logs parameters, metrics, tags, and artifacts. If MLflow is unavailable, local artifacts still save and the workflow continues with a warning.

## Tradeoffs

- Deterministic services are less flexible than LLM reasoning, but they are reproducible and testable.
- The model set is intentionally lightweight for fast demos.
- Reports are artifact-driven, so they avoid hallucination but cannot add unavailable insights.
- The UI is optimized for local portfolio demos, not multi-user production access.

## Limitations

- No paid LLM API calls or natural-language reasoning in the current workflow.
- No deep learning, time-series, geospatial, or text modeling.
- Temporal leakage checks are still future work.
- Docker Compose is for local demos, not hardened production deployment.
- Authentication, authorization, and multi-tenant storage are out of scope.

## Future Improvements

- Add stronger leakage detection and validation strategies.
- Add user-facing model selection controls and lightweight hyperparameter options.
- Add optional LLM-assisted report narration behind explicit configuration.
- Add production auth and object storage.
- Add monitoring for run failures, model drift, and data quality trends.

## Example Questions And Suggested Answers

### Why did you use deterministic services instead of LLMs for computation?

Because profiling, cleaning, modeling, and evaluation need reproducibility. Deterministic services make tests, artifacts, and debugging straightforward. LLMs could later help with narrative reasoning, but they should not replace core statistical computation.

### How do you prevent data leakage?

The project preserves the raw dataset, separates structural cleaning from learned preprocessing, excludes ID-like columns from modeling, performs model selection with CV on the training split only, and evaluates the selected model once on the untouched holdout. I would still add stronger future checks for temporal leakage and domain-specific target leakage.

### How do approval gates work?

The workflow can pause before cleaning or modeling. The state file records the waiting step, reason, details, and approval status. The user can approve, reject, or retry through the API or Streamlit UI.

### How do you decide regression vs. classification?

The preprocessing layer uses deterministic target heuristics and saves the inference reason. Boolean targets, low-cardinality categorical strings, and low-cardinality discrete numeric targets are classification; continuous numeric and numeric-string targets are regression. Identifier-like, constant, high-cardinality text, and rare-class targets are rejected. Explicit classification requests are validated instead of blindly accepted.

### How do you choose the best model?

Successful candidates are compared with cross-validation on the training partition only. Regression selects the lowest mean CV RMSE; classification selects the highest mean CV macro F1. After selection, the chosen model is fit on the complete training partition and evaluated once on the untouched holdout test set.

### What makes this agentic?

The system can run a multi-step analysis workflow from upload to report generation, with persisted state, retries, approval gates, trace logs, and deterministic agent boundaries around each specialist step.

### How would you add LLM-based reasoning later?

I would keep deterministic computation in services and add optional LLM agents for narrative synthesis, hypothesis explanation, and user Q&A over saved artifacts. LLM calls would be explicitly configured and traceable.

### How would you deploy this?

I would containerize the backend and frontend separately, use managed object storage for artifacts, a database for run metadata, authentication for users, and a production model registry or experiment store.

### How would you scale it?

I would move long-running workflow steps to a job queue, store artifacts outside the app container, persist workflow state in a database, and run workers horizontally.

### How would you monitor it?

I would track workflow failures, step durations, data quality warnings, model metrics, report completeness, and MLflow experiment health. Production monitoring would include logs, traces, and alerts.
