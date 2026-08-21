# AutoDS Agent

AutoDS Agent is a compact, auditable agent-vs-deterministic machine-learning workflow for supervised classification and regression on tabular CSV datasets. It combines an LLM-based data scientist with deterministic checks so that model training never starts from an unexamined language-model suggestion.

The central idea is simple:

1. An independent planning agent reads the question and a dataset profile and proposes a target, task type, preprocessing, and model family.
2. A deterministic recommender independently inspects the schema, missingness, cardinality, and dataset size and proposes its own plan.
3. A validation gate compares both plans before any model is fit, then applies the same deterministic feasibility and leakage contract even when the plans agree.
4. Agreement records a provisional choice. Disagreement triggers a second agent call that investigates the deterministic recommendation and records a justified final choice; either path must pass deterministic validation.
5. Specialist calls help plan cleaning, interpret computed EDA, and write the final report. All data transformations, metrics, plots, and saved model artifacts remain deterministic and inspectable.

This makes the boundary between probabilistic reasoning and reproducible data science visible rather than hiding it behind an autonomous chain.

### Agent roles

- Modeling agent: independently proposes the target, classification/regression task, preprocessing, and model family.
- Deterministic validator: independently checks the target/schema and recommends a fitting family before training.
- Reconciliation agent: investigates only disagreements and records the final choice with evidence.
- Cleaning agent: selects safe structural cleaning actions from an allow-list.
- EDA agent: interprets deterministic summaries and plots without inventing measurements.
- Report agent: turns the saved evidence into the final narrative and next steps.

The actual fit is deliberately deterministic after the gate: scikit-learn trains the approved model with a reproducible preprocessing pipeline and evaluation protocol.

## What is included

- OpenAI Responses API calls with strict JSON-schema outputs for planning, reconciliation, cleaning, EDA interpretation, and report drafting.
- Deterministic schema profiling and classification/regression inference.
- A pre-training validation gate persisted in `decision.json`.
- Safe structural cleaning with an allow-list; learned imputation and encoding live inside scikit-learn pipelines.
- Cross-validation on the training partition only, followed by one untouched holdout evaluation.
- A saved model, plots, compact JSON artifacts, Markdown report, and generated reproduction script for every run.
- Small tests covering recommendation, model training, artifact persistence, and the full offline path.
- Bundled public benchmark datasets for a repeatable walkthrough.

## Architecture

The code is deliberately small:

```text
app/
  deterministic.py  # profile, task inference, recommendation, cleaning, EDA, plots
  validation.py     # fail-closed target, leakage, preprocessing, and split/CV invariants
  preprocessing.py  # typed contract, requirement derivation, and canonical builder
  llm.py            # OpenAI Responses API specialist agents
  modeling.py       # training-only preprocessing and approved model evaluation
  pipeline.py       # orchestration and the pre-training gate
  reporting.py      # Markdown report and reproduction script
  schemas.py        # structured agent outputs
  cli.py            # command-line interface
```

The important control flow is:

```text
CSV + question
    |
    v
profile -> independent agent plan
    |                  \
    |                   \ mismatch
    v                    -> reconciliation agent
deterministic plan ----/
    |
    v
VALIDATION GATE
    |
    v
clean -> EDA -> approved training -> report + artifacts
```

The deterministic recommendation is intentionally not a model-selection benchmark. It is a pre-training policy that makes a defensible recommendation from observable data characteristics. Once the gate approves a method, a separate deterministic contract still has to prove that training may safely proceed. Agreement between two recommenders is not proof of model quality or data validity.

## Deterministic validation boundary

`app/validation.py` is the hard boundary before fitting. It evaluates the approved target, task, method, feature matrix, holdout fraction, and cross-validation strategy against the actual dataframe. Every check has a stable code, pass/fail status, compact evidence, and an actionable message; the evidence is saved in `decision.json` and repeated in `modeling.json` for successful runs.

The contract stops training when:

- the target is missing, ambiguous, invalid for the approved task, has too few valid values/classes, or contains invalid/non-finite regression values;
- missing target rows would otherwise become artificial string labels;
- the target appears in the feature matrix, a feature is a proven target copy, or identical usable feature rows have conflicting targets;
- schema safeguards leave no usable feature, or a requested method is unsupported;
- `test_size`, stratification, holdout size, training-fold class coverage, or regression CV feasibility is invalid;
- numeric infinities cannot be handled under the documented policy.

Target rows are filtered deterministically before classification labels are converted to strings. Numeric feature infinities become missing values before training-only imputation. Imputation, scaling, encoding, and model fitting remain inside the scikit-learn pipeline; the holdout is reserved for the final evaluation and is not used for reconciliation, preprocessing fitting, or cross-validation. Boosted trees use bounded ordinal encoding for categorical features instead of a dense one-hot expansion.

Name-based indicators such as a feature containing the target name are recorded as warnings for domain review. They do not block a run without stronger deterministic evidence. Semantic leakage, post-outcome variables, temporal leakage, proxy variables, and whether a feature was available at prediction time still require subject-matter review. These checks prove that a run meets the workflow's safety and feasibility preconditions; they do not prove that leakage is impossible, that the chosen family is empirically optimal, or that the model is fit for deployment.

## Preprocessing contract and validation gate

Structural cleaning and learned preprocessing are intentionally separate. The cleaning plan may trim strings, coerce numeric strings, remove exact duplicates, drop all-null or constant columns, and remove rows with missing targets. These operations happen before the modeling view is validated. Imputation, scaling, categorical encoding, and any other learned transformation are never cleaning actions; they are fitted inside the scikit-learn pipeline after the train/holdout split and inside every cross-validation fold.

The modeling agent and deterministic recommender independently return the typed `PreprocessingContract` persisted in `decision.json`. Its compact vocabulary includes numeric and categorical imputation, numeric scaling, one-hot or ordinal encoding, safe unknown-category handling, infinity handling, identifier/high-cardinality/text/datetime policies, and the invariant `fit_inside_pipeline=true`. Known legacy names are accepted only as an allow-listed migration and are serialized back to the typed object; arbitrary transformation names and Python code are rejected.

The deterministic validator derives requirements from the observed feature schema, missingness, infinities, cardinality, task, and model family. Some requirements are safety invariants and some are optional preferences. Numeric and categorical missing values require their corresponding imputers; usable categories require a supported encoder with safe unknown handling; linear methods require standard numeric scaling under this project policy; tree ensembles are not forced to scale; boosted trees use bounded ordinal encoding; identifiers, unsupported text, datetimes, and high-cardinality features remain excluded; and oversized one-hot matrices fail closed. No holdout row is used to fit preprocessing.

The gate compares normalized material preprocessing behavior in addition to target, task, and method. Ordering or absent-feature differences are recorded as immaterial. For example, if the agent proposes no imputation but the data contains missing numeric values, the difference is material and reconciliation is required. A reconciliation response must return a complete supported contract and explicitly discuss material preprocessing differences. The selected contract is validated again after structural cleaning; failed invariants stop the run before model fitting regardless of confidence or justification.

Example:

```text
Agent:       numeric_imputation=none, categorical_encoding=one_hot
Deterministic: numeric_imputation=median, categorical_imputation=most_frequent,
               categorical_encoding=one_hot (missing values observed)
Gate:        preprocessing disagreement -> reconciliation
Approved:    median + most_frequent imputation, one_hot(handle_unknown=ignore),
             identifier_handling=exclude, fit_inside_pipeline=true
Executed:    the approved contract builds the saved ColumnTransformer/Pipeline
```

The final approved contract, both independent proposals, normalized comparison, deterministic requirements, executed pipeline steps, feature exclusions and reasons, and training-only status appear in `decision.json`, `modeling.json`, and the final report. `reproduce_analysis.py` loads that recorded contract, validates it against the dataset, and passes it to the same canonical pipeline builder; it never asks an agent to make a new preprocessing decision.

## Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For API-backed runs, set the key in the environment. Do not commit it:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "gpt-4.1-mini"
```

The implementation follows the official OpenAI Responses API pattern for structured outputs. See the [OpenAI API quickstart](https://platform.openai.com/docs/quickstart) and [Responses API reference](https://developers.openai.com/api/reference/typescript/resources/beta/subresources/responses/methods/create).

## Run it

API-backed example:

```powershell
python -m app --data examples/sample_data/breast_cancer_wisconsin.csv --target diagnosis --question "Can we classify diagnosis from the measured cell features?"
```

Offline example:

```powershell
python -m app --data examples/sample_data/diabetes_progression.csv --target disease_progression --question "Can we estimate disease progression from the patient measurements?" --offline
```

A run folder is created under `runs/<run_id>/`. The CLI prints the report path. To run both bundled demos:

```powershell
python scripts/run_demo.py --offline
```

See [the demo notes and generated figures](docs/demo.md) for a compact walkthrough.

## Reading a run

The most important files are:

- `decision.json`: independent plan, deterministic plan, final gate status, reconciliation evidence, and every deterministic invariant check.
- `profile.json`: compact dataset facts supplied to the agents.
- `cleaning.json`: requested and applied structural actions.
- `eda.json`: deterministic EDA values plus agent findings and plot paths.
- `modeling.json`: selected model, CV metrics, holdout metrics, feature handling, and artifact path.
- `report.md`: analyst-style narrative.
- `reproduce_analysis.py`: executable replay of the approved decision plus the deterministic cleaning, EDA, and modeling stages.
- `model/selected_model.joblib`: fitted scikit-learn pipeline.

Generated run folders are ignored by Git so datasets, models, and reports do not accidentally become repository history. The agent decisions remain inspectable in `decision.json`; replay does not silently make a new semantic decision.

## Method and evaluation details

Supported method families are `linear`, `regularized_linear`, `tree_ensemble`, and `boosted_tree`. They map to logistic/linear regression, Ridge or regularized logistic regression, random forests, and histogram gradient boosting.

The workflow excludes obvious identifiers, free text, datetime columns, constant features, and extremely high-cardinality features from the compact baseline, recording each exclusion reason. The approved contract determines whether observed numeric and categorical missing values are imputed, whether numeric features are scaled, and whether categories use one-hot or bounded ordinal encoding. Cross-validation and preprocessing are fitted only on the training partition. The final holdout is not used to choose the method or fit preprocessing. These are hard preconditions for safe execution, not a claim that the approved family is the best empirical model.

Metrics:

- Classification: macro F1, weighted F1, balanced accuracy, and accuracy.
- Regression: RMSE, MAE, and R-squared.

These are baseline engineering choices, not universal defaults.

## Security and data handling

The API key is read from `OPENAI_API_KEY` and is never written to a run artifact. The prompts contain a compact profile and computed summaries rather than the full CSV. Because profile values can still contain sensitive information, do not run confidential data through a hosted model without reviewing your organization's data policy.

Rotate any key that has been pasted into chat, a terminal transcript, or a repository.

## Limitations and expansion points

This is a strong foundation for experimentation, not an autonomous production data scientist. It currently focuses on CSV tables, supervised classification/regression, a small allow-list of model families, and one holdout split. Natural next extensions are schema adapters for Parquet/SQL, richer target inference, uncertainty and calibration, experiment tracking, formal evaluation of agent-vs-deterministic agreement rates, and domain-specific semantic leakage policies. The deterministic gate validates whether training may safely proceed; it does not replace domain review or empirical model comparison.

The design keeps those extensions localized: schema/recommendation policies live in `app/deterministic.py`, hard training invariants in `app/validation.py`, new models in `app/modeling.py`, and new specialist outputs in `app/llm.py` and `app/schemas.py`.
