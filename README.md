# AutoDS Agent

AutoDS Agent is a compact, auditable agent-vs-deterministic machine-learning workflow for supervised classification and regression on tabular CSV datasets. It combines an LLM-based data scientist with deterministic checks so that model training never starts from an unexamined language-model suggestion.

The central idea is simple:

1. Establish the target and task type. A supplied target is validated directly; an omitted target is inferred from the question and dataset schema.
2. Freeze one deterministic supervised train/holdout partition using positional row membership, the configured seed, and stratification when required.
3. Give the independent modeling agent and deterministic recommender the same compact profile built from training rows only.
4. A validation gate compares both independent plans before any model is fit. Disagreement triggers a second agent call that investigates the deterministic recommendation using the same training-only evidence.
5. Apply structural cleaning while preserving original row positions, then compute EDA and plots from cleaned training rows only. The EDA agent interprets those training-only summaries before the approved model is fit.
6. Specialist calls write the final report. All data transformations, metrics, plots, and saved model artifacts remain deterministic and inspectable.

This makes the boundary between probabilistic reasoning and reproducible data science visible rather than hiding it behind an autonomous chain.

The agent recommendation can never substitute for a failed deterministic recommendation. If the independent deterministic recommender is unavailable, the run fails closed before model training. A recorded agreement means both independent recommendation paths completed successfully and materially agreed.

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
- Training-only modeling-agent planning, deterministic diagnostics/recommendation, reconciliation, preprocessing requirements, structural-cleaning decisions, EDA, and plots, followed by training-only cross-validation and one untouched holdout evaluation.
- A saved model, plots, compact JSON artifacts, Markdown report, and generated reproduction script for every run.
- Small tests covering recommendation, model training, artifact persistence, and the full offline path.
- Bundled public benchmark datasets for a repeatable walkthrough.

## Architecture

The code is deliberately small:

```text
app/
  deterministic.py  # profile, task inference, public recommendation, cleaning, EDA, plots
  deterministic_diagnostics.py  # compact training-only structural and target diagnostics
  deterministic_policy.py  # versioned thresholds, compatibility scoring, eligibility
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
target/task establishment
    |
    v
freeze supervised train/holdout partition
    |
    +---- holdout locked until final evaluation
    |
    v
training-only profile
    |                  \
    +--> independent modeling agent
    |                   \ mismatch
    +--> deterministic recommender -> reconciliation agent
    |
    v
VALIDATION GATE
    |
    v
structural cleaning -> training-only EDA + plots
    |
    v
training-only CV + preprocessing
    |
    v
one final holdout evaluation -> report + artifacts
```

Target establishment is the one unavoidable pre-split stage: the target and task are needed to construct the correct supervised split, especially a stratified classification split. It does not choose a model family. After target/task establishment, the holdout membership is frozen before any model-family, preprocessing, reconciliation, structural-cleaning, or model-selection reasoning. Modeling-agent planning, deterministic diagnostics/recommendation, reconciliation, preprocessing requirements, structural-cleaning decisions, pre-evaluation EDA and plots, and cross-validation use training-partition evidence only. The fail-closed validation gate may inspect the full raw or cleaned frame only to enforce target, schema, feasibility, and frozen-membership invariants; those checks are guardrails rather than planning evidence. The holdout is reserved for the one final model evaluation.

The deterministic recommendation is intentionally not a model-selection benchmark or a second predictive model. Policy version 2 profiles only the frozen training partition and scores the compatibility of `linear`, `regularized_linear`, `tree_ensemble`, and `boosted_tree` using explicit, auditable factors: dataset scale, usable feature count, sample-to-feature ratio, numeric/categorical/binary composition, exclusions, missingness pattern, categorical cardinality and estimated one-hot expansion, candidate post-preprocessing dimensionality, numeric multicollinearity, Pearson/Spearman and binned-target nonlinearity signals, heuristic interaction potential, numeric outlier burden, and task-appropriate target balance or robustness diagnostics. Compatibility scores are bounded policy points, not probabilities and not claims of optimality. The recommendation persists the score for every family, ranked methods, eligibility and safety reasons, diagnostics, score contributions, confidence/margin, preprocessing contract, and policy version. It remains independent of the LLM, holdout, empirical reference, and prior live results.

The policy uses the same canonical categorical-cardinality and one-hot safety limits as `app.preprocessing`. Linear and regularized-linear families use the production one-hot path; boosted trees use the production ordinal path. Candidate one-hot families are marked ineligible when the estimated encoded dimension exceeds the canonical safe bound; this is distinct from merely ranking a valid family lower. The validation gate remains the fail-closed authority after recommendation. In classification, `linear` uses unregularized logistic regression (`penalty=None`) while `regularized_linear` uses L2-regularized logistic regression; regression uses `LinearRegression` versus `Ridge`. Boosted trees can rank first when nonlinear evidence, adequate sample size, and manageable effective dimensionality jointly support them, but the policy does not force that outcome.

The policy can be wrong and is evaluated separately against the evaluation-only empirical CV reference. That reference never enters runtime recommendation or reconciliation, and the holdout is never used for model-family recommendation.

If the deterministic recommender fails, the validation gate remains incomplete and the run stops before training; the agent plan is retained for auditability, but it is never copied into a deterministic recommendation.

## Deterministic validation boundary

`app/validation.py` is the hard boundary before fitting. It evaluates the approved target, task, method, feature matrix, frozen positional membership, holdout fraction, and cross-validation strategy against the actual dataframe. Every check has a stable code, pass/fail status, compact evidence, and an actionable message; the evidence is saved in `decision.json` and repeated in `modeling.json` for successful runs. The split contract records a source-data fingerprint and digests of the train/holdout positions, so reproduction fails if the source data changes or a different partition would be constructed.

The contract stops training when:

- the target is missing, ambiguous, invalid for the approved task, has too few valid values/classes, or contains invalid/non-finite regression values;
- missing target rows would otherwise become artificial string labels;
- the target appears in the feature matrix, a feature is a proven target copy, or identical usable feature rows have conflicting targets;
- schema safeguards leave no usable feature, or a requested method is unsupported;
- `test_size`, stratification, holdout size, training-fold class coverage, or regression CV feasibility is invalid;
- numeric infinities cannot be handled under the documented policy.

Target rows are filtered deterministically before classification labels are converted to strings. Numeric feature infinities become missing values before training-only imputation. Imputation, scaling, encoding, and model fitting remain inside the scikit-learn pipeline; the holdout is reserved for the final evaluation and is not used for reconciliation, preprocessing fitting, pre-evaluation EDA, plots, or cross-validation. Boosted trees use bounded ordinal encoding for categorical features instead of a dense one-hot expansion.

Name-based indicators such as a feature containing the target name are recorded as warnings for domain review. They do not block a run without stronger deterministic evidence. Semantic leakage, post-outcome variables, temporal leakage, proxy variables, and whether a feature was available at prediction time still require subject-matter review. These checks prove that a run meets the workflow's safety and feasibility preconditions; they do not prove that leakage is impossible, that the chosen family is empirically optimal, or that the model is fit for deployment.

## Preprocessing contract and validation gate

Structural cleaning and learned preprocessing are intentionally separate. The cleaning plan is selected from the training-only profile and may trim strings, coerce numeric strings, remove exact duplicates, drop all-null or constant columns, and remove rows with missing targets. The same deterministic structural actions are applied with original row positions carried through, so validation and fitting can verify the frozen membership after cleaning. Imputation, scaling, categorical encoding, and any other learned transformation are never cleaning actions; they are fitted inside the scikit-learn pipeline after the train/holdout split and inside every cross-validation fold. Global deterministic normalization is safe to apply consistently; training-derived structural decisions use training evidence; learned preprocessing is always fit inside the training pipeline.

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

## Evaluation harness

The repository includes a separate, evaluation-only harness for the central research question:

> Given identical training-only evidence, does deterministic validation and reconciliation improve the safety and quality of an LLM data-science agent's modeling decision?

It compares three auditable conditions:

```text
agent_initial  →  the independent proposal before reconciliation
gated_final    →  the existing deterministic validation/reconciliation path
empirical reference →  post-hoc training-only CV across all supported families
```

Runtime validation does not search every candidate model before approving a plan. The empirical reference is evaluation-only code and cannot influence the runtime decision. It compares `linear`, `regularized_linear`, `tree_ensemble`, and `boosted_tree` with the canonical preprocessing builder. Classification uses macro F1 as the primary metric; regression uses RMSE. The holdout is scored only after plans and candidate rankings are frozen.

The decision sequence is deliberately ordered as:

```text
establish target/task
  → freeze train/holdout split
  → build training-only profile
  → agent_initial
  → deterministic recommendation
  → compare / reconcile
  → deterministic validation gate
  → model training
```

The initial modeling request contains only the question, established target/task, and frozen training-only profile. It does not receive the deterministic recommendation, empirical-reference scores, holdout values, or previous repetitions. Repetitions for a case use the same split seed, split membership, and profile; only the stochastic LLM response is intended to vary. Each trial records the split contract, source, model, timestamp, repository commit, generation settings, prompt/schema version, structured initial and reconciliation responses, validation evidence, empirical CV table, and final holdout metrics.

### Benchmark suite and run presets

The default local suite needs no network download:

| Task | Cases |
|---|---|
| Classification | `breast_cancer`, `wine` |
| Regression | `diabetes`, `synthetic_regression` |

```powershell
# Small offline smoke test
python -m evaluation.run --offline --case wine --repetitions 1 --output evaluation_results/live_smoke

# Main live study: 10 independent OpenAI responses per case
python -m evaluation.run --live --repetitions 10 --output evaluation_results/live_main

# Larger research-style study
python -m evaluation.run --live --repetitions 25 --output evaluation_results/live_extended
```

For a live run, set `OPENAI_API_KEY` and optionally `OPENAI_MODEL`. `--live` is explicit but the default when `--offline` is absent. If credentials are unavailable, the run records `offline_fallback`; if an API request fails and production fallback handles it, the row records that failure separately. Neither fallback nor mock rows are included in OpenAI-only metrics.

Interrupted runs can be continued safely with `--resume` using the same configuration:

```powershell
python -m evaluation.run --live --repetitions 10 --output evaluation_results/live_main --resume
```

The harness verifies configuration compatibility, skips completed trial IDs, writes after each completed trial, and refuses to overwrite an existing bundle without `--resume`. `empirical_reference.json` stores the in-run training-only reference cache so repeated identical-evidence trials do not recompute candidate CV.

### Metrics and interpretation

The primary live denominator is `agent_source == "openai"`. The summary separately reports requested live trials, successful OpenAI trials, offline fallbacks, failed executions, provider-request failures, and mock trials. Operational/all-source metrics remain available for safety and pipeline coverage, but conclusions about LLM behavior use the OpenAI-only view.

The empirical reference ranks all four supported families using identical training-only CV folds whenever possible. Classification retains macro F1, balanced accuracy, and accuracy; regression retains RMSE, MAE, and R². The ranking is frozen before either method is scored on the untouched holdout. It is a benchmark under this candidate set and CV design, not an oracle or universal optimum.

For each comparable trial, normalized regret is larger-is-worse: classification regret is `max(0, best_macro_f1 - selected_macro_f1)`; regression regret is `max(0, selected_rmse - best_rmse) / max(abs(best_rmse), 1e-12)`. A gate outcome is `improved`, `worsened`, or `tie` when gated regret differs from initial regret by more than the configured tolerance (`0.02` by default); otherwise it is a tie. Paired CV improvement is `gated_macro_f1 - initial_macro_f1` for classification and `initial_rmse - gated_rmse` for regression, so positive always means gating helped. The summary reports mean, median, standard deviation, counts, method distributions, modal-method frequency, pairwise consistency, reference-match rates, reconciliation sides, and dataset-level breakdowns.

“Potentially unnecessary intervention” is intentionally cautious: the initial plan was valid, the agent and deterministic methods disagreed, the final method changed, and the initial method was within the task-specific reference tolerance. It does not prove that the gate was universally wrong. Holdout results are retained as a descriptive external check, not used to label the gate outcome.

Add the deterministic missing-value, infinity, identifier, target-copy, invalid-regression-target, and classification-feasibility scenarios with:

```powershell
python -m evaluation.run --offline --include-perturbations --repetitions 1 --output evaluation_results/perturbations
```

Results are written as `config.json`, `trials.jsonl`, `summary.json`, `summary.md`, and `empirical_reference.json`. `summary.md` is generated from structured rows, never by an LLM, and includes experiment configuration, trial coverage, OpenAI-only stability, agreement, empirical-reference comparison, gate outcomes, reconciliation, predictive performance, dataset results, safety interceptions, and limitations. Meaningful claims about LLM behavior require successful `openai` trials; offline fallback and mocks are useful for pipeline testing only.

## Reading a run

The most important files are:

- `decision.json`: independent plan, deterministic plan, final gate status, reconciliation evidence, and every deterministic invariant check.
- `profile.json`: compact full-dataset facts recorded after final evaluation for report context; `planning_profile.json` is the training-only profile supplied to modeling and reconciliation.
- `planning_profile.json`: the compact profile supplied to modeling and reconciliation, built from frozen training rows only.
- `split_contract` in `decision.json`: target/task, seed, holdout policy, source fingerprint, and train/holdout position digests.
- `cleaning.json`: requested and applied structural actions.
- `eda.json`: deterministic training-partition-only EDA values, scope metadata, agent findings, and plot paths.
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

The design keeps those extensions localized: the public recommendation interface lives in `app/deterministic.py`, diagnostics in `app/deterministic_diagnostics.py`, the versioned scoring policy in `app/deterministic_policy.py`, hard training invariants in `app/validation.py`, new models in `app/modeling.py`, and new specialist outputs in `app/llm.py` and `app/schemas.py`.
