# AutoDS Agent Analysis Report

## Question

Estimate disease progression from the patient measurements.

## Validation gate: Agreement

The workflow intentionally made a modeling decision before fitting any model.

| Source | Target | Task | Method |
| --- | --- | --- | --- |
| Independent agent | <code>disease_progression</code> | <code>regression</code> | <code>regularized_linear</code> |
| Deterministic recommender | <code>disease_progression</code> | <code>regression</code> | <code>regularized_linear</code> |
| Final approved plan | <code>disease_progression</code> | <code>regression</code> | <code>regularized_linear</code> |

Deterministic reasoning: The dataset is numeric, has enough rows for stable validation, and has a feature count where regularization controls coefficient variance.

Validation decision: The independent agent and deterministic recommender agreed on target, task type, method, and material preprocessing behavior before training.

Holdout boundary: target/task establishment completed before the supervised split; the frozen holdout was reserved for final evaluation. Modeling-agent evidence, deterministic recommendation evidence, preprocessing requirements, and any reconciliation used the training partition only.

Deterministic contract: <code>passed</code> (35/35 checks passed); target rows removed: <code>0</code>; direct leakage detected: <code>False</code>.

Excluded features and reasons: <code>none</code>.

### Preprocessing contract

| Source | Contract |
| --- | --- |
| Independent agent | <code>{"categorical_encoding": "none", "categorical_imputation": "none", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "none", "numeric_scaling": "standard", "unsupported_text_handling": "exclude"}</code> |
| Deterministic recommender | <code>{"categorical_encoding": "none", "categorical_imputation": "none", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "none", "numeric_scaling": "standard", "unsupported_text_handling": "exclude"}</code> |
| Final approved executable plan | <code>{"categorical_encoding": "none", "categorical_imputation": "none", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "none", "numeric_scaling": "standard", "unsupported_text_handling": "exclude"}</code> |

Preprocessing comparison: <code>agreement</code>; material differences: <code>none</code>.

Reconciliation output: <code>not invoked</code>.

The deterministic requirements and checks are persisted with the gate. All learned transformations are fitted inside the training pipeline; the holdout is used only for final evaluation. The executed pipeline components are <code>{"categorical": [], "infinity_handling_before_pipeline": "replace_with_missing", "numeric": ["standard_scaling"], "remainder": "drop", "unknown_category_handling": "ignore"}</code>.

## Data profile

- Rows before cleaning: <code>442</code>
- Columns before cleaning: <code>11</code>
- Duplicate rows detected: <code>0</code>
- Rows after cleaning: <code>442</code>
- Columns after cleaning: <code>11</code>

## Cleaning agent

The cleaning agent requested: <code>trim_strings, drop_exact_duplicates, drop_all_null_columns, drop_constant_features, drop_rows_missing_target</code>.

Applied structural actions: <code>trim_strings, drop_exact_duplicates, drop_all_null_columns, drop_constant_features, drop_rows_missing_target</code>.

The offline cleaning fallback applies only structural operations and leaves learned imputation inside the model pipeline.

## Exploratory data analysis

- The cleaned analysis frame contains 442 rows and 11 columns.
- No missing values were present in the cleaned frame.
- The strongest absolute numeric relationship observed was s1 with s2 (absolute r=0.897).

The numeric relationships and target distribution are computed deterministically; the EDA agent only interprets those values.

## Modeling

The approved model was <strong>ridge</strong> using training-only preprocessing and <code>5</code>-fold <code>kfold</code> validation.

### Cross-validation

- <code>cv_rmse_mean</code>: <code>55.8012</code>
- <code>cv_rmse_std</code>: <code>4.1346</code>
- <code>cv_mae_mean</code>: <code>45.0061</code>
- <code>cv_mae_std</code>: <code>3.0448</code>
- <code>cv_r2_mean</code>: <code>0.4714</code>
- <code>cv_r2_std</code>: <code>0.1010</code>

### Untouched holdout

- <code>rmse</code>: <code>53.7775</code>
- <code>mae</code>: <code>42.8120</code>
- <code>r2</code>: <code>0.4541</code>

### Baseline holdout

<code>rmse=72.8862</code>, <code>mae=62.7416</code>, <code>r2=-0.0027</code>

Cross-validation metrics are used for model selection, while the holdout metrics are reported once after the approved method was selected.

## Limitations and next steps

- This is a compact baseline workflow, not a production deployment or causal analysis.
- Performance estimates depend on the supplied data and split seed.

Recommended next steps:

- Review feature definitions and leakage risks with a domain expert.
- Compare additional models and calibration strategies on a representative evaluation set.

## Artifacts

- <code>profile.json</code>
- <code>planning_profile.json</code>
- <code>decision.json</code>
- <code>cleaning.json</code>
- <code>data/cleaned.csv</code>
- <code>eda.json</code>
- <code>modeling.json</code>
- <code>model/selected_model.joblib</code>
- <code>report.md</code>
- <code>reproduce_analysis.py</code>

<code>reproduce_analysis.py</code> replays the approved deterministic stages and verifies the recorded gate decision. The agent decisions themselves are preserved in <code>decision.json</code> rather than regenerated during replay.
