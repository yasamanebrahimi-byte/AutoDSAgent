# AutoDS Agent Analysis Report

## Question

classify target

## Validation gate: Disagreement investigated

The workflow intentionally made a modeling decision before fitting any model.

| Source | Target | Task | Method |
| --- | --- | --- | --- |
| Independent agent | <code>target</code> | <code>classification</code> | <code>tree_ensemble</code> |
| Deterministic recommender | <code>target</code> | <code>classification</code> | <code>linear</code> |
| Final approved plan | <code>target</code> | <code>classification</code> | <code>linear</code> |

Deterministic reasoning: The compact deterministic baseline favors an interpretable linear family when the schema is mostly numeric and no stronger structural signal is available before fitting.

Validation decision: The offline validation fallback selected the deterministic target, task, method, and preprocessing contract because they are tied directly to the observed schema and do not invent a new executable transformation.

Holdout boundary: target/task establishment completed before the supervised split; the frozen holdout was reserved for final evaluation. Modeling-agent evidence, deterministic recommendation evidence, preprocessing requirements, and any reconciliation used the training partition only.

Deterministic contract: <code>passed</code> (37/37 checks passed); target rows removed: <code>0</code>; direct leakage detected: <code>False</code>.

Excluded features and reasons: <code>none</code>.

### Preprocessing contract

| Source | Contract |
| --- | --- |
| Independent agent | <code>{"categorical_encoding": "one_hot", "categorical_imputation": "most_frequent", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "median", "numeric_scaling": "none", "unsupported_text_handling": "exclude"}</code> |
| Deterministic recommender | <code>{"categorical_encoding": "one_hot", "categorical_imputation": "most_frequent", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "median", "numeric_scaling": "standard", "unsupported_text_handling": "exclude"}</code> |
| Final approved executable plan | <code>{"categorical_encoding": "one_hot", "categorical_imputation": "most_frequent", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "median", "numeric_scaling": "standard", "unsupported_text_handling": "exclude"}</code> |

Preprocessing comparison: <code>disagreement</code>; material differences: <code>[{"agent": "none", "deterministic": "standard", "field": "numeric_scaling", "material": true, "reason": "Executable preprocessing behavior differs."}]</code>.

Reconciliation output: <code>{"checks": ["deterministic_target_exists", "deterministic_task_is_feasible", "deterministic_method_is_allow_listed"], "confidence": 0.5, "justification": "The offline validation fallback selected the deterministic target, task, method, and preprocessing contract because they are tied directly to the observed schema and do not invent a new executable transformation.", "selected_method": "linear", "selected_preprocessing": {"categorical_encoding": "one_hot", "categorical_imputation": "most_frequent", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "median", "numeric_scaling": "standard", "unsupported_text_handling": "exclude"}, "selected_target_column": "target", "selected_task_type": "classification"}</code>.

The deterministic requirements and checks are persisted with the gate. All learned transformations are fitted inside the training pipeline; the holdout is used only for final evaluation. The executed pipeline components are <code>{"categorical": ["most_frequent_imputation", "one_hot"], "infinity_handling_before_pipeline": "replace_with_missing", "numeric": ["median_imputation", "standard_scaling"], "remainder": "drop", "unknown_category_handling": "ignore"}</code>.

## Data profile

- Rows before cleaning: <code>100</code>
- Columns before cleaning: <code>3</code>
- Duplicate rows detected: <code>1</code>
- Rows after cleaning: <code>99</code>
- Columns after cleaning: <code>3</code>

## Cleaning agent

The cleaning agent requested: <code>trim_strings, drop_exact_duplicates, drop_all_null_columns, drop_constant_features, drop_rows_missing_target</code>.

Applied structural actions: <code>trim_strings, drop_exact_duplicates, drop_all_null_columns, drop_constant_features, drop_rows_missing_target</code>.

The offline cleaning fallback applies only structural operations and leaves learned imputation inside the model pipeline.

## Exploratory data analysis

- The cleaned analysis frame contains 99 rows and 3 columns.
- Missing values remain visible to the modeling pipeline, where imputation is learned from training folds only.

The numeric relationships and target distribution are computed deterministically; the EDA agent only interprets those values.

## Modeling

The approved model was <strong>logistic_regression</strong> using training-only preprocessing and <code>5</code>-fold <code>stratified_kfold</code> validation.

### Cross-validation

- <code>cv_macro_f1_mean</code>: <code>1.0000</code>
- <code>cv_macro_f1_std</code>: <code>0.0000</code>
- <code>cv_balanced_accuracy_mean</code>: <code>1.0000</code>
- <code>cv_balanced_accuracy_std</code>: <code>0.0000</code>
- <code>cv_accuracy_mean</code>: <code>1.0000</code>
- <code>cv_accuracy_std</code>: <code>0.0000</code>

### Untouched holdout

- <code>accuracy</code>: <code>1.0000</code>
- <code>macro_f1</code>: <code>1.0000</code>
- <code>weighted_f1</code>: <code>1.0000</code>
- <code>balanced_accuracy</code>: <code>1.0000</code>

### Baseline holdout

<code>accuracy=0.5000</code>, <code>macro_f1=0.3333</code>, <code>weighted_f1=0.3333</code>, <code>balanced_accuracy=0.5000</code>

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
