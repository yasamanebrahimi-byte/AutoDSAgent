# AutoDS Agent Analysis Report

## Question

classify target

## Problem formulation gate: agreement

The workflow first validated what supervised problem to solve, before creating any train/holdout split.

### User question

classify target

### Agent formulation

- Target: <code>target</code>
- Task: <code>classification</code>
- Reasoning: Offline fallback uses the deterministic schema formulation as a local heuristic; this is not evidence of independent LLM reasoning. The target 'target' is treated as classification because its semantic type is categorical, or its numeric values are low-cardinality label-like (2 unique values across 100 valid rows).
- Confidence: <code>0.4</code>

### Deterministic formulation

- Target: <code>target</code>
- Task: <code>classification</code>
- Reasoning: The target 'target' is treated as classification because its semantic type is categorical, or its numeric values are low-cardinality label-like (2 unique values across 100 valid rows).
- Evidence: <code>["target_column=target", "target_source=user_supplied", "target_dtype=object", "target_semantic_type=categorical", "valid_target_rows=100", "unique_values=2", "unique_value_fraction=0.0200", "numeric_or_coercible_fraction=0.0000", "model_selection_diagnostics_not_used"]</code>

### Formulation result

- Target agreement: <code>True</code>
- Task agreement: <code>True</code>
- Result: <strong>Agreement</strong>
- Reconciliation: <code>not invoked</code>
- Approved target: <code>target</code>
- Approved task: <code>classification</code>
- Target source: <code>user_supplied</code>; mutable: <code>False</code>
- Justification: The independent formulation agent and deterministic formulation engine agreed on the target and supported task before split construction.

## Modeling decision gate: Agreement

The workflow intentionally made a modeling decision before fitting any model.

Hard deterministic validation is authoritative for safety and executability. The deterministic model-family recommendation is an advisory soft challenger; disagreement alone is not an invalid plan.

| Source | Approved formulation | Method |
| --- | --- | --- |
| Modeling agent | <code>target</code> / <code>classification</code> | <code>tree_ensemble</code> |
| Deterministic recommender | <code>target</code> / <code>classification</code> | <code>regularized_linear</code> |
| Final approved plan | <code>target</code> / <code>classification</code> | <code>tree_ensemble</code> |

Deterministic reasoning: The training-only deterministic policy ranked regularized_linear highest with compatibility score 68 (low confidence). It considered 2 usable features across 80 training rows, estimated 4 post-one-hot features, and observed nominal class association measured with classification_eta_squared_and_cramers_v; maximum class-separation strength was 1.00. The structural-complexity signal was low (0.20). Key positive factors: numeric and categorical predictors are mixed; maximum categorical cardinality 77 favors shrinkage; ratio 20.00 is healthy. Compatibility scores are policy rankings, not probabilities.

### Deterministic compatibility evidence

Policy version: <code>4</code>; confidence: <code>low</code>; score margin: <code>3.0</code>.

Method-family compatibility scores (bounded policy points, not probabilities):

- <code>linear</code>: <code>61.0</code>
- <code>regularized_linear</code>: <code>68.0</code>
- <code>tree_ensemble</code>: <code>61.0</code>
- <code>boosted_tree</code>: <code>65.0</code>

Training-only diagnostics: <code>{"association_measure": "classification_eta_squared_and_cramers_v", "binary_feature_count": 1, "boosted_effective_features_estimate": 2, "categorical_feature_count": 1, "class_separation_strength": 1.0, "effective_features_estimate": 4, "estimated_one_hot_dimensionality": 4, "excluded_feature_types": {}, "excluded_features": 0, "features_with_missing_count": 2, "features_with_missing_fraction": 1.0, "high_cardinality_feature_count": 0, "high_cardinality_feature_fraction": 0.0, "high_correlation_pair_count": 0, "high_correlation_pair_fraction": 0.0, "interaction_signals": {"candidate_feature_count": 0, "candidate_features": [], "interaction_applicable": false, "interaction_pairs_evaluated": 0, "interaction_score": 0.0, "interaction_strength": "low", "skipped_pair_count": 0, "skipped_pair_reasons": {}, "strong_interaction_pair_count": 0, "strong_interaction_pair_fraction": 0.0, "top_interaction_pairs": []}, "linear_effective_features_estimate": 4, "marginal_association_strength": 0.5000091229848717, "max_abs_numeric_correlation": 0.0, "max_categorical_cardinality": 77, "max_feature_missing_fraction": 0.0375, "mean_categorical_cardinality": 39.5, "mean_univariate_signal": 0.5000091229848717, "missingness_pattern": "widespread", "nonlinear_feature_count": 0, "nonlinear_feature_fraction": 0.0, "nonlinearity_applicable": false, "nonlinearity_heterogeneity": 0.0, "nonlinearity_score": 0.0, "nonlinearity_signal": "low", "numeric_feature_count": 1, "numeric_outlier_cell_fraction": 0.0, "numeric_outlier_feature_fraction": 0.0, "overall_missing_fraction": 0.03125, "pearson_spearman_gap": 0.0, "rows": 80, "sample_to_feature_ratio": 20.0, "structural_complexity_score": 0.19999999999999998, "structural_complexity_signal": "low", "target": {"classification": {"classes": 2, "imbalance_ratio": 1.0, "majority_class_fraction": 0.5, "minimum_class_size": 40, "minority_class_fraction": 0.5, "samples_per_class": {"class_1": 40, "class_2": 40}}, "regression": null}, "text_feature_count": 0, "training_row_count": 80, "tree_effective_features_estimate": 4, "usable_features": 2}</code>.

Classification diagnostics use label-order-invariant eta-squared/Cramér's V class-association measures and do not treat nominal class labels as ordered numeric values.

The structural-complexity score is a bounded training-only compatibility heuristic; it summarizes observable feature structure and does not prove the presence of statistical feature interactions.

Selected-family score contributions:

- <code>feature_composition</code> (+2): numeric and categorical predictors are mixed
- <code>categorical_cardinality</code> (+2): maximum categorical cardinality 77 favors shrinkage
- <code>sample_to_feature_ratio</code> (+5): ratio 20.00 is healthy
- <code>encoded_dimensionality</code> (+3): estimated encoded dimension 4 is low
- <code>multicollinearity</code> (+4): max absolute correlation 0.00 is limited
- <code>structural_complexity</code> (+2): structural complexity score 0.20 reflects mixed/categorical structure and weak marginal class association; it does not assert multiclass nonlinearity

Hard validation: <code>passed</code>; intervention required: <code>False</code>; initial proposal hard-invalid: <code>False</code>.

Soft challenge: <code>disagreement</code>; decision: <code>abstain</code>; decision reason: <code>low_deterministic_confidence</code>; method disagreement: <code>True</code>; preprocessing disagreement: <code>True</code>; deterministic confidence: <code>low</code>; score margin: <code>3.0</code>; empirical reliability: <code>0.14285714285714285</code>; calibration support: <code>20</code>; reconciliation invoked: <code>False</code>.

Final selection source: <code>agent</code>.

Validation decision: The independent modeling agent and deterministic challenger agreed on model family and material preprocessing behavior after both proposals passed hard safety and executability validation.

Holdout boundary: the formulation gate completed and was validated before the supervised split. Modeling-agent evidence, deterministic recommendation evidence, modeling reconciliation, preprocessing requirements, structural-cleaning decisions, pre-evaluation EDA and plots, and cross-validation used training-partition evidence only. The EDA artifact contains <code>79</code> cleaned training rows and no holdout rows. The fail-closed validation gate may inspect the full raw or cleaned frame only to enforce target, schema, feasibility, and frozen-membership invariants; those guardrail checks are not planning evidence. The frozen holdout was scored once for final model evaluation.

Deterministic contract: <code>passed</code> (37/37 checks passed); target rows removed: <code>0</code>; direct leakage detected: <code>False</code>.

Excluded features and reasons: <code>none</code>.

### Preprocessing contract

| Source | Contract |
| --- | --- |
| Independent agent | <code>{"categorical_encoding": "one_hot", "categorical_imputation": "most_frequent", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "median", "numeric_scaling": "none", "unsupported_text_handling": "exclude"}</code> |
| Deterministic recommender | <code>{"categorical_encoding": "one_hot", "categorical_imputation": "most_frequent", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "median", "numeric_scaling": "standard", "unsupported_text_handling": "exclude"}</code> |
| Final approved executable plan | <code>{"categorical_encoding": "one_hot", "categorical_imputation": "most_frequent", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "median", "numeric_scaling": "none", "unsupported_text_handling": "exclude"}</code> |

Preprocessing comparison: <code>disagreement</code>; material differences: <code>[{"agent": "none", "deterministic": "standard", "field": "numeric_scaling", "material": true, "reason": "Executable preprocessing behavior differs."}]</code>.

Reconciliation output: <code>not invoked</code>.

The deterministic requirements and checks are persisted with the gate. All learned transformations are fitted inside the training pipeline; the holdout is used only for final evaluation. The executed pipeline components are <code>{"categorical": ["most_frequent_imputation", "one_hot"], "infinity_handling_before_pipeline": "replace_with_missing", "numeric": ["median_imputation"], "remainder": "drop", "unknown_category_handling": "ignore"}</code>.

## Data profile

- Rows before cleaning: <code>100</code>
- Columns before cleaning: <code>3</code>
- Duplicate rows detected: <code>1</code>
- Rows after cleaning: <code>99</code>
- Columns after cleaning: <code>3</code>

## Cleaning agent

The cleaning agent requested: <code>trim_strings, drop_exact_duplicates, drop_all_null_columns, drop_constant_features, drop_rows_missing_target, coerce_numeric_strings</code>.

Applied structural actions: <code>trim_strings, drop_exact_duplicates, drop_all_null_columns, drop_constant_features, drop_rows_missing_target, coerce_numeric_strings</code>.

The structural-cleaning specification was fitted from <code>training_partition_only</code> evidence and then transformed independently on each partition. Holdout values were not used to derive column removals, coercion eligibility, thresholds, or training duplicate membership. Exact-duplicate removal uses the policy <code>within_partition_only_keep_first</code>.

The offline cleaning fallback applies only structural operations and leaves learned imputation inside the model pipeline.

## Exploratory data analysis

- The cleaned analysis frame contains 79 rows and 3 columns.
- Missing values remain visible to the modeling pipeline, where imputation is learned from training folds only.

The pre-evaluation numeric relationships, target distribution, missingness, and plots were computed deterministically from the frozen training partition only; the EDA agent received only those training-only summaries and interpreted those values without inspecting holdout data.

## Modeling

The approved model was <strong>random_forest</strong> using training-only preprocessing and <code>5</code>-fold <code>stratified_kfold</code> validation.

### Cross-validation

- <code>cv_macro_f1_mean</code>: <code>0.9875</code>
- <code>cv_macro_f1_std</code>: <code>0.0251</code>
- <code>cv_balanced_accuracy_mean</code>: <code>0.9875</code>
- <code>cv_balanced_accuracy_std</code>: <code>0.0250</code>
- <code>cv_accuracy_mean</code>: <code>0.9875</code>
- <code>cv_accuracy_std</code>: <code>0.0250</code>

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
- <code>formulation_profile.json</code>
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
