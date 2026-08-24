# AutoDS Agent Analysis Report

## Question

Estimate disease progression from the patient measurements.

## Problem formulation gate: agreement

The workflow first validated what supervised problem to solve, before creating any train/holdout split.

### User question

Estimate disease progression from the patient measurements.

### Agent formulation

- Target: <code>disease_progression</code>
- Task: <code>regression</code>
- Reasoning: Offline fallback uses the deterministic schema formulation as a local heuristic; this is not evidence of independent LLM reasoning. The target 'disease_progression' is treated as regression because it is numeric/coercible (1.000 valid numeric fraction) and has 214 distinct values across 442 valid rows, rather than appearing to be a low-cardinality label.
- Confidence: <code>0.4</code>

### Deterministic formulation

- Target: <code>disease_progression</code>
- Task: <code>regression</code>
- Reasoning: The target 'disease_progression' is treated as regression because it is numeric/coercible (1.000 valid numeric fraction) and has 214 distinct values across 442 valid rows, rather than appearing to be a low-cardinality label.
- Evidence: <code>["target_column=disease_progression", "target_source=user_supplied", "target_dtype=object", "target_semantic_type=numeric_like", "valid_target_rows=442", "unique_values=214", "unique_value_fraction=0.4842", "numeric_or_coercible_fraction=1.0000", "model_selection_diagnostics_not_used"]</code>

### Formulation result

- Target agreement: <code>True</code>
- Task agreement: <code>True</code>
- Result: <strong>Agreement</strong>
- Reconciliation: <code>not invoked</code>
- Approved target: <code>disease_progression</code>
- Approved task: <code>regression</code>
- Target source: <code>user_supplied</code>; mutable: <code>False</code>
- Justification: The independent formulation agent and deterministic formulation engine agreed on the target and supported task before split construction.

## Modeling decision gate: Agreement

The workflow intentionally made a modeling decision before fitting any model.

Hard deterministic validation is authoritative for safety and executability. The deterministic model-family recommendation is an advisory soft challenger; disagreement alone is not an invalid plan.

| Source | Approved formulation | Method |
| --- | --- | --- |
| Modeling agent | <code>disease_progression</code> / <code>regression</code> | <code>regularized_linear</code> |
| Deterministic recommender | <code>disease_progression</code> / <code>regression</code> | <code>regularized_linear</code> |
| Final approved plan | <code>disease_progression</code> / <code>regression</code> | <code>regularized_linear</code> |

Deterministic reasoning: The training-only deterministic policy ranked regularized_linear highest with compatibility score 77 (medium confidence). It considered 10 usable features across 353 training rows, estimated 10 post-one-hot features, and observed low numeric-target nonlinearity and regression_pearson_spearman_binned; low interaction evidence (0.05). The structural-complexity signal was low (0.07). Key positive factors: numeric predictors predominate; ratio 35.30 is healthy; estimated encoded dimension 10 is low. Compatibility scores are policy rankings, not probabilities.

### Deterministic compatibility evidence

Policy version: <code>4</code>; confidence: <code>medium</code>; score margin: <code>10.0</code>.

Method-family compatibility scores (bounded policy points, not probabilities):

- <code>linear</code>: <code>65.0</code>
- <code>regularized_linear</code>: <code>77.0</code>
- <code>tree_ensemble</code>: <code>58.0</code>
- <code>boosted_tree</code>: <code>67.0</code>

Training-only diagnostics: <code>{"association_measure": "regression_pearson_spearman_binned", "binary_feature_count": 1, "boosted_effective_features_estimate": 10, "categorical_feature_count": 0, "class_separation_strength": 0.0, "effective_features_estimate": 10, "estimated_one_hot_dimensionality": 10, "excluded_feature_types": {}, "excluded_features": 0, "features_with_missing_count": 0, "features_with_missing_fraction": 0.0, "high_cardinality_feature_count": 5, "high_cardinality_feature_fraction": 0.5, "high_correlation_pair_count": 1, "high_correlation_pair_fraction": 0.022222222222222223, "interaction_signals": {"candidate_feature_count": 10, "candidate_features": ["age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6", "sex"], "interaction_applicable": true, "interaction_pairs_evaluated": 45, "interaction_score": 0.04901376438327085, "interaction_strength": "low", "skipped_pair_count": 0, "skipped_pair_reasons": {}, "strong_interaction_pair_count": 0, "strong_interaction_pair_fraction": 0.0, "top_interaction_pairs": []}, "linear_effective_features_estimate": 10, "marginal_association_strength": 0.4727966924369298, "max_abs_numeric_correlation": 0.8910632483212737, "max_categorical_cardinality": 259, "max_feature_missing_fraction": 0.0, "mean_categorical_cardinality": 102.6, "mean_univariate_signal": 0.4727966924369298, "missingness_pattern": "none", "nonlinear_feature_count": 1, "nonlinear_feature_fraction": 0.1, "nonlinearity_applicable": true, "nonlinearity_heterogeneity": 0.11149925697206048, "nonlinearity_score": 0.11160660192620266, "nonlinearity_signal": "low", "numeric_feature_count": 10, "numeric_outlier_cell_fraction": 0.010198300283286119, "numeric_outlier_feature_fraction": 0.0, "overall_missing_fraction": 0.0, "pearson_spearman_gap": 0.029510305570872752, "rows": 353, "sample_to_feature_ratio": 35.3, "structural_complexity_score": 0.0744292919040139, "structural_complexity_signal": "low", "target": {"classification": null, "regression": {"heavy_tail_signal": "low", "outlier_fraction": 0.0, "skewness": 0.4257040191566251, "variance": 6093.660507339687}}, "text_feature_count": 0, "training_row_count": 353, "tree_effective_features_estimate": 10, "usable_features": 10}</code>.

Regression diagnostics use numeric-target correlation, rank, and binned-target evidence.

The structural-complexity score is a bounded training-only compatibility heuristic; it summarizes observable feature structure and does not prove the presence of statistical feature interactions.

Selected-family score contributions:

- <code>feature_composition</code> (+6): numeric predictors predominate
- <code>categorical_cardinality</code> (-2): at least half of categorical predictors exceed the canonical cardinality band
- <code>sample_to_feature_ratio</code> (+5): ratio 35.30 is healthy
- <code>encoded_dimensionality</code> (+3): estimated encoded dimension 10 is low
- <code>multicollinearity</code> (+8): max absolute correlation 0.89 favors shrinkage
- <code>nonlinearity</code> (+5): nonlinearity score 0.11 is low
- <code>structural_complexity</code> (+2): structural complexity score 0.07 suggests heterogeneous or nonlinear feature relationships

Hard validation: <code>passed</code>; intervention required: <code>False</code>; initial proposal hard-invalid: <code>False</code>.

Soft challenge: <code>agreement</code>; decision: <code>agree</code>; decision reason: <code>model_family_agreement</code>; method disagreement: <code>False</code>; preprocessing disagreement: <code>False</code>; deterministic confidence: <code>medium</code>; score margin: <code>10.0</code>; empirical reliability: <code>0.25</code>; calibration support: <code>14</code>; reconciliation invoked: <code>False</code>.

Final selection source: <code>agent</code>.

Validation decision: The independent modeling agent and deterministic challenger agreed on model family and material preprocessing behavior after both proposals passed hard safety and executability validation.

Holdout boundary: the formulation gate completed and was validated before the supervised split. Modeling-agent evidence, deterministic recommendation evidence, modeling reconciliation, preprocessing requirements, structural-cleaning decisions, pre-evaluation EDA and plots, and cross-validation used training-partition evidence only. The EDA artifact contains <code>353</code> cleaned training rows and no holdout rows. The fail-closed validation gate may inspect the full raw or cleaned frame only to enforce target, schema, feasibility, and frozen-membership invariants; those guardrail checks are not planning evidence. The frozen holdout was scored once for final model evaluation.

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

The cleaning agent requested: <code>trim_strings, drop_exact_duplicates, drop_all_null_columns, drop_constant_features, drop_rows_missing_target, coerce_numeric_strings</code>.

Applied structural actions: <code>trim_strings, drop_exact_duplicates, drop_all_null_columns, drop_constant_features, drop_rows_missing_target, coerce_numeric_strings</code>.

The structural-cleaning specification was fitted from <code>training_partition_only</code> evidence and then transformed independently on each partition. Holdout values were not used to derive column removals, coercion eligibility, thresholds, or training duplicate membership. Exact-duplicate removal uses the policy <code>within_partition_only_keep_first</code>.

The offline cleaning fallback applies only structural operations and leaves learned imputation inside the model pipeline.

## Exploratory data analysis

- The cleaned analysis frame contains 353 rows and 11 columns.
- No missing values were present in the cleaned frame.
- The strongest absolute numeric relationship observed was s1 with s2 (absolute r=0.891).

The pre-evaluation numeric relationships, target distribution, missingness, and plots were computed deterministically from the frozen training partition only; the EDA agent received only those training-only summaries and interpreted those values without inspecting holdout data.

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
