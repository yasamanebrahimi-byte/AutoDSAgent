# AutoDS Agent Analysis Report

## Question

Classify diagnosis from the measured features.

## Problem formulation gate: agreement

The workflow first validated what supervised problem to solve, before creating any train/holdout split.

### User question

Classify diagnosis from the measured features.

### Agent formulation

- Target: <code>diagnosis</code>
- Task: <code>classification</code>
- Reasoning: The formulation proposal uses the question and compact schema evidence only.
- Confidence: <code>0.8</code>

### Deterministic formulation

- Target: <code>diagnosis</code>
- Task: <code>classification</code>
- Reasoning: The target 'diagnosis' is treated as classification because its semantic type is categorical, or its numeric values are low-cardinality label-like (2 unique values across 64 valid rows).
- Evidence: <code>["target_column=diagnosis", "target_source=inferred", "target_dtype=object", "target_semantic_type=categorical", "valid_target_rows=64", "unique_values=2", "unique_value_fraction=0.0312", "numeric_or_coercible_fraction=0.0000", "model_selection_diagnostics_not_used"]</code>

### Formulation result

- Target agreement: <code>True</code>
- Task agreement: <code>True</code>
- Result: <strong>Agreement</strong>
- Reconciliation: <code>not invoked</code>
- Approved target: <code>diagnosis</code>
- Approved task: <code>classification</code>
- Target source: <code>inferred</code>; mutable: <code>True</code>
- Justification: The independent formulation agent and deterministic formulation engine agreed on the target and supported task before split construction.

## Modeling decision gate: Agreement

The workflow intentionally made a modeling decision before fitting any model.

Hard deterministic validation is authoritative for safety and executability. The deterministic model-family recommendation is an advisory soft challenger; disagreement alone is not an invalid plan.

| Source | Approved formulation | Method |
| --- | --- | --- |
| Modeling agent | <code>diagnosis</code> / <code>classification</code> | <code>linear</code> |
| Deterministic recommender | <code>diagnosis</code> / <code>classification</code> | <code>regularized_linear</code> |
| Final approved plan | <code>diagnosis</code> / <code>classification</code> | <code>linear</code> |

Deterministic reasoning: The training-only deterministic policy ranked regularized_linear highest with compatibility score 76 (low confidence). It considered 2 usable features across 51 training rows, estimated 2 post-one-hot features, and observed nominal class association measured with classification_eta_squared; maximum class-separation strength was 0.01. The structural-complexity signal was low (0.00). Key positive factors: numeric predictors predominate; maximum categorical cardinality 51 favors shrinkage; ratio 25.50 is healthy. Compatibility scores are policy rankings, not probabilities.

### Deterministic compatibility evidence

Policy version: <code>4</code>; confidence: <code>low</code>; score margin: <code>2.0</code>.

Method-family compatibility scores (bounded policy points, not probabilities):

- <code>linear</code>: <code>74.0</code>
- <code>regularized_linear</code>: <code>76.0</code>
- <code>tree_ensemble</code>: <code>43.0</code>
- <code>boosted_tree</code>: <code>51.0</code>

Training-only diagnostics: <code>{"association_measure": "classification_eta_squared", "binary_feature_count": 0, "boosted_effective_features_estimate": 2, "categorical_feature_count": 0, "class_separation_strength": 0.012790434680174323, "effective_features_estimate": 2, "estimated_one_hot_dimensionality": 2, "excluded_feature_types": {}, "excluded_features": 0, "features_with_missing_count": 0, "features_with_missing_fraction": 0.0, "high_cardinality_feature_count": 0, "high_cardinality_feature_fraction": 0.0, "high_correlation_pair_count": 0, "high_correlation_pair_fraction": 0.0, "interaction_signals": {"candidate_feature_count": 0, "candidate_features": [], "interaction_applicable": false, "interaction_pairs_evaluated": 0, "interaction_score": 0.0, "interaction_strength": "low", "skipped_pair_count": 0, "skipped_pair_reasons": {}, "strong_interaction_pair_count": 0, "strong_interaction_pair_fraction": 0.0, "top_interaction_pairs": []}, "linear_effective_features_estimate": 2, "marginal_association_strength": 0.007478872115300312, "max_abs_numeric_correlation": 0.018690195175477015, "max_categorical_cardinality": 51, "max_feature_missing_fraction": 0.0, "mean_categorical_cardinality": 31.0, "mean_univariate_signal": 0.007478872115300312, "missingness_pattern": "none", "nonlinear_feature_count": 0, "nonlinear_feature_fraction": 0.0, "nonlinearity_applicable": false, "nonlinearity_heterogeneity": 0.0, "nonlinearity_score": 0.0, "nonlinearity_signal": "low", "numeric_feature_count": 2, "numeric_outlier_cell_fraction": 0.0, "numeric_outlier_feature_fraction": 0.0, "overall_missing_fraction": 0.0, "pearson_spearman_gap": 0.0, "rows": 51, "sample_to_feature_ratio": 25.5, "structural_complexity_score": 0.0, "structural_complexity_signal": "low", "target": {"classification": {"classes": 2, "imbalance_ratio": 1.04, "majority_class_fraction": 0.5098039215686274, "minimum_class_size": 25, "minority_class_fraction": 0.49019607843137253, "samples_per_class": {"class_1": 26, "class_2": 25}}, "regression": null}, "text_feature_count": 0, "training_row_count": 51, "tree_effective_features_estimate": 2, "usable_features": 2}</code>.

Classification diagnostics use label-order-invariant eta-squared/Cramér's V class-association measures and do not treat nominal class labels as ordered numeric values.

The structural-complexity score is a bounded training-only compatibility heuristic; it summarizes observable feature structure and does not prove the presence of statistical feature interactions.

Selected-family score contributions:

- <code>feature_composition</code> (+6): numeric predictors predominate
- <code>categorical_cardinality</code> (+2): maximum categorical cardinality 51 favors shrinkage
- <code>sample_to_feature_ratio</code> (+5): ratio 25.50 is healthy
- <code>dataset_scale</code> (+4): 51 rows favor a compact regularized baseline
- <code>encoded_dimensionality</code> (+3): estimated encoded dimension 2 is low
- <code>multicollinearity</code> (+4): max absolute correlation 0.02 is limited
- <code>structural_complexity</code> (+2): structural complexity score 0.00 reflects mixed/categorical structure and weak marginal class association; it does not assert multiclass nonlinearity

Hard validation: <code>passed</code>; intervention required: <code>False</code>; initial proposal hard-invalid: <code>False</code>.

Soft challenge: <code>disagreement</code>; decision: <code>abstain</code>; decision reason: <code>low_deterministic_confidence</code>; method disagreement: <code>True</code>; preprocessing disagreement: <code>False</code>; deterministic confidence: <code>low</code>; score margin: <code>2.0</code>; empirical reliability: <code>0.14285714285714285</code>; calibration support: <code>20</code>; reconciliation invoked: <code>False</code>.

Final selection source: <code>agent</code>.

Validation decision: The independent modeling agent and deterministic challenger agreed on model family and material preprocessing behavior after both proposals passed hard safety and executability validation.

Holdout boundary: the formulation gate completed and was validated before the supervised split. Modeling-agent evidence, deterministic recommendation evidence, modeling reconciliation, preprocessing requirements, structural-cleaning decisions, pre-evaluation EDA and plots, and cross-validation used training-partition evidence only. The EDA artifact contains <code>51</code> cleaned training rows and no holdout rows. The fail-closed validation gate may inspect the full raw or cleaned frame only to enforce target, schema, feasibility, and frozen-membership invariants; those guardrail checks are not planning evidence. The frozen holdout was scored once for final model evaluation.

Deterministic contract: <code>passed</code> (37/37 checks passed); target rows removed: <code>0</code>; direct leakage detected: <code>False</code>.

Excluded features and reasons: <code>none</code>.

### Preprocessing contract

| Source | Contract |
| --- | --- |
| Independent agent | <code>{"categorical_encoding": "one_hot", "categorical_imputation": "most_frequent", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "median", "numeric_scaling": "standard", "unsupported_text_handling": "exclude"}</code> |
| Deterministic recommender | <code>{"categorical_encoding": "none", "categorical_imputation": "none", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "none", "numeric_scaling": "standard", "unsupported_text_handling": "exclude"}</code> |
| Final approved executable plan | <code>{"categorical_encoding": "one_hot", "categorical_imputation": "most_frequent", "categorical_unknown_handling": "ignore", "datetime_handling": "exclude", "fit_inside_pipeline": true, "high_cardinality_handling": "exclude", "identifier_handling": "exclude", "infinity_handling": "replace_with_missing", "numeric_imputation": "median", "numeric_scaling": "standard", "unsupported_text_handling": "exclude"}</code> |

Preprocessing comparison: <code>agreement</code>; material differences: <code>none</code>.

Reconciliation output: <code>not invoked</code>.

The deterministic requirements and checks are persisted with the gate. All learned transformations are fitted inside the training pipeline; the holdout is used only for final evaluation. The executed pipeline components are <code>{"categorical": ["most_frequent_imputation", "one_hot"], "infinity_handling_before_pipeline": "replace_with_missing", "numeric": [], "remainder": "drop", "unknown_category_handling": "ignore"}</code>.

## Data profile

- Rows before cleaning: <code>64</code>
- Columns before cleaning: <code>3</code>
- Duplicate rows detected: <code>0</code>
- Rows after cleaning: <code>64</code>
- Columns after cleaning: <code>3</code>

## Cleaning agent

The cleaning agent requested: <code>no actions</code>.

Applied structural actions: <code>none</code>.

The structural-cleaning specification was fitted from <code>training_partition_only</code> evidence and then transformed independently on each partition. Holdout values were not used to derive column removals, coercion eligibility, thresholds, or training duplicate membership. Exact-duplicate removal uses the policy <code>within_partition_only_keep_first</code>.

No structural cleaning is required for this fixture.

## Exploratory data analysis

- The training-only summary was inspected.

The pre-evaluation numeric relationships, target distribution, missingness, and plots were computed deterministically from the frozen training partition only; the EDA agent received only those training-only summaries and interpreted those values without inspecting holdout data.

## Modeling

The approved model was <strong>logistic_regression</strong> using training-only preprocessing and <code>5</code>-fold <code>stratified_kfold</code> validation.

### Cross-validation

- <code>cv_macro_f1_mean</code>: <code>0.2397</code>
- <code>cv_macro_f1_std</code>: <code>0.1027</code>
- <code>cv_balanced_accuracy_mean</code>: <code>0.2800</code>
- <code>cv_balanced_accuracy_std</code>: <code>0.1166</code>
- <code>cv_accuracy_mean</code>: <code>0.2764</code>
- <code>cv_accuracy_std</code>: <code>0.1193</code>

### Untouched holdout

- <code>accuracy</code>: <code>0.2308</code>
- <code>macro_f1</code>: <code>0.1875</code>
- <code>weighted_f1</code>: <code>0.1731</code>
- <code>balanced_accuracy</code>: <code>0.2500</code>

### Baseline holdout

<code>accuracy=0.4615</code>, <code>macro_f1=0.3158</code>, <code>weighted_f1=0.2915</code>, <code>balanced_accuracy=0.5000</code>

The approved model was evaluated with training-only cross-validation.

## Limitations and next steps

- This is a test fixture.

Recommended next steps:

- Review the persisted decision artifact.

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
