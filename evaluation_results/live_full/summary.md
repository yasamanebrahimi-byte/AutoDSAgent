# AutoDS Validation Architecture Evaluation

This report is generated deterministically from `config.json`, `trials.jsonl`, and the computed summary. Offline fallback and mock rows are not evidence of live LLM performance.

## Experiment Configuration

- Repetitions per benchmark/scenario: **10**.
- Base seed: **42**; holdout fraction: **0.2**.
- Requested model: `gpt-4.1-mini`; prompt/schema version: `2026-08-23.formulation-modeling-gates.v1`.
- Repository commit: `128d222f93abd53d130e24ec45e3e13d05fe3bc6`.
- Each repetition keeps the case, frozen train/holdout membership, and training-only profile fixed; the intended varying factor is the stochastic LLM response.
- These rows are `modeling_gate` evaluations: benchmark target/task values are fixed context, while `agent_initial` represents only the post-split model-family and preprocessing proposal. `gated_final` is the approved plan after comparison, optional reconciliation, and deterministic validation. Formulation accuracy requires a separate formulation-gate evaluation mode.
- `empirical_reference` is an evaluation-only ranking of the four supported families using training-only CV; it is not an oracle and never enters runtime decisions.

## Trial Coverage

| Trial category | Count |
|---|---:|
| Requested live trials | 180 |
| Successful OpenAI trials | 170 |
| Offline fallback trials | 10 |
| Failed trials | 0 |
| Mock trials | 0 |
| Completed trials | 180 |

Claims about LLM behavior below use `agent_source == "openai"` only.

## LLM Decision Stability

| Dataset | OpenAI trials | Unique initial methods | Modal method | Modal frequency | Pairwise consistency |
|---|---:|---:|---|---:|---:|
| breast_cancer | 10 | 2 | regularized_linear | 80.0% | 64.4% |
| diabetes | 10 | 1 | regularized_linear | 100.0% | 100.0% |
| digits_subset | 10 | 1 | boosted_tree | 100.0% | 100.0% |
| final_interaction_regression | 10 | 1 | boosted_tree | 100.0% | 100.0% |
| final_low_n_high_p_classification | 10 | 1 | regularized_linear | 100.0% | 100.0% |
| final_shifted_nonlinear_regression | 10 | 1 | boosted_tree | 100.0% | 100.0% |
| synthetic_binary_linear | 10 | 1 | linear | 100.0% | 100.0% |
| synthetic_binary_nonlinear | 10 | 1 | boosted_tree | 100.0% | 100.0% |
| synthetic_high_dim_regression | 10 | 1 | regularized_linear | 100.0% | 100.0% |
| synthetic_imbalanced_classification | 10 | 2 | regularized_linear | 60.0% | 46.7% |
| synthetic_linear_regression | 10 | 2 | linear | 90.0% | 80.0% |
| synthetic_missingness | 10 | 1 | regularized_linear | 100.0% | 100.0% |
| synthetic_multiclass | 10 | 1 | regularized_linear | 100.0% | 100.0% |
| synthetic_nonlinear_regression | 10 | 2 | boosted_tree | 90.0% | 80.0% |
| synthetic_outlier_regression | 10 | 1 | regularized_linear | 100.0% | 100.0% |
| synthetic_regression | 10 | 2 | regularized_linear | 90.0% | 80.0% |
| wine | 10 | 2 | regularized_linear | 90.0% | 80.0% |

## Agent vs Deterministic Agreement

- All operational trials: agreement **50.6%**, disagreement **49.4%**.
- OpenAI-only method agreement: **53.5%**; preprocessing agreement: **77.6%**.

| Method distribution | Initial agent | Gated final |
|---|---|---|
| All trials | {'boosted_tree': {'count': 57, 'rate': 0.31666666666666665}, 'linear': {'count': 19, 'rate': 0.10555555555555556}, 'regularized_linear': {'count': 93, 'rate': 0.5166666666666667}, 'tree_ensemble': {'count': 11, 'rate': 0.06111111111111111}} | {'boosted_tree': {'count': 34, 'rate': 0.18888888888888888}, 'linear': {'count': 51, 'rate': 0.2833333333333333}, 'regularized_linear': {'count': 95, 'rate': 0.5277777777777778}} |
| OpenAI only | {'boosted_tree': {'count': 57, 'rate': 0.3352941176470588}, 'linear': {'count': 19, 'rate': 0.11176470588235295}, 'regularized_linear': {'count': 93, 'rate': 0.5470588235294118}, 'tree_ensemble': {'count': 1, 'rate': 0.0058823529411764705}} | {'boosted_tree': {'count': 24, 'rate': 0.1411764705882353}, 'linear': {'count': 51, 'rate': 0.3}, 'regularized_linear': {'count': 95, 'rate': 0.5588235294117647}} |

## Empirical Reference Comparison

- All operational trials: initial reference match **43.9%**; gated reference match **55.6%**.
- OpenAI only: initial reference match **46.5%**; gated reference match **58.8%**.
- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.

## Effect of the Validation Gate

- OpenAI-only gate outcomes: **10 improved**, **28 worsened**, **132 tied**.
- OpenAI-only potentially unnecessary interventions: **56**.
- Operational outcomes: improved **10**, worsened **28**, tie **142**.
- Improved/worsened/tie is defined from paired training-only CV regret using the configured tolerance; holdout results do not define this label.

## Reconciliation Outcomes

- Reconciliation invocation rate: **49.4%**; success rate: **100.0%**.
- Sided with agent: **10.1%**; sided with deterministic validator: **89.9%**.
- Every disagreement row retains the initial plan, deterministic plan, preprocessing comparison, reconciliation response, selected source, and final validation result.

## Predictive Performance

- OpenAI-only mean paired CV improvement: **0.0887**; median: **0.0000**; standard deviation: **1.5705**.
- OpenAI-only mean paired holdout improvement: **0.0304** (descriptive only; not used to define gate outcomes).
- Classification improvement is `gated_macro_f1 - initial_macro_f1`; regression improvement is `initial_rmse - gated_rmse`, so positive always means gating helped.
- Untouched holdout metrics are retained per trial as a descriptive external check after decisions and the empirical ranking are frozen.

## Dataset-Level Results

| Dataset | Trials | Initial match | Gated match | Improved / worsened / tie | Mean paired CV improvement |
|---|---:|---:|---:|---|---:|
| breast_cancer | 10 | 80.0% | 100.0% | 0 / 0 / 10 | 0.0037 |
| diabetes | 10 | 100.0% | 100.0% | 0 / 0 / 10 | 0.0000 |
| digits_subset | 10 | 0.0% | 0.0% | 0 / 0 / 10 | -0.0004 |
| final_interaction_regression | 10 | 100.0% | 30.0% | 0 / 7 / 3 | -0.4738 |
| final_low_n_high_p_classification | 10 | 0.0% | 0.0% | 0 / 0 / 10 | 0.0000 |
| final_mixed_type_classification | 0 | n/a | n/a | 0 / 0 / 0 | n/a |
| final_shifted_nonlinear_regression | 10 | 100.0% | 100.0% | 0 / 0 / 10 | 0.0000 |
| synthetic_binary_linear | 10 | 0.0% | 100.0% | 0 / 0 / 10 | 0.0042 |
| synthetic_binary_nonlinear | 10 | 0.0% | 0.0% | 0 / 9 / 1 | -0.1055 |
| synthetic_high_dim_regression | 10 | 0.0% | 0.0% | 0 / 0 / 10 | 0.0000 |
| synthetic_imbalanced_classification | 10 | 40.0% | 0.0% | 0 / 4 / 6 | -0.0185 |
| synthetic_linear_regression | 10 | 90.0% | 90.0% | 0 / 0 / 10 | 0.0000 |
| synthetic_missingness | 10 | 100.0% | 100.0% | 0 / 0 / 10 | 0.0000 |
| synthetic_multiclass | 10 | 0.0% | 0.0% | 0 / 8 / 2 | -0.0222 |
| synthetic_nonlinear_regression | 10 | 90.0% | 100.0% | 1 / 0 / 9 | 0.0478 |
| synthetic_outlier_regression | 10 | 0.0% | 100.0% | 0 / 0 / 10 | 0.0086 |
| synthetic_regression | 10 | 0.0% | 80.0% | 8 / 0 / 2 | 2.0579 |
| wine | 10 | 90.0% | 100.0% | 1 / 0 / 9 | 0.0057 |

## Validation / Safety Interceptions

- Initial invalid proposals: **0**; intercepted without proceeding unchanged: **0**.
- Final invalid trials: **0**; validation failure codes: `{}`.
- Intentionally unsafe perturbations intercepted: **0** / **0** perturbation trials where applicable.

## Limitations

- The benchmark suite is small and local; it is not representative of every tabular data-science domain.
- The empirical reference is not a universal optimum or ground truth; it ranks only the supported families under one CV design.
- Method-family match is not equivalent to predictive or deployment quality, and a one-split study cannot establish generalization.
- Offline fallback and mock rows must not be used to make claims about live LLM behavior.
- Semantic leakage, feature availability, and domain-specific safety still require expert review.
