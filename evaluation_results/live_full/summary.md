# AutoDS Validation Architecture Evaluation

This report is generated deterministically from `config.json`, `trials.jsonl`, and the computed summary. Offline fallback and mock rows are not evidence of live LLM performance.

## Experiment Configuration

- Repetitions per benchmark/scenario: **10**.
- Base seed: **42**; holdout fraction: **0.2**.
- Requested model: `gpt-4.1-mini`; prompt/schema version: `2026-08-24.blinded-evidence-comparison.v2-empirical-probe`.
- Gate objective version: `intervention-quality-v1`; neutrality tolerance: `0.02`; catastrophic threshold: `0.1`.
- Repository commit: `ccfae2f36dae23671e39d31b28c7927ba1dce3af`.
- Each repetition keeps the case, frozen train/holdout membership, and training-only profile fixed; the intended varying factor is the stochastic LLM response.
- These rows are `modeling_gate` evaluations: benchmark target/task values are fixed context, while `agent_initial` represents only the post-split model-family and preprocessing proposal. `gated_final` is the approved plan after comparison, optional reconciliation, and deterministic validation. Formulation accuracy requires a separate formulation-gate evaluation mode.
- `empirical_reference` is an evaluation-only ranking of the four supported families using training-only CV; it is not an oracle and never enters runtime decisions.

## Trial Coverage

| Trial category | Count |
|---|---:|
| Requested live trials | 180 |
| Successful OpenAI trials | 173 |
| Offline fallback trials | 0 |
| Failed trials | 7 |
| Mock trials | 0 |
| Completed trials | 173 |

Claims about LLM behavior below use `agent_source == "openai"` only.

## Gate Health

Intervention quality is primary. Exact family match and top-2 compatibility below are secondary diagnostics.
- Challenges: **1 / 63 disagreements**; abstentions: **62**.
- Improved: **1**; worsened: **0**; neutral: **0**.
- Intervention precision: **100.0%**; challenge yield: **100.0%**; harmful-intervention rate: **0.0%**; unnecessary-intervention rate: **0.0%**.
- Challenge recall: **50.0%**; missed rescues: **1**.
- Mean regret reduction: **0.0012**; median: **0.0000**; uncertainty interval: `{'lower': 0.0, 'upper': 0.0035143768185567146, 'support': 173, 'stable': True}`.
- Catastrophic regret: initial **1**, final **0**, prevented **1**, introduced **0**, net **1**.
- Utility contribution: `{'improvement_reward': 1.0, 'worsening_penalty': 0.0, 'unnecessary_intervention_penalty': 0.0, 'catastrophic_prevention_reward': 3.0, 'catastrophic_introduction_penalty': 0.0, 'missed_rescue_penalty': -1.0, 'total_utility': 3.0, 'weights': {'improvement': 1.0, 'worsening': 2.0, 'neutral_intervention': 0.25, 'catastrophic_prevention': 3.0, 'catastrophic_introduction': 5.0, 'missed_rescue': 1.0}}`.

| Metric | Trial-weighted | Dataset-weighted |
|---|---:|---:|
| Intervention precision | 100.0% | 100.0% |
| Harmful-intervention rate | 0.0% | 0.0% |
| Mean regret reduction | 0.0012 | 0.0011 |

## LLM Decision Stability

| Dataset | OpenAI trials | Unique initial methods | Modal method | Modal frequency | Pairwise consistency |
|---|---:|---:|---|---:|---:|
| breast_cancer | 10 | 2 | regularized_linear | 70.0% | 53.3% |
| diabetes | 10 | 1 | regularized_linear | 100.0% | 100.0% |
| digits_subset | 10 | 1 | boosted_tree | 100.0% | 100.0% |
| final_interaction_regression | 10 | 1 | boosted_tree | 100.0% | 100.0% |
| final_low_n_high_p_classification | 10 | 1 | regularized_linear | 100.0% | 100.0% |
| final_mixed_type_classification | 3 | 2 | regularized_linear | 66.7% | 33.3% |
| final_shifted_nonlinear_regression | 10 | 1 | boosted_tree | 100.0% | 100.0% |
| synthetic_binary_linear | 10 | 2 | linear | 80.0% | 64.4% |
| synthetic_binary_nonlinear | 10 | 1 | boosted_tree | 100.0% | 100.0% |
| synthetic_high_dim_regression | 10 | 1 | regularized_linear | 100.0% | 100.0% |
| synthetic_imbalanced_classification | 10 | 2 | boosted_tree | 60.0% | 46.7% |
| synthetic_linear_regression | 10 | 1 | linear | 100.0% | 100.0% |
| synthetic_missingness | 10 | 2 | regularized_linear | 90.0% | 80.0% |
| synthetic_multiclass | 10 | 2 | regularized_linear | 80.0% | 64.4% |
| synthetic_nonlinear_regression | 10 | 2 | boosted_tree | 90.0% | 80.0% |
| synthetic_outlier_regression | 10 | 1 | regularized_linear | 100.0% | 100.0% |
| synthetic_regression | 10 | 1 | regularized_linear | 100.0% | 100.0% |
| wine | 10 | 2 | regularized_linear | 90.0% | 80.0% |

## Agent vs Deterministic Soft Challenge

- All operational trials: soft agreement **63.6%**, soft disagreement **36.4%**.
- Model-family disagreement rate: **36.4%**; preprocessing disagreement rate: **15.0%**.
- OpenAI-only method agreement: **63.6%**; preprocessing agreement: **85.0%**.

| Method distribution | Initial agent | Gated final |
|---|---|---|
| All trials | {'boosted_tree': {'count': 62, 'rate': 0.3583815028901734}, 'linear': {'count': 18, 'rate': 0.10404624277456648}, 'regularized_linear': {'count': 91, 'rate': 0.5260115606936416}, 'tree_ensemble': {'count': 2, 'rate': 0.011560693641618497}} | {'boosted_tree': {'count': 63, 'rate': 0.36416184971098264}, 'linear': {'count': 18, 'rate': 0.10404624277456648}, 'regularized_linear': {'count': 91, 'rate': 0.5260115606936416}, 'tree_ensemble': {'count': 1, 'rate': 0.005780346820809248}} |
| OpenAI only | {'boosted_tree': {'count': 62, 'rate': 0.3583815028901734}, 'linear': {'count': 18, 'rate': 0.10404624277456648}, 'regularized_linear': {'count': 91, 'rate': 0.5260115606936416}, 'tree_ensemble': {'count': 2, 'rate': 0.011560693641618497}} | {'boosted_tree': {'count': 63, 'rate': 0.36416184971098264}, 'linear': {'count': 18, 'rate': 0.10404624277456648}, 'regularized_linear': {'count': 91, 'rate': 0.5260115606936416}, 'tree_ensemble': {'count': 1, 'rate': 0.005780346820809248}} |

## Empirical Reference Comparison

- All operational trials: initial reference match **48.6%**; gated reference match **49.1%**.
- OpenAI only: initial reference match **48.6%**; gated reference match **49.1%**.
- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.

## Effect of the Validation Gate

- OpenAI-only gate outcomes: **1 improved**, **0 worsened**, **172 neutral**.
- OpenAI-only potentially unnecessary interventions: **0**.
- Operational outcomes: improved **1**, worsened **0**, neutral **172**.
- Improved/worsened/neutral is defined from normalized regret reduction using the configured neutrality tolerance; holdout results do not define this label.

## Soft-Challenge Reconciliation Outcomes

- Total disagreements: **63**; challenges: **1**; abstentions: **62**.
- Challenge rate: **1.6%**; abstention rate: **98.4%**.
- Soft-challenge reconciliation invocation rate: **1.6%**.
- Reconciliation invocation rate: **0.6%**; success rate: **100.0%**.
- Sided with agent: **0.0%**; sided with deterministic challenger: **100.0%**.
- Proposal A selected: **0.0%**; Proposal B selected: **100.0%**; A/B selection imbalance: **100.0%**.
- Order-swap consistency: **n/a**; order-flip rate: **n/a** over **0** paired cases.
- Reconciliation modes observed: **['blinded_evidence_comparison']**.
- Soft-challenge outcomes: **1 improved**, **0 worsened**, **0 neutral**.
- Challenge outcomes: **1 improved**, **0 worsened**, **0 neutral**; intervention precision: **100.0%**.
- Abstentions where agent was better: **18**; where deterministic was better: **1**.
- Mean challenge regret improvement: **0.2027**; unnecessary interventions: **0** (**0.0%**).
- Catastrophic-regret rate: **0.6%**; catastrophic cases prevented by challenge: **1** (**100.0%**).
- A soft disagreement is competing advisory evidence, not an invalid plan. Every challenge row retains the initial plan, deterministic plan, preprocessing comparison, reconciliation response, selected source, and final hard-validation result.

## Predictive Performance

- OpenAI-only mean paired CV improvement: **0.0028**; median: **0.0000**; standard deviation: **0.0362**.
- OpenAI-only mean paired holdout improvement: **0.0019** (descriptive only; not used to define gate outcomes).
- Classification improvement is `gated_macro_f1 - initial_macro_f1`; regression improvement is `initial_rmse - gated_rmse`, so positive always means gating helped.
- Untouched holdout metrics are retained per trial as a descriptive external check after decisions and the empirical ranking are frozen.

## Dataset-Level Results

| Dataset | Trials | Challenges | Abstentions | Improved / worsened / neutral | Precision | Harm | Mean regret reduction | Catastrophic prevented / introduced | Exact match (diagnostic) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| breast_cancer | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 70.0% -> 70.0% |
| diabetes | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| digits_subset | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| final_interaction_regression | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| final_low_n_high_p_classification | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| final_mixed_type_classification | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| final_shifted_nonlinear_regression | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| synthetic_binary_linear | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 20.0% -> 20.0% |
| synthetic_binary_nonlinear | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| synthetic_high_dim_regression | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| synthetic_imbalanced_classification | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 60.0% -> 60.0% |
| synthetic_linear_regression | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| synthetic_missingness | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 90.0% -> 90.0% |
| synthetic_multiclass | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 20.0% -> 20.0% |
| synthetic_nonlinear_regression | 10 | 0 | 0 | 1 / 0 / 9 | 100.0% | 0.0% | 0.0203 | 1 / 0 | 90.0% -> 100.0% |
| synthetic_outlier_regression | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| synthetic_regression | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| wine | 10 | 0 | 0 | 0 / 0 / 10 | n/a | n/a | 0.0000 | 0 / 0 | 90.0% -> 90.0% |

## Validation / Safety Interceptions

- Initial hard-invalid proposals: **0**; hard-validation interventions: **0**; interception rate: **n/a**.
- Final hard-invalid trials: **0**; validation failure codes: `{}`.
- Hard validation is authoritative for safety and executability. Model-family disagreement is reported above as a soft challenge and is not counted as an invalid plan by itself.
- Intentionally unsafe perturbations intercepted: **0** / **0** perturbation trials where applicable.

## Limitations

- The benchmark suite is small and local; it is not representative of every tabular data-science domain.
- The empirical reference is not a universal optimum or ground truth; it ranks only the supported families under one CV design.
- Method-family match is not equivalent to predictive or deployment quality, and a one-split study cannot establish generalization.
- Offline fallback and mock rows must not be used to make claims about live LLM behavior.
- Semantic leakage, feature availability, and domain-specific safety still require expert review.
