# AutoDS Validation Architecture Evaluation

This report is generated deterministically from `config.json`, `trials.jsonl`, and the computed summary. Offline fallback and mock rows are not evidence of live LLM performance.

## Experiment Configuration

- Repetitions per benchmark/scenario: **5**.
- Base seed: **42**; holdout fraction: **0.2**.
- Requested model: `gpt-4.1-mini`; prompt/schema version: `2026-08-24.blinded-evidence-comparison.v3-empirical-probe`.
- Gate objective version: `intervention-quality-v1`; neutrality tolerance: `0.02`; catastrophic threshold: `0.1`.
- Repository commit: `e8195f8d815ef13316fe5a594a1f83d6ca868b44`.
- Each repetition keeps the case, frozen train/holdout membership, and training-only profile fixed; the intended varying factor is the stochastic LLM response.
- These rows are `modeling_gate` evaluations: benchmark target/task values are fixed context, while `agent_initial` represents only the post-split model-family and preprocessing proposal. `gated_final` is the approved plan after comparison, optional reconciliation, and deterministic validation. Formulation accuracy requires a separate formulation-gate evaluation mode.
- `empirical_reference` is an evaluation-only ranking of the four supported families using training-only CV; it is not an oracle and never enters runtime decisions.

## Trial Coverage

| Trial category | Count |
|---|---:|
| Requested live trials | 270 |
| Successful OpenAI trials | 270 |
| Offline fallback trials | 0 |
| Failed trials | 0 |
| Mock trials | 0 |
| Completed trials | 270 |

Claims about LLM behavior below use `agent_source == "openai"` only.

## Gate Health

Intervention quality is primary. Exact family match and top-2 compatibility below are secondary diagnostics.
- Challenges: **99 / 101 disagreements**; abstentions: **0**.
- Improved: **3**; worsened: **13**; neutral: **80**.
- Intervention precision: **18.8%**; challenge yield: **3.0%**; harmful-intervention rate: **13.1%**; unnecessary-intervention rate: **80.8%**.
- Challenge recall: **100.0%**; missed rescues: **0**.
- Mean regret reduction: **-0.0265**; median: **0.0000**; uncertainty interval: `{'lower': -0.06605816360345054, 'upper': 0.0004650734098510614, 'support': 265, 'stable': True}`.
- Catastrophic regret: initial **14**, final **16**, prevented **0**, introduced **2**, net **-2**.
- Utility contribution: `{'improvement_reward': 3.0, 'worsening_penalty': -26.0, 'unnecessary_intervention_penalty': -20.0, 'catastrophic_prevention_reward': 0.0, 'catastrophic_introduction_penalty': -10.0, 'missed_rescue_penalty': 0.0, 'total_utility': -53.0, 'weights': {'improvement': 1.0, 'worsening': 2.0, 'neutral_intervention': 0.25, 'catastrophic_prevention': 3.0, 'catastrophic_introduction': 5.0, 'missed_rescue': 1.0}}`.

| Metric | Trial-weighted | Dataset-weighted |
|---|---:|---:|
| Intervention precision | 18.8% | 25.0% |
| Harmful-intervention rate | 13.1% | 15.8% |
| Mean regret reduction | -0.0265 | -0.0260 |

## LLM Decision Stability

| Dataset | OpenAI trials | Unique initial methods | Modal method | Modal frequency | Pairwise consistency |
|---|---:|---:|---|---:|---:|
| breast_cancer | 15 | 1 | regularized_linear | 100.0% | 100.0% |
| diabetes | 15 | 1 | regularized_linear | 100.0% | 100.0% |
| digits_subset | 15 | 3 | boosted_tree | 66.7% | 46.7% |
| final_interaction_regression | 15 | 1 | boosted_tree | 100.0% | 100.0% |
| final_low_n_high_p_classification | 15 | 1 | regularized_linear | 100.0% | 100.0% |
| final_mixed_type_classification | 15 | 3 | regularized_linear | 60.0% | 43.8% |
| final_shifted_nonlinear_regression | 15 | 2 | boosted_tree | 80.0% | 65.7% |
| synthetic_binary_linear | 15 | 1 | linear | 100.0% | 100.0% |
| synthetic_binary_nonlinear | 15 | 1 | boosted_tree | 100.0% | 100.0% |
| synthetic_high_dim_regression | 15 | 2 | regularized_linear | 93.3% | 86.7% |
| synthetic_imbalanced_classification | 15 | 2 | regularized_linear | 60.0% | 48.6% |
| synthetic_linear_regression | 15 | 1 | linear | 100.0% | 100.0% |
| synthetic_missingness | 15 | 1 | regularized_linear | 100.0% | 100.0% |
| synthetic_multiclass | 15 | 1 | regularized_linear | 100.0% | 100.0% |
| synthetic_nonlinear_regression | 15 | 1 | boosted_tree | 100.0% | 100.0% |
| synthetic_outlier_regression | 15 | 2 | regularized_linear | 93.3% | 86.7% |
| synthetic_regression | 15 | 2 | linear | 66.7% | 52.4% |
| wine | 15 | 2 | regularized_linear | 86.7% | 75.2% |

## Agent vs Deterministic Soft Challenge

- All operational trials: soft agreement **62.6%**, soft disagreement **37.4%**.
- Model-family disagreement rate: **36.7%**; preprocessing disagreement rate: **15.9%**.
- OpenAI-only method agreement: **63.3%**; preprocessing agreement: **84.1%**.

| Method distribution | Initial agent | Gated final |
|---|---|---|
| All trials | {'boosted_tree': {'count': 79, 'rate': 0.29259259259259257}, 'linear': {'count': 41, 'rate': 0.15185185185185185}, 'regularized_linear': {'count': 142, 'rate': 0.5259259259259259}, 'tree_ensemble': {'count': 8, 'rate': 0.02962962962962963}} | {'boosted_tree': {'count': 63, 'rate': 0.23333333333333334}, 'linear': {'count': 30, 'rate': 0.1111111111111111}, 'regularized_linear': {'count': 174, 'rate': 0.6444444444444445}, 'tree_ensemble': {'count': 3, 'rate': 0.011111111111111112}} |
| OpenAI only | {'boosted_tree': {'count': 79, 'rate': 0.29259259259259257}, 'linear': {'count': 41, 'rate': 0.15185185185185185}, 'regularized_linear': {'count': 142, 'rate': 0.5259259259259259}, 'tree_ensemble': {'count': 8, 'rate': 0.02962962962962963}} | {'boosted_tree': {'count': 63, 'rate': 0.23333333333333334}, 'linear': {'count': 30, 'rate': 0.1111111111111111}, 'regularized_linear': {'count': 174, 'rate': 0.6444444444444445}, 'tree_ensemble': {'count': 3, 'rate': 0.011111111111111112}} |

## Empirical Reference Comparison

- All operational trials: initial reference match **58.5%**; gated reference match **61.9%**.
- OpenAI only: initial reference match **58.5%**; gated reference match **61.9%**.
- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.

## Effect of the Validation Gate

- OpenAI-only gate outcomes: **3 improved**, **13 worsened**, **249 neutral**.
- OpenAI-only potentially unnecessary interventions: **31**.
- Operational outcomes: improved **3**, worsened **13**, neutral **249**.
- Improved/worsened/neutral is defined from normalized regret reduction using the configured neutrality tolerance; holdout results do not define this label.

## Soft-Challenge Reconciliation Outcomes

- Total disagreements: **101**; challenges: **99**; abstentions: **0**.
- Challenge rate: **98.0%**; abstention rate: **0.0%**.
- Soft-challenge reconciliation invocation rate: **100.0%**.
- Reconciliation invocation rate: **37.4%**; success rate: **100.0%**.
- Sided with agent: **52.5%**; sided with deterministic challenger: **47.5%**.
- Proposal A selected: **n/a**; Proposal B selected: **n/a**; A/B selection imbalance: **n/a**.
- Order-swap consistency: **n/a**; order-flip rate: **n/a** over **0** paired cases.
- Reconciliation modes observed: **['legacy_reconciliation']**.
- Soft-challenge outcomes: **3 improved**, **13 worsened**, **80 neutral**.
- Challenge outcomes: **3 improved**, **13 worsened**, **80 neutral**; intervention precision: **18.8%**.
- Abstentions where agent was better: **0**; where deterministic was better: **0**.
- Mean challenge regret improvement: **-0.2059**; unnecessary interventions: **80** (**80.8%**).
- Catastrophic-regret rate: **5.2%**; catastrophic cases prevented by challenge: **0** (**0.0%**).
- A soft disagreement is competing advisory evidence, not an invalid plan. Every challenge row retains the initial plan, deterministic plan, preprocessing comparison, reconciliation response, selected source, and final hard-validation result.

## Predictive Performance

- OpenAI-only mean paired CV improvement: **-0.5681**; median: **0.0000**; standard deviation: **6.5380**.
- OpenAI-only mean paired holdout improvement: **-0.6452** (descriptive only; not used to define gate outcomes).
- Classification improvement is `gated_macro_f1 - initial_macro_f1`; regression improvement is `initial_rmse - gated_rmse`, so positive always means gating helped.
- Untouched holdout metrics are retained per trial as a descriptive external check after decisions and the empirical ranking are frozen.

## Dataset-Level Results

| Dataset | Trials | Challenges | Abstentions | Improved / worsened / neutral | Precision | Harm | Mean regret reduction | Catastrophic prevented / introduced | Exact match (diagnostic) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| breast_cancer | 15 | 0 | 0 | 0 / 0 / 15 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| diabetes | 15 | 0 | 0 | 0 / 0 / 15 | n/a | n/a | 0.0000 | 0 / 0 | 66.7% -> 66.7% |
| digits_subset | 15 | 12 | 0 | 0 / 0 / 15 | n/a | 0.0% | -0.0011 | 0 / 0 | 13.3% -> 0.0% |
| final_interaction_regression | 15 | 0 | 0 | 0 / 0 / 15 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| final_low_n_high_p_classification | 15 | 0 | 0 | 0 / 0 / 15 | n/a | n/a | 0.0000 | 0 / 0 | 66.7% -> 66.7% |
| final_mixed_type_classification | 15 | 11 | 0 | 0 / 0 / 10 | n/a | 0.0% | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| final_shifted_nonlinear_regression | 15 | 3 | 0 | 3 / 0 / 12 | 100.0% | 0.0% | 0.0198 | 0 / 0 | 80.0% -> 100.0% |
| synthetic_binary_linear | 15 | 15 | 0 | 0 / 0 / 15 | n/a | 0.0% | 0.0070 | 0 / 0 | 0.0% -> 100.0% |
| synthetic_binary_nonlinear | 15 | 10 | 0 | 0 / 0 / 15 | n/a | 0.0% | 0.0000 | 0 / 0 | 66.7% -> 66.7% |
| synthetic_high_dim_regression | 15 | 5 | 0 | 0 / 2 / 13 | 0.0% | 40.0% | -0.4621 | 0 / 2 | 26.7% -> 20.0% |
| synthetic_imbalanced_classification | 15 | 6 | 0 | 0 / 6 / 9 | 0.0% | 100.0% | -0.0232 | 0 / 0 | 40.0% -> 0.0% |
| synthetic_linear_regression | 15 | 0 | 0 | 0 / 0 / 15 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| synthetic_missingness | 15 | 0 | 0 | 0 / 0 / 15 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| synthetic_multiclass | 15 | 15 | 0 | 0 / 5 / 10 | 0.0% | 33.3% | -0.0093 | 0 / 0 | 33.3% -> 20.0% |
| synthetic_nonlinear_regression | 15 | 0 | 0 | 0 / 0 / 15 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| synthetic_outlier_regression | 15 | 15 | 0 | 0 / 0 / 15 | n/a | 0.0% | 0.0000 | 0 / 0 | 6.7% -> 6.7% |
| synthetic_regression | 15 | 5 | 0 | 0 / 0 / 15 | n/a | 0.0% | 0.0000 | 0 / 0 | 66.7% -> 66.7% |
| wine | 15 | 2 | 0 | 0 / 0 / 15 | n/a | 0.0% | 0.0007 | 0 / 0 | 86.7% -> 100.0% |

## Validation / Safety Interceptions

- Initial hard-invalid proposals: **5**; hard-validation interventions: **5**; interception rate: **100.0%**.
- Final hard-invalid trials: **0**; validation failure codes: `{'boosted_tree_encoding_is_compatible': 5}`.
- Hard validation is authoritative for safety and executability. Model-family disagreement is reported above as a soft challenge and is not counted as an invalid plan by itself.
- Intentionally unsafe perturbations intercepted: **0** / **0** perturbation trials where applicable.

## Limitations

- The benchmark suite is small and local; it is not representative of every tabular data-science domain.
- The empirical reference is not a universal optimum or ground truth; it ranks only the supported families under one CV design.
- Method-family match is not equivalent to predictive or deployment quality, and a one-split study cannot establish generalization.
- Offline fallback and mock rows must not be used to make claims about live LLM behavior.
- Semantic leakage, feature availability, and domain-specific safety still require expert review.
