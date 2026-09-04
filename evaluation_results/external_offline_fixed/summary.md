# AutoDS Validation Architecture Evaluation

This report is generated deterministically from `config.json`, `trials.jsonl`, and the computed summary. Offline fallback and mock rows are not evidence of live LLM performance.

## Experiment Configuration

- Repetitions per benchmark/scenario: **1**.
- Base seed: **42**; holdout fraction: **0.2**.
- Benchmark suite: `external`; tier: `None`.
- Planner model: `gpt-4.1-mini`; reconciler model: `gpt-4.1-mini`; prompt/schema version: `2026-08-24.blinded-evidence-comparison.v3-empirical-probe`.
- Gate objective version: `intervention-quality-v1`; neutrality tolerance: `0.02`; catastrophic threshold: `0.1`.
- Repository commit: `dfb7208a446368df52cce6a25835531b9ed09863`.
- Each repetition keeps the case, frozen train/holdout membership, and training-only profile fixed; the intended varying factor is the stochastic LLM response.
- These rows are `modeling_gate` evaluations: benchmark target/task values are fixed context, while `agent_initial` represents only the post-split model-family and preprocessing proposal. `gated_final` is the approved plan after comparison, optional reconciliation, and deterministic validation. Formulation accuracy requires a separate formulation-gate evaluation mode.
- `empirical_reference` is an evaluation-only ranking of the four supported families using training-only CV; it is not an oracle and never enters runtime decisions.

- External benchmark suite version: **1.0.0**; source: AMLB/OpenML task IDs.
- External Benchmark v1 uses AutoDS deterministic train/holdout splits rather than AMLB predefined folds; results are not directly comparable to AMLB leaderboard numbers.
- External results are evaluation-only and must not be used for policy calibration or threshold/prompt tuning.

## Trial Coverage

| Trial category | Count |
|---|---:|
| Requested live trials | 0 |
| Successful OpenAI trials | 0 |
| Offline fallback trials | 40 |
| Failed trials | 0 |
| Mock trials | 0 |
| Completed trials | 40 |

Claims about LLM behavior below use `agent_source == "openai"` only.

## Gate Health

Intervention quality is primary. Exact family match and top-2 compatibility below are secondary diagnostics.
- Challenges: **7 / 13 disagreements**; abstentions: **6**.
- Improved: **5**; worsened: **2**; neutral: **0**.
- Intervention precision: **71.4%**; challenge yield: **71.4%**; harmful-intervention rate: **28.6%**; unnecessary-intervention rate: **0.0%**.
- Challenge recall: **62.5%**; missed rescues: **3**.
- Mean regret reduction: **0.0294**; median: **0.0000**; uncertainty interval: `{'lower': -0.014516099783238538, 'upper': 0.0746117881515322, 'support': 28, 'stable': True}`.
- Catastrophic regret: initial **12**, final **10**, prevented **4**, introduced **2**, net **2**.
- Utility contribution: `{'improvement_reward': 5.0, 'worsening_penalty': -4.0, 'unnecessary_intervention_penalty': 0.0, 'catastrophic_prevention_reward': 12.0, 'catastrophic_introduction_penalty': -10.0, 'missed_rescue_penalty': -3.0, 'total_utility': 0.0, 'weights': {'improvement': 1.0, 'worsening': 2.0, 'neutral_intervention': 0.25, 'catastrophic_prevention': 3.0, 'catastrophic_introduction': 5.0, 'missed_rescue': 1.0}}`.

| Metric | Trial-weighted | Dataset-weighted |
|---|---:|---:|
| Intervention precision | 71.4% | 71.4% |
| Harmful-intervention rate | 28.6% | 28.6% |
| Mean regret reduction | 0.0294 | 0.0294 |

## LLM Decision Stability

No successful OpenAI trials were recorded, so live decision stability is not estimable.

## Agent vs Deterministic Soft Challenge

- All operational trials: soft agreement **45.0%**, soft disagreement **55.0%**.
- Model-family disagreement rate: **55.0%**; preprocessing disagreement rate: **35.0%**.
- OpenAI-only method agreement: **n/a**; preprocessing agreement: **n/a**.

| Method distribution | Initial agent | Gated final |
|---|---|---|
| All trials | {'regularized_linear': {'count': 26, 'rate': 0.65}, 'tree_ensemble': {'count': 14, 'rate': 0.35}} | {'boosted_tree': {'count': 6, 'rate': 0.21428571428571427}, 'regularized_linear': {'count': 14, 'rate': 0.5}, 'tree_ensemble': {'count': 8, 'rate': 0.2857142857142857}} |
| OpenAI only | {} | {} |

## Empirical Reference Comparison

- All operational trials: initial reference match **10.7%**; gated reference match **17.9%**.
- OpenAI only: initial reference match **n/a**; gated reference match **n/a**.
- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.

## Effect of the Validation Gate

- OpenAI-only gate outcomes: **0 improved**, **0 worsened**, **0 neutral**.
- OpenAI-only potentially unnecessary interventions: **0**.
- Operational outcomes: improved **5**, worsened **2**, neutral **21**.
- Improved/worsened/neutral is defined from normalized regret reduction using the configured neutrality tolerance; holdout results do not define this label.

## Soft-Challenge Reconciliation Outcomes

- Total disagreements: **13**; challenges: **7**; abstentions: **6**.
- Challenge rate: **53.8%**; abstention rate: **46.2%**.
- Soft-challenge reconciliation invocation rate: **53.8%**.
- Reconciliation invocation rate: **40.0%**; success rate: **43.8%**.
- Sided with agent: **0.0%**; sided with deterministic challenger: **100.0%**.
- Proposal A selected: **28.6%**; Proposal B selected: **71.4%**; A/B selection imbalance: **42.9%**.
- Order-swap consistency: **n/a**; order-flip rate: **n/a** over **0** paired cases.
- Reconciliation modes observed: **['blinded_evidence_comparison']**.
- Soft-challenge outcomes: **5 improved**, **2 worsened**, **0 neutral**.
- Challenge outcomes: **5 improved**, **2 worsened**, **0 neutral**; intervention precision: **71.4%**.
- Abstentions where agent was better: **0**; where deterministic was better: **3**.
- Mean deterministic-challenger regret advantage: **0.1176**; this is `agent normalized regret - deterministic challenger normalized regret`, so it is not final gated intervention improvement; unnecessary interventions: **0** (**0.0%**).
- Catastrophic-regret rate: **30.0%**; catastrophic cases prevented by challenge: **4** (**100.0%**).
- A soft disagreement is competing advisory evidence, not an invalid plan. Every challenge row retains the initial plan, deterministic plan, preprocessing comparison, reconciliation response, selected source, and final hard-validation result.

## Predictive Performance

- OpenAI-only mean paired CV improvement: **n/a**; median: **n/a**; standard deviation: **n/a**.
- OpenAI-only mean paired holdout improvement: **n/a** (descriptive only; not used to define gate outcomes).
- Classification improvement is `gated_macro_f1 - initial_macro_f1`; regression improvement is `initial_rmse - gated_rmse`, so positive always means gating helped.
- Untouched holdout metrics are retained per trial as a descriptive external check after decisions and the empirical ranking are frozen.

## Dataset-Level Results

| Dataset | Trials | Challenges | Abstentions | Improved / worsened / neutral | Precision | Harm | Mean regret reduction | Catastrophic prevented / introduced | Exact match (diagnostic) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| APSFailure | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| Amazon_employee_access | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| Australian | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| Bioresponse | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| Brazilian_houses | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| Click_prediction_small | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| GesturePhaseSegmentationProcessed | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| Internet-Advertisements | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| MIP-2016-regression | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| Mercedes_Benz_Greener_Manufacturing | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| Moneyball | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| OnlineNewsPopularity | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| PhishingWebsites | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| abalone | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| adult | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| bank-marketing | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| blood-transfusion-service-center | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| car | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| churn | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| colleges | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| credit-g | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| diamonds | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| dna | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| elevators | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| eucalyptus | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| house_16H | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| house_prices_nominal | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| house_sales | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| kc1 | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| ozone-level-8hr | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| phoneme | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| qsar-biodeg | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| quake | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| sensory | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| socmob | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| space_ga | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| steel-plates-fault | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| tecator | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| wine-quality-white | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |
| wine_quality | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |

## Validation / Safety Interceptions

- Initial hard-invalid proposals: **12**; hard-validation interventions: **12**; interception rate: **100.0%**.
- Final hard-invalid trials: **12**; validation failure codes: `{'identical_feature_rows_have_consistent_targets': 12, 'one_hot_matrix_is_memory_safe': 2}`.
- Hard validation is authoritative for safety and executability. Model-family disagreement is reported above as a soft challenge and is not counted as an invalid plan by itself.
- Intentionally unsafe perturbations intercepted: **0** / **0** perturbation trials where applicable.

## Limitations

- The benchmark suite is small and local; it is not representative of every tabular data-science domain.
- The empirical reference is not a universal optimum or ground truth; it ranks only the supported families under one CV design.
- Method-family match is not equivalent to predictive or deployment quality, and One train/holdout split and a small benchmark suite still do not establish broad domain generalization.
- Offline fallback and mock rows must not be used to make claims about live LLM behavior.
- Semantic leakage, feature availability, and domain-specific safety still require expert review.
