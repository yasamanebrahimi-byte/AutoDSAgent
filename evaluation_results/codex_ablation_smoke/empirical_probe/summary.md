# AutoDS Validation Architecture Evaluation

This report is generated deterministically from `config.json`, `trials.jsonl`, and the computed summary. Offline fallback and mock rows are not evidence of live LLM performance.

## Experiment Configuration

- Repetitions per benchmark/scenario: **2**.
- Base seed: **42**; holdout fraction: **0.2**.
- Requested model: `gpt-4.1-mini`; prompt/schema version: `2026-08-24.blinded-evidence-comparison.v2-empirical-probe`.
- Gate objective version: `intervention-quality-v1`; neutrality tolerance: `0.02`; catastrophic threshold: `0.1`.
- Repository commit: `e1ecee9319479caf82ac76180a9e13cae328f8ef`.
- Each repetition keeps the case, frozen train/holdout membership, and training-only profile fixed; the intended varying factor is the stochastic LLM response.
- These rows are `modeling_gate` evaluations: benchmark target/task values are fixed context, while `agent_initial` represents only the post-split model-family and preprocessing proposal. `gated_final` is the approved plan after comparison, optional reconciliation, and deterministic validation. Formulation accuracy requires a separate formulation-gate evaluation mode.
- `empirical_reference` is an evaluation-only ranking of the four supported families using training-only CV; it is not an oracle and never enters runtime decisions.

## Trial Coverage

| Trial category | Count |
|---|---:|
| Requested live trials | 0 |
| Successful OpenAI trials | 0 |
| Offline fallback trials | 4 |
| Failed trials | 0 |
| Mock trials | 0 |
| Completed trials | 4 |

Claims about LLM behavior below use `agent_source == "openai"` only.

## Gate Health

Intervention quality is primary. Exact family match and top-2 compatibility below are secondary diagnostics.
- Challenges: **0 / 0 disagreements**; abstentions: **0**.
- Improved: **0**; worsened: **0**; neutral: **0**.
- Intervention precision: **n/a**; challenge yield: **n/a**; harmful-intervention rate: **n/a**; unnecessary-intervention rate: **n/a**.
- Challenge recall: **n/a**; missed rescues: **0**.
- Mean regret reduction: **0.0000**; median: **0.0000**; uncertainty interval: `{'lower': 0.0, 'upper': 0.0, 'support': 4, 'stable': False}`.
- Catastrophic regret: initial **0**, final **0**, prevented **0**, introduced **0**, net **0**.
- Utility contribution: `{'improvement_reward': 0.0, 'worsening_penalty': 0.0, 'unnecessary_intervention_penalty': 0.0, 'catastrophic_prevention_reward': 0.0, 'catastrophic_introduction_penalty': 0.0, 'missed_rescue_penalty': 0.0, 'total_utility': 0.0, 'weights': {'improvement': 1.0, 'worsening': 2.0, 'neutral_intervention': 0.25, 'catastrophic_prevention': 3.0, 'catastrophic_introduction': 5.0, 'missed_rescue': 1.0}}`.

| Metric | Trial-weighted | Dataset-weighted |
|---|---:|---:|
| Intervention precision | n/a | n/a |
| Harmful-intervention rate | n/a | n/a |
| Mean regret reduction | 0.0000 | 0.0000 |

## LLM Decision Stability

No successful OpenAI trials were recorded, so live decision stability is not estimable.

## Agent vs Deterministic Soft Challenge

- All operational trials: soft agreement **100.0%**, soft disagreement **0.0%**.
- Model-family disagreement rate: **0.0%**; preprocessing disagreement rate: **0.0%**.
- OpenAI-only method agreement: **n/a**; preprocessing agreement: **n/a**.

| Method distribution | Initial agent | Gated final |
|---|---|---|
| All trials | {'regularized_linear': {'count': 4, 'rate': 1.0}} | {'regularized_linear': {'count': 4, 'rate': 1.0}} |
| OpenAI only | {} | {} |

## Empirical Reference Comparison

- All operational trials: initial reference match **100.0%**; gated reference match **100.0%**.
- OpenAI only: initial reference match **n/a**; gated reference match **n/a**.
- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.

## Effect of the Validation Gate

- OpenAI-only gate outcomes: **0 improved**, **0 worsened**, **0 neutral**.
- OpenAI-only potentially unnecessary interventions: **0**.
- Operational outcomes: improved **0**, worsened **0**, neutral **4**.
- Improved/worsened/neutral is defined from normalized regret reduction using the configured neutrality tolerance; holdout results do not define this label.

## Soft-Challenge Reconciliation Outcomes

- Total disagreements: **0**; challenges: **0**; abstentions: **0**.
- Challenge rate: **n/a**; abstention rate: **n/a**.
- Soft-challenge reconciliation invocation rate: **n/a**.
- Reconciliation invocation rate: **0.0%**; success rate: **n/a**.
- Sided with agent: **n/a**; sided with deterministic challenger: **n/a**.
- Proposal A selected: **n/a**; Proposal B selected: **n/a**; A/B selection imbalance: **n/a**.
- Order-swap consistency: **n/a**; order-flip rate: **n/a** over **0** paired cases.
- Reconciliation modes observed: **['none']**.
- Soft-challenge outcomes: **0 improved**, **0 worsened**, **0 neutral**.
- Challenge outcomes: **0 improved**, **0 worsened**, **0 neutral**; intervention precision: **n/a**.
- Abstentions where agent was better: **0**; where deterministic was better: **0**.
- Mean challenge regret improvement: **n/a**; unnecessary interventions: **0** (**n/a**).
- Catastrophic-regret rate: **0.0%**; catastrophic cases prevented by challenge: **0** (**n/a**).
- A soft disagreement is competing advisory evidence, not an invalid plan. Every challenge row retains the initial plan, deterministic plan, preprocessing comparison, reconciliation response, selected source, and final hard-validation result.

## Predictive Performance

- OpenAI-only mean paired CV improvement: **n/a**; median: **n/a**; standard deviation: **n/a**.
- OpenAI-only mean paired holdout improvement: **n/a** (descriptive only; not used to define gate outcomes).
- Classification improvement is `gated_macro_f1 - initial_macro_f1`; regression improvement is `initial_rmse - gated_rmse`, so positive always means gating helped.
- Untouched holdout metrics are retained per trial as a descriptive external check after decisions and the empirical ranking are frozen.

## Dataset-Level Results

| Dataset | Trials | Challenges | Abstentions | Improved / worsened / neutral | Precision | Harm | Mean regret reduction | Catastrophic prevented / introduced | Exact match (diagnostic) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| wine | 0 | 0 | 0 | 0 / 0 / 0 | n/a | n/a | n/a | 0 / 0 | n/a -> n/a |

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
