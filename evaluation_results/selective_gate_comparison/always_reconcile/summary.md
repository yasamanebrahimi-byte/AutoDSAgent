# AutoDS Validation Architecture Evaluation

This report is generated deterministically from `config.json`, `trials.jsonl`, and the computed summary. Offline fallback and mock rows are not evidence of live LLM performance.

## Experiment Configuration

- Repetitions per benchmark/scenario: **1**.
- Base seed: **42**; holdout fraction: **0.2**.
- Requested model: `gpt-4.1-mini`; prompt/schema version: `2026-08-23.formulation-modeling-gates.v2`.
- Repository commit: `4f9fed67a0a31743b8f84f1d60ff34f9bb324e65`.
- Each repetition keeps the case, frozen train/holdout membership, and training-only profile fixed; the intended varying factor is the stochastic LLM response.
- These rows are `modeling_gate` evaluations: benchmark target/task values are fixed context, while `agent_initial` represents only the post-split model-family and preprocessing proposal. `gated_final` is the approved plan after comparison, optional reconciliation, and deterministic validation. Formulation accuracy requires a separate formulation-gate evaluation mode.
- `empirical_reference` is an evaluation-only ranking of the four supported families using training-only CV; it is not an oracle and never enters runtime decisions.

## Trial Coverage

| Trial category | Count |
|---|---:|
| Requested live trials | 0 |
| Successful OpenAI trials | 0 |
| Offline fallback trials | 3 |
| Failed trials | 0 |
| Mock trials | 0 |
| Completed trials | 3 |

Claims about LLM behavior below use `agent_source == "openai"` only.

## LLM Decision Stability

No successful OpenAI trials were recorded, so live decision stability is not estimable.

## Agent vs Deterministic Soft Challenge

- All operational trials: soft agreement **33.3%**, soft disagreement **66.7%**.
- Model-family disagreement rate: **66.7%**; preprocessing disagreement rate: **0.0%**.
- OpenAI-only method agreement: **n/a**; preprocessing agreement: **n/a**.

| Method distribution | Initial agent | Gated final |
|---|---|---|
| All trials | {'regularized_linear': {'count': 3, 'rate': 1.0}} | {'boosted_tree': {'count': 1, 'rate': 0.3333333333333333}, 'linear': {'count': 1, 'rate': 0.3333333333333333}, 'regularized_linear': {'count': 1, 'rate': 0.3333333333333333}} |
| OpenAI only | {} | {} |

## Empirical Reference Comparison

- All operational trials: initial reference match **0.0%**; gated reference match **33.3%**.
- OpenAI only: initial reference match **n/a**; gated reference match **n/a**.
- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.

## Effect of the Validation Gate

- OpenAI-only gate outcomes: **0 improved**, **0 worsened**, **0 tied**.
- OpenAI-only potentially unnecessary interventions: **0**.
- Operational outcomes: improved **1**, worsened **0**, tie **2**.
- Improved/worsened/tie is defined from paired training-only CV regret using the configured tolerance; holdout results do not define this label.

## Soft-Challenge Reconciliation Outcomes

- Total disagreements: **2**; challenges: **2**; abstentions: **0**.
- Challenge rate: **100.0%**; abstention rate: **0.0%**.
- Soft-challenge reconciliation invocation rate: **100.0%**.
- Reconciliation invocation rate: **66.7%**; success rate: **100.0%**.
- Sided with agent: **0.0%**; sided with deterministic challenger: **100.0%**.
- Soft-challenge outcomes: **1 improved**, **0 worsened**, **1 neutral**.
- Challenge outcomes: **1 improved**, **0 worsened**, **1 neutral**; intervention precision: **100.0%**.
- Abstentions where agent was better: **0**; where deterministic was better: **0**.
- Mean challenge regret improvement: **0.1533**; unnecessary interventions: **0** (**0.0%**).
- Catastrophic-regret rate: **66.7%**; catastrophic cases prevented by challenge: **1** (**50.0%**).
- A soft disagreement is competing advisory evidence, not an invalid plan. Every challenge row retains the initial plan, deterministic plan, preprocessing comparison, reconciliation response, selected source, and final hard-validation result.

## Predictive Performance

- OpenAI-only mean paired CV improvement: **n/a**; median: **n/a**; standard deviation: **n/a**.
- OpenAI-only mean paired holdout improvement: **n/a** (descriptive only; not used to define gate outcomes).
- Classification improvement is `gated_macro_f1 - initial_macro_f1`; regression improvement is `initial_rmse - gated_rmse`, so positive always means gating helped.
- Untouched holdout metrics are retained per trial as a descriptive external check after decisions and the empirical ranking are frozen.

## Dataset-Level Results

| Dataset | Trials | Initial match | Gated match | Improved / worsened / tie | Mean paired CV improvement |
|---|---:|---:|---:|---|---:|
| synthetic_binary_nonlinear | 0 | n/a | n/a | 0 / 0 / 0 | n/a |
| synthetic_high_dim_regression | 0 | n/a | n/a | 0 / 0 / 0 | n/a |
| synthetic_nonlinear_regression | 0 | n/a | n/a | 0 / 0 / 0 | n/a |

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
