# AutoDS Validation Architecture Evaluation

This report is generated deterministically from `config.json`, `trials.jsonl`, and the computed summary. Offline fallback and mock rows are not evidence of live LLM performance.

## Experiment Configuration

- Repetitions per benchmark/scenario: **1**.
- Base seed: **42**; holdout fraction: **0.2**.
- Requested model: `gpt-4.1-mini`; prompt/schema version: `2026-08-21.modeling-reconciliation.v1`.
- Repository commit: `c9efbb7a378850564ae1238267fe5ca2c65c9244`.
- Each repetition keeps the case, frozen train/holdout membership, and training-only profile fixed; the intended varying factor is the stochastic LLM response.
- `agent_initial` is the independent modeling response before deterministic recommendation or reconciliation. `gated_final` is the approved plan after comparison, optional reconciliation, and deterministic validation.
- `empirical_reference` is an evaluation-only ranking of the four supported families using training-only CV; it is not an oracle and never enters runtime decisions.

## Trial Coverage

| Trial category | Count |
|---|---:|
| Requested live trials | 0 |
| Successful OpenAI trials | 0 |
| Offline fallback trials | 1 |
| Failed trials | 0 |
| Mock trials | 0 |
| Completed trials | 1 |

Claims about LLM behavior below use `agent_source == "openai"` only.

## LLM Decision Stability

No successful OpenAI trials were recorded, so live decision stability is not estimable.

## Agent vs Deterministic Agreement

- All operational trials: agreement **100.0%**, disagreement **0.0%**.
- OpenAI-only method agreement: **n/a**; preprocessing agreement: **n/a**.

| Method distribution | Initial agent | Gated final |
|---|---|---|
| All trials | {'regularized_linear': {'count': 1, 'rate': 1.0}} | {'regularized_linear': {'count': 1, 'rate': 1.0}} |
| OpenAI only | {} | {} |

## Empirical Reference Comparison

- All operational trials: initial reference match **100.0%**; gated reference match **100.0%**.
- OpenAI only: initial reference match **n/a**; gated reference match **n/a**.
- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.

## Effect of the Validation Gate

- OpenAI-only gate outcomes: **0 improved**, **0 worsened**, **0 tied**.
- OpenAI-only potentially unnecessary interventions: **0**.
- Operational outcomes: improved **0**, worsened **0**, tie **1**.
- Improved/worsened/tie is defined from paired training-only CV regret using the configured tolerance; holdout results do not define this label.

## Reconciliation Outcomes

- Reconciliation invocation rate: **0.0%**; success rate: **n/a**.
- Sided with agent: **n/a**; sided with deterministic validator: **n/a**.
- Every disagreement row retains the initial plan, deterministic plan, preprocessing comparison, reconciliation response, selected source, and final validation result.

## Predictive Performance

- OpenAI-only mean paired CV improvement: **n/a**; median: **n/a**; standard deviation: **n/a**.
- OpenAI-only mean paired holdout improvement: **n/a** (descriptive only; not used to define gate outcomes).
- Classification improvement is `gated_macro_f1 - initial_macro_f1`; regression improvement is `initial_rmse - gated_rmse`, so positive always means gating helped.
- Untouched holdout metrics are retained per trial as a descriptive external check after decisions and the empirical ranking are frozen.

## Dataset-Level Results

| Dataset | Trials | Initial match | Gated match | Improved / worsened / tie | Mean paired CV improvement |
|---|---:|---:|---:|---|---:|
| wine | 0 | n/a | n/a | 0 / 0 / 0 | n/a |

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
