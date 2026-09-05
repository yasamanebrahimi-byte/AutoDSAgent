# AutoDS Validation Architecture Evaluation

This report is generated deterministically from `config.json`, `trials.jsonl`, and the computed summary. Offline fallback and mock rows are not evidence of live LLM performance.

## Experiment Configuration

- Repetitions per benchmark/scenario: **1**.
- Base seed: **42**; holdout fraction: **0.2**.
- Benchmark suite: `local`; tier: `None`.
- Planner model: `gpt-5.4-mini-2026-03-17`; reconciler model: `gpt-5.4-mini-2026-03-17`; planner prompt schema: `2026-09-04.training-profile-diagnostics.v1`; reconciler prompt schema: `2026-09-04.blinded-canonical-proposals.v1-empirical-probe`.
- Gate objective version: `intervention-quality-v1`; training/reference neutrality tolerance: `0.02`; holdout tolerances: classification `0.02` macro-F1 points, regression `0.02` relative RMSE; catastrophic threshold: `0.1`.
- Repository commit: `19d753612509beb490a0a8b2981e4aa8b7e3231f`.
- Each repetition keeps the case, frozen train/holdout membership, and training-only profile fixed; the intended varying factor is the stochastic LLM response.
- These rows are `modeling_gate` evaluations: benchmark target/task values are fixed context, while `agent_initial` represents only the post-split model-family and preprocessing proposal. `gated_final` is the approved plan after comparison, optional reconciliation, and deterministic validation. Formulation accuracy requires a separate formulation-gate evaluation mode.
- Primary uncertainty: 95% confidence intervals use a nonparametric dataset/task-cluster bootstrap with replacement. All split seeds and stochastic repetitions belonging to a sampled dataset are retained together; the benchmark dataset/task is the independent sampling unit.
- `empirical_reference` is an evaluation-only ranking of the four supported families using training-only CV; it is not an oracle and never enters runtime decisions.

## Trial Coverage

| Trial category | Count |
|---|---:|
| Requested live trials | 1 |
| Successful OpenAI trials | 1 |
| Offline fallback trials | 0 |
| Failed trials | 0 |
| Mock trials | 0 |
| Completed trials | 1 |
| Live required | True |
| Fallback rows | 0 |
| Planner live successes | 1 |
| Reconciler live successes | 0 |
| Strict-live validity | True |

Claims about LLM behavior below use `agent_source == "openai"` only.

## Gate Health

Untouched-holdout intervention outcome is primary. The headline estimate is dataset-macro: each eligible `benchmark_case` contributes one equally weighted dataset/task summary. Trial-weighted values and training/reference regret are secondary diagnostics. Exact family match and top-2 compatibility are also secondary diagnostics.
- Challenges: **0 / 0 disagreements**; abstentions: **0**.
- Holdout intervention outcomes: beneficial **0**; harmful **0**; neutral **0**.
- Training-reference diagnostics: challenge yield **n/a**; harmful-reference rate **n/a**; unnecessary-reference rate **n/a**.
- Paper-facing rates: challenge **n/a**, intervention **0.0%**, abstention/preservation **n/a**, beneficial **n/a**, harmful **n/a**, neutral **n/a**.
- Challenge recall: **n/a**; missed rescues: **0**.
- Mean regret reduction: **0.0000**; median: **0.0000**; uncertainty interval: `{'lower': None, 'upper': None, 'ci_low': None, 'ci_high': None, 'support': 1, 'n_clusters': 1, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': False, 'status': 'unavailable'}`.
- Catastrophic regret: initial **0**, final **0**, prevented **0**, introduced **0**, net **0**.
- Utility contribution: `{'improvement_reward': 0.0, 'worsening_penalty': 0.0, 'unnecessary_intervention_penalty': 0.0, 'catastrophic_prevention_reward': 0.0, 'catastrophic_introduction_penalty': 0.0, 'missed_rescue_penalty': 0.0, 'total_utility': 0.0, 'weights': {'improvement': 1.0, 'worsening': 2.0, 'neutral_intervention': 0.25, 'catastrophic_prevention': 3.0, 'catastrophic_introduction': 5.0, 'missed_rescue': 1.0}}`.
- Descriptive pooled paper holdout delta (dataset-macro mean): **n/a**; median: **n/a**; clustered CI: `{'lower': None, 'upper': None, 'ci_low': None, 'ci_high': None, 'support': 1, 'n_clusters': 1, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': False, 'status': 'unavailable'}`. Classification and regression magnitudes are reported separately below because their units differ.

| Metric | Dataset-macro (primary) | Trial-weighted (secondary diagnostic) |
|---|---:|---:|
| Intervention precision | n/a | n/a |
| Holdout harmful-intervention rate | n/a | n/a |
| Mean regret reduction (secondary diagnostic) | 0.0000 | 0.0000 |
| Descriptive pooled paper holdout delta | n/a | n/a |

Paper-facing rate denominators: challenge rate = challenged eligible initial plans / eligible soft disagreements; intervention rate = actual soft final-plan changes / completed eligible trials; abstention rate = challenged eligible initial plans preserved for insufficient evidence / eligible soft disagreements; beneficial, harmful, and neutral rates = the corresponding actual interventions / actual interventions with evaluable holdout outcomes; intervention precision = beneficial / comparable interventions; harm rate = harmful / comparable interventions. Zero denominators are reported as `null`/`n/a`.

## LLM Decision Stability

| Dataset | OpenAI trials | Unique initial methods | Modal method | Modal frequency | Pairwise consistency |
|---|---:|---:|---|---:|---:|
| synthetic_binary_linear | 1 | 1 | regularized_linear | 100.0% | n/a |

## Model-Condition Summaries

Each planner condition is reported separately. Repetitions are nested within dataset/task; no model conditions are silently pooled.

| Model condition | Intervention rate | Abstention rate | Beneficial | Harmful | Neutral | Precision | Holdout delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a |

## Agent vs Deterministic Soft Challenge

- All operational trials: soft agreement **100.0%**, soft disagreement **0.0%**.
- Model-family disagreement rate: **0.0%**; preprocessing disagreement rate: **0.0%**.
- OpenAI-only method agreement: **100.0%**; preprocessing agreement: **100.0%**.

| Method distribution | Initial agent | Gated final |
|---|---|---|
| All trials | {'regularized_linear': {'count': 1, 'rate': 1.0}} | {'regularized_linear': {'count': 1, 'rate': 1.0}} |
| OpenAI only | {'regularized_linear': {'count': 1, 'rate': 1.0}} | {'regularized_linear': {'count': 1, 'rate': 1.0}} |

## Empirical Reference Comparison

- All operational trials: initial reference match **100.0%**; gated reference match **100.0%**.
- OpenAI only: initial reference match **100.0%**; gated reference match **100.0%**.
- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.

## Effect of the Validation Gate

- OpenAI-only gate outcomes: **0 improved**, **0 worsened**, **1 neutral**.
- OpenAI-only potentially unnecessary interventions: **0**.
- Operational outcomes: improved **0**, worsened **0**, neutral **1**.
- Training-side improved/worsened/neutral is a secondary normalized-regret diagnostic; the primary realized intervention label is defined from the untouched-holdout `paper_holdout_delta` and task-specific tolerance.

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
- Training-reference challenge outcomes: **0 improved**, **0 worsened**, **0 neutral**; holdout intervention precision: **n/a**.
- Abstentions where agent was better: **0**; where deterministic was better: **0**.
- Mean deterministic-challenger regret advantage: **n/a**; this is `agent normalized regret - deterministic challenger normalized regret`, so it is not final gated intervention improvement; training-reference unnecessary interventions: **0** (**n/a**).
- Catastrophic-regret rate: **0.0%**; catastrophic cases prevented by challenge: **0** (**n/a**).
- A soft disagreement is competing advisory evidence, not an invalid plan. Every challenge row retains the initial plan, deterministic plan, preprocessing comparison, reconciliation response, selected source, and final hard-validation result.

## Predictive Performance

- OpenAI-only mean paired CV improvement: **0.0000**; median: **0.0000**; standard deviation: **0.0000**.
- OpenAI-only mean paired paper holdout delta: **0.0000** (dimensionless; subgroup descriptive only).
- Classification paper delta is `final_holdout_macro_f1 - initial_holdout_macro_f1`; regression paper delta is relative RMSE improvement `(initial_rmse-final_rmse)/max(abs(initial_rmse), epsilon)`. Positive always means the final plan helped.
- The untouched holdout is first used only after the final plan and all intervention decisions are frozen.
- Classification dataset-macro macro-F1 change: **n/a**; clustered CI: `{'lower': None, 'upper': None, 'ci_low': None, 'ci_high': None, 'support': 1, 'n_clusters': 1, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': False, 'status': 'unavailable'}`.
- Regression dataset-macro relative RMSE improvement: **n/a**; clustered CI: `{'lower': None, 'upper': None, 'ci_low': None, 'ci_high': None, 'support': 0, 'n_clusters': 0, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': False, 'status': 'unavailable'}`.

## Dataset-Level Results

| Dataset | Trials | Challenges | Abstentions | Improved / worsened / neutral | Precision | Harm | Mean regret reduction | Catastrophic prevented / introduced | Exact match (diagnostic) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| synthetic_binary_linear | 1 | 0 | 0 | 0 / 0 / 1 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |

## Validation / Safety Interceptions

- Initial hard-invalid proposals: **0**; hard-validation interventions: **0**; interception rate: **n/a**.
- Final hard-invalid trials: **0**; validation failure codes: `{}`.
- Hard validation is authoritative for safety and executability. Model-family disagreement is reported above as a soft challenge and is not counted as an invalid plan by itself.
- Intentionally unsafe perturbations intercepted: **0** / **0** perturbation trials where applicable.

## Limitations

- The local benchmark suite is intended for reproducible development and is not representative of every tabular data-science domain.
- The empirical reference is not a universal optimum or ground truth; it ranks only the supported families under one CV design.
- Method-family match is not equivalent to predictive or deployment quality, and One train/holdout split and a small benchmark suite still do not establish broad domain generalization.
- Offline fallback and mock rows must not be used to make claims about live LLM behavior.
- Semantic leakage, feature availability, and domain-specific safety still require expert review.
