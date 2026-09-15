# AutoDS Validation Architecture Evaluation

This report is generated deterministically from `config.json`, `trials.jsonl`, and the computed summary. Offline fallback and mock rows are not evidence of live LLM performance.

## Experiment Configuration

- Repetitions per benchmark/scenario: **3**.
- Base seed: **42**; holdout fraction: **0.2**.
- Benchmark suite: `external`; tier: `None`.
- Provider: `google`; planner model: `gemini-3.5-flash-lite`; reconciler model: `gemini-3.5-flash-lite`; planner prompt schema: `2026-09-06.contract-aware-modeling.v2`; reconciler prompt schema: `2026-09-04.blinded-canonical-proposals.v1-empirical-probe`.
- Gate objective version: `intervention-quality-v1`; training/reference neutrality tolerance: `0.02`; holdout tolerances: classification `0.02` macro-F1 points, regression `0.02` relative RMSE; catastrophic threshold: `0.1`.
- Repository commit: `df4f85f138a2deb69f61e32d5130a6a403b96da2`.
- Each repetition keeps the case, frozen train/holdout membership, and training-only profile fixed; repetition IDs are aligned slots for balanced analysis, not shared-seed stochastic matches across separate planner calls.
- These rows are `modeling_gate` evaluations: benchmark target/task values are fixed context, while `agent_initial` represents only the post-split model-family and preprocessing proposal. `gated_final` is the approved plan after comparison, optional reconciliation, and deterministic validation. Formulation accuracy requires a separate formulation-gate evaluation mode.
- Primary uncertainty: 95% confidence intervals use a nonparametric dataset/task-cluster bootstrap with replacement. All split seeds and stochastic repetitions belonging to a sampled dataset are retained together; the benchmark dataset/task is the independent sampling unit.
- `empirical_reference` is an evaluation-only ranking of the four supported families using training-only CV; it is not an oracle and never enters runtime decisions.

- External benchmark suite version: **1.0.0**; source: AMLB/OpenML task IDs.
- External Benchmark v1 uses AutoDS deterministic train/holdout splits rather than AMLB predefined folds; results are not directly comparable to AMLB leaderboard numbers.
- External results are evaluation-only and must not be used for policy calibration or threshold/prompt tuning.

## Trial Coverage

| Trial category | Count |
|---|---:|
| Requested live trials | 0 |
| Successful requested-provider live calls | 0 |
| Successful OpenAI trials | 0 |
| Offline fallback trials | 0 |
| Failed trials | 0 |
| Mock trials | 0 |
| Completed trials | 120 |
| Live required | True |
| Fallback rows | 0 |
| Planner live successes | 0 |
| Reconciler live successes | 0 |
| Successful reconciliation live calls | 0 |
| Strict-live validity | True |

Claims about successful live LLM behavior below use `agent_source` equal to the requested provider; legacy OpenAI-only diagnostics are labeled separately.

## Gate Health

Untouched-holdout intervention outcome is primary. The headline estimate is dataset-macro: each eligible `benchmark_case` contributes one equally weighted dataset/task summary. Trial-weighted values and training/reference regret are secondary diagnostics. Exact family match and top-2 compatibility are also secondary diagnostics.
- Challenges: **0 / 0 actionable model-family disagreements**; abstentions: **0**.
- Holdout intervention outcomes: beneficial **0**; harmful **0**; neutral **0**.
- Training-reference diagnostics: challenge yield **n/a**; harmful-reference rate **n/a**; unnecessary-reference rate **n/a**.
- Paper-facing rates: challenge **n/a**, intervention **0.0%**, abstention/preservation **n/a**, beneficial **n/a**, harmful **n/a**, neutral **n/a**.
- Challenge recall: **n/a**; missed rescues: **0**.
- Mean regret reduction: **0.0000**; median: **0.0000**; uncertainty interval: `{'lower': 0.0, 'upper': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'support': 40, 'n_clusters': 40, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': True, 'status': 'ok'}`.
- Catastrophic regret: initial **33**, final **33**, prevented **0**, introduced **0**, net **0**.
- Utility contribution: `{'improvement_reward': 0.0, 'worsening_penalty': 0.0, 'unnecessary_intervention_penalty': 0.0, 'catastrophic_prevention_reward': 0.0, 'catastrophic_introduction_penalty': 0.0, 'missed_rescue_penalty': 0.0, 'total_utility': 0.0, 'weights': {'improvement': 1.0, 'worsening': 2.0, 'neutral_intervention': 0.25, 'catastrophic_prevention': 3.0, 'catastrophic_introduction': 5.0, 'missed_rescue': 1.0}}`.
- Within-model-condition paper holdout delta (dataset-macro mean): **n/a**; median: **n/a**; clustered CI: `{'lower': None, 'upper': None, 'ci_low': None, 'ci_high': None, 'support': 40, 'n_clusters': 40, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': True, 'status': 'unavailable'}`. Classification and regression magnitudes are reported separately below because their units differ.

| Metric | Dataset-macro (primary) | Trial-weighted (secondary diagnostic) |
|---|---:|---:|
| Intervention precision | n/a | n/a |
| Holdout harmful-intervention rate | n/a | n/a |
| Mean regret reduction (secondary diagnostic) | 0.0000 | 0.0000 |
| Within-model-condition paper holdout delta | n/a | n/a |

Paper-facing rate denominators: challenge rate = challenged actionable model-family disagreements / actionable model-family disagreements; intervention rate conditional on disagreement = actual soft final-plan changes / actionable model-family disagreements; abstention rate = actionable model-family disagreements preserved for insufficient evidence / actionable model-family disagreements; beneficial, harmful, and neutral rates = the corresponding actual interventions / actual interventions with evaluable holdout outcomes; intervention precision = beneficial / comparable interventions; harm rate = harmful / comparable interventions. Preprocessing-only disagreements are reported separately and excluded from these denominators. Zero denominators are reported as `null`/`n/a`.

## LLM Decision Stability

No successful requested-provider trials were recorded, so live decision stability is not estimable.

## Agent vs Deterministic Soft Challenge

- All operational trials: soft agreement **n/a**, soft disagreement **n/a**.
- Model-family disagreement rate: **n/a**; preprocessing disagreement rate: **0.0%**.
- OpenAI-only method agreement: **n/a**; preprocessing agreement: **n/a**.

| Method distribution | Initial agent | Gated final |
|---|---|---|
| All trials | {'boosted_tree': {'count': 42, 'rate': 0.35}, 'linear': {'count': 6, 'rate': 0.05}, 'regularized_linear': {'count': 54, 'rate': 0.45}, 'tree_ensemble': {'count': 18, 'rate': 0.15}} | {'boosted_tree': {'count': 42, 'rate': 0.35}, 'linear': {'count': 6, 'rate': 0.05}, 'regularized_linear': {'count': 54, 'rate': 0.45}, 'tree_ensemble': {'count': 18, 'rate': 0.15}} |
| OpenAI only | {} | {} |

## Empirical Reference Comparison

- All operational trials: initial reference match **30.0%**; gated reference match **30.0%**.
- OpenAI only: initial reference match **n/a**; gated reference match **n/a**.
- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.

## Effect of the Validation Gate

- OpenAI-only gate outcomes: **0 improved**, **0 worsened**, **0 neutral**.
- OpenAI-only potentially unnecessary interventions: **0**.
- Operational outcomes: improved **0**, worsened **0**, neutral **117**.
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
- Catastrophic-regret rate: **27.5%**; catastrophic cases prevented by challenge: **0** (**n/a**).
- A soft disagreement is competing advisory evidence, not an invalid plan. Every challenge row retains the initial plan, deterministic plan, preprocessing comparison, reconciliation response, selected source, and final hard-validation result.

## Predictive Performance

- OpenAI-only mean paired CV improvement: **n/a**; median: **n/a**; standard deviation: **n/a**.
- OpenAI-only mean paired paper holdout delta: **n/a** (dimensionless; subgroup descriptive only).
- Classification paper delta is `final_holdout_macro_f1 - initial_holdout_macro_f1`; regression paper delta is relative RMSE improvement `(initial_rmse-final_rmse)/max(abs(initial_rmse), epsilon)`. Positive always means the final plan helped.
- The untouched holdout is first used only after the final plan and all intervention decisions are frozen.
- Classification dataset-macro macro-F1 change: **n/a**; clustered CI: `{'lower': None, 'upper': None, 'ci_low': None, 'ci_high': None, 'support': 22, 'n_clusters': 22, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': True, 'status': 'unavailable'}`.
- Regression dataset-macro relative RMSE improvement: **n/a**; clustered CI: `{'lower': None, 'upper': None, 'ci_low': None, 'ci_high': None, 'support': 18, 'n_clusters': 18, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': False, 'status': 'unavailable'}`.

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

- Initial hard-invalid proposals: **3**; hard interceptions: **3**; hard repairs: **0**; soft interventions: **0**.
- Hard interception means an invalid initial LLM plan was detected and prevented from training. Hard repair means that invalid plan was replaced by a valid deterministic alternative. Soft intervention means a valid LLM plan changed after a valid model-family disagreement; preprocessing-only disagreement remains diagnostic.
- Final hard-invalid trials: **3**; validation failure codes: `{'one_hot_matrix_is_memory_safe': 3}`.
- Hard validation is authoritative for safety and executability. Model-family disagreement is reported above as a soft challenge and is not counted as an invalid plan by itself.
- Intentionally unsafe perturbations intercepted: **0** / **0** perturbation trials where applicable.

## Limitations

- The frozen external AMLB/OpenML suite is not representative of every tabular data-science domain.
- The empirical reference is not a universal optimum or ground truth; it ranks only the supported families under one CV design.
- Method-family match is not equivalent to predictive or deployment quality, and One train/holdout split and the frozen external AMLB/OpenML benchmark suite still do not establish broad domain generalization.
- Offline fallback and mock rows must not be used to make claims about live LLM behavior.
- Semantic leakage, feature availability, and domain-specific safety still require expert review.
