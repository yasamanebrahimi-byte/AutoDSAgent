# AutoDS Validation Architecture Evaluation

This report is generated deterministically from `config.json`, `trials.jsonl`, and the computed summary. Offline fallback and mock rows are not evidence of live LLM performance.

## Experiment Configuration

- Repetitions per benchmark/scenario: **3**.
- Base seed: **42**; holdout fraction: **0.2**.
- Benchmark suite: `external`; tier: `None`.
- Planner model: `gpt-5.6-terra`; reconciler model: `gpt-5.6-terra`; planner prompt schema: `2026-09-06.contract-aware-modeling.v2`; reconciler prompt schema: `2026-09-04.blinded-canonical-proposals.v1-empirical-probe`.
- Gate objective version: `intervention-quality-v1`; training/reference neutrality tolerance: `0.02`; holdout tolerances: classification `0.02` macro-F1 points, regression `0.02` relative RMSE; catastrophic threshold: `0.1`.
- Repository commit: `dd7514a54f7b9e8472bc7b3ad0515e1875ae5e07`.
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
| Requested live trials | 120 |
| Successful OpenAI trials | 120 |
| Offline fallback trials | 0 |
| Failed trials | 0 |
| Mock trials | 0 |
| Completed trials | 120 |
| Live required | True |
| Fallback rows | 0 |
| Planner live successes | 120 |
| Reconciler live successes | 0 |
| Strict-live validity | True |

Claims about LLM behavior below use `agent_source == "openai"` only.

## Gate Health

Untouched-holdout intervention outcome is primary. The headline estimate is dataset-macro: each eligible `benchmark_case` contributes one equally weighted dataset/task summary. Trial-weighted values and training/reference regret are secondary diagnostics. Exact family match and top-2 compatibility are also secondary diagnostics.
- Challenges: **0 / 0 actionable model-family disagreements**; abstentions: **0**.
- Holdout intervention outcomes: beneficial **0**; harmful **0**; neutral **0**.
- Training-reference diagnostics: challenge yield **n/a**; harmful-reference rate **n/a**; unnecessary-reference rate **n/a**.
- Paper-facing rates: challenge **n/a**, intervention **0.0%**, abstention/preservation **n/a**, beneficial **n/a**, harmful **n/a**, neutral **n/a**.
- Challenge recall: **n/a**; missed rescues: **0**.
- Mean regret reduction: **0.0000**; median: **0.0000**; uncertainty interval: `{'lower': 0.0, 'upper': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'support': 40, 'n_clusters': 40, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': True, 'status': 'ok'}`.
- Catastrophic regret: initial **6**, final **6**, prevented **0**, introduced **0**, net **0**.
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

| Dataset | OpenAI trials | Unique initial methods | Modal method | Modal frequency | Pairwise consistency |
|---|---:|---:|---|---:|---:|
| APSFailure | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Amazon_employee_access | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Australian | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Bioresponse | 3 | 1 | regularized_linear | 100.0% | 100.0% |
| Brazilian_houses | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Click_prediction_small | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| GesturePhaseSegmentationProcessed | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Internet-Advertisements | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| MIP-2016-regression | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Mercedes_Benz_Greener_Manufacturing | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Moneyball | 3 | 1 | regularized_linear | 100.0% | 100.0% |
| OnlineNewsPopularity | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| PhishingWebsites | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| abalone | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| adult | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| bank-marketing | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| blood-transfusion-service-center | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| car | 3 | 2 | boosted_tree | 66.7% | 33.3% |
| churn | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| colleges | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| credit-g | 3 | 1 | regularized_linear | 100.0% | 100.0% |
| diamonds | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| dna | 3 | 2 | regularized_linear | 66.7% | 33.3% |
| elevators | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| eucalyptus | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| house_16H | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| house_prices_nominal | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| house_sales | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| kc1 | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| ozone-level-8hr | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| phoneme | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| qsar-biodeg | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| quake | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| sensory | 3 | 1 | regularized_linear | 100.0% | 100.0% |
| socmob | 3 | 3 | boosted_tree | 33.3% | 0.0% |
| space_ga | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| steel-plates-fault | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| tecator | 3 | 1 | regularized_linear | 100.0% | 100.0% |
| wine-quality-white | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| wine_quality | 3 | 1 | boosted_tree | 100.0% | 100.0% |

## Agent vs Deterministic Soft Challenge

- All operational trials: soft agreement **n/a**, soft disagreement **n/a**.
- Model-family disagreement rate: **n/a**; preprocessing disagreement rate: **53.3%**.
- OpenAI-only method agreement: **39.2%**; preprocessing agreement: **46.7%**.

| Method distribution | Initial agent | Gated final |
|---|---|---|
| All trials | {'boosted_tree': {'count': 100, 'rate': 0.8333333333333334}, 'regularized_linear': {'count': 18, 'rate': 0.15}, 'tree_ensemble': {'count': 2, 'rate': 0.016666666666666666}} | {'boosted_tree': {'count': 100, 'rate': 0.8333333333333334}, 'regularized_linear': {'count': 18, 'rate': 0.15}, 'tree_ensemble': {'count': 2, 'rate': 0.016666666666666666}} |
| OpenAI only | {'boosted_tree': {'count': 100, 'rate': 0.8333333333333334}, 'regularized_linear': {'count': 18, 'rate': 0.15}, 'tree_ensemble': {'count': 2, 'rate': 0.016666666666666666}} | {'boosted_tree': {'count': 100, 'rate': 0.8333333333333334}, 'regularized_linear': {'count': 18, 'rate': 0.15}, 'tree_ensemble': {'count': 2, 'rate': 0.016666666666666666}} |

## Empirical Reference Comparison

- All operational trials: initial reference match **58.3%**; gated reference match **58.3%**.
- OpenAI only: initial reference match **58.3%**; gated reference match **58.3%**.
- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.

## Effect of the Validation Gate

- OpenAI-only gate outcomes: **0 improved**, **0 worsened**, **120 neutral**.
- OpenAI-only potentially unnecessary interventions: **0**.
- Operational outcomes: improved **0**, worsened **0**, neutral **120**.
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
- Catastrophic-regret rate: **5.0%**; catastrophic cases prevented by challenge: **0** (**n/a**).
- A soft disagreement is competing advisory evidence, not an invalid plan. Every challenge row retains the initial plan, deterministic plan, preprocessing comparison, reconciliation response, selected source, and final hard-validation result.

## Predictive Performance

- OpenAI-only mean paired CV improvement: **0.0000**; median: **0.0000**; standard deviation: **0.0000**.
- OpenAI-only mean paired paper holdout delta: **-0.0000** (dimensionless; subgroup descriptive only).
- Classification paper delta is `final_holdout_macro_f1 - initial_holdout_macro_f1`; regression paper delta is relative RMSE improvement `(initial_rmse-final_rmse)/max(abs(initial_rmse), epsilon)`. Positive always means the final plan helped.
- The untouched holdout is first used only after the final plan and all intervention decisions are frozen.
- Classification dataset-macro macro-F1 change: **n/a**; clustered CI: `{'lower': None, 'upper': None, 'ci_low': None, 'ci_high': None, 'support': 22, 'n_clusters': 22, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': True, 'status': 'unavailable'}`.
- Regression dataset-macro relative RMSE improvement: **n/a**; clustered CI: `{'lower': None, 'upper': None, 'ci_low': None, 'ci_high': None, 'support': 18, 'n_clusters': 18, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': False, 'status': 'unavailable'}`.

## Dataset-Level Results

| Dataset | Trials | Challenges | Abstentions | Improved / worsened / neutral | Precision | Harm | Mean regret reduction | Catastrophic prevented / introduced | Exact match (diagnostic) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| APSFailure | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| Amazon_employee_access | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| Australian | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| Bioresponse | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| Brazilian_houses | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| Click_prediction_small | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| GesturePhaseSegmentationProcessed | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| Internet-Advertisements | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| MIP-2016-regression | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| Mercedes_Benz_Greener_Manufacturing | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| Moneyball | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| OnlineNewsPopularity | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| PhishingWebsites | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| abalone | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| adult | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| bank-marketing | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| blood-transfusion-service-center | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| car | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 66.7% -> 66.7% |
| churn | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| colleges | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| credit-g | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| diamonds | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| dna | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 33.3% -> 33.3% |
| elevators | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| eucalyptus | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| house_16H | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| house_prices_nominal | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| house_sales | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| kc1 | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| ozone-level-8hr | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| phoneme | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| qsar-biodeg | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| quake | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| sensory | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| socmob | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 33.3% -> 33.3% |
| space_ga | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| steel-plates-fault | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| tecator | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| wine-quality-white | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| wine_quality | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |

## Validation / Safety Interceptions

- Initial hard-invalid proposals: **0**; hard interceptions: **0**; hard repairs: **0**; soft interventions: **0**.
- Hard interception means an invalid initial LLM plan was detected and prevented from training. Hard repair means that invalid plan was replaced by a valid deterministic alternative. Soft intervention means a valid LLM plan changed after a valid model-family disagreement; preprocessing-only disagreement remains diagnostic.
- Final hard-invalid trials: **0**; validation failure codes: `{}`.
- Hard validation is authoritative for safety and executability. Model-family disagreement is reported above as a soft challenge and is not counted as an invalid plan by itself.
- Intentionally unsafe perturbations intercepted: **0** / **0** perturbation trials where applicable.

## Limitations

- The frozen external AMLB/OpenML suite is not representative of every tabular data-science domain.
- The empirical reference is not a universal optimum or ground truth; it ranks only the supported families under one CV design.
- Method-family match is not equivalent to predictive or deployment quality, and One train/holdout split and the frozen external AMLB/OpenML benchmark suite still do not establish broad domain generalization.
- Offline fallback and mock rows must not be used to make claims about live LLM behavior.
- Semantic leakage, feature availability, and domain-specific safety still require expert review.
