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
| Requested live trials | 120 |
| Successful requested-provider live calls | 0 |
| Successful OpenAI trials | 0 |
| Offline fallback trials | 0 |
| Failed trials | 0 |
| Mock trials | 0 |
| Completed trials | 120 |
| Live required | True |
| Fallback rows | 0 |
| Planner live successes | 120 |
| Reconciler live successes | 72 |
| Successful reconciliation live calls | 72 |
| Strict-live validity | True |

Claims about successful live LLM behavior below use `agent_source` equal to the requested provider; legacy OpenAI-only diagnostics are labeled separately.

## Gate Health

Untouched-holdout intervention outcome is primary. The headline estimate is dataset-macro: each eligible `benchmark_case` contributes one equally weighted dataset/task summary. Trial-weighted values and training/reference regret are secondary diagnostics. Exact family match and top-2 compatibility are also secondary diagnostics.
- Challenges: **69 / 69 actionable model-family disagreements**; abstentions: **0**.
- Holdout intervention outcomes: beneficial **9**; harmful **36**; neutral **21**.
- Training-reference diagnostics: challenge yield **12.5%**; harmful-reference rate **63.9%**; unnecessary-reference rate **23.6%**.
- Paper-facing rates: challenge **100.0%**, intervention **55.0%**, abstention/preservation **0.0%**, beneficial **13.0%**, harmful **56.5%**, neutral **30.4%**.
- Challenge recall: **100.0%**; missed rescues: **0**.
- Mean regret reduction: **35.1403**; median: **35.1406**; uncertainty interval: `{'lower': -0.10318014080246243, 'upper': 105.56908329307822, 'ci_low': -0.10318014080246243, 'ci_high': 105.56908329307822, 'support': 40, 'n_clusters': 40, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': True, 'status': 'ok'}`.
- Catastrophic regret: initial **9**, final **30**, prevented **0**, introduced **21**, net **-21**.
- Utility contribution: `{'improvement_reward': 9.0, 'worsening_penalty': -86.0, 'unnecessary_intervention_penalty': -4.25, 'catastrophic_prevention_reward': 0.0, 'catastrophic_introduction_penalty': -105.0, 'missed_rescue_penalty': 0.0, 'total_utility': -186.25, 'weights': {'improvement': 1.0, 'worsening': 2.0, 'neutral_intervention': 0.25, 'catastrophic_prevention': 3.0, 'catastrophic_introduction': 5.0, 'missed_rescue': 1.0}}`.
- Within-model-condition paper holdout delta (dataset-macro mean): **-0.0691**; median: **-0.0701**; clustered CI: `{'lower': None, 'upper': None, 'ci_low': None, 'ci_high': None, 'support': 40, 'n_clusters': 40, 'n_bootstrap': 10000, 'confidence_level': 0.95, 'uncertainty_method': 'dataset_cluster_bootstrap_percentile', 'cluster_column': 'benchmark_case', 'stable': True, 'status': 'unavailable'}`. Classification and regression magnitudes are reported separately below because their units differ.

| Metric | Dataset-macro (primary) | Trial-weighted (secondary diagnostic) |
|---|---:|---:|
| Intervention precision | 13.0% | 13.6% |
| Holdout harmful-intervention rate | 56.5% | 54.5% |
| Mean regret reduction (secondary diagnostic) | 35.1403 | 35.1403 |
| Within-model-condition paper holdout delta | -0.0691 | -0.0700 |

Paper-facing rate denominators: challenge rate = challenged actionable model-family disagreements / actionable model-family disagreements; intervention rate conditional on disagreement = actual soft final-plan changes / actionable model-family disagreements; abstention rate = actionable model-family disagreements preserved for insufficient evidence / actionable model-family disagreements; beneficial, harmful, and neutral rates = the corresponding actual interventions / actual interventions with evaluable holdout outcomes; intervention precision = beneficial / comparable interventions; harm rate = harmful / comparable interventions. Preprocessing-only disagreements are reported separately and excluded from these denominators. Zero denominators are reported as `null`/`n/a`.

## LLM Decision Stability

| Dataset | Live-provider trials | Unique initial methods | Modal method | Modal frequency | Pairwise consistency |
|---|---:|---:|---|---:|---:|
| APSFailure | 3 | 2 | boosted_tree | 66.7% | 33.3% |
| Amazon_employee_access | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Australian | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Bioresponse | 3 | 1 | regularized_linear | 100.0% | 100.0% |
| Brazilian_houses | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Click_prediction_small | 3 | 2 | boosted_tree | 66.7% | 33.3% |
| GesturePhaseSegmentationProcessed | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Internet-Advertisements | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| MIP-2016-regression | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Mercedes_Benz_Greener_Manufacturing | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| Moneyball | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| OnlineNewsPopularity | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| PhishingWebsites | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| abalone | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| adult | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| bank-marketing | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| blood-transfusion-service-center | 3 | 2 | boosted_tree | 66.7% | 33.3% |
| car | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| churn | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| colleges | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| credit-g | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| diamonds | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| dna | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| elevators | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| eucalyptus | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| house_16H | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| house_prices_nominal | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| house_sales | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| kc1 | 3 | 2 | boosted_tree | 66.7% | 33.3% |
| ozone-level-8hr | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| phoneme | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| qsar-biodeg | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| quake | 3 | 1 | tree_ensemble | 100.0% | 100.0% |
| sensory | 3 | 2 | boosted_tree | 66.7% | 33.3% |
| socmob | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| space_ga | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| steel-plates-fault | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| tecator | 3 | 1 | regularized_linear | 100.0% | 100.0% |
| wine-quality-white | 3 | 1 | boosted_tree | 100.0% | 100.0% |
| wine_quality | 3 | 1 | boosted_tree | 100.0% | 100.0% |

## Agent vs Deterministic Soft Challenge

- All operational trials: soft agreement **n/a**, soft disagreement **59.0%**.
- Model-family disagreement rate: **59.0%**; preprocessing disagreement rate: **59.2%**.
- OpenAI-only method agreement: **n/a**; preprocessing agreement: **n/a**.

| Method distribution | Initial agent | Gated final |
|---|---|---|
| All trials | {'boosted_tree': {'count': 106, 'rate': 0.8833333333333333}, 'regularized_linear': {'count': 6, 'rate': 0.05}, 'tree_ensemble': {'count': 8, 'rate': 0.06666666666666667}} | {'boosted_tree': {'count': 48, 'rate': 0.4}, 'linear': {'count': 3, 'rate': 0.025}, 'regularized_linear': {'count': 51, 'rate': 0.425}, 'tree_ensemble': {'count': 18, 'rate': 0.15}} |
| OpenAI only | {} | {} |

## Empirical Reference Comparison

- All operational trials: initial reference match **58.3%**; gated reference match **35.0%**.
- OpenAI only: initial reference match **n/a**; gated reference match **n/a**.
- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.

## Effect of the Validation Gate

- OpenAI-only gate outcomes: **0 improved**, **0 worsened**, **0 neutral**.
- OpenAI-only potentially unnecessary interventions: **0**.
- Operational outcomes: improved **9**, worsened **43**, neutral **68**.
- Training-side improved/worsened/neutral is a secondary normalized-regret diagnostic; the primary realized intervention label is defined from the untouched-holdout `paper_holdout_delta` and task-specific tolerance.

## Soft-Challenge Reconciliation Outcomes

- Total disagreements: **69**; challenges: **69**; abstentions: **0**.
- Challenge rate: **100.0%**; abstention rate: **0.0%**.
- Soft-challenge reconciliation invocation rate: **100.0%**.
- Reconciliation invocation rate: **60.0%**; success rate: **100.0%**.
- Sided with agent: **8.3%**; sided with deterministic challenger: **91.7%**.
- Proposal A selected: **40.3%**; Proposal B selected: **59.7%**; A/B selection imbalance: **19.4%**.
- Order-swap consistency: **n/a**; order-flip rate: **n/a** over **0** paired cases.
- Reconciliation modes observed: **['blinded_evidence_comparison']**.
- Soft-challenge outcomes: **9 improved**, **43 worsened**, **17 neutral**.
- Training-reference challenge outcomes: **9 improved**, **43 worsened**, **17 neutral**; holdout intervention precision: **13.0%**.
- Abstentions where agent was better: **0**; where deterministic was better: **0**.
- Mean deterministic-challenger regret advantage: **-3059619.3689**; this is `agent normalized regret - deterministic challenger normalized regret`, so it is not final gated intervention improvement; training-reference unnecessary interventions: **17** (**23.6%**).
- Catastrophic-regret rate: **7.5%**; catastrophic cases prevented by challenge: **0** (**0.0%**).
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
| APSFailure | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 100.0% | -0.0360 | 0 / 0 | 33.3% -> 0.0% |
| Amazon_employee_access | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| Australian | 3 | 3 | 0 | 0 / 0 / 3 | 0.0% | 0.0% | 0.0077 | 0 / 0 | 0.0% -> 100.0% |
| Bioresponse | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| Brazilian_houses | 3 | 3 | 0 | 3 / 0 / 0 | 100.0% | 0.0% | 1407.9539 | 0 / 0 | 0.0% -> 0.0% |
| Click_prediction_small | 3 | 1 | 0 | 0 / 1 / 2 | 0.0% | 100.0% | -0.0164 | 0 / 0 | 33.3% -> 0.0% |
| GesturePhaseSegmentationProcessed | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 100.0% | -0.2638 | 0 / 3 | 100.0% -> 0.0% |
| Internet-Advertisements | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| MIP-2016-regression | 3 | 3 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| Mercedes_Benz_Greener_Manufacturing | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 0.0% | -0.0320 | 0 / 0 | 100.0% -> 0.0% |
| Moneyball | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 100.0% | -0.0581 | 0 / 0 | 0.0% -> 0.0% |
| OnlineNewsPopularity | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 0.0% | -0.2829 | 0 / 3 | 100.0% -> 0.0% |
| PhishingWebsites | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| abalone | 3 | 3 | 0 | 0 / 0 / 3 | 100.0% | 0.0% | -0.0150 | 0 / 0 | 0.0% -> 0.0% |
| adult | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| bank-marketing | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| blood-transfusion-service-center | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 100.0% | -0.0662 | 0 / 0 | 66.7% -> 0.0% |
| car | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| churn | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| colleges | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| credit-g | 3 | 3 | 0 | 0 / 0 / 3 | 0.0% | 0.0% | -0.0049 | 0 / 0 | 0.0% -> 0.0% |
| diamonds | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 100.0% | -1.0206 | 0 / 3 | 0.0% -> 0.0% |
| dna | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 0.0% | -0.0205 | 0 / 0 | 100.0% -> 0.0% |
| elevators | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 100.0% | -0.1963 | 0 / 3 | 100.0% -> 0.0% |
| eucalyptus | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 0.0% | -0.0405 | 0 / 0 | 100.0% -> 0.0% |
| house_16H | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| house_prices_nominal | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| house_sales | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| kc1 | 3 | 3 | 0 | 0 / 1 / 2 | 0.0% | 100.0% | -0.0157 | 0 / 0 | 33.3% -> 0.0% |
| ozone-level-8hr | 3 | 3 | 0 | 3 / 0 / 0 | 0.0% | 0.0% | 0.0419 | 0 / 0 | 0.0% -> 0.0% |
| phoneme | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| qsar-biodeg | 3 | 3 | 0 | 0 / 0 / 3 | 0.0% | 100.0% | 0.0084 | 0 / 0 | 0.0% -> 100.0% |
| quake | 3 | 3 | 0 | 3 / 0 / 0 | 100.0% | 0.0% | 0.0577 | 0 / 0 | 0.0% -> 0.0% |
| sensory | 3 | 2 | 0 | 0 / 2 / 1 | 0.0% | 100.0% | -0.0134 | 0 / 0 | 66.7% -> 0.0% |
| socmob | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |
| space_ga | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 100.0% | -0.1376 | 0 / 3 | 100.0% -> 0.0% |
| steel-plates-fault | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 100.0% | -0.1084 | 0 / 3 | 100.0% -> 0.0% |
| tecator | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 100.0% -> 100.0% |
| wine-quality-white | 3 | 3 | 0 | 0 / 3 / 0 | 0.0% | 100.0% | -0.1297 | 0 / 3 | 100.0% -> 0.0% |
| wine_quality | 3 | 0 | 0 | 0 / 0 / 3 | n/a | n/a | 0.0000 | 0 / 0 | 0.0% -> 0.0% |

## Validation / Safety Interceptions

- Initial hard-invalid proposals: **0**; hard interceptions: **0**; hard repairs: **0**; soft interventions: **66**.
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
