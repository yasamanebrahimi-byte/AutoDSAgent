# AutoDSAgent Validation Architecture Evaluation

Offline/mock results are not evidence of live LLM performance.

## Evaluation setup

- Trials: **6** (1 clean, 5 perturbation).
- Repetitions per case/scenario: **1**.
- Agent mode: **offline**; requested model: `gpt-4.1-mini`.
- The final holdout was frozen before model-family decisions and was reserved for final evaluation.

## Benchmark composition

| Case | Task | Rows | Source |
|---|---|---:|---|
| unsafe_perturbation | classification | 48 | in-memory test fixture |

## Agent vs deterministic agreement

- Agreement rate: **0.0%**; disagreement rate: **100.0%**.

| Dimension | Disagreements |
|---|---:|
| target | 0 |
| task | 0 |
| method | 6 |
| preprocessing | 1 |

## Safety / validity results

- Final valid/invalid trials: **4 / 2**.
- Initial agent validity rate: **66.7%**.
- Initial invalid proposals: **2**.
- Unsafe proposal interception: **2** / invalid initial proposals (**100.0%**).
- Intentionally unsafe perturbation interception: **2** (**100.0%**).
- Final invalid rate: **33.3%**.
- Validation failures by code: `{'classification_split_and_stratification_feasible': 1, 'classification_training_supports_cross_validation': 1, 'no_direct_target_copy_features': 1}`.

## Reconciliation results

- Reconciliation invocation rate: **100.0%**.
- Reconciliation success rate: **66.7%**.

| Selection source | Successful reconciliations |
|---|---:|
| agent | 0 |
| deterministic | 4 |
| other | 0 |

## Empirical model-family comparison

- Empirical-reference match rate: agent initial **0.0%**; gated final **75.0%**.
- The empirical reference is a post-hoc training-only benchmark over the supported candidate set, not a universal optimum.

| Case | Trial | Best family | Candidate ranking |
|---|---:|---|---|
| unsafe_perturbation / clean | 0 | linear | linear, regularized_linear, tree_ensemble, boosted_tree |
| unsafe_perturbation / missing_values | 0 | linear | linear, regularized_linear, tree_ensemble, boosted_tree |
| unsafe_perturbation / infinity_values | 0 | tree_ensemble | tree_ensemble, linear, regularized_linear, boosted_tree |
| unsafe_perturbation / identifier_column | 0 | linear | linear, regularized_linear, tree_ensemble, boosted_tree |
| unsafe_perturbation / target_copy_leakage | 0 | None | none |
| unsafe_perturbation / classification_feasibility | 0 | None | none |

## Regret analysis

- Mean normalized regret: agent **0.0220**, gated **0.0146**.
- Median normalized regret: agent **0.0149**, gated **0.0000**.
- Paired comparison: gated better **1**, agent better **0**, tie **3** (eligible **4**).
- Classification regret is `best_macro_f1 - selected_macro_f1`; regression regret is `selected_rmse - best_rmse`. Regression aggregate regret is normalized by the best RMSE for that trial.

| Task | Trials | Agent mean normalized regret | Gated mean normalized regret | Improved / worse / unchanged |
|---|---:|---:|---:|---|
| classification | 6 | 0.0220 | 0.0146 | 1 / 0 / 3 |
| regression | 0 | n/a | n/a | 0 / 0 / 0 |

## Perturbation results

| Scenario | Kind | Trials | Expected failed checks observed |
|---|---|---:|---:|
| classification_feasibility | deterministic_invariant_violation | 1 | 1 |
| identifier_column | deterministic_safe_exclusion | 1 | 0 |
| infinity_values | agent_preprocessing_challenge | 1 | 0 |
| missing_values | agent_preprocessing_challenge | 1 | 0 |
| target_copy_leakage | deterministic_invariant_violation | 1 | 1 |

## Potentially unnecessary intervention

- Count: **2**; denominator is valid initial plans that materially disagreed on model family (**50.0%**).
- This is an exploratory heuristic, not a correctness theorem: the configured approximation thresholds are shown in `config.json`.

## Limitations

- The benchmark suite is small and may not represent real project domains.
- Only the currently supported tabular model families and one CV procedure are compared.
- The empirical reference is not ground truth or a universal optimum.
- LLM behavior is stochastic; meaningful live conclusions require repeated live-agent trials.
- Model-family choice is only one component of data-science quality.
- Semantic/domain leakage and feature availability cannot be fully validated automatically.
- Offline fallback and mock rows must be filtered out before making claims about actual LLM behavior.
