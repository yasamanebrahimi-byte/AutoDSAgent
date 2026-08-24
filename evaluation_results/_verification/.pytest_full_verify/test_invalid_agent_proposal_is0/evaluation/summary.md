# AutoDSAgent Validation Architecture Evaluation

Offline/mock results are not evidence of live LLM performance.

## Evaluation setup

- Trials: **1** (1 clean, 0 perturbation).
- Repetitions per case/scenario: **1**.
- Agent mode: **live_or_fallback**; requested model: `gpt-4.1-mini`.
- The final holdout was frozen before model-family decisions and was reserved for final evaluation.

## Benchmark composition

| Case | Task | Rows | Source |
|---|---|---:|---|
| invalid_agent | classification | 60 | in-memory test fixture |

## Agent vs deterministic agreement

- Agreement rate: **0.0%**; disagreement rate: **100.0%**.

| Dimension | Disagreements |
|---|---:|
| target | 0 |
| task | 0 |
| method | 0 |
| preprocessing | 1 |

## Safety / validity results

- Final valid/invalid trials: **1 / 0**.
- Initial agent validity rate: **0.0%**.
- Initial invalid proposals: **1**.
- Unsafe proposal interception: **1** / invalid initial proposals (**100.0%**).
- Intentionally unsafe perturbation interception: **0** (**n/a**).
- Final invalid rate: **0.0%**.
- Validation failures by code: `{'numeric_missing_values_are_handled': 1}`.

## Reconciliation results

- Reconciliation invocation rate: **100.0%**.
- Reconciliation success rate: **100.0%**.

| Selection source | Successful reconciliations |
|---|---:|
| agent | 1 |
| deterministic | 0 |
| other | 0 |

## Empirical model-family comparison

- Empirical-reference match rate: agent initial **0.0%**; gated final **0.0%**.
- The empirical reference is a post-hoc training-only benchmark over the supported candidate set, not a universal optimum.

| Case | Trial | Best family | Candidate ranking |
|---|---:|---|---|
| invalid_agent / clean | 0 | tree_ensemble | tree_ensemble, linear, regularized_linear, boosted_tree |

## Regret analysis

- Mean normalized regret: agent **n/a**, gated **0.0580**.
- Median normalized regret: agent **n/a**, gated **0.0580**.
- Paired comparison: gated better **0**, agent better **0**, tie **0** (eligible **0**).
- Classification regret is `best_macro_f1 - selected_macro_f1`; regression regret is `selected_rmse - best_rmse`. Regression aggregate regret is normalized by the best RMSE for that trial.

| Task | Trials | Agent mean normalized regret | Gated mean normalized regret | Improved / worse / unchanged |
|---|---:|---:|---:|---|
| classification | 1 | n/a | 0.0580 | 0 / 0 / 0 |
| regression | 0 | n/a | n/a | 0 / 0 / 0 |

## Perturbation results

| Scenario | Kind | Trials | Expected failed checks observed |
|---|---|---:|---:|

## Potentially unnecessary intervention

- Count: **0**; denominator is valid initial plans that materially disagreed on model family (**n/a**).
- This is an exploratory heuristic, not a correctness theorem: the configured approximation thresholds are shown in `config.json`.

## Limitations

- The benchmark suite is small and may not represent real project domains.
- Only the currently supported tabular model families and one CV procedure are compared.
- The empirical reference is not ground truth or a universal optimum.
- LLM behavior is stochastic; meaningful live conclusions require repeated live-agent trials.
- Model-family choice is only one component of data-science quality.
- Semantic/domain leakage and feature availability cannot be fully validated automatically.
- Offline fallback and mock rows must be filtered out before making claims about actual LLM behavior.
