# AutoDSAgent Validation Architecture Evaluation

Offline/mock results are not evidence of live LLM performance.

## Evaluation setup

- Trials: **1** (1 clean, 0 perturbation).
- Repetitions per case/scenario: **1**.
- Agent mode: **offline**; requested model: `gpt-4.1-mini`.
- The final holdout was frozen before model-family decisions and was reserved for final evaluation.

## Benchmark composition

| Case | Task | Rows | Source |
|---|---|---:|---|
| wine | classification | 178 | sklearn.datasets.load_wine(as_frame=True) |

## Agent vs deterministic agreement

- Agreement rate: **0.0%**; disagreement rate: **100.0%**.

| Dimension | Disagreements |
|---|---:|
| target | 0 |
| task | 0 |
| method | 1 |
| preprocessing | 0 |

## Safety / validity results

- Initial agent validity rate: **100.0%**.
- Initial invalid proposals: **0**.
- Unsafe proposal interception: **0** / invalid initial proposals (**n/a**).
- Final invalid rate: **0.0%**.

## Reconciliation results

- Reconciliation success rate: **100.0%**.

| Selection source | Successful reconciliations |
|---|---:|
| agent | 0 |
| deterministic | 1 |
| other | 0 |

## Empirical model-family comparison

- Empirical-reference match rate: agent initial **100.0%**; gated final **0.0%**.
- The empirical reference is a post-hoc training-only benchmark over the supported candidate set, not a universal optimum.

| Case | Trial | Best family | Candidate ranking |
|---|---:|---|---|
| wine / clean | 0 | regularized_linear | regularized_linear, linear, tree_ensemble, boosted_tree |

## Regret analysis

- Mean normalized regret: agent **0.0000**, gated **0.0073**.
- Median normalized regret: agent **0.0000**, gated **0.0073**.
- Paired comparison: gated better **0**, agent better **0**, tie **1** (eligible **1**).
- Classification regret is `best_macro_f1 - selected_macro_f1`; regression regret is `selected_rmse - best_rmse`. Regression aggregate regret is normalized by the best RMSE for that trial.

| Task | Trials | Agent mean normalized regret | Gated mean normalized regret | Improved / worse / unchanged |
|---|---:|---:|---:|---|
| classification | 1 | 0.0000 | 0.0073 | 0 / 0 / 1 |
| regression | 0 | n/a | n/a | 0 / 0 / 0 |

## Perturbation results

| Scenario | Kind | Trials | Expected failed checks observed |
|---|---|---:|---:|

## Potentially unnecessary intervention

- Count: **1**; denominator is valid initial plans that materially disagreed on model family (**100.0%**).
- This is an exploratory heuristic, not a correctness theorem: the configured approximation thresholds are shown in `config.json`.

## Limitations

- The benchmark suite is small and may not represent real project domains.
- Only the currently supported tabular model families and one CV procedure are compared.
- The empirical reference is not ground truth or a universal optimum.
- LLM behavior is stochastic; meaningful live conclusions require repeated live-agent trials.
- Model-family choice is only one component of data-science quality.
- Semantic/domain leakage and feature availability cannot be fully validated automatically.
- Offline fallback and mock rows must be filtered out before making claims about actual LLM behavior.
