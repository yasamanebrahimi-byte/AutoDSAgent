# AutoDSAgent Validation Architecture Evaluation

Offline/mock results are not evidence of live LLM performance.

## Evaluation setup

- Trials: **4** (4 clean, 0 perturbation).
- Repetitions per case/scenario: **1**.
- Agent mode: **offline**; requested model: `gpt-4.1-mini`.
- The final holdout was frozen before model-family decisions and was reserved for final evaluation.

## Benchmark composition

| Case | Task | Rows | Source |
|---|---|---:|---|
| breast_cancer | classification | 569 | sklearn.datasets.load_breast_cancer(as_frame=True) |
| wine | classification | 178 | sklearn.datasets.load_wine(as_frame=True) |
| diabetes | regression | 442 | sklearn.datasets.load_diabetes(as_frame=True, scaled=False) |
| synthetic_regression | regression | 240 | sklearn.datasets.make_regression(random_state=123) |

## Agent vs deterministic agreement

- Agreement rate: **50.0%**; disagreement rate: **50.0%**.

| Dimension | Disagreements |
|---|---:|
| target | 0 |
| task | 0 |
| method | 2 |
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
| deterministic | 2 |
| other | 0 |

## Empirical model-family comparison

- Empirical-reference match rate: agent initial **50.0%**; gated final **50.0%**.
- The empirical reference is a post-hoc training-only benchmark over the supported candidate set, not a universal optimum.

| Case | Trial | Best family | Candidate ranking |
|---|---:|---|---|
| breast_cancer / clean | 0 | linear | linear, regularized_linear, tree_ensemble, boosted_tree |
| wine / clean | 0 | regularized_linear | regularized_linear, linear, tree_ensemble, boosted_tree |
| diabetes / clean | 0 | regularized_linear | regularized_linear, linear, tree_ensemble, boosted_tree |
| synthetic_regression / clean | 0 | linear | linear, regularized_linear, boosted_tree, tree_ensemble |

## Regret analysis

- Mean normalized regret: agent **0.0004**, gated **0.0018**.
- Median normalized regret: agent **0.0000**, gated **0.0000**.
- Paired comparison: gated better **0**, agent better **0**, tie **4** (eligible **4**).
- Classification regret is `best_macro_f1 - selected_macro_f1`; regression regret is `selected_rmse - best_rmse`. Regression aggregate regret is normalized by the best RMSE for that trial.

| Task | Trials | Agent mean normalized regret | Gated mean normalized regret | Improved / worse / unchanged |
|---|---:|---:|---:|---|
| classification | 2 | 0.0000 | 0.0037 | 0 / 0 / 2 |
| regression | 2 | 0.0007 | 0.0000 | 0 / 0 / 2 |

## Perturbation results

| Scenario | Kind | Trials | Expected failed checks observed |
|---|---|---:|---:|

## Potentially unnecessary intervention

- Count: **2**; denominator is valid initial plans that materially disagreed on model family (**100.0%**).
- This is an exploratory heuristic, not a correctness theorem: the configured approximation thresholds are shown in `config.json`.

## Limitations

- The benchmark suite is small and may not represent real project domains.
- Only the currently supported tabular model families and one CV procedure are compared.
- The empirical reference is not ground truth or a universal optimum.
- LLM behavior is stochastic; meaningful live conclusions require repeated live-agent trials.
- Model-family choice is only one component of data-science quality.
- Semantic/domain leakage and feature availability cannot be fully validated automatically.
- Offline fallback and mock rows must be filtered out before making claims about actual LLM behavior.
