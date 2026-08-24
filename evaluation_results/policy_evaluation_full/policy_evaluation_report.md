# Frozen Deterministic Policy Final Evaluation

- Evaluation role: `final_evaluation`
- Frozen policy version: `4`
- Benchmark suite: `2`
- Unique datasets: **6**
- Split seeds: `[42, 123, 2027]`
- Git commit: `c9efbb7a378850564ae1238267fe5ca2c65c9244`

## Frozen evaluation protocol

Only final-evaluation cases are accepted. Diagnostics and empirical-reference CV use each case's training partition; the holdout is scored after the policy decision and cannot modify policy parameters.

## Policy quality metrics

- Mean dataset-level normalized regret: `0.11422111989711344`
- Median dataset-level normalized regret: `0.018989589934565343`
- Empirical-reference match rate: `0.2222222222222222`
- Catastrophic-regret rate: `0.27777777777777773`
- Top-2 compatibility success: `0.38888888888888884`
- Family-selection distribution: `{"boosted_tree": {"count": 3, "rate": 0.16666666666666666}, "linear": {"count": 5, "rate": 0.2777777777777778}, "regularized_linear": {"count": 10, "rate": 0.5555555555555556}}`

## Per-dataset final results

| Dataset | Seeds | Mean regret | Catastrophic rate | Top-2 rate | Selected families |
|---|---:|---:|---:|---:|---|
| `digits_subset` | 3 | 0.007249427480332156 | 0.0 | 0.3333333333333333 | regularized_linear |
| `final_interaction_regression` | 3 | 0.5469466475566896 | 1.0 | 0.0 | linear |
| `final_low_n_high_p_classification` | 3 | 0.013765063504588873 | 0.0 | 1.0 | regularized_linear |
| `final_mixed_type_classification` | 3 | 0.02421411636454181 | 0.0 | 0.3333333333333333 | boosted_tree, regularized_linear |
| `final_shifted_nonlinear_regression` | 3 | 0.09074322259684559 | 0.6666666666666666 | 0.3333333333333333 | boosted_tree, linear |
| `wine` | 3 | 0.002408241879682582 | 0.0 | 0.3333333333333333 | regularized_linear |

## Final holdout metrics

- `digits_subset`: `{'accuracy': 0.9629629629629629, 'balanced_accuracy': 0.9629629629629631, 'macro_f1': 0.962667153603686, 'weighted_f1': 0.9626671536036863}`
- `final_interaction_regression`: `{'mae': 1.482624283412485, 'r2': 0.08347246380458025, 'rmse': 1.9007769217076416}`
- `final_low_n_high_p_classification`: `{'accuracy': 0.7962962962962963, 'balanced_accuracy': 0.7962962962962963, 'macro_f1': 0.7950882966362842, 'weighted_f1': 0.7950882966362842}`
- `final_mixed_type_classification`: `{'accuracy': 0.9944444444444445, 'balanced_accuracy': 0.9952380952380953, 'macro_f1': 0.9943165672065928, 'weighted_f1': 0.994458653026428}`
- `final_shifted_nonlinear_regression`: `{'mae': 3.0200048274161726, 'r2': 0.4356639657399734, 'rmse': 3.9288944654281948}`
- `wine`: `{'accuracy': 0.9814814814814815, 'balanced_accuracy': 0.980952380952381, 'macro_f1': 0.9809143975306508, 'weighted_f1': 0.9814464499548858}`

## Largest policy failure cases

- `final_interaction_regression` seed `123`: deterministic `linear`, empirical best `boosted_tree`, regret `0.6476254692686403`.
- `final_interaction_regression` seed `42`: deterministic `linear`, empirical best `boosted_tree`, regret `0.5146366003831132`.
- `final_interaction_regression` seed `2027`: deterministic `linear`, empirical best `boosted_tree`, regret `0.47857787301831556`.
- `final_shifted_nonlinear_regression` seed `2027`: deterministic `linear`, empirical best `boosted_tree`, regret `0.14448187625874198`.
- `final_shifted_nonlinear_regression` seed `42`: deterministic `linear`, empirical best `boosted_tree`, regret `0.12774779153179477`.

Final benchmark results are descriptive evidence for the frozen policy. They are not used to tune or rewrite the policy version evaluated here.
