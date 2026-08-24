# Frozen Deterministic Policy Final Evaluation

- Evaluation role: `final_evaluation`
- Frozen policy version: `4`
- Benchmark suite: `2`
- Unique datasets: **6**
- Split seeds: `[42, 123, 2027]`
- Git commit: `d63d43961c5ba0a00957cd9fb2f0ed573312fa39`

## Frozen evaluation protocol

Only final-evaluation cases are accepted. Diagnostics and empirical-reference CV use each case's training partition; the holdout is scored after the policy decision and cannot modify policy parameters.

## Policy quality metrics

- Mean dataset-level normalized regret: `0.023063345304331836`
- Median dataset-level normalized regret: `0.010507245492460515`
- Empirical-reference match rate: `0.38888888888888884`
- Catastrophic-regret rate: `0.1111111111111111`
- Top-2 compatibility success: `0.5555555555555555`
- Family-selection distribution: `{"boosted_tree": {"count": 6, "rate": 0.3333333333333333}, "linear": {"count": 2, "rate": 0.1111111111111111}, "regularized_linear": {"count": 10, "rate": 0.5555555555555556}}`

## Per-dataset final results

| Dataset | Seeds | Interaction | Score | Mean regret | Catastrophic rate | Top-2 rate | Selected families |
|---|---:|---|---:|---:|---:|---:|---|
| `digits_subset` | 3 | low | 0.0 | 0.007249427480332156 | 0.0 | 0.3333333333333333 | regularized_linear |
| `final_interaction_regression` | 3 | high | 0.3751544565529337 | 0.0 | 0.0 | 1.0 | boosted_tree |
| `final_low_n_high_p_classification` | 3 | low | 0.0 | 0.013765063504588873 | 0.0 | 1.0 | regularized_linear |
| `final_mixed_type_classification` | 3 | low | 0.0 | 0.02421411636454181 | 0.0 | 0.3333333333333333 | boosted_tree, regularized_linear |
| `final_shifted_nonlinear_regression` | 3 | low | 0.06773744671840111 | 0.09074322259684559 | 0.6666666666666666 | 0.3333333333333333 | boosted_tree, linear |
| `wine` | 3 | low | 0.0 | 0.002408241879682582 | 0.0 | 0.3333333333333333 | regularized_linear |

## Interaction diagnostics

- Mean interaction score: `0.07381531721188914`; median: `0.0`
- Strength distribution: `{"high": 2, "low": 15, "moderate": 1}`
- Mean evaluated pairs: `10.0`; mean strong-pair fraction: `0.018518518518518517`
- Regime metrics: `{"high": {"catastrophic_regret_rate": 0.0, "dataset_count": 1, "exact_reference_match_rate": 1.0, "mean_normalized_regret": 0.0, "median_normalized_regret": 0.0, "top2_compatibility_rate": 1.0}, "low": {"catastrophic_regret_rate": 0.13333333333333333, "dataset_count": 5, "exact_reference_match_rate": 0.26666666666666666, "mean_normalized_regret": 0.027676014365198204, "median_normalized_regret": 0.013765063504588873, "top2_compatibility_rate": 0.4666666666666666}}`
- Top interaction evidence: `[{"features": ["feature_2", "feature_3"], "incremental_strength": 0.6564555942591729, "interaction_strength": 0.5146424198634678, "joint_strength": 0.7839714130919321, "marginal_strength": 0.12751581883275923, "sample_count": 240, "transform": "product"}, {"features": ["feature_0", "feature_1"], "incremental_strength": 0.5498652215951517, "interaction_strength": 0.33660116149674496, "joint_strength": 0.6121521206965399, "marginal_strength": 0.062286899101388155, "sample_count": 240, "transform": "product"}]`

## Classification boundary diagnostics

- Mean boundary complexity score: `0.0`; median: `0.0`
- Mean linear probe score: `0.8886400202941358`; mean normalized linear separability: `0.8099148030228687`
- Mean local class consistency: `0.8298359195955776`; mean nonlinear advantage: `0.0`
- Category distribution: `{"low": 12}`
- Confidence distribution: `{"high": 11, "medium": 1}`
- Regime metrics: `{"low": {"catastrophic_regret_rate": 0.1111111111111111, "dataset_count": 6, "exact_reference_match_rate": 0.38888888888888884, "mean_normalized_regret": 0.023063345304331836, "median_normalized_regret": 0.010507245492460515, "top2_compatibility_rate": 0.5555555555555555}}`

## Final holdout metrics

- `digits_subset`: `{'accuracy': 0.9629629629629629, 'balanced_accuracy': 0.9629629629629631, 'macro_f1': 0.962667153603686, 'weighted_f1': 0.9626671536036863}`
- `final_interaction_regression`: `{'mae': 0.909090495701432, 'r2': 0.6205376399713599, 'rmse': 1.2074044087266513}`
- `final_low_n_high_p_classification`: `{'accuracy': 0.7962962962962963, 'balanced_accuracy': 0.7962962962962963, 'macro_f1': 0.7950882966362842, 'weighted_f1': 0.7950882966362842}`
- `final_mixed_type_classification`: `{'accuracy': 0.9944444444444445, 'balanced_accuracy': 0.9952380952380953, 'macro_f1': 0.9943165672065928, 'weighted_f1': 0.994458653026428}`
- `final_shifted_nonlinear_regression`: `{'mae': 3.0200048274161726, 'r2': 0.4356639657399734, 'rmse': 3.9288944654281948}`
- `wine`: `{'accuracy': 0.9814814814814815, 'balanced_accuracy': 0.980952380952381, 'macro_f1': 0.9809143975306508, 'weighted_f1': 0.9814464499548858}`

## Largest policy failure cases

- `final_shifted_nonlinear_regression` seed `2027`: deterministic `linear`, empirical best `boosted_tree`, regret `0.14448187625874198`.
- `final_shifted_nonlinear_regression` seed `42`: deterministic `linear`, empirical best `boosted_tree`, regret `0.12774779153179477`.
- `final_low_n_high_p_classification` seed `42`: deterministic `regularized_linear`, empirical best `tree_ensemble`, regret `0.04129519051376662`.
- `final_mixed_type_classification` seed `2027`: deterministic `boosted_tree`, empirical best `linear`, regret `0.02980098938876441`.
- `final_mixed_type_classification` seed `123`: deterministic `regularized_linear`, empirical best `linear`, regret `0.02556819398924659`.

Final benchmark results are descriptive evidence for the frozen policy. They are not used to tune or rewrite the policy version evaluated here.
