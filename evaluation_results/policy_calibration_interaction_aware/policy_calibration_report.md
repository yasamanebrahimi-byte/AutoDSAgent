# Deterministic Policy Calibration Report

- Policy version under test: `4`
- Benchmark suite: `2`
- Role: `policy_development`
- Unique datasets: **12**; repeated seeds are not treated as independent datasets
- Split seeds: `[42, 123, 2027]`
- Git commit: `90a09f252022755a592157f2b37ef812703b57d5`

## Development benchmark composition

The registry assigns these cases permanently to policy development. Final-evaluation cases are rejected by the calibration runner.

- `breast_cancer`
- `diabetes`
- `synthetic_regression`
- `synthetic_linear_regression`
- `synthetic_nonlinear_regression`
- `synthetic_high_dim_regression`
- `synthetic_binary_linear`
- `synthetic_binary_nonlinear`
- `synthetic_imbalanced_classification`
- `synthetic_multiclass`
- `synthetic_missingness`
- `synthetic_outlier_regression`

## Candidate configurations and selection criterion

rank by lowest dataset-level mean normalized regret, then lowest dataset-level catastrophic-regret rate, then highest dataset-level top-2 reference inclusion, then lowest policy complexity; retain the current policy unless the selected candidate clears the predefined promotion margin.

| Candidate | Mean regret | Catastrophic rate | Exact match | Top-2 rate | Collapse warning |
|---|---:|---:|---:|---:|---|
| `current` | 0.14936245125377104 | 0.2222222222222222 | 0.5555555555555556 | 0.6666666666666666 | no |
| `nonlinear_sensitive` | 0.39423767503511753 | 0.27777777777777773 | 0.5277777777777778 | 0.6666666666666666 | no |
| `high_dimensional_sensitive` | 0.14936245125377104 | 0.2222222222222222 | 0.5555555555555556 | 0.6666666666666666 | no |
| `missingness_sensitive` | 0.14936245125377104 | 0.2222222222222222 | 0.5555555555555556 | 0.6666666666666666 | no |

## Sensitivity analysis

The four candidates are a deliberately small neighborhood around the current interpretable thresholds; no continuous optimizer or LLM is used.

- `current`: nonlinear thresholds 0.15/0.35; interaction thresholds 0.18/0.38; interaction limits 12 features/48 pairs; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.14936245125377104`.
- `nonlinear_sensitive`: nonlinear thresholds 0.12/0.3; interaction thresholds 0.18/0.38; interaction limits 12 features/48 pairs; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.39423767503511753`.
- `high_dimensional_sensitive`: nonlinear thresholds 0.15/0.35; interaction thresholds 0.18/0.38; interaction limits 12 features/48 pairs; sample-feature bands `[5.0, 10.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.14936245125377104`.
- `missingness_sensitive`: nonlinear thresholds 0.15/0.35; interaction thresholds 0.18/0.38; interaction limits 12 features/48 pairs; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.08, 0.2]`; mean regret `0.14936245125377104`.

## Per-dataset results and failure cases

Per-dataset means below preserve dataset identity before the across-dataset summary.

| Dataset | Seeds | Interaction | Score | Mean regret | Catastrophic rate | Top-2 rate | Selected families |
|---|---:|---|---:|---:|---:|---:|---|
| `breast_cancer` | 3 | low | 0.0 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `diabetes` | 3 | low | 0.045945672510372355 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `synthetic_binary_linear` | 3 | low | 0.0 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `synthetic_binary_nonlinear` | 3 | low | 0.0 | 0.12111926542292761 | 1.0 | 0.0 | linear |
| `synthetic_high_dim_regression` | 3 | low | 0.12913167114252627 | 1.301876192234007 | 0.3333333333333333 | 0.3333333333333333 | regularized_linear, tree_ensemble |
| `synthetic_imbalanced_classification` | 3 | low | 0.0 | 0.1079835566220844 | 0.6666666666666666 | 1.0 | regularized_linear |
| `synthetic_linear_regression` | 3 | low | 0.08181988401692038 | 0.0 | 0.0 | 1.0 | linear |
| `synthetic_missingness` | 3 | low | 0.0 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `synthetic_multiclass` | 3 | low | 0.0 | 0.03629675488267451 | 0.0 | 0.0 | linear |
| `synthetic_nonlinear_regression` | 3 | low | 0.06376901688497728 | 0.20954077593692558 | 0.6666666666666666 | 0.3333333333333333 | boosted_tree, linear |
| `synthetic_outlier_regression` | 3 | low | 0.05014359004841903 | 0.015532869946633492 | 0.0 | 0.3333333333333333 | linear |
| `synthetic_regression` | 3 | low | 0.07502004328085758 | 0.0 | 0.0 | 1.0 | linear |

## Interaction diagnostics

- Mean interaction score: `0.03715248982367274`; median: `0.021800015744920662`
- Strength distribution: `{"low": 36}`
- Mean evaluated pairs: `16.0`; mean strong-pair fraction: `0.0`
- Regime metrics: `{"low": {"catastrophic_regret_rate": 0.2222222222222222, "dataset_count": 12, "exact_reference_match_rate": 0.5555555555555556, "mean_normalized_regret": 0.14936245125377104, "median_normalized_regret": 0.007766434973316746, "top2_compatibility_rate": 0.6666666666666666}}`
- Top interaction evidence: `[{"features": ["feature_10", "feature_20"], "incremental_strength": 0.22102339124672443, "interaction_strength": 0.17169800815252229, "joint_strength": 0.7768318420237199, "marginal_strength": 0.5558084507769955, "sample_count": 192, "transform": "sum"}, {"features": ["feature_10", "feature_26"], "incremental_strength": 0.16472512095263947, "interaction_strength": 0.11868997975360145, "joint_strength": 0.720533571729635, "marginal_strength": 0.5558084507769955, "sample_count": 192, "transform": "sum"}, {"features": ["feature_10", "feature_9"], "incremental_strength": 0.15021661062697378, "interaction_strength": 0.1060566917418053, "joint_strength": 0.7060250614039693, "marginal_strength": 0.5558084507769955, "sample_count": 192, "transform": "sum"}]`

Largest current-policy regret cases:

- `synthetic_high_dim_regression` seed `2027`: deterministic `tree_ensemble`, empirical best `linear`, regret `3.9015933567290895`.
- `synthetic_nonlinear_regression` seed `2027`: deterministic `linear`, empirical best `boosted_tree`, regret `0.32857791766543515`.
- `synthetic_nonlinear_regression` seed `42`: deterministic `linear`, empirical best `boosted_tree`, regret `0.30004441014534156`.
- `synthetic_imbalanced_classification` seed `123`: deterministic `regularized_linear`, empirical best `boosted_tree`, regret `0.14078094109833128`.
- `synthetic_binary_nonlinear` seed `42`: deterministic `linear`, empirical best `tree_ensemble`, regret `0.13342028985507248`.

## Family-selection distribution

{"boosted_tree": {"count": 1, "rate": 0.027777777777777776}, "linear": {"count": 17, "rate": 0.4722222222222222}, "regularized_linear": {"count": 17, "rate": 0.4722222222222222}, "tree_ensemble": {"count": 1, "rate": 0.027777777777777776}}

## Recommendation

- Objective-selected candidate: `current`
- Recommendation: **retain_current**
- Final benchmark results are not used to modify the policy version under test.
- Compatibility scores remain interpretable compatibility points, not probabilities of empirical optimality.

## Selective soft-challenge calibration

- Calibration artifact: `soft-challenge-calibration-v1-v1`
- Development records with a model-family disagreement: `19`
- Reliability is the deterministic challenger win rate among non-tied development disagreements; ties remain in support but are excluded from that denominator.

| Regime | Support | Challenger wins | Agent wins | Ties | Win rate | Mean regret delta | Catastrophic prevention |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all/all/all/all` | 19 | 1 | 4 | 14 | 0.2 | -0.1927978155841069 | 0.2 |
| `classification/all/all/all` | 6 | 0 | 3 | 3 | 0.0 | -0.013034229106088924 | 0.0 |
| `classification/all/all/low` | 6 | 0 | 3 | 3 | 0.0 | -0.013034229106088924 | 0.0 |
| `classification/low/all/low` | 6 | 0 | 3 | 3 | 0.0 | -0.013034229106088924 | 0.0 |
| `classification/low/low/low` | 6 | 0 | 3 | 3 | 0.0 | -0.013034229106088924 | 0.0 |
| `regression/all/all/all` | 13 | 1 | 1 | 11 | 0.5 | -0.2757656247278075 | 0.3333333333333333 |
| `regression/all/all/low` | 12 | 1 | 1 | 10 | 0.5 | -0.29939004819375065 | 0.3333333333333333 |
| `regression/all/all/medium` | 1 | 0 | 0 | 1 | None | 0.00772745686351031 | None |
| `regression/low/all/low` | 11 | 1 | 0 | 10 | 1.0 | 0.027955018835499893 | 0.3333333333333333 |
| `regression/low/all/medium` | 1 | 0 | 0 | 1 | None | 0.00772745686351031 | None |
| `regression/low/high/low` | 1 | 1 | 0 | 0 | 1.0 | 0.297601320830194 | 1.0 |
| `regression/low/low/medium` | 1 | 0 | 0 | 1 | None | 0.00772745686351031 | None |
| `regression/low/moderate/low` | 10 | 0 | 0 | 10 | None | 0.0009903886360304788 | 0.0 |
| `regression/medium/all/low` | 1 | 0 | 1 | 0 | 0.0 | -3.900185785515507 | None |
| `regression/medium/high/low` | 1 | 0 | 1 | 0 | 0.0 | -3.900185785515507 | None |
