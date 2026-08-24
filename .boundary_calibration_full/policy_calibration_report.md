# Deterministic Policy Calibration Report

- Policy version under test: `4`
- Benchmark suite: `2`
- Role: `policy_development`
- Unique datasets: **12**; repeated seeds are not treated as independent datasets
- Split seeds: `[42, 123, 2027]`
- Git commit: `d63d43961c5ba0a00957cd9fb2f0ed573312fa39`

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
| `current` | 0.2924657311766499 | 0.2222222222222222 | 0.5555555555555556 | 0.6666666666666666 | no |
| `nonlinear_sensitive` | 0.3843692662144345 | 0.19444444444444442 | 0.5833333333333334 | 0.6944444444444445 | no |
| `high_dimensional_sensitive` | 0.2924657311766499 | 0.2222222222222222 | 0.5555555555555556 | 0.6666666666666666 | no |
| `missingness_sensitive` | 0.2924657311766499 | 0.2222222222222222 | 0.5555555555555556 | 0.6666666666666666 | no |

## Sensitivity analysis

The four candidates are a deliberately small neighborhood around the current interpretable thresholds; no continuous optimizer or LLM is used.

- `current`: nonlinear thresholds 0.15/0.35; classification-boundary thresholds 0.1/0.2; interaction thresholds 0.18/0.38; interaction limits 12 features/48 pairs; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.2924657311766499`.
- `nonlinear_sensitive`: nonlinear thresholds 0.12/0.3; classification-boundary thresholds 0.1/0.2; interaction thresholds 0.18/0.38; interaction limits 12 features/48 pairs; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.3843692662144345`.
- `high_dimensional_sensitive`: nonlinear thresholds 0.15/0.35; classification-boundary thresholds 0.1/0.2; interaction thresholds 0.18/0.38; interaction limits 12 features/48 pairs; sample-feature bands `[5.0, 10.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.2924657311766499`.
- `missingness_sensitive`: nonlinear thresholds 0.15/0.35; classification-boundary thresholds 0.1/0.2; interaction thresholds 0.18/0.38; interaction limits 12 features/48 pairs; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.08, 0.2]`; mean regret `0.2924657311766499`.

## Per-dataset results and failure cases

Per-dataset means below preserve dataset identity before the across-dataset summary.

| Dataset | Seeds | Interaction | Score | Mean regret | Catastrophic rate | Top-2 rate | Selected families |
|---|---:|---|---:|---:|---:|---:|---|
| `breast_cancer` | 3 | low | 0.0 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `diabetes` | 3 | low | 0.045945672510372355 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `synthetic_binary_linear` | 3 | low | 0.0 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `synthetic_binary_nonlinear` | 3 | low | 0.0 | 0.03911257790878123 | 0.3333333333333333 | 0.3333333333333333 | boosted_tree, linear |
| `synthetic_high_dim_regression` | 3 | low | 0.12913167114252627 | 2.6257960658653974 | 0.6666666666666666 | 0.3333333333333333 | regularized_linear, tree_ensemble |
| `synthetic_imbalanced_classification` | 3 | low | 0.0 | 0.1079835566220844 | 0.6666666666666666 | 1.0 | regularized_linear |
| `synthetic_linear_regression` | 3 | low | 0.08181988401692038 | 0.0 | 0.0 | 1.0 | linear |
| `synthetic_missingness` | 3 | low | 0.0 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `synthetic_multiclass` | 3 | low | 0.0 | 0.03629675488267451 | 0.0 | 0.0 | linear |
| `synthetic_nonlinear_regression` | 3 | low | 0.06376901688497728 | 0.20954077593692558 | 0.6666666666666666 | 0.3333333333333333 | boosted_tree, linear |
| `synthetic_outlier_regression` | 3 | low | 0.05014359004841903 | 0.015532869946633492 | 0.0 | 0.3333333333333333 | linear |
| `synthetic_regression` | 3 | low | 0.07502004328085758 | 0.47532617295730223 | 0.3333333333333333 | 0.6666666666666666 | boosted_tree, linear |

## Interaction diagnostics

- Mean interaction score: `0.03715248982367274`; median: `0.021800015744920662`
- Strength distribution: `{"low": 36}`
- Mean evaluated pairs: `16.0`; mean strong-pair fraction: `0.0`
- Regime metrics: `{"low": {"catastrophic_regret_rate": 0.2222222222222222, "dataset_count": 12, "exact_reference_match_rate": 0.5555555555555556, "mean_normalized_regret": 0.2924657311766499, "median_normalized_regret": 0.025914812414654, "top2_compatibility_rate": 0.6666666666666666}}`
- Top interaction evidence: `[{"features": ["feature_10", "feature_20"], "incremental_strength": 0.22102339124672443, "interaction_strength": 0.17169800815252229, "joint_strength": 0.7768318420237199, "marginal_strength": 0.5558084507769955, "sample_count": 192, "transform": "sum"}, {"features": ["feature_10", "feature_26"], "incremental_strength": 0.16472512095263947, "interaction_strength": 0.11868997975360145, "joint_strength": 0.720533571729635, "marginal_strength": 0.5558084507769955, "sample_count": 192, "transform": "sum"}, {"features": ["feature_10", "feature_9"], "incremental_strength": 0.15021661062697378, "interaction_strength": 0.1060566917418053, "joint_strength": 0.7060250614039693, "marginal_strength": 0.5558084507769955, "sample_count": 192, "transform": "sum"}]`

## Classification boundary diagnostics

- Mean boundary complexity score: `0.036196522566737326`; median: `0.0`
- Mean linear probe score: `0.8949042012006098`; mean normalized linear separability: `0.7954064037709502`
- Mean local class consistency: `0.8767326918763251`; mean nonlinear advantage: `0.03949101905561377`
- Category distribution: `{"high": 2, "low": 15, "moderate": 1}`
- Confidence distribution: `{"high": 16, "medium": 2}`
- Regime metrics: `{"high": {"catastrophic_regret_rate": 0.3333333333333333, "dataset_count": 1, "exact_reference_match_rate": 0.3333333333333333, "mean_normalized_regret": 0.03911257790878123, "median_normalized_regret": 0.03911257790878123, "top2_compatibility_rate": 0.3333333333333333}, "low": {"catastrophic_regret_rate": 0.2121212121212121, "dataset_count": 11, "exact_reference_match_rate": 0.5757575757575757, "mean_normalized_regret": 0.3154978360191834, "median_normalized_regret": 0.015532869946633492, "top2_compatibility_rate": 0.696969696969697}}`

Largest current-policy regret cases:

- `synthetic_high_dim_regression` seed `123`: deterministic `tree_ensemble`, empirical best `linear`, regret `3.9757948408671027`.
- `synthetic_high_dim_regression` seed `2027`: deterministic `tree_ensemble`, empirical best `linear`, regret `3.9015933567290895`.
- `synthetic_regression` seed `42`: deterministic `boosted_tree`, empirical best `linear`, regret `1.4259785188719067`.
- `synthetic_nonlinear_regression` seed `2027`: deterministic `linear`, empirical best `boosted_tree`, regret `0.32857791766543515`.
- `synthetic_nonlinear_regression` seed `42`: deterministic `linear`, empirical best `boosted_tree`, regret `0.30004441014534156`.

## Family-selection distribution

{"boosted_tree": {"count": 4, "rate": 0.1111111111111111}, "linear": {"count": 14, "rate": 0.3888888888888889}, "regularized_linear": {"count": 16, "rate": 0.4444444444444444}, "tree_ensemble": {"count": 2, "rate": 0.05555555555555555}}

## Recommendation

- Objective-selected candidate: `current`
- Recommendation: **retain_current**
- Final benchmark results are not used to modify the policy version under test.
- Compatibility scores remain interpretable compatibility points, not probabilities of empirical optimality.

## Selective soft-challenge calibration

- Calibration artifact: `soft-challenge-calibration-v1-v1`
- Development records with a model-family disagreement: `20`
- Reliability is the deterministic challenger win rate among non-tied development disagreements; ties remain in support but are excluded from that denominator.

| Regime | Support | Challenger wins | Agent wins | Ties | Win rate | Mean regret delta | Catastrophic prevention |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all/all/all/all` | 20 | 3 | 6 | 11 | 0.3333333333333333 | -0.4407438286660835 | 0.6 |
| `classification/all/all/all` | 6 | 2 | 3 | 1 | 0.4 | 0.027969114650984266 | 1.0 |
| `classification/all/all/low` | 6 | 2 | 3 | 1 | 0.4 | 0.027969114650984266 | 1.0 |
| `classification/low/all/low` | 6 | 2 | 3 | 1 | 0.4 | 0.027969114650984266 | 1.0 |
| `classification/low/low/low` | 6 | 2 | 3 | 1 | 0.4 | 0.027969114650984266 | 1.0 |
| `regression/all/all/all` | 14 | 1 | 3 | 10 | 0.25 | -0.6416208043733983 | 0.3333333333333333 |
| `regression/all/all/low` | 12 | 1 | 3 | 8 | 0.25 | -0.7496960020108591 | 0.3333333333333333 |
| `regression/all/all/medium` | 2 | 0 | 0 | 2 | None | 0.006830381451366595 | None |
| `regression/low/all/low` | 10 | 1 | 1 | 8 | 0.5 | -0.11244066177206309 | 0.3333333333333333 |
| `regression/low/all/medium` | 2 | 0 | 0 | 2 | None | 0.006830381451366595 | None |
| `regression/low/high/low` | 2 | 1 | 1 | 0 | 0.5 | -0.5637537899553178 | 1.0 |
| `regression/low/low/medium` | 2 | 0 | 0 | 2 | None | 0.006830381451366595 | None |
| `regression/low/moderate/low` | 8 | 0 | 0 | 8 | None | 0.0003876202737505812 | 0.0 |
| `regression/medium/all/low` | 2 | 0 | 2 | 0 | 0.0 | -3.935972703204839 | None |
| `regression/medium/high/low` | 2 | 0 | 2 | 0 | 0.0 | -3.935972703204839 | None |
