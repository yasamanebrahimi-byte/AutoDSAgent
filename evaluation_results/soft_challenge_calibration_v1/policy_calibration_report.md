# Deterministic Policy Calibration Report

- Policy version under test: `4`
- Benchmark suite: `2`
- Role: `policy_development`
- Unique datasets: **12**; repeated seeds are not treated as independent datasets
- Split seeds: `[42, 123, 2027]`
- Git commit: `4f9fed67a0a31743b8f84f1d60ff34f9bb324e65`

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
| `current` | 0.2992996218028286 | 0.27777777777777773 | 0.5277777777777778 | 0.638888888888889 | no |
| `nonlinear_sensitive` | 0.39120315684061313 | 0.25 | 0.5555555555555556 | 0.6944444444444445 | no |
| `high_dimensional_sensitive` | 0.2992996218028286 | 0.27777777777777773 | 0.5277777777777778 | 0.638888888888889 | no |
| `missingness_sensitive` | 0.2992996218028286 | 0.27777777777777773 | 0.5277777777777778 | 0.638888888888889 | no |

## Sensitivity analysis

The four candidates are a deliberately small neighborhood around the current interpretable thresholds; no continuous optimizer or LLM is used.

- `current`: nonlinear thresholds 0.15/0.35; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.2992996218028286`.
- `nonlinear_sensitive`: nonlinear thresholds 0.12/0.3; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.39120315684061313`.
- `high_dimensional_sensitive`: nonlinear thresholds 0.15/0.35; sample-feature bands `[5.0, 10.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.2992996218028286`.
- `missingness_sensitive`: nonlinear thresholds 0.15/0.35; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.08, 0.2]`; mean regret `0.2992996218028286`.

## Per-dataset results and failure cases

Per-dataset means below preserve dataset identity before the across-dataset summary.

| Dataset | Seeds | Mean regret | Catastrophic rate | Top-2 rate | Selected families |
|---|---:|---:|---:|---:|---|
| `breast_cancer` | 3 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `diabetes` | 3 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `synthetic_binary_linear` | 3 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `synthetic_binary_nonlinear` | 3 | 0.12111926542292761 | 1.0 | 0.0 | linear |
| `synthetic_high_dim_regression` | 3 | 2.6257960658653947 | 0.6666666666666666 | 0.3333333333333333 | regularized_linear, tree_ensemble |
| `synthetic_imbalanced_classification` | 3 | 0.1079835566220844 | 0.6666666666666666 | 1.0 | regularized_linear |
| `synthetic_linear_regression` | 3 | 0.0 | 0.0 | 1.0 | linear |
| `synthetic_missingness` | 3 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `synthetic_multiclass` | 3 | 0.03629675488267451 | 0.0 | 0.0 | linear |
| `synthetic_nonlinear_regression` | 3 | 0.20954077593692566 | 0.6666666666666666 | 0.3333333333333333 | boosted_tree, linear |
| `synthetic_outlier_regression` | 3 | 0.015532869946633611 | 0.0 | 0.3333333333333333 | linear |
| `synthetic_regression` | 3 | 0.47532617295730223 | 0.3333333333333333 | 0.6666666666666666 | boosted_tree, linear |

Largest current-policy regret cases:

- `synthetic_high_dim_regression` seed `123`: deterministic `tree_ensemble`, empirical best `linear`, regret `3.9757948408670973`.
- `synthetic_high_dim_regression` seed `2027`: deterministic `tree_ensemble`, empirical best `linear`, regret `3.9015933567290864`.
- `synthetic_regression` seed `42`: deterministic `boosted_tree`, empirical best `linear`, regret `1.4259785188719067`.
- `synthetic_nonlinear_regression` seed `2027`: deterministic `linear`, empirical best `boosted_tree`, regret `0.32857791766543515`.
- `synthetic_nonlinear_regression` seed `42`: deterministic `linear`, empirical best `boosted_tree`, regret `0.3000444101453418`.

## Family-selection distribution

{"boosted_tree": {"count": 2, "rate": 0.05555555555555555}, "linear": {"count": 16, "rate": 0.4444444444444444}, "regularized_linear": {"count": 16, "rate": 0.4444444444444444}, "tree_ensemble": {"count": 2, "rate": 0.05555555555555555}}

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
| `all/all/all/all` | 20 | 1 | 6 | 13 | 0.14285714285714285 | -0.4530448317932055 | 0.2 |
| `classification/all/all/all` | 6 | 0 | 3 | 3 | 0.0 | -0.013034229106088924 | 0.0 |
| `classification/all/all/low` | 6 | 0 | 3 | 3 | 0.0 | -0.013034229106088924 | 0.0 |
| `classification/low/all/low` | 6 | 0 | 3 | 3 | 0.0 | -0.013034229106088924 | 0.0 |
| `classification/low/low/low` | 6 | 0 | 3 | 3 | 0.0 | -0.013034229106088924 | 0.0 |
| `regression/all/all/all` | 14 | 1 | 3 | 10 | 0.25 | -0.6416208043733983 | 0.3333333333333333 |
| `regression/all/all/low` | 12 | 1 | 3 | 8 | 0.25 | -0.7496960020108587 | 0.3333333333333333 |
| `regression/all/all/medium` | 2 | 0 | 0 | 2 | None | 0.0068303814513648795 | None |
| `regression/low/all/low` | 10 | 1 | 1 | 8 | 0.5 | -0.11244066177206317 | 0.3333333333333333 |
| `regression/low/all/medium` | 2 | 0 | 0 | 2 | None | 0.0068303814513648795 | None |
| `regression/low/high/low` | 2 | 1 | 1 | 0 | 0.5 | -0.5637537899553179 | 1.0 |
| `regression/low/low/medium` | 2 | 0 | 0 | 2 | None | 0.0068303814513648795 | None |
| `regression/low/moderate/low` | 8 | 0 | 0 | 8 | None | 0.0003876202737505079 | 0.0 |
| `regression/medium/all/low` | 2 | 0 | 2 | 0 | 0.0 | -3.9359727032048357 | None |
| `regression/medium/high/low` | 2 | 0 | 2 | 0 | 0.0 | -3.9359727032048357 | None |
