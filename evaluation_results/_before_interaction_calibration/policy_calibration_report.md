# Deterministic Policy Calibration Report

- Policy version under test: `4`
- Benchmark suite: `2`
- Role: `policy_development`
- Unique datasets: **2**; repeated seeds are not treated as independent datasets
- Split seeds: `[42]`
- Git commit: `90a09f252022755a592157f2b37ef812703b57d5`

## Development benchmark composition

The registry assigns these cases permanently to policy development. Final-evaluation cases are rejected by the calibration runner.

- `breast_cancer`
- `diabetes`

## Candidate configurations and selection criterion

rank by lowest dataset-level mean normalized regret, then lowest dataset-level catastrophic-regret rate, then highest dataset-level top-2 reference inclusion, then lowest policy complexity; retain the current policy unless the selected candidate clears the predefined promotion margin.

| Candidate | Mean regret | Catastrophic rate | Exact match | Top-2 rate | Collapse warning |
|---|---:|---:|---:|---:|---|
| `current` | 0.0 | 0.0 | 1.0 | 1.0 | yes |
| `nonlinear_sensitive` | 0.0 | 0.0 | 1.0 | 1.0 | yes |
| `high_dimensional_sensitive` | 0.0 | 0.0 | 1.0 | 1.0 | yes |
| `missingness_sensitive` | 0.0 | 0.0 | 1.0 | 1.0 | yes |

## Sensitivity analysis

The four candidates are a deliberately small neighborhood around the current interpretable thresholds; no continuous optimizer or LLM is used.

- `current`: nonlinear thresholds 0.15/0.35; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.0`.
- `nonlinear_sensitive`: nonlinear thresholds 0.12/0.3; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.0`.
- `high_dimensional_sensitive`: nonlinear thresholds 0.15/0.35; sample-feature bands `[5.0, 10.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.0`.
- `missingness_sensitive`: nonlinear thresholds 0.15/0.35; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.08, 0.2]`; mean regret `0.0`.

## Per-dataset results and failure cases

Per-dataset means below preserve dataset identity before the across-dataset summary.

| Dataset | Seeds | Mean regret | Catastrophic rate | Top-2 rate | Selected families |
|---|---:|---:|---:|---:|---|
| `breast_cancer` | 1 | 0.0 | 0.0 | 1.0 | regularized_linear |
| `diabetes` | 1 | 0.0 | 0.0 | 1.0 | regularized_linear |

Largest current-policy regret cases:

- `breast_cancer` seed `42`: deterministic `regularized_linear`, empirical best `regularized_linear`, regret `0.0`.
- `diabetes` seed `42`: deterministic `regularized_linear`, empirical best `regularized_linear`, regret `0.0`.

## Family-selection distribution

{"regularized_linear": {"count": 2, "rate": 1.0}}

## Recommendation

- Objective-selected candidate: `current`
- Recommendation: **retain_current**
- Final benchmark results are not used to modify the policy version under test.
- Compatibility scores remain interpretable compatibility points, not probabilities of empirical optimality.

## Selective soft-challenge calibration

- Calibration artifact: `soft-challenge-calibration-v1-v1`
- Development records with a model-family disagreement: `0`
- Reliability is the deterministic challenger win rate among non-tied development disagreements; ties remain in support but are excluded from that denominator.

| Regime | Support | Challenger wins | Agent wins | Ties | Win rate | Mean regret delta | Catastrophic prevention |
|---|---:|---:|---:|---:|---:|---:|---:|
