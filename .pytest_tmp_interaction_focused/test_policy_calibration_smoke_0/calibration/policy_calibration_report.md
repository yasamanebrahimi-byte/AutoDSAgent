# Deterministic Policy Calibration Report

- Policy version under test: `4`
- Benchmark suite: `2`
- Role: `policy_development`
- Unique datasets: **1**; repeated seeds are not treated as independent datasets
- Split seeds: `[11]`
- Git commit: `90a09f252022755a592157f2b37ef812703b57d5`

## Development benchmark composition

The registry assigns these cases permanently to policy development. Final-evaluation cases are rejected by the calibration runner.

- `calibration_fixture`

## Candidate configurations and selection criterion

rank by lowest dataset-level mean normalized regret, then lowest dataset-level catastrophic-regret rate, then highest dataset-level top-2 reference inclusion, then lowest policy complexity; retain the current policy unless the selected candidate clears the predefined promotion margin.

| Candidate | Mean regret | Catastrophic rate | Exact match | Top-2 rate | Collapse warning |
|---|---:|---:|---:|---:|---|
| `current` | 0.04232238349885409 | 0.0 | 0.0 | 0.0 | yes |
| `nonlinear_sensitive` | 0.04232238349885409 | 0.0 | 0.0 | 0.0 | yes |

## Sensitivity analysis

The four candidates are a deliberately small neighborhood around the current interpretable thresholds; no continuous optimizer or LLM is used.

- `current`: nonlinear thresholds 0.15/0.35; interaction thresholds 0.18/0.38; interaction limits 12 features/48 pairs; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.04232238349885409`.
- `nonlinear_sensitive`: nonlinear thresholds 0.12/0.3; interaction thresholds 0.18/0.38; interaction limits 12 features/48 pairs; sample-feature bands `[3.0, 8.0, 20.0]`; missingness bands `[0.1, 0.25]`; mean regret `0.04232238349885409`.

## Per-dataset results and failure cases

Per-dataset means below preserve dataset identity before the across-dataset summary.

| Dataset | Seeds | Interaction | Score | Mean regret | Catastrophic rate | Top-2 rate | Selected families |
|---|---:|---|---:|---:|---:|---:|---|
| `calibration_fixture` | 1 | low | 0.0 | 0.04232238349885409 | 0.0 | 0.0 | linear |

## Interaction diagnostics

- Mean interaction score: `0.0`; median: `0.0`
- Strength distribution: `{"low": 1}`
- Mean evaluated pairs: `0.0`; mean strong-pair fraction: `0.0`
- Regime metrics: `{"low": {"catastrophic_regret_rate": 0.0, "dataset_count": 1, "exact_reference_match_rate": 0.0, "mean_normalized_regret": 0.04232238349885409, "median_normalized_regret": 0.04232238349885409, "top2_compatibility_rate": 0.0}}`
- Top interaction evidence: `[]`

Largest current-policy regret cases:

- `calibration_fixture` seed `11`: deterministic `linear`, empirical best `boosted_tree`, regret `0.04232238349885409`.

## Family-selection distribution

{"linear": {"count": 1, "rate": 1.0}}

## Recommendation

- Objective-selected candidate: `current`
- Recommendation: **retain_current**
- Final benchmark results are not used to modify the policy version under test.
- Compatibility scores remain interpretable compatibility points, not probabilities of empirical optimality.

## Selective soft-challenge calibration

- Calibration artifact: `soft-challenge-calibration-v1-v1`
- Development records with a model-family disagreement: `1`
- Reliability is the deterministic challenger win rate among non-tied development disagreements; ties remain in support but are excluded from that denominator.

| Regime | Support | Challenger wins | Agent wins | Ties | Win rate | Mean regret delta | Catastrophic prevention |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all/all/all/all` | 1 | 1 | 0 | 0 | 1.0 | 0.03658119658119663 | None |
| `classification/all/all/all` | 1 | 1 | 0 | 0 | 1.0 | 0.03658119658119663 | None |
| `classification/all/all/low` | 1 | 1 | 0 | 0 | 1.0 | 0.03658119658119663 | None |
| `classification/low/all/low` | 1 | 1 | 0 | 0 | 1.0 | 0.03658119658119663 | None |
| `classification/low/low/low` | 1 | 1 | 0 | 0 | 1.0 | 0.03658119658119663 | None |
