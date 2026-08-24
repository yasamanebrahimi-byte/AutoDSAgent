# Frozen Deterministic Policy Final Evaluation

- Evaluation role: `final_evaluation`
- Frozen policy version: `4`
- Benchmark suite: `2`
- Unique datasets: **1**
- Split seeds: `[11]`
- Git commit: `90a09f252022755a592157f2b37ef812703b57d5`

## Frozen evaluation protocol

Only final-evaluation cases are accepted. Diagnostics and empirical-reference CV use each case's training partition; the holdout is scored after the policy decision and cannot modify policy parameters.

## Policy quality metrics

- Mean dataset-level normalized regret: `0.04232238349885409`
- Median dataset-level normalized regret: `0.04232238349885409`
- Empirical-reference match rate: `0.0`
- Catastrophic-regret rate: `0.0`
- Top-2 compatibility success: `0.0`
- Family-selection distribution: `{"linear": {"count": 1, "rate": 1.0}}`

## Per-dataset final results

| Dataset | Seeds | Interaction | Score | Mean regret | Catastrophic rate | Top-2 rate | Selected families |
|---|---:|---|---:|---:|---:|---:|---|
| `final_fixture` | 1 | low | 0.0 | 0.04232238349885409 | 0.0 | 0.0 | linear |

## Interaction diagnostics

- Mean interaction score: `0.0`; median: `0.0`
- Strength distribution: `{"low": 1}`
- Mean evaluated pairs: `0.0`; mean strong-pair fraction: `0.0`
- Regime metrics: `{"low": {"catastrophic_regret_rate": 0.0, "dataset_count": 1, "exact_reference_match_rate": 0.0, "mean_normalized_regret": 0.04232238349885409, "median_normalized_regret": 0.04232238349885409, "top2_compatibility_rate": 0.0}}`
- Top interaction evidence: `[]`

## Final holdout metrics

- `final_fixture`: `{'accuracy': 1.0, 'balanced_accuracy': 1.0, 'macro_f1': 1.0, 'weighted_f1': 1.0}`

## Largest policy failure cases

- `final_fixture` seed `11`: deterministic `linear`, empirical best `boosted_tree`, regret `0.04232238349885409`.

Final benchmark results are descriptive evidence for the frozen policy. They are not used to tune or rewrite the policy version evaluated here.
