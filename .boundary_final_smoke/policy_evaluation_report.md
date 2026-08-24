# Frozen Deterministic Policy Final Evaluation

- Evaluation role: `final_evaluation`
- Frozen policy version: `4`
- Benchmark suite: `2`
- Unique datasets: **2**
- Split seeds: `[42]`
- Git commit: `d63d43961c5ba0a00957cd9fb2f0ed573312fa39`

## Frozen evaluation protocol

Only final-evaluation cases are accepted. Diagnostics and empirical-reference CV use each case's training partition; the holdout is scored after the policy decision and cannot modify policy parameters.

## Policy quality metrics

- Mean dataset-level normalized regret: `0.007508680990055638`
- Median dataset-level normalized regret: `0.007508680990055638`
- Empirical-reference match rate: `0.0`
- Catastrophic-regret rate: `0.0`
- Top-2 compatibility success: `0.0`
- Family-selection distribution: `{"regularized_linear": {"count": 2, "rate": 1.0}}`

## Per-dataset final results

| Dataset | Seeds | Interaction | Score | Mean regret | Catastrophic rate | Top-2 rate | Selected families |
|---|---:|---|---:|---:|---:|---:|---|
| `digits_subset` | 1 | low | 0.0 | 0.008039459708091501 | 0.0 | 0.0 | regularized_linear |
| `wine` | 1 | low | 0.0 | 0.006977902272019776 | 0.0 | 0.0 | regularized_linear |

## Interaction diagnostics

- Mean interaction score: `0.0`; median: `0.0`
- Strength distribution: `{"low": 2}`
- Mean evaluated pairs: `0.0`; mean strong-pair fraction: `0.0`
- Regime metrics: `{"low": {"catastrophic_regret_rate": 0.0, "dataset_count": 2, "exact_reference_match_rate": 0.0, "mean_normalized_regret": 0.007508680990055638, "median_normalized_regret": 0.007508680990055638, "top2_compatibility_rate": 0.0}}`
- Top interaction evidence: `[]`

## Classification boundary diagnostics

- Mean boundary complexity score: `0.0`; median: `0.0`
- Mean linear probe score: `0.9280481820493263`; mean normalized linear separability: `0.9177793315817724`
- Mean local class consistency: `0.8797679143391988`; mean nonlinear advantage: `0.0`
- Category distribution: `{"low": 2}`
- Confidence distribution: `{"high": 2}`
- Regime metrics: `{"low": {"catastrophic_regret_rate": 0.0, "dataset_count": 2, "exact_reference_match_rate": 0.0, "mean_normalized_regret": 0.007508680990055638, "median_normalized_regret": 0.007508680990055638, "top2_compatibility_rate": 0.0}}`

## Final holdout metrics

- `digits_subset`: `{'accuracy': 0.9666666666666667, 'balanced_accuracy': 0.9666666666666666, 'macro_f1': 0.9664606667702642, 'weighted_f1': 0.9664606667702642}`
- `wine`: `{'accuracy': 0.9722222222222222, 'balanced_accuracy': 0.9666666666666667, 'macro_f1': 0.9709618874773139, 'weighted_f1': 0.9719701552732407}`

## Largest policy failure cases

- `digits_subset` seed `42`: deterministic `regularized_linear`, empirical best `tree_ensemble`, regret `0.008039459708091501`.
- `wine` seed `42`: deterministic `regularized_linear`, empirical best `tree_ensemble`, regret `0.006977902272019776`.

Final benchmark results are descriptive evidence for the frozen policy. They are not used to tune or rewrite the policy version evaluated here.
