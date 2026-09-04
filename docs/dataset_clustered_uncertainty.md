# Dataset-clustered uncertainty

Paper-facing benchmark uncertainty treats `benchmark_case` as the canonical
dataset/task cluster. The independent resampling unit is the benchmark
dataset/task, not an individual trial. Confidence intervals use a percentile
bootstrap: for every replicate, the observed number of datasets is sampled
with replacement and every saved row belonging to each sampled dataset is
retained. Duplicate dataset draws therefore contribute duplicate complete
clusters. Split seeds, repetitions, interventions, and ablation rows are not
resampled independently.

The default is 10,000 replicates with seed `20260824` and confidence level
0.95. A CI is unavailable when fewer than two independent datasets are
eligible. Statistics are recomputed inside every replicate. Existing
trial-weighted point estimates are preserved; the separately exposed
`dataset_weighted_gate_health` provides dataset-macro descriptive estimates.
Conditional rates retain their event-count denominator semantics and missing
holdout outcomes are excluded rather than converted to zero.

The old row-level helper is retained as explicitly named `iid_bootstrap_ci`
for diagnostics and is not used for paper-facing intervals.
