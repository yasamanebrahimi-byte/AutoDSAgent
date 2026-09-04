# External benchmark selection and freeze record

This document records what can be established from the repository about the
frozen external confirmatory suite. It is intentionally transparent about
what cannot be recovered from the checked-in history.

## What is frozen

The confirmatory manifest is `EXTERNAL_BENCHMARK_MANIFEST` in
`evaluation/external_benchmarks.py`, version `1.0.0`. It contains 40 OpenML
tasks drawn from AMLB classification suite `271` and AMLB regression suite
`269`, identified by immutable OpenML task ID:

- 22 classification tasks from suite 271.
- 18 regression tasks from suite 269.
- 30 `core` tasks and 10 `stress` tasks (5 classification and 5 regression
  tasks in each tier).

The manifest was first checked into repository history on 2026-09-03 in commit
`dfb7208` (`added more benchmarks`). The current manifest version and task list
are frozen before a confirmatory live run. Task IDs, expected row/feature
counts, expected class counts, task type, and tier are serialized into each
evaluation configuration and result row.

## Why these tasks?

The supported, repository-grounded answer is: these are the AMLB/OpenML tasks
that make up the checked-in external suite for supervised tabular
classification and regression. The code does not contain a separate
machine-readable inclusion script, a dated sampling seed, or a complete
historical selection log from the AMLB suites. Therefore this repository does
not claim a more specific historical sampling rule.

The objective inclusion constraints that are visible in code are:

- a task belongs to the AMLB classification or regression suite represented by
  the corresponding source-suite ID;
- classification tasks declare an expected class count, while regression tasks
  require a numeric target;
- each task declares expected rows and features and is rejected if the loader
  does not match those dimensions;
- the `core`/`stress` labels are static manifest metadata. The stress entries
  cover larger, wider, high-dimensional, low-N/high-P, imbalanced, or
  multiclass cases as stated in their manifest notes.

The repository contains no code that uses AutoDSAgent performance, LLM
performance, holdout scores, regret, or intervention outcomes to choose task
membership. The manifest is a static tuple, and loading is lazy. Prefetching
validates only task identity, dimensions, target type, and class count; it does
not run the planner or evaluate model performance.

## Confirmatory boundary

Benchmark membership is frozen before the first external confirmatory
evaluation. The external suite is not a development set. Routine live API
smoke tests use local, synthetic, or other predesignated development tasks.
Schema-only external prefetching is allowed before confirmation.

If an external task must be used for operational live debugging, it must be
predeclared as exploratory and excluded from the confirmatory headline
analysis. Inspecting such outcomes must not be followed by policy, prompt,
threshold, candidate-family, reconciliation, or benchmark-membership tuning
while still presenting that task as untouched confirmation.

## Source and comparability limitation

The source benchmark is AMLB/OpenML, with the original AMLB reference cited in
[`external_benchmark.md`](external_benchmark.md). AutoDSAgent uses its own
frozen train/holdout split and training-side candidate procedure rather than
AMLB's predefined folds. Results therefore must not be presented as directly
comparable to AMLB leaderboard numbers.
