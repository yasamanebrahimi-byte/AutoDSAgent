# Prospective MLSys experiment preparation

This repository now contains preparation infrastructure for a fresh study of
the reviewer question “Why not simply cross-validate all four fixed candidate
model families?” The historical `trials.jsonl` files are unavailable, so the
old experiment cannot be reconstructed from summary tables. The new study is
prospective: its task panel, seeds, model conditions, repetitions, and
reporting rules must be reviewed and frozen before execution.

## Policies

The live runtime source arms are deliberately limited to:

* `llm_only`: retain the initial LLM plan after universal hard validation.
* `probe_direct`: use the existing selective two-candidate probe-direct gate.
* `full`: use the existing probe plus reconciliation path.

The two conventional policies are derived after the run from raw training-only
evidence:

* `pairwise_cv_always`: compare the same LLM incumbent and deterministic
  challenger as the probe, ignore the selective evidence threshold, and choose
  the better raw mean CV score. Classification maximizes macro-F1; regression
  minimizes RMSE; exact ties retain the LLM incumbent.
* `all_four_cv`: choose among exactly `linear`, `regularized_linear`,
  `tree_ensemble`, and `boosted_tree` using the same training split and CV
  contract, then evaluate once on the untouched holdout. With four families and
  three folds this is 12 conceptual CV fits.

Neither conventional policy invokes an LLM or reads holdout values during
selection. Missing raw evidence is an error, not an invitation to infer a
winner from summary prose.

## Draft panel construction

Build a deterministic draft without freezing it:

```text
python scripts/build_mlsys_prospective_panel.py \
  --output evaluation_results/mlsys_panel_draft.json \
  --selection-seed <predeclared-selection-seed> \
  --classification-count 20 \
  --regression-count 20
```

The builder uses a lazy OpenML dependency, excludes the checked-in historical
panel, applies explicit row/feature/class filters, samples independently by
task type after canonical sorting, and records rejected tasks and reasons. The
manifest always starts with `"status": "draft"`; this command never freezes a
panel. Reviewers must later choose the final sampling frame, eligibility
filters, duplicate/version policy, panel membership, and exclusion record.

The checked-in configuration checklist is
`evaluation/configs/mlsys_prospective_template.json`. It contains placeholders
for the panel hash, provider/model condition, repetitions, split seeds, runtime
source arms, CV folds, output directory, and provenance fields. Do not replace
those placeholders with scientific choices as part of repository preparation.

## Local run and resume

The repository includes a no-network synthetic smoke workflow covering one
classification realization, one regression-style path in the fixture helpers,
one split, one repetition, all three source arms, resume behavior, and strict
validator failures:

```text
python -m pytest -q tests/test_mlsys_prospective.py tests/test_conventional_baselines.py
```

The tests use fixture records and fake/mock behavior; they do not call OpenML
or a paid LLM API. After a panel has been explicitly reviewed and frozen, a
tiny end-to-end smoke run can also be performed with two local/mock cases and
source arms only. A future real run
must use a new output directory and the reviewed frozen panel:

```text
python -m evaluation.prospective \
  --panel-manifest path/to/frozen_prospective_task_panel.json \
  --output evaluation_results/mlsys_prospective_run \
  --split-seed <predeclared-seed> \
  --repetitions 1 \
  --ablation llm_only \
  --ablation probe_direct \
  --ablation full \
  --offline
```

`--offline` is for development and tests only. The runner writes each trial
checkpoint immediately using an atomic, flushed local-file replacement. It
retains run configuration and panel copies, rejects incompatible resume
metadata, skips completed trial IDs, and fails on conflicting completed
duplicates. Run status is recorded as `initialized`, `running`,
`incomplete/interrupted`, or `complete`. For prospective runs, the root
directory is marked `complete` only after evidence validation is ready; source
arm checkpoints can be individually complete while the overall run remains
incomplete.

## Completeness validation

Validate before deriving baselines:

```text
python -m evaluation.validate_mlsys_run path/to/mlsys_prospective_run --strict
```

The command writes `mlsys_validation_report.json` and a human-readable text
report. It checks the three source arms, initial plans and preprocessing
contracts, normalized raw pairwise evidence, all four family CV records, split
and repetition identities, panel/config hashes, holdout results, supported
schema versions, duplicate trial IDs, and run status. Strict mode exits
nonzero unless the report says:

```text
READY_FOR_BASELINE_DERIVATION: YES
```

Only then should the existing analyzer be used:

```text
python -m evaluation.conventional_baselines \
  --source path/to/mlsys_prospective_run \
  --output path/to/new/mlsys_prospective_baselines \
  --strict
```

Derived rows retain source run/trial IDs, panel/config hashes, split seed,
provider/model/repetition, baseline rule, selected family, raw CV values,
holdout source, schema version, fit counts, and LLM/reconciler call counts.

## What must be frozen later

Before the real scientific run, a human review must explicitly decide and
record the final panel and its immutable OpenML task/dataset versions, provider
and model condition(s), repetition count, split-seed list, CV-fold policy,
eligibility and exclusion rules, intervention thresholds, source-arm matrix,
resume/output location, and primary reporting strata. No final values are
selected by this preparation change. No paid LLM run, cloud backup, or real
panel freeze is performed here.
