# Conventional baseline analysis

The conventional baselines are a retrospective, post-hoc analysis layer. They
do not alter the runtime gate and do not change the meaning of
`paper_confirmatory_v3` or the completed Gemini replication.

`all_four_cv` is conventional full-portfolio model-family selection over
`linear`, `regularized_linear`, `tree_ensemble`, and `boosted_tree`. It selects
the highest training-only mean macro-F1 for classification or the lowest
training-only mean RMSE for regression, using the existing deterministic tie
break. Its selected family is then evaluated once on the untouched holdout.

`pairwise_cv_always` is the two-candidate control. On a valid LLM/challenger
family disagreement it always selects the raw mean-CV winner, ignoring the
selective probe's evidence-strength threshold. A strict numerical tie retains
the initial LLM plan. It never invokes a reconciler and never uses holdout
information during selection.

## Retrospective commands

The source is read-only and the output must be a new directory:

```text
python -m evaluation.conventional_baselines \
  --source evaluation_results/external_ablation_live_v3 \
  --output evaluation_results/retrospective_posthoc_openai
```

For the completed Gemini replication, use a separate output directory:

```text
python -m evaluation.conventional_baselines \
  --source evaluation_results/external_ablation_gemini_v1 \
  --output evaluation_results/retrospective_posthoc_gemini
```

If a source row does not contain the selected baseline's holdout metric, add
`--recompute-missing`. This loads only the deterministic benchmark data and
verifies that the raw frame reproduces the persisted split contract before a
single final holdout fit. It does not make an LLM call or rewrite the source.
Use `--strict` when a paper pipeline should stop instead of writing explicit
missing-artifact rows.

Each output contains:

* `baseline_trials.jsonl` — one provenance-rich derived row per logical trial
  and baseline, after all ablation arms under the source directory have been
  grouped together;
* `baseline_summary.json` — machine-readable comparisons, coverage, fit and
  LLM-call accounting, missing-arm diagnostics, and dataset-cluster bootstrap
  metadata;
* `baseline_summary.csv` — flat baseline-level summary rows;
* `baseline_summary.md` — human-readable audit report.

Historical source trees must contain `trials.jsonl`. The analyzer discovers all
such files below the supplied directory before deriving any baseline. A logical
trial is keyed by model condition, provider/model, benchmark case and dataset
identity, task type, perturbation, split seed/random state, LLM repetition and
trial slot, evaluation variant, and order-swap pair identity. The ablation arm
is deliberately excluded. Each derived row records the source file, source
ablation, source model condition, and the arm supplying the LLM-only holdout,
empirical probe, probe-direct result, and full empirical reference.
The checked-in historical external bundles currently contain configs and
summaries but not those trial-level files; the CLI fails explicitly rather than
inventing baseline decisions from aggregate prose.

Counterfactual policy cost is separate from work performed by this analyzer.
For example, cached historical CV scores produce zero
`actual_analysis_fit_count`, while `all_four_cv` still reports its deployed
four-family CV cost. That cost is keyed by dataset/task, frozen split, and CV
configuration, excluding LLM repetition (and model condition), so two
repetitions on one split charge one four-family search. `all_four_cv` search
cost is therefore 12 fits for four families with three-fold CV, not 24 for two
paired repetitions. Primary comparisons are exposed under
`comparisons_by_model_condition` and
`comparisons_by_model_condition_and_task_type`. Any combined-condition view is
named `descriptive_combined_condition_comparisons` and is audit-only.

## Prospective fresh splits and task panels

Use `evaluation/configs/prospective_generalization_template.json` as a draft
checklist. The existing runner already propagates multiple split seeds into
split creation, proposal-cache identity, trial IDs, empirical-reference cache
identity, resume checks, and summaries. A future run should provide several
seeds selected before outcomes are observed.

For a fresh-split study on the existing local cases, declare the seeds and
conditions in the run command and write to a new output directory:

```text
python -m evaluation.ablation \
  --output evaluation_results/prospective_fresh_splits_v1 \
  --suite local \
  --split-seed <predeclared_seed_1> \
  --split-seed <predeclared_seed_2> \
  --split-seed <predeclared_seed_3> \
  --repetitions 3 \
  --provider openai \
  --model <declared_planner_model> \
  --ablation llm_only \
  --ablation probe_direct \
  --ablation full \
  --offline
```

For a real prospective run, set the declared provider/model conditions and
replace `--offline` with the reviewed live-execution policy. Do not pass a
historical confirmatory manifest to this fresh-split path, and do not reuse a
historical output directory.

A new task panel must be supplied as a separate manifest with
`analysis_role: "prospective_generalization"`. It must declare a sampling
frame, non-empty inclusion criteria, exclusion criteria, selection seed,
immutable OpenML task/dataset identifiers and versions, task type, target,
dimensions, provenance, per-task selection metadata, and the content hash.
The original 40-task AMLB/OpenML manifest is not modified by this machinery.

After human review and freezing that panel, a future run has the form:

```text
python -m evaluation.prospective \
  --panel-manifest path/to/frozen_prospective_task_panel.json \
  --output evaluation_results/prospective_generalization_v1 \
  --split-seed <predeclared_seed_1> \
  --split-seed <predeclared_seed_2> \
  --split-seed <predeclared_seed_3> \
  --repetitions 3 \
  --provider openai \
  --model <declared_planner_model> \
  --ablation llm_only \
  --ablation probe_direct \
  --ablation full \
  --require-live
```

Before freezing, a human must choose the sampling frame, classification and
regression counts, eligibility bounds, exclusions, duplicate/version policy,
provider/model conditions, repetition count, split seeds, and the intended
primary reporting strata. Those choices are intentionally not invented here.

## Paper mapping

* Why not all-four CV? Compare `llm_only`, `pairwise_cv_always`, and
  `all_four_cv` holdout performance against training normalized regret and
  counterfactual selection-fit counts.
* Does abstention help? Compare `probe_direct` with
  `pairwise_cv_always`, including beneficial/harmful/neutral intervention rates
  under the frozen task-specific holdout tolerances.
* Does the two-candidate set cover the all-four winner? Use
  `llm_hit_all4_winner`, `challenger_incremental_hit`, and
  `two_candidate_coverage`, reported separately by task type and model
  condition with dataset-macro denominators.
* What compute is saved? Use the counterfactual CV-fit, final-fit, empirical
  probe-fit, planner-call, reconciler-call, and distinct-family fields.
* Does the result persist? Use the separate prospective run's
  `analysis_role`, frozen panel hash, explicit split-seed list, and the same
  dataset-clustered reporting hierarchy.

## Small accounting example

For one dataset/task, split seed `42`, and two LLM repetitions, the analyzer
emits two paired observations for each repetition-dependent performance
comparison. With three-fold CV and the four candidates listed above,
`all_four_cv` has `4 × 3 = 12` counterfactual selection fits for the shared
dataset/split. The second repetition reuses that same search key, so the
summary remains at 12 rather than 24. If the persisted reference and holdout
artifacts are reused, retrospective work is 0 new fits; the deployed policy
cost still reports 12 selection fits plus the final-fit cost separately.
