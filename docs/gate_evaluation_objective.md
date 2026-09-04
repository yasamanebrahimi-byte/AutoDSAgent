# Gate evaluation objective

The evaluation objective is versioned as `intervention-quality-v1`. The
paper-facing claim is deliberately narrow: selective reliability safeguards
for LLM-based model-family and preprocessing planning in supervised tabular
classification and regression. AutoDSAgent remains a broader end-to-end
data-science product; this evaluation does not validate arbitrary EDA,
cleaning, reporting, causal reasoning, time series, clustering, or target
discovery capabilities.

## Runtime/evaluation boundary

The initial LLM plan is generated from training-side information. An independent
deterministic/non-LLM structural challenger also operates only on the frozen
training partition, but the representations are intentionally asymmetric: the
LLM receives a compact profile while the challenger computes richer
pre-specified structural diagnostics from that same partition. Disagreements
trigger a bounded empirical arbitration stage when configured. Weak or tied
evidence leads to abstention and preservation of the LLM plan; sufficiently
strong evidence can trigger blinded reconciliation before the final plan is
frozen. The secondary `llm_with_diagnostics` ablation gives the initial LLM the
canonical training-only diagnostics without enabling the intervention gate, to
test whether the information advantage alone explains an effect.

These are pre-final-training validation stages: small training-only fits/CV
probes are allowed inside the safeguard. They are not a claim of zero model
fitting of any kind. The train/holdout split is frozen before planning, and the
untouched holdout target values, scores, winner comparisons, and intervention
outcomes are unavailable to the planner, challenger, hard validator, probe,
reconciler, abstention logic, and final-plan selection. Only after the final
plan is frozen are the initial and final plans evaluated on the same holdout.

## Trial outcomes

The primary intervention outcome is the paired exact-plan untouched-holdout
result. `paper_holdout_delta` is dimensionless and positive always means the
final/intervened plan is better:

```text
classification:
    holdout_macro_f1_delta = final_holdout_macro_f1 - initial_holdout_macro_f1
regression:
    holdout_rmse_relative_improvement =
        (initial_holdout_rmse - final_holdout_rmse)
        / max(abs(initial_holdout_rmse), holdout_rmse_epsilon)
```

`holdout_rmse_delta_raw = initial_holdout_rmse - final_holdout_rmse` remains
available for regression diagnostics, but is native-unit and is never averaged
with classification deltas for a paper-facing cross-dataset estimate.

Classification neutrality uses
`classification_holdout_neutral_tolerance` in absolute macro-F1 points.
Regression neutrality uses `regression_holdout_neutral_tolerance` in relative
RMSE-improvement units. A changed soft plan is `beneficial`, `harmful`, or
`neutral`; an unchanged plan is `not_intervened`, and a missing/failed pair is
`not_comparable`.

The holdout-based intervention precision, harmful-intervention rate, neutral
rate, scale-free mean/median delta, dataset-macro estimate, clustered CI, and
valid paired denominator are reported alongside the training-side diagnostics
below.

The result schema remains `holdout-metrics-v2`: the explicit training-reference
names and incidence fields are additive, while historical short names remain
deprecated aliases with their semantics recorded in the serialized formulas.

For training-side diagnostic analysis, `normalized_gate_delta` is
`initial_normalized_regret - final_normalized_regret`. Positive means that the
gate reduced regret. The same direction is obtained directly from primary
scores with `final_macro_f1 - initial_macro_f1` for classification and
`initial_rmse - final_rmse` for regression.

The default, configurable `neutral_tolerance` is `0.02` normalized-regret
units. A delta whose absolute value is at most that tolerance is `neutral`;
larger positive and negative deltas are `improved` and `worsened`.

Normalized regret is zero for the best available training-only empirical
reference result. Classification regret is `max(0, best_macro_f1 - selected_macro_f1)`;
regression regret is `max(0, selected_rmse - best_rmse) / max(abs(best_rmse), 1e-12)`.
Regret reduction is `initial_regret - final_regret`.

These regret-based values remain diagnostics of what training-side evidence
predicted; they are not the primary definition of realized intervention
success or harm.

## Paper-facing intervention metrics

- An eligible soft disagreement is a completed disagreement with a valid
  initial proposal after hard-validation eligibility; the denominator is the
  set of eligible soft disagreements, represented by challenge plus abstention
  records.
- `challenge_rate = challenged eligible initial plans / eligible soft
  disagreements`.
- `intervention_rate = actual changed soft plans / eligible completed trials`.
- `abstention_rate` and `abstention_preservation_rate` are conditional:
  `challenged eligible initial plans preserved because evidence was insufficient
  / eligible soft disagreements`.
- `beneficial_intervention_rate = beneficial actual interventions / actual
  interventions with evaluable holdout outcome`; `harmful_intervention_rate`
  and `neutral_intervention_rate` use the same denominator and their respective
  untouched-holdout outcomes. Their incidence counterparts use all eligible
  completed trials as denominator.
- `intervention_precision = beneficial actual interventions / actual
  interventions with evaluable holdout outcome`.
- `probe_invocation_rate_conditional_on_disagreement` is the fraction of
  eligible LLM/challenger disagreements that receive the bounded training-only
  probe.
- `abstention_rate_conditional_on_disagreement` is the fraction of eligible
  disagreements preserved because the evidence was insufficient.
- `mean_beneficial_holdout_magnitude` and
  `mean_harmful_holdout_magnitude` summarize the positive improvement and
  absolute harm among comparable actual interventions; they are null when
  support is absent.
- `harm_rate = harmful actual interventions / actual interventions with
  evaluable holdout outcome` (also exposed as `harmful_intervention_rate`).
- A zero denominator returns `null` and renders as `n/a`; no rate silently
  changes its denominator.

All rates return `null` when their denominator is zero. Primary estimates are
dataset-macro; trial-weighted values are retained and explicitly labeled as
secondary diagnostics.

## Training-side diagnostic metrics

- `training_reference_intervention_precision = improved / (improved +
  worsened)`. Neutral interventions are excluded from this denominator; this
  is distinct from the primary holdout `intervention_precision`.
- `training_reference_challenge_yield = training-reference improved challenges
  / total challenges`.
- `training_reference_harmful_intervention_rate = training-reference worsened
  challenges / total challenges`.
- `training_reference_unnecessary_intervention_rate = training-reference
  neutral challenges / total challenges`.
- `challenge_yield` is a deprecated alias for
  `training_reference_challenge_yield`; `unnecessary_intervention_rate` is a
  deprecated alias for `training_reference_unnecessary_intervention_rate`.
  New paper-facing harm is always the untouched-holdout
  `harmful_intervention_rate`; historical rows without holdout pairs retain a
  documented compatibility fallback rather than being reinterpreted.
- `challenge_recall` (also exposed as `rescue_recall`) is an optional post-hoc
  diagnostic whose denominator is a training-only empirical-reference
  opportunity: the deterministic alternative is better than the initial plan
  under the configured training CV reference. It never uses holdout outcomes
  to make a runtime decision and is not an oracle claim or true intervention
  recall. The experiment does not evaluate an oracle alternative for every
  non-intervened case, so complete harmful-plan recall is not identifiable.
- `mean_regret_reduction` and `median_regret_reduction` are both reported.
- Catastrophic regret is normalized regret at or above the configurable
  `catastrophic_regret_threshold` (default `0.10`). Prevention is
  catastrophic-to-safe; introduction is safe-to-catastrophic; net prevention
  is prevention minus introduction.
- Abstained disagreements are classified post hoc as `good_abstention`,
  `missed_improvement`, or `neutral_abstention` by comparing the deterministic
  alternative with the original agent choice. This is evaluation-only.

Exact family match, top-2 compatibility, and method agreement remain reported
as secondary diagnostics. They do not determine the primary calibration rank.

## Utility used for development calibration

The transparent utility is the sum of these components:

```text
+ improvement_count * improvement
- worsening_count * worsening
- neutral_intervention_count * neutral_intervention
+ catastrophic_prevented_count * catastrophic_prevention
- catastrophic_introduced_count * catastrophic_introduction
- missed_rescue_count * missed_rescue
```

The versioned default weights are:

```text
improvement=1.0
worsening=2.0
neutral_intervention=0.25
catastrophic_prevention=3.0
catastrophic_introduction=5.0
missed_rescue=1.0
```

They are selected from development/calibration data and frozen before final
evaluation. Reports include each contribution and the total. Final evaluation
never tunes weights or thresholds.

## Aggregation and uncertainty

Summary JSON reports both trial-weighted and dataset-macro gate-health metrics.
Dataset-macro values first summarize all eligible repeated trials within each
dataset/task, then give every dataset/task one equal weight. A clustered
percentile bootstrap samples dataset/task IDs with replacement and retains all
rows belonging to each sampled task, including repeated splits and LLM
repetitions. It is never an IID row bootstrap. Key mean metrics include a
fixed-seed dataset-clustered interval; support below 20 is marked unstable.

Paired ablation effects follow the same independent-unit rule: trials are
paired by case, perturbation, split seed, repetition, and evaluation variant;
paired holdout deltas are averaged within each dataset/task; the headline is
the mean of those dataset effects; and the paired CI resamples one complete
dataset effect at a time. Trial-weighted paired values remain diagnostics only.

The pooled `paper_holdout_delta` is descriptive because classification uses
absolute macro-F1 points while regression uses relative RMSE improvement.
Paper reporting therefore emphasizes challenge/intervention/abstention and
holdout outcome rates, dataset-level win/tie/loss, and separate classification
and regression magnitude summaries with clustered CIs.

Reports include decision-path, deterministic-confidence, empirical-probe-
strength, task/regime, per-dataset, repetition-aware, concentration, and
leave-one-dataset-out diagnostics when the available support permits them.
