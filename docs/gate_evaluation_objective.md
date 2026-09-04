# Gate evaluation objective

The evaluation objective is versioned as `intervention-quality-v1`. It is used
only by the evaluation and policy-development packages. Runtime decisions do
not receive empirical-reference outcomes, holdout scores, regret, or whether a
prior intervention helped.

## Runtime/evaluation boundary

Runtime intervention decisions use training-side validation, cross-validation,
and bounded empirical probes only. Untouched holdout data is never provided to
the gate, probe, abstention logic, reconciliation, prompts, thresholds, or
final-plan selection. The final plan is frozen first; only then are the initial
and final complete plans (including preprocessing) evaluated on the same
untouched holdout split.

## Trial outcomes

The primary intervention outcome is the paired exact-plan holdout result. For
classification, `holdout_intervention_delta` is
`final_holdout_macro_f1 - initial_holdout_macro_f1`. For regression it is
`initial_holdout_rmse - final_holdout_rmse`. Positive always means the
safeguard improved performance. A changed soft plan is `beneficial`, `harmful`,
or `neutral` using the configurable native-metric
`holdout_neutral_tolerance`; an unchanged plan is `not_intervened`, and a
missing/failed pair is `not_comparable`.

The holdout-based intervention precision, harmful-intervention rate, neutral
rate, mean delta, median delta, and valid paired denominator are reported
alongside the training-side diagnostics below.

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

## Training-side diagnostic metrics

- `intervention_precision = improved / (improved + worsened)`. Neutral
  interventions are excluded from this denominator.
- `challenge_yield = improved / total_challenges`.
- `harmful_intervention_rate = worsened / total_challenges`.
- `unnecessary_intervention_rate = neutral / total_challenges`.
- `challenge_recall = beneficial_challenges_made / beneficial_disagreement_opportunities`.
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

Summary JSON reports both trial-weighted and dataset-weighted gate-health
metrics. Dataset-weighted values first summarize repeated trials within each
dataset, then average the dataset summaries. Key mean metrics include a fixed-
seed percentile bootstrap interval; support below 20 is marked unstable.

Reports include decision-path, deterministic-confidence, empirical-probe-
strength, task/regime, per-dataset, repetition-aware, concentration, and
leave-one-dataset-out diagnostics when the available support permits them.
