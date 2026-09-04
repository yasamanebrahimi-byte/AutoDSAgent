import pytest

from evaluation.metrics import (
    classify_holdout_intervention_outcome,
    holdout_intervention_delta,
    summarize_gate_health,
)


def _record(task_type, initial, final, *, occurred=True):
    metric = "macro_f1" if task_type == "classification" else "rmse"
    return {
        "trial_status": "completed",
        "task_type": task_type,
        "benchmark_case": "holdout-fixture",
        "intervention_occurred": occurred,
        "initial_holdout_metric": initial,
        "final_holdout_metric": final,
        "holdout_metric_name": metric,
        "holdout_intervention_delta": holdout_intervention_delta(task_type, initial, final),
    }


def test_signed_holdout_delta_and_outcomes_for_classification_and_regression():
    assert holdout_intervention_delta("classification", 0.70, 0.75) == pytest.approx(0.05)
    assert holdout_intervention_delta("regression", 10, 8) == pytest.approx(2)
    assert classify_holdout_intervention_outcome(0.05, 0.01, intervention_occurred=True) == "beneficial"
    assert classify_holdout_intervention_outcome(-2, 0.01, intervention_occurred=True) == "harmful"


def test_holdout_health_uses_changed_soft_plans_and_explicit_valid_denominator():
    records = [
        _record("classification", 0.70, 0.75),
        _record("classification", 0.75, 0.70),
        _record("regression", 10, 10.005),
        _record("classification", 0.70, 0.70, occurred=False),
        {"trial_status": "completed", "intervention_occurred": True},
    ]
    health = summarize_gate_health(records, neutral_tolerance=0.02)
    assert health["intervention_precision"] == pytest.approx(1 / 3)
    assert health["harmful_intervention_rate"] == pytest.approx(1 / 3)
    assert health["holdout_intervention_metrics"]["valid_paired_holdout_comparison_count"] == 3
    assert health["holdout_intervention_metrics"]["missing_or_failed_holdout_count"] == 1
    assert health["holdout_intervention_metrics"]["neutral_intervention_count"] == 1


def test_reconciliation_that_preserves_initial_plan_is_not_intervention():
    record = _record("classification", 0.70, 0.90, occurred=False)
    record.update({
        "reconciliation_invoked": True,
        "reconciliation_status": "succeeded",
        "proceeded_unchanged": True,
        "gate_changed_initial_plan": False,
    })
    health = summarize_gate_health([record])
    assert health["holdout_intervention_metrics"]["intervention_count"] == 0
    assert health["holdout_intervention_metrics"]["valid_paired_holdout_comparison_count"] == 0

