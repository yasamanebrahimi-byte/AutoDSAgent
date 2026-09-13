from __future__ import annotations

import pytest

import evaluation.ablation as ablation_module
import evaluation.metrics as metrics_module
from evaluation.ablation import _paired_comparison
from evaluation.metrics import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    HOLDOUT_METRIC_SCHEMA_VERSION,
    HOLDOUT_RMSE_EPSILON,
    classify_holdout_intervention_outcome,
    paper_holdout_delta,
    relative_rmse_improvement,
    summarize_gate_health,
    summarize_trials,
)
from evaluation.statistics import cluster_bootstrap_ci


def _holdout_record(
    dataset: str,
    task_type: str,
    initial: float,
    final: float,
    *,
    occurred: bool = True,
    repetitions: int = 1,
) -> dict:
    return {
        "benchmark_case": dataset,
        "task_type": task_type,
        "trial_status": "completed",
        "intervention_occurred": occurred,
        "initial_holdout_metric": initial,
        "final_holdout_metric": final,
        "holdout_metric_name": "macro_f1" if task_type == "classification" else "rmse",
        "paper_holdout_delta": paper_holdout_delta(task_type, initial, final),
        "repetitions_fixture": repetitions,
    }


def test_paper_holdout_delta_sign_and_regression_scale_are_explicit():
    assert paper_holdout_delta("classification", 0.60, 0.70) == pytest.approx(0.10)
    assert paper_holdout_delta("classification", 0.70, 0.60) == pytest.approx(-0.10)
    assert relative_rmse_improvement(100.0, 90.0) == pytest.approx(0.10)
    assert relative_rmse_improvement(100.0, 110.0) == pytest.approx(-0.10)


def test_zero_initial_rmse_uses_explicit_epsilon_without_division_error():
    assert relative_rmse_improvement(0.0, 0.0) == pytest.approx(0.0)
    assert relative_rmse_improvement(0.0, HOLDOUT_RMSE_EPSILON) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "delta, expected",
    [(0.02, "neutral"), (0.020001, "beneficial"), (-0.02, "neutral"), (-0.020001, "harmful")],
)
def test_holdout_neutrality_boundaries_are_inclusive(delta: float, expected: str):
    assert classify_holdout_intervention_outcome(
        delta, 0.02, intervention_occurred=True
    ) == expected


def test_regression_neutrality_uses_relative_not_native_rmse_units():
    records = [
        _holdout_record("regression", "regression", 100.0, 97.0),
        _holdout_record("regression", "regression", 10.0, 10.15),
    ]
    summary = summarize_gate_health(
        records,
        holdout_tolerances={"classification": 0.02, "regression": 0.02},
    )
    holdout = summary["holdout_intervention_metrics"]
    assert holdout["beneficial_intervention_count"] == 1
    assert holdout["neutral_intervention_count"] == 1
    assert holdout["harmful_intervention_count"] == 0


def test_training_reference_harm_and_holdout_harm_are_distinct_metrics():
    records = []
    for index, (initial, final) in enumerate(((0.20, 0.00), (0.00, 0.20), (0.05, 0.045))):
        record = {
            "benchmark_case": "metric-separation",
            "task_type": "classification",
            "trial_status": "completed",
            "agreement_status": "disagreement",
            "method_disagreement": True,
            "soft_challenge": {"status": "disagreement", "decision": "challenge"},
            "agent_initial_valid": True,
            "intervention_occurred": True,
            "agent_normalized_regret": initial,
            "gated_normalized_regret": final,
            "initial_holdout_metric": 0.60,
            "final_holdout_metric": 0.70,
            "paper_holdout_delta": 0.10,
            "trial": index,
        }
        records.append(record)

    health = summarize_gate_health(records)
    assert health["training_reference_challenge_yield"] == pytest.approx(1 / 3)
    assert health["training_reference_harmful_intervention_rate"] == pytest.approx(1 / 3)
    assert health["harmful_intervention_rate"] == pytest.approx(0.0)
    assert health["holdout_intervention_metrics"]["harmful_intervention_rate"] == pytest.approx(0.0)


def test_dataset_macro_holdout_estimate_does_not_weight_extra_repetitions():
    records = [
        _holdout_record("small", "classification", 0.60, 0.70),
        *[_holdout_record("large", "classification", 0.70, 0.60) for _ in range(9)],
    ]
    summary = summarize_gate_health(records)
    # Direct gate-health is trial-level; use the same public aggregator used
    # by paper reports to verify the equal-weighted dataset estimate.
    from evaluation.metrics import summarize_trials

    paper_summary = summarize_trials(records)
    assert paper_summary["dataset_macro_paper_holdout_delta_mean"] == pytest.approx(0.0)
    assert paper_summary["paper_holdout_delta_mean"] == pytest.approx((0.10 - 9 * 0.10) / 10)
    assert summary["holdout_intervention_metrics"]["valid_paired_holdout_comparison_count"] == 10


def test_ablation_primary_pairing_uses_holdout_delta_and_keeps_regret_diagnostic():
    base = {
        "trial_status": "completed",
        "benchmark_case": "paired",
        "task_type": "classification",
        "perturbation_id": "clean",
        "split_seed": 42,
        "trial": 0,
        "evaluation_variant": "standard",
    }
    rows = {
        "first": [{
            **base,
            "paper_holdout_delta": 0.10,
            "gated_normalized_regret": 0.40,
        }],
        "second": [{
            **base,
            "paper_holdout_delta": -0.10,
            "gated_normalized_regret": 0.00,
        }],
    }
    result = _paired_comparison(rows, "first", "second")
    assert result["first_better"] == 1
    assert result["second_better"] == 0
    assert result["mean_paired_holdout_delta_difference_first_advantage"] == pytest.approx(0.20)
    assert result["training_reference_comparison_role"].startswith("secondary")
    assert result["mean_paired_regret_difference_first_advantage"] == pytest.approx(-0.40)


def test_ablation_pairing_is_dataset_macro_with_unequal_repetitions():
    def row(dataset: str, trial: int, delta: float) -> dict:
        return {
            "trial_status": "completed",
            "benchmark_case": dataset,
            "task_type": "classification",
            "perturbation_id": "clean",
            "split_seed": 42,
            "trial": trial,
            "evaluation_variant": "standard",
            "paper_holdout_delta": delta,
        }

    rows = {
        "first": [row("A", trial, 0.10) for trial in range(9)] + [row("B", 0, -0.10)],
        "second": [row("A", trial, 0.00) for trial in range(9)] + [row("B", 0, 0.00)],
    }
    result = _paired_comparison(rows, "first", "second", tolerance=0.02)

    assert result["n_paired_datasets"] == 2
    assert result["mean_paired_holdout_delta_difference_first_advantage"] == pytest.approx(0.0)
    assert result["median_paired_holdout_delta_difference_first_advantage"] == pytest.approx(0.0)
    assert result["trial_weighted_mean_paired_holdout_delta_difference_first_advantage"] == pytest.approx(0.08)
    assert (result["first_better"], result["second_better"], result["tied"]) == (1, 1, 0)
    assert result["paired_holdout_delta_ci"]["n_clusters"] == 2
    expected_ci = cluster_bootstrap_ci(
        result["paired_holdout_dataset_effects"],
        lambda sample: sum(item["difference"] for item in sample) / len(sample) if sample else None,
        "benchmark_case",
    )
    assert result["paired_holdout_delta_ci"] == expected_ci
    assert [item["paired_trial_count"] for item in result["paired_holdout_dataset_effects"]] == [9, 1]


def test_descriptive_paired_comparison_skips_bootstrap_but_preserves_point_estimates(
    monkeypatch,
):
    base = {
        "trial_status": "completed",
        "benchmark_case": "paired",
        "task_type": "classification",
        "perturbation_id": "clean",
        "split_seed": 42,
        "trial": 0,
        "evaluation_variant": "standard",
        "paper_holdout_delta": 0.10,
        "gated_normalized_regret": 0.40,
        "final_holdout_metric": 0.70,
    }
    rows = {
        "first": [base],
        "second": [{**base, "paper_holdout_delta": -0.10, "gated_normalized_regret": 0.00}],
    }
    primary = _paired_comparison(rows, "first", "second")
    assert primary["paired_holdout_delta_ci"]["n_bootstrap"] == DEFAULT_BOOTSTRAP_REPLICATES
    assert primary["paired_regret_difference_ci"]["n_bootstrap"] == DEFAULT_BOOTSTRAP_REPLICATES

    bootstrap_calls = []

    def fake_cluster_bootstrap_ci(*args, **kwargs):
        bootstrap_calls.append((args, kwargs))
        raise AssertionError("descriptive paired comparisons must not bootstrap")

    monkeypatch.setattr(ablation_module, "cluster_bootstrap_ci", fake_cluster_bootstrap_ci)
    descriptive = _paired_comparison(
        rows,
        "first",
        "second",
        compute_confidence_intervals=False,
    )

    assert bootstrap_calls == []
    for key in (
        "paired_units",
        "n_paired_datasets",
        "first_better",
        "second_better",
        "tied",
        "mean_paired_holdout_delta_difference_first_advantage",
        "mean_paired_regret_difference_first_advantage",
    ):
        assert descriptive[key] == primary[key]
    assert descriptive["paired_holdout_delta_ci"] == {}
    assert descriptive["paired_regret_difference_ci"] == {}
    assert descriptive["trial_weighted_paired_regret_difference_ci"] == {}
    descriptive_final = _paired_comparison(
        rows,
        "first",
        "second",
        comparison_estimand="final_plan_holdout_performance",
        compute_confidence_intervals=False,
    )
    assert descriptive_final["paired_final_holdout_performance_ci"] == {}
    assert descriptive_final["paired_holdout_delta_ci"] is None
    assert set(descriptive) >= {
        "paired_holdout_delta_ci",
        "paired_regret_difference_ci",
        "trial_weighted_paired_regret_difference_ci",
    }


def test_ablation_pairing_win_tie_loss_is_dataset_level():
    base = {
        "trial_status": "completed",
        "task_type": "classification",
        "perturbation_id": "clean",
        "split_seed": 42,
        "evaluation_variant": "standard",
    }
    rows = {
        "first": [
            {**base, "benchmark_case": "better", "trial": 0, "paper_holdout_delta": 0.05},
            {**base, "benchmark_case": "tie", "trial": 0, "paper_holdout_delta": 0.01},
            {**base, "benchmark_case": "worse", "trial": 0, "paper_holdout_delta": -0.05},
        ],
        "second": [
            {**base, "benchmark_case": "better", "trial": 0, "paper_holdout_delta": 0.00},
            {**base, "benchmark_case": "tie", "trial": 0, "paper_holdout_delta": 0.00},
            {**base, "benchmark_case": "worse", "trial": 0, "paper_holdout_delta": 0.00},
        ],
    }
    result = _paired_comparison(rows, "first", "second", tolerance=0.02)
    assert (result["first_better"], result["second_better"], result["tied"]) == (1, 1, 1)


def test_summary_exposes_versioned_paper_fields_and_marks_strict_failures():
    completed = _holdout_record("classification", "classification", 0.60, 0.70)
    completed.update({
        "require_live": True,
        "requested_live_trial": True,
        "agent_source": "openai",
        "fallback_row": False,
        "planner_model_effective": "planner-id",
        "reconciler_model_effective": "reconciler-id",
    })
    failed = {**completed, "trial_status": "failed", "agent_source": "failed"}
    summary = summarize_trials([completed, failed])

    assert summary["strict_live_valid"] is False
    assert summary["result_schema_version"] == HOLDOUT_METRIC_SCHEMA_VERSION
    assert summary["paper_holdout_delta_mean"] == pytest.approx(0.10)
    assert summary["harm_rate"] == pytest.approx(0.0)
    assert "dataset_macro_confidence_intervals" in summary["paper_metrics_by_task"]["classification"]


def test_summarize_trials_default_computes_dataset_cluster_confidence_intervals():
    summary = summarize_trials([_holdout_record("single", "classification", 0.60, 0.70)])

    dataset_ci = summary["dataset_macro_gate_health"]["confidence_intervals"]
    assert dataset_ci["mean_paper_holdout_delta"]["n_bootstrap"] == DEFAULT_BOOTSTRAP_REPLICATES
    assert (
        summary["paper_metrics_by_task"]["classification"]
        ["dataset_macro_confidence_intervals"]["mean_paper_holdout_delta"]["n_bootstrap"]
        == DEFAULT_BOOTSTRAP_REPLICATES
    )


def test_summarize_trials_can_skip_all_bootstrap_confidence_intervals(monkeypatch):
    records = [
        _holdout_record("first", "classification", 0.60, 0.70),
        _holdout_record("second", "classification", 0.70, 0.60),
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("bootstrap CI machinery must not run for checkpoint summaries")

    monkeypatch.setattr(metrics_module, "_dataset_macro_ci", fail_if_called)
    monkeypatch.setattr(metrics_module, "cluster_bootstrap_ci", fail_if_called)
    monkeypatch.setattr(metrics_module, "cluster_bootstrap_multi_ci", fail_if_called)

    summary = summarize_trials(records, compute_confidence_intervals=False)

    assert summary["dataset_macro_gate_health"]["confidence_intervals"] == {}
    assert summary["dataset_macro_paper_holdout_delta_ci"] == {}
    assert summary["regret_reduction_ci"] == {}
    assert summary["paper_metrics_by_task"]["classification"]["dataset_macro_confidence_intervals"] == {}
    assert summary["dataset_macro_paper_holdout_delta_mean"] == pytest.approx(0.0)


def test_final_dataset_macro_cis_use_one_batched_pass_per_population(monkeypatch):
    records = [
        _holdout_record("classification-a", "classification", 0.60, 0.70),
        _holdout_record("classification-b", "classification", 0.70, 0.60),
        _holdout_record("regression-a", "regression", 100.0, 90.0),
        _holdout_record("regression-b", "regression", 100.0, 110.0),
    ]
    batched_calls = []
    single_metric_calls = []
    original_multi_ci = metrics_module.cluster_bootstrap_multi_ci

    def multi_ci_spy(data, statistic_fns, cluster_col, **kwargs):
        batched_calls.append((len(data), tuple(statistic_fns), cluster_col))
        return original_multi_ci(
            data, statistic_fns, cluster_col, n_bootstrap=7, **kwargs
        )

    def single_ci_spy(*args, **kwargs):
        single_metric_calls.append((args, kwargs))
        return {
            "lower": None,
            "upper": None,
            "ci_low": None,
            "ci_high": None,
            "support": 0,
            "n_clusters": 0,
            "n_bootstrap": 10_000,
            "confidence_level": 0.95,
            "uncertainty_method": "dataset_cluster_bootstrap_percentile",
            "cluster_column": "benchmark_case",
            "stable": False,
            "status": "unavailable",
        }

    def fail_if_old_dataset_macro_ci(*args, **kwargs):
        raise AssertionError("final dataset-macro CIs must use the batched helper")

    monkeypatch.setattr(metrics_module, "cluster_bootstrap_multi_ci", multi_ci_spy)
    monkeypatch.setattr(metrics_module, "cluster_bootstrap_ci", single_ci_spy)
    monkeypatch.setattr(metrics_module, "_dataset_macro_ci", fail_if_old_dataset_macro_ci)

    summary = summarize_trials(records, include_model_condition_breakdown=False)

    assert len(batched_calls) == 3
    assert [len(call[1]) for call in batched_calls] == [23, 6, 6]
    assert all(call[2] == "benchmark_case" for call in batched_calls)
    assert single_metric_calls  # Existing paired/holdout summaries remain unchanged.
    assert set(summary["dataset_macro_gate_health"]["confidence_intervals"]) >= {
        "mean_regret_reduction",
        "mean_paper_holdout_delta",
    }
    assert set(summary["paper_metrics_by_task"]["classification"]["dataset_macro_confidence_intervals"]) == {
        "beneficial_intervention_rate",
        "harmful_intervention_rate",
        "neutral_intervention_rate",
        "intervention_precision",
        "harm_rate",
        "mean_paper_holdout_delta",
    }


def test_historical_classification_delta_is_read_without_reusing_regression_units():
    historical_classification = {
        "benchmark_case": "historical-classification",
        "task_type": "classification",
        "trial_status": "completed",
        "intervention_occurred": True,
        "holdout_intervention_delta": 0.05,
    }
    historical_regression = {
        "benchmark_case": "historical-regression",
        "task_type": "regression",
        "trial_status": "completed",
        "intervention_occurred": True,
        "holdout_intervention_delta": 5.0,
    }

    classification = summarize_gate_health([historical_classification])
    regression = summarize_gate_health([historical_regression])
    assert classification["holdout_intervention_metrics"]["beneficial_intervention_count"] == 1
    assert regression["holdout_intervention_metrics"]["valid_paired_holdout_comparison_count"] == 0
