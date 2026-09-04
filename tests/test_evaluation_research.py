from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.benchmarks import BenchmarkCase
from evaluation.empirical_reference import evaluate_holdout_plan
from evaluation.metrics import normalized_regret, summarize_trials
from evaluation.runner import run_evaluation
from app.schemas import ModelingPlan, PreprocessingContract
from app.validation import freeze_supervised_split


def _case() -> BenchmarkCase:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "signal": [float(index) for index in range(48)],
            "noise": [float(index % 5) for index in range(48)],
            "target": ["yes" if index % 2 else "no" for index in range(48)],
        }
    )
    return BenchmarkCase(
        name="research_fixture",
        dataframe=frame,
        target_column="target",
        question="Classify target from the supplied features.",
        expected_task_type="classification",
        dataset_source="in-memory test fixture",
    )


def test_repetitions_reuse_identical_evidence_and_cached_reference(tmp_path: Path):
    output = tmp_path / "evaluation"
    run_evaluation(output, cases=[_case()], repetitions=3, offline=True, seed=19)
    rows = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]

    assert len(rows) == 3
    assert len({row["split_seed"] for row in rows}) == 1
    assert len({json.dumps(row["split_contract"], sort_keys=True) for row in rows}) == 1
    assert len({json.dumps(row["agent_initial_input"], sort_keys=True) for row in rows}) == 1
    assert len({row["empirical_reference_cache_key"] for row in rows}) == 1
    assert all(row["holdout_policy"]["used_for_agent_planning"] is False for row in rows)


def test_holdout_evaluation_aligns_masks_after_invalid_target_rows_are_removed():
    frame = _case().load()
    frame.loc[3, "target"] = None
    split = freeze_supervised_split(frame, "target", "classification", random_state=19)

    result = evaluate_holdout_plan(
        frame,
        split,
        "target",
        "classification",
        "linear",
        PreprocessingContract(numeric_scaling="standard"),
        random_state=19,
        row_positions=list(range(len(frame))),
    )

    assert result["status"] == "evaluated"


def test_initial_agent_factory_context_excludes_downstream_decisions(tmp_path: Path):
    captured: list[dict] = []

    def plan_factory(context):
        captured.append(context)
        return ModelingPlan(
            recommended_method="linear",
            preprocessing=PreprocessingContract(numeric_scaling="standard"),
            reasoning="The initial agent selects a simple baseline from the training-only schema evidence.",
            confidence=0.7,
        )

    run_evaluation(tmp_path / "evaluation", cases=[_case()], repetitions=2, modeling_plan_factory=plan_factory)

    assert len(captured) == 2
    for context in captured:
        assert "deterministic_recommendation" not in context
        assert "empirical_reference" not in context
        assert "previous_repetitions" not in context
        assert context["training_profile"]["rows"] < 48


def test_openai_only_metrics_exclude_fallback_mock_and_failed_rows():
    base = {
        "task_type": "classification",
        "perturbation_id": "clean",
        "agent_initial_valid": True,
        "final_valid": True,
        "agreement_status": "agreement",
        "method_disagreement": False,
        "agent_initial_method": "linear",
        "final_method": "linear",
        "empirical_best_method": "linear",
        "agent_normalized_regret": 0.0,
        "gated_normalized_regret": 0.0,
        "paired_cv_improvement": 0.0,
    }
    trials = [
        {**base, "agent_source": "openai", "trial_id": "openai"},
        {**base, "agent_source": "offline_fallback", "trial_id": "fallback"},
        {**base, "agent_source": "mock", "trial_id": "mock"},
        {"trial_id": "failed", "trial_status": "failed", "agent_source": "failed"},
    ]

    summary = summarize_trials(trials)

    assert summary["requested_live_trials"] == 0
    assert summary["successful_openai_trials"] == 1
    assert summary["offline_fallback_trials"] == 1
    assert summary["mock_trials"] == 1
    assert summary["failed_trials"] == 1
    assert summary["openai_only"]["completed_trial_count"] == 1
    assert summary["openai_only_match_rates"]["initial_reference_match_rate"] == 1.0


def test_llm_only_stale_reconciliation_flag_is_not_counted_as_an_invocation():
    base = {
        "trial_status": "completed",
        "agent_source": "openai",
        "task_type": "classification",
        "perturbation_id": "clean",
        "agent_initial_valid": True,
        "final_valid": True,
        "agent_initial_method": "linear",
        "final_method": "linear",
        "empirical_best_method": "linear",
        "agent_normalized_regret": 0.0,
        "gated_normalized_regret": 0.0,
        "paired_cv_improvement": 0.0,
    }
    llm_only_summary = summarize_trials(
        [
            {
                **base,
                "agreement_status": "llm_only",
                "method_disagreement": True,
                "reconciliation_invoked": True,
                "reconciliation_status": "not_invoked",
            }
        ]
    )
    actual_reconciliation_summary = summarize_trials(
        [
            {
                **base,
                "agreement_status": "disagreement",
                "method_disagreement": True,
                "reconciliation_invoked": True,
                "reconciliation_status": "succeeded",
                "reconciliation_method_source": "agent",
            }
        ]
    )

    assert llm_only_summary["reconciliation_invocation_rate"] == 0.0
    assert llm_only_summary["reconciliation_success_rate"] is None
    assert llm_only_summary["openai_only_reconciliation_invocation_rate"] == 0.0
    assert actual_reconciliation_summary["reconciliation_invocation_rate"] == 1.0
    assert actual_reconciliation_summary["reconciliation_success_rate"] == 1.0


def test_regret_direction_and_gate_outcome_tolerance_are_explicit():
    assert normalized_regret("classification", 0.8, 0.7) == pytest.approx(0.1)
    assert normalized_regret("classification", 0.8, 0.9) == 0.0
    assert normalized_regret("regression", 0.0, 0.0) == 0.0
    assert normalized_regret("regression", 2.0, 2.4) == pytest.approx(0.2)

    records = [
        {
            "task_type": "regression",
            "perturbation_id": "clean",
            "agent_source": "openai",
            "agent_initial_valid": True,
            "final_valid": True,
            "agent_initial_method": "linear",
            "final_method": "tree_ensemble",
            "empirical_best_method": "tree_ensemble",
            "agent_normalized_regret": 0.2,
            "gated_normalized_regret": 0.0,
            "paired_cv_improvement": 1.0,
            "gate_outcome": "improved",
        },
        {
            "task_type": "regression",
            "perturbation_id": "clean",
            "agent_source": "openai",
            "agent_initial_valid": True,
            "final_valid": True,
            "agent_initial_method": "tree_ensemble",
            "final_method": "linear",
            "empirical_best_method": "tree_ensemble",
            "agent_normalized_regret": 0.0,
            "gated_normalized_regret": 0.2,
            "paired_cv_improvement": -1.0,
            "gate_outcome": "worsened",
        },
    ]
    summary = summarize_trials(records)
    assert summary["openai_only_paired_stats"]["improved_count"] == 1
    assert summary["openai_only_paired_stats"]["worsened_count"] == 1
    assert summary["openai_only_paired_stats"]["tie_count"] == 0
    assert summary["openai_only_paired_stats"]["mean_paired_improvement"] == 0.0


def test_empirical_reference_is_called_after_the_runtime_gate(monkeypatch, tmp_path: Path):
    import evaluation.runner as runner

    events: list[str] = []
    original_gate = runner._validate_modeling_gate
    original_reference = runner.evaluate_empirical_reference

    def gate(*args, **kwargs):
        events.append("gate")
        return original_gate(*args, **kwargs)

    def reference(*args, **kwargs):
        events.append("reference")
        return original_reference(*args, **kwargs)

    monkeypatch.setattr(runner, "_validate_modeling_gate", gate)
    monkeypatch.setattr(runner, "evaluate_empirical_reference", reference)
    run_evaluation(tmp_path / "evaluation", cases=[_case()], offline=True)

    assert events == ["gate", "reference"]
