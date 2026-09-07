from __future__ import annotations

import pandas as pd
import pytest

import evaluation.runner as runner
from app.preprocessing import build_preprocessor
from app.schemas import PreprocessingContract
from app.validation import (
    build_execution_contract,
    validate_training_plan,
)
from evaluation.ablation import _paired_comparison
from evaluation.benchmarks import BenchmarkCase
from evaluation.empirical_reference import evaluate_plan_cv


def _classification_case(*, random_seed: int = 42) -> BenchmarkCase:
    rows = 48
    return BenchmarkCase(
        name="confirmatory-v2-fixture",
        dataframe=pd.DataFrame(
            {
                "numeric": list(range(rows)),
                "category": ["a", "b", "c"] * (rows // 3),
                "target": ["yes", "no"] * (rows // 2),
            }
        ),
        target_column="target",
        question="Classify the target using the observed training data.",
        expected_task_type="classification",
        dataset_source="test",
        random_seed=random_seed,
    )


def _regression_frame(rows: int = 48) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric": list(range(rows)),
            "category": ["a", "b", "c"] * (rows // 3),
            "target": [float(index * 0.75 + (index % 3)) for index in range(rows)],
        }
    )


def _shared_row(*, task_type: str, initial: float, final: float, delta: float) -> dict:
    return {
        "benchmark_case": f"{task_type}-fixture",
        "perturbation_id": "clean",
        "split_seed": 42,
        "trial": 0,
        "evaluation_variant": "standard",
        "model_condition_id": "default",
        "llm_repetition_id": "rep_001",
        "trial_status": "completed",
        "task_type": task_type,
        "initial_holdout_metric": initial,
        "final_holdout_metric": final,
        "paper_holdout_delta": delta,
    }


def test_deterministic_only_row_uses_one_coherent_deterministic_plan(tmp_path):
    row = runner.run_evaluation(
        tmp_path / "deterministic-only",
        cases=[_classification_case()],
        gate_mode="deterministic_only",
        offline=False,
        modeling_plan_factory=lambda _context: pytest.fail("deterministic_only requested an LLM plan"),
    )["trials"][0]

    assert row["agent_source"] == "deterministic_only"
    assert row["initial_modeling_call_made"] is False
    assert row["final_selection_source"] == "deterministic"
    assert row["agent_initial_method"] == row["final_method"] == row["deterministic_method"]
    assert row["agent_initial_preprocessing"] == row["final_preprocessing"] == row["deterministic_preprocessing"]
    assert row["agent_initial_valid"] is True
    assert row["final_valid"] is True
    assert row["agent_initial_validation"]["status"] == row["final_validation"]["status"] == "passed"
    assert row["hard_validation_status"] == "passed"
    assert row["paper_holdout_delta"] == 0.0


def test_primary_planner_is_called_before_deterministic_recommendation(tmp_path, monkeypatch):
    events: list[str] = []
    captured: dict = {}
    original = runner.deterministic_recommendation

    def wrapped(*args, **kwargs):
        events.append("deterministic")
        return original(*args, **kwargs)

    def planner(context):
        events.append("planner")
        captured.update(context)
        return runner._fallback_modeling_plan(
            context["training_profile"],
            context["question"],
            context["target_column"],
            context["task_type"],
        )

    monkeypatch.setattr(runner, "deterministic_recommendation", wrapped)
    runner.run_evaluation(
        tmp_path / "ordering",
        cases=[_classification_case()],
        gate_mode="llm_only",
        modeling_plan_factory=planner,
    )

    assert events[:2] == ["planner", "deterministic"]
    assert "deterministic_recommendation" not in captured
    assert "deterministic_method" not in captured
    assert captured["execution_contract"] is not None


def test_nonshared_baselines_use_final_plan_performance_not_intervention_delta():
    deterministic = _shared_row(task_type="classification", initial=0.80, final=0.80, delta=0.0)
    hard_validation = _shared_row(task_type="classification", initial=0.70, final=0.60, delta=-0.10)
    comparison = _paired_comparison(
        {"deterministic_only": [deterministic], "hard_validation_only": [hard_validation]},
        "deterministic_only",
        "hard_validation_only",
        tolerance={"classification": 0.02, "regression": 0.02},
    )

    assert comparison["comparison_estimand"] == "final_plan_holdout_performance"
    assert comparison["mean_paired_holdout_delta_difference_first_advantage"] is None
    assert comparison["mean_paired_final_holdout_performance_difference_first_advantage"] == pytest.approx(0.20)
    assert comparison["first_better"] == 1


def test_final_plan_comparison_sign_is_direction_normalized_for_regression():
    deterministic = _shared_row(task_type="regression", initial=2.0, final=2.0, delta=0.0)
    hard_validation = _shared_row(task_type="regression", initial=2.5, final=3.0, delta=-0.2)
    comparison = _paired_comparison(
        {"deterministic_only": [deterministic], "hard_validation_only": [hard_validation]},
        "deterministic_only",
        "hard_validation_only",
        tolerance={"classification": 0.02, "regression": 0.02},
    )

    assert comparison["comparison_estimand"] == "final_plan_holdout_performance"
    assert comparison["mean_paired_final_holdout_performance_difference_first_advantage"] == pytest.approx(1 / 3)
    assert comparison["first_better"] == 1


@pytest.mark.parametrize(
    ("task_type", "frame"),
    [
        ("classification", _classification_case().load()),
        ("regression", _regression_frame()),
    ],
)
def test_contract_declared_executable_plans_execute(task_type: str, frame: pd.DataFrame):
    contract = build_execution_contract(frame, "target", task_type, random_state=42)
    for method, details in contract["model_families"].items():
        if not details["executable_for_dataset"]:
            continue
        preprocessing = PreprocessingContract.model_validate(
            details["compatible_contract_for_observed_schema"]
        )
        validation = validate_training_plan(
            frame,
            "target",
            task_type,
            method,
            preprocessing=preprocessing,
            random_state=42,
            training_only=True,
        )
        assert validation.status == "passed", (method, validation.as_dict())
        execution = evaluate_plan_cv(
            frame,
            "target",
            task_type,
            method,
            preprocessing,
            random_state=42,
        )
        assert execution["status"] == "evaluated", (method, execution)


def test_contract_rejects_illegal_boosted_tree_categorical_encoding():
    frame = _classification_case().load()
    illegal = PreprocessingContract(
        numeric_imputation="none",
        categorical_imputation="none",
        numeric_scaling="none",
        categorical_encoding="one_hot",
        categorical_unknown_handling="ignore",
        identifier_handling="exclude",
        high_cardinality_handling="exclude",
        unsupported_text_handling="exclude",
        datetime_handling="exclude",
        infinity_handling="replace_with_missing",
        fit_inside_pipeline=True,
    )
    validation = validate_training_plan(
        frame,
        "target",
        "classification",
        "boosted_tree",
        preprocessing=illegal,
        random_state=42,
        training_only=True,
    )
    assert validation.status == "failed"
    assert any(check.code == "boosted_tree_encoding_is_compatible" for check in validation.failed_checks)
    with pytest.raises(ValueError, match="Boosted trees require ordinal"):
        build_preprocessor(illegal, ["numeric"], ["category"], "boosted_tree")


def test_external_case_seed_does_not_offset_effective_split_random_state(tmp_path):
    row = runner.run_evaluation(
        tmp_path / "split-seed",
        cases=[_classification_case(random_seed=42)],
        gate_mode="deterministic_only",
        offline=True,
        split_seeds=[42],
    )["trials"][0]

    assert row["split_seed"] == 42
    assert row["case_random_seed"] == 42
    assert row["split_random_state"] == 42
    assert row["split_contract"]["random_state"] == 42
