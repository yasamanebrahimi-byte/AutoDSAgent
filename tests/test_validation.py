import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.pipeline import _validate_before_training, run_analysis
from app.schemas import AgentPlan, ConflictResolution, DeterministicRecommendation
from app.validation import (
    InvariantViolation,
    freeze_supervised_split,
    training_partition_frame,
    validate_training_plan,
)


def _classification_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "signal": rng.normal(size=rows),
            "segment": np.where(np.arange(rows) % 2, "b", "a"),
            "target": np.where(np.arange(rows) % 2, "yes", "no"),
        }
    )


def _regression_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "signal": rng.normal(size=rows),
            "category": np.where(np.arange(rows) % 2, "b", "a"),
            "target": rng.normal(size=rows),
        }
    )


def _check(result, code: str) -> dict:
    return next(check for check in result.as_dict()["checks"] if check["code"] == code)


def test_missing_classification_targets_are_filtered_before_string_encoding():
    frame = _classification_frame()
    frame.loc[[1, 3], "target"] = None

    result = validate_training_plan(frame, "target", "classification", "tree_ensemble")

    assert result.status == "passed"
    assert result.target_rows_removed == 2
    assert "nan" not in _check(result, "classification_target_has_two_classes")["evidence"][
        "class_counts"
    ]
    assert "None" not in _check(result, "classification_target_has_two_classes")["evidence"][
        "class_counts"
    ]


def test_training_partition_frame_fails_closed_for_unreconciled_positions():
    frame = _classification_frame()
    split = freeze_supervised_split(frame, "target", "classification")

    with pytest.raises(InvariantViolation):
        training_partition_frame(frame.iloc[:-1], split, np.arange(len(frame)))
    unknown_positions = np.arange(len(frame))
    unknown_positions[0] = len(frame)
    with pytest.raises(InvariantViolation):
        training_partition_frame(frame, split, unknown_positions)


@pytest.mark.parametrize(
    ("target", "failed_code"),
    [
        (pd.Series(["1", "2", "bad", "4"]), "regression_target_is_numeric_or_coercible"),
        (pd.Series([1.0, 2.0, np.inf, 4.0]), "regression_target_is_finite"),
    ],
)
def test_invalid_regression_targets_fail_closed(target, failed_code):
    frame = pd.DataFrame({"feature": np.arange(len(target), dtype=float), "target": target})

    result = validate_training_plan(frame, "target", "regression", "linear")

    assert result.status == "failed"
    assert _check(result, failed_code)["passed"] is False
    with pytest.raises(InvariantViolation):
        result.raise_if_failed()


def test_single_class_and_small_class_fail_before_cv():
    single = _classification_frame()
    single["target"] = "only"
    result = validate_training_plan(single, "target", "classification", "linear")
    assert _check(result, "classification_target_has_two_classes")["passed"] is False

    small_class = _classification_frame(30)
    small_class["target"] = ["rare"] * 2 + ["common"] * 28
    result = validate_training_plan(small_class, "target", "classification", "linear")
    assert result.status == "failed"
    assert _check(result, "classification_training_supports_cross_validation")["passed"] is False


def test_small_regression_dataset_and_invalid_test_size_fail():
    frame = _regression_frame(3)
    result = validate_training_plan(frame, "target", "regression", "linear")
    assert _check(result, "regression_split_and_cross_validation_feasible")["passed"] is False

    result = validate_training_plan(_regression_frame(), "target", "regression", "linear", test_size=0.01)
    assert _check(result, "test_size_is_supported")["passed"] is False


def test_nonexistent_target_and_target_in_explicit_features_fail():
    frame = _classification_frame()
    missing = validate_training_plan(frame, "missing", "classification", "linear")
    assert _check(missing, "target_exists")["passed"] is False

    present = validate_training_plan(
        frame,
        "target",
        "classification",
        "linear",
        feature_columns=["signal", "target"],
    )
    assert _check(present, "target_absent_from_feature_matrix")["passed"] is False


def test_exact_target_copies_are_detected_across_dtypes():
    frame = _classification_frame()
    frame["target_copy"] = frame["target"]
    result = validate_training_plan(frame, "target", "classification", "tree_ensemble")
    assert result.status == "failed"
    assert "target_copy" in _check(result, "no_direct_target_copy_features")["evidence"][
        "direct_target_copy_features"
    ]

    numeric = _regression_frame()
    numeric["target_copy"] = numeric["target"].map(str)
    result = validate_training_plan(numeric, "target", "regression", "linear")
    assert "target_copy" in _check(result, "no_direct_target_copy_features")["evidence"][
        "direct_target_copy_features"
    ]


def test_identical_features_with_conflicting_targets_fail():
    frame = pd.DataFrame(
        {
            "feature": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
            "noise": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "target": ["yes", "no", "yes", "yes", "no", "no", "yes", "yes", "no", "no"],
        }
    )
    result = validate_training_plan(frame, "target", "classification", "linear")
    assert _check(result, "identical_feature_rows_have_consistent_targets")["passed"] is False


def test_all_features_excluded_are_reported():
    frame = _classification_frame(40)
    frame["id"] = np.arange(len(frame))
    frame["description"] = [f"{index}-" + "x" * 60 for index in range(len(frame))]
    frame["when"] = pd.date_range("2024-01-01", periods=len(frame))
    frame = frame.drop(columns=["signal", "segment"])
    result = validate_training_plan(frame, "target", "classification", "tree_ensemble")
    assert _check(result, "usable_feature_remains")["passed"] is False
    reasons = {item["reason_code"] for item in result.excluded_features}
    assert {"identifier_like", "unsupported_text", "unsupported_datetime"} <= reasons


def test_target_name_indicator_is_advisory_only():
    frame = _classification_frame()
    frame["target_history"] = np.random.default_rng(8).normal(size=len(frame))
    result = validate_training_plan(frame, "target", "classification", "linear")
    assert result.status == "passed"
    assert _check(result, "target_name_leakage_review")["severity"] == "warning"
    assert result.direct_leakage_detected is False


class _ResolutionAgent:
    def __init__(self, resolution: ConflictResolution):
        self.resolution = resolution

    def reconcile(self, question, profile, agent_plan, deterministic):
        return self.resolution


def _agent_plan(target: str, task: str, method: str) -> AgentPlan:
    return AgentPlan(
        target_column=target,
        task_type=task,
        recommended_method=method,
        preprocessing=["training_only_imputation"],
        reasoning="The independent planner selected a compact, inspectable baseline for the supplied schema.",
        confidence=0.7,
    )


def _recommendation(target: str, task: str, method: str) -> DeterministicRecommendation:
    return DeterministicRecommendation(
        target_column=target,
        task_type=task,
        recommended_method=method,
        preprocessing=["training_only_imputation"],
        reasoning="The deterministic policy selected a supported baseline from observed schema evidence.",
        evidence=["rows=40"],
    )


def test_agreement_path_still_runs_deterministic_validation():
    frame = _classification_frame()
    frame["target"] = "only"
    plan = _agent_plan("target", "classification", "linear")
    recommendation = _recommendation("target", "classification", "linear")

    with pytest.raises(InvariantViolation) as error:
        _validate_before_training(
            _ResolutionAgent(
                ConflictResolution(
                    selected_target_column="target",
                    selected_task_type="classification",
                    selected_method="linear",
                    checks=["agreement"],
                    justification="The selected proposal matches the deterministic recommendation and remains inspectable.",
                    confidence=0.9,
                )
            ),
            {"rows": len(frame)},
            "classify target",
            plan,
            recommendation,
            [],
            {},
            offline=False,
            dataframe=frame,
        )
    assert "classification_target_has_two_classes" in str(error.value)


def test_reconciliation_cannot_select_incoherent_pair_or_unproposed_method():
    frame = _classification_frame()
    frame["other_target"] = np.arange(len(frame), dtype=float)
    agent = _agent_plan("target", "classification", "linear")
    deterministic = _recommendation("other_target", "regression", "tree_ensemble")
    incoherent = ConflictResolution(
        selected_target_column="target",
        selected_task_type="regression",
        selected_method="tree_ensemble",
        checks=["pair_checked"],
        justification="The reconciliation agent selected fields from the two proposals for further review.",
        confidence=0.9,
    )
    with pytest.raises(InvariantViolation) as error:
        _validate_before_training(
            _ResolutionAgent(incoherent),
            {"rows": len(frame)},
            "choose a target",
            agent,
            deterministic,
            [],
            {},
            offline=False,
            dataframe=frame,
        )
    assert "regression_target_is_numeric_or_coercible" in str(error.value)

    unsupported = ConflictResolution(
        selected_target_column="target",
        selected_task_type="classification",
        selected_method="boosted_tree",
        checks=["method_checked"],
        justification="The reconciliation agent returned a method that was not among the proposed choices.",
        confidence=0.9,
    )
    with pytest.raises(InvariantViolation) as error:
        _validate_before_training(
            _ResolutionAgent(unsupported),
            {"rows": len(frame)},
            "choose a target",
            agent,
            deterministic,
            [],
            {},
            offline=False,
            dataframe=frame,
        )
    assert "reconciliation_method_is_proposed" in str(error.value)


def test_offline_invalid_run_persists_failure_and_no_model(tmp_path: Path):
    frame = _classification_frame()
    frame["target"] = ["only"] * 39 + ["other"]
    dataset = tmp_path / "invalid.csv"
    frame.to_csv(dataset, index=False)

    with pytest.raises(InvariantViolation):
        run_analysis(
            dataset,
            "classify target",
            target_column="target",
            output_dir=tmp_path / "runs",
            offline=True,
        )

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    decision = json.loads((run_dirs[0] / "decision.json").read_text(encoding="utf-8"))
    assert decision["gate_completed_before_training"] is False
    assert decision["validation"]["overall_status"] == "failed"
    assert decision["failure"]["check_codes"]
    assert not (run_dirs[0] / "model" / "selected_model.joblib").exists()
