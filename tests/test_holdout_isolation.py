from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.deterministic import (
    deterministic_recommendation,
    eda_summary,
    fit_cleaning_spec,
    profile_dataframe,
    transform_cleaning,
)
from app.pipeline import _validate_before_training, run_analysis
from app.schemas import AgentPlan, ConflictResolution, DeterministicRecommendation
from app.validation import (
    freeze_supervised_split,
    training_partition_frame,
    training_profile_frame,
    validate_training_plan,
)


def _frame(rows: int = 80) -> pd.DataFrame:
    positions = np.arange(rows)
    return pd.DataFrame(
        {
            "signal": positions.astype(float),
            "segment": np.where(positions % 2, "b", "a"),
            "region": np.where(positions % 3, "west", "east"),
            "target": np.where(positions % 2, "yes", "no"),
        }
    )


def _read_run(run_dir: str | Path) -> tuple[dict, dict, dict]:
    run_path = Path(run_dir)
    decision = json.loads((run_path / "decision.json").read_text(encoding="utf-8"))
    planning = json.loads((run_path / "planning_profile.json").read_text(encoding="utf-8"))
    modeling = json.loads((run_path / "modeling.json").read_text(encoding="utf-8"))
    return decision, planning, modeling


def test_explicit_target_uses_the_frozen_training_rows_for_planning_and_fit(tmp_path: Path):
    dataset = tmp_path / "explicit.csv"
    frame = _frame()
    frame.to_csv(dataset, index=False)

    result = run_analysis(
        dataset,
        "Classify target from the measured features.",
        target_column="target",
        output_dir=tmp_path / "runs",
        offline=True,
    )
    decision, planning, modeling = _read_run(result["run_dir"])
    contract = decision["split_contract"]

    assert decision["target_establishment"]["target_source"] == "user_supplied"
    assert planning["rows"] == contract["train_rows"]
    assert modeling["train_rows"] == contract["train_rows"]
    assert modeling["test_rows"] == contract["holdout_rows"]
    assert modeling["split_evidence"]["contract"] == contract
    assert decision["holdout_policy"] == {
        "frozen_before_modeling_recommendations": True,
        "planning_data": "training_partition_only",
        "holdout_used_for": "final_evaluation_only",
    }


def test_inferred_target_is_established_before_the_training_only_profile(tmp_path: Path):
    dataset = tmp_path / "inferred.csv"
    _frame().to_csv(dataset, index=False)

    result = run_analysis(
        dataset,
        "Please classify target using the available measurements.",
        output_dir=tmp_path / "runs",
        offline=True,
    )
    decision, planning, modeling = _read_run(result["run_dir"])

    assert decision["target_establishment"]["target_column"] == "target"
    assert decision["target_establishment"]["completed_before_holdout_freeze"] is True
    assert planning["rows"] == decision["split_contract"]["train_rows"]
    assert modeling["split_evidence"]["contract"] == decision["split_contract"]


def test_classification_and_regression_reuse_the_same_frozen_membership():
    classification = _frame()
    classification_split = freeze_supervised_split(
        classification, "target", "classification", test_size=0.2, random_state=17
    )
    classification_validation = validate_training_plan(
        classification,
        "target",
        "classification",
        "tree_ensemble",
        test_size=0.2,
        random_state=17,
        split=classification_split,
        row_positions=np.arange(len(classification)),
    )
    classification_validation.raise_if_failed()
    assert classification_validation.split["contract"] == classification_split.as_dict()
    assert classification_validation.split["strategy"] == "stratified"

    regression = classification.drop(columns=["segment", "region"]).copy()
    regression["target"] = np.linspace(0.0, 1.0, len(regression))
    regression_split = freeze_supervised_split(
        regression, "target", "regression", test_size=0.2, random_state=17
    )
    regression_validation = validate_training_plan(
        regression,
        "target",
        "regression",
        "linear",
        test_size=0.2,
        random_state=17,
        split=regression_split,
        row_positions=np.arange(len(regression)),
    )
    regression_validation.raise_if_failed()
    assert regression_validation.split["contract"] == regression_split.as_dict()
    assert regression_validation.split["strategy"] == "seeded_random"


def test_reconciliation_receives_training_only_evidence():
    frame = _frame()
    split = freeze_supervised_split(frame, "target", "classification")
    planning_frame = training_profile_frame(
        frame,
        "target",
        "classification",
        test_size=0.2,
        random_state=42,
        split=split,
    )
    planning_profile = profile_dataframe(planning_frame)
    agent_plan = AgentPlan(
        target_column="target",
        task_type="classification",
        recommended_method="linear",
        reasoning="The independent planner selected an interpretable baseline from the training schema.",
        confidence=0.7,
    )
    deterministic = DeterministicRecommendation(
        target_column="target",
        task_type="classification",
        recommended_method="tree_ensemble",
        reasoning="The deterministic policy selected a robust family from the training schema evidence.",
        evidence=["training_rows=64"],
    )

    captured: dict[str, object] = {}

    class Reconciler:
        def reconcile(self, question, profile, agent_plan, deterministic):
            captured["profile"] = profile
            return ConflictResolution(
                selected_target_column="target",
                selected_task_type="classification",
                selected_method="tree_ensemble",
                selected_preprocessing=deterministic["preprocessing"],
                checks=["training_profile_checked"],
                justification="The tree recommendation is better aligned with the categorical training features and safe preprocessing contract.",
                confidence=0.8,
            )

    result = _validate_before_training(
        Reconciler(),
        planning_profile,
        "classify target",
        agent_plan,
        deterministic,
        [],
        {},
        offline=False,
        dataframe=frame,
        split=split,
        row_positions=list(range(len(frame))),
        reconciliation_profile=planning_profile,
        established_target="target",
        established_task="classification",
    )

    assert result["status"] == "disagreement_resolved"
    assert captured["profile"]["rows"] == split.as_dict()["train_rows"]
    assert captured["profile"] == planning_profile


def test_holdout_only_perturbation_does_not_change_planning_evidence_or_recommendation():
    original = _frame(100)
    split = freeze_supervised_split(original, "target", "classification")
    perturbed = original.copy()
    holdout = list(split.holdout_row_positions)
    perturbed.loc[holdout, "signal"] = np.linspace(1e9, 2e9, len(holdout))
    perturbed.loc[holdout, "segment"] = [f"holdout-{index}" for index in range(len(holdout))]
    perturbed.loc[holdout, "region"] = [f"region-{index}" for index in range(len(holdout))]

    original_training = training_profile_frame(
        original, "target", "classification", test_size=0.2, random_state=42, split=split
    )
    perturbed_split = freeze_supervised_split(perturbed, "target", "classification")
    perturbed_training = training_profile_frame(
        perturbed,
        "target",
        "classification",
        test_size=0.2,
        random_state=42,
        split=perturbed_split,
    )
    original_profile = profile_dataframe(original_training)
    perturbed_profile = profile_dataframe(perturbed_training)
    original_recommendation = deterministic_recommendation(
        original_training, "classify target", "target", task_type="classification"
    )
    perturbed_recommendation = deterministic_recommendation(
        perturbed_training, "classify target", "target", task_type="classification"
    )

    assert original_profile == perturbed_profile
    assert original_recommendation.model_dump(mode="json") == perturbed_recommendation.model_dump(mode="json")
    assert split.train_row_positions == perturbed_split.train_row_positions
    assert split.holdout_row_positions == perturbed_split.holdout_row_positions


def test_holdout_only_perturbation_does_not_change_training_only_eda_input():
    original = _frame(100)
    split = freeze_supervised_split(original, "target", "classification")
    holdout = np.asarray(split.holdout_row_positions, dtype=int)
    perturbed = original.copy()
    perturbed.loc[holdout, "signal"] = np.linspace(1e9, 2e9, len(holdout))
    perturbed.loc[holdout, "segment"] = [f"unseen-category-{index}" for index in range(len(holdout))]
    perturbed.loc[holdout, "region"] = [f"unseen-region-{index}" for index in range(len(holdout))]

    # Emulate structural cleaning having removed rows while retaining the
    # original source positions rather than relying on reset dataframe indices.
    removed = np.array([split.train_row_positions[0], split.holdout_row_positions[0]])
    keep = ~np.isin(np.arange(len(original)), removed)
    original_cleaned = original.loc[keep].reset_index(drop=True)
    perturbed_cleaned = perturbed.loc[keep].reset_index(drop=True)
    cleaned_positions = np.flatnonzero(keep)

    original_eda_frame = training_partition_frame(original_cleaned, split, cleaned_positions)
    perturbed_eda_frame = training_partition_frame(perturbed_cleaned, split, cleaned_positions)
    original_agent_input = eda_summary(original_eda_frame, "target")
    perturbed_agent_input = eda_summary(perturbed_eda_frame, "target")
    expected_training_positions = cleaned_positions[
        np.isin(cleaned_positions, split.train_row_positions)
    ]

    assert original_agent_input == perturbed_agent_input
    assert original_agent_input["rows"] == len(split.train_row_positions) - 1
    assert set(cleaned_positions).issubset(set(split.valid_row_positions))
    assert np.array_equal(
        original_eda_frame["signal"].to_numpy(),
        original.loc[expected_training_positions, "signal"].to_numpy(),
    )
    assert not set(expected_training_positions).intersection(set(split.holdout_row_positions))


def test_reconciliation_receives_structured_deterministic_evidence():
    frame = _frame()
    split = freeze_supervised_split(frame, "target", "classification")
    planning_frame = training_profile_frame(
        frame,
        "target",
        "classification",
        test_size=0.2,
        random_state=42,
        split=split,
    )
    planning_profile = profile_dataframe(planning_frame)
    deterministic = deterministic_recommendation(
        planning_frame,
        "classify target",
        "target",
        task_type="classification",
    )
    agent_plan = AgentPlan(
        target_column="target",
        task_type="classification",
        recommended_method="linear",
        reasoning="The independent planner selected a linear baseline from the training schema.",
        confidence=0.7,
    )
    captured: dict[str, object] = {}

    class Reconciler:
        def reconcile(self, question, profile, agent_plan, deterministic):
            captured["deterministic"] = deterministic
            return ConflictResolution(
                selected_target_column="target",
                selected_task_type="classification",
                selected_method=deterministic["recommended_method"],
                selected_preprocessing=deterministic["preprocessing"],
                checks=["structured_evidence_checked"],
                justification="The deterministic score evidence is more compatible with the observed training feature structure and preprocessing contract.",
                confidence=0.7,
            )

    result = _validate_before_training(
        Reconciler(),
        planning_profile,
        "classify target",
        agent_plan,
        deterministic,
        [],
        {},
        offline=False,
        dataframe=frame,
        split=split,
        row_positions=list(range(len(frame))),
        reconciliation_profile=planning_profile,
        established_target="target",
        established_task="classification",
    )

    evidence = captured["deterministic"]
    assert result["status"] == "disagreement_resolved"
    assert evidence["policy_version"] == "4"
    assert set(evidence["method_scores"]) == {
        "linear",
        "regularized_linear",
        "tree_ensemble",
        "boosted_tree",
    }
    assert evidence["diagnostics"]["rows"] == split.as_dict()["train_rows"]
    assert "holdout" not in str(evidence).lower()


def _partition_cleaning(frame: pd.DataFrame, split, actions: list[str]):
    row_position_column = "__autods_row_position__"
    with_positions = frame.copy()
    with_positions[row_position_column] = np.arange(len(with_positions))
    training = with_positions.iloc[list(split.train_row_positions)].copy()
    specification = fit_cleaning_spec(
        training,
        "target",
        actions,
        row_position_column=row_position_column,
    )
    transformed_training, training_log = transform_cleaning(
        training,
        specification,
        partition="training",
    )
    transformed_holdout, holdout_log = transform_cleaning(
        with_positions.iloc[list(split.holdout_row_positions)].copy(),
        specification,
        partition="holdout",
    )
    return specification, transformed_training, training_log, transformed_holdout, holdout_log


def test_cleaning_spec_constant_and_all_null_decisions_ignore_holdout_values():
    frame = _frame(100)
    split = freeze_supervised_split(frame, "target", "classification")
    frame["constant_feature"] = "constant"
    frame["null_only_feature"] = None
    changed = frame.copy()
    changed.loc[list(split.holdout_row_positions), "constant_feature"] = [
        f"holdout-{index}" for index in range(len(split.holdout_row_positions))
    ]
    changed.loc[list(split.holdout_row_positions), "null_only_feature"] = [
        f"populated-{index}" for index in range(len(split.holdout_row_positions))
    ]

    original_spec, original_training, _, _, _ = _partition_cleaning(
        frame,
        split,
        ["drop_all_null_columns", "drop_constant_features"],
    )
    changed_spec, changed_training, _, _, _ = _partition_cleaning(
        changed,
        split,
        ["drop_all_null_columns", "drop_constant_features"],
    )

    assert original_spec.model_dump(mode="json") == changed_spec.model_dump(mode="json")
    assert "constant_feature" in original_spec.constant_columns
    assert "null_only_feature" in original_spec.all_null_columns
    pd.testing.assert_frame_equal(original_training, changed_training)


def test_numeric_string_coercion_threshold_is_fitted_from_training_only():
    frame = _frame(100)
    split = freeze_supervised_split(frame, "target", "classification")
    frame["numeric_text"] = [str(value) for value in range(len(frame))]
    changed = frame.copy()
    changed.loc[list(split.holdout_row_positions), "numeric_text"] = "not-a-number"

    original_spec, original_training, _, _, _ = _partition_cleaning(
        frame,
        split,
        ["coerce_numeric_strings"],
    )
    changed_spec, changed_training, _, _, _ = _partition_cleaning(
        changed,
        split,
        ["coerce_numeric_strings"],
    )

    assert original_spec.model_dump(mode="json") == changed_spec.model_dump(mode="json")
    assert original_spec.numeric_coercion_columns == ["numeric_text"]
    assert pd.api.types.is_numeric_dtype(original_training["numeric_text"])
    pd.testing.assert_frame_equal(original_training, changed_training)


def test_duplicate_cleaning_never_compares_training_and_holdout_rows():
    frame = _frame(100)
    split = freeze_supervised_split(frame, "target", "classification")
    train_positions = list(split.train_row_positions)
    holdout_positions = list(split.holdout_row_positions)
    same_class_train = next(
        (left, right)
        for index, left in enumerate(train_positions)
        for right in train_positions[index + 1 :]
        if frame.loc[left, "target"] == frame.loc[right, "target"]
    )
    same_class_cross = next(
        (train_position, holdout_position)
        for train_position in train_positions
        for holdout_position in holdout_positions
        if frame.loc[train_position, "target"] == frame.loc[holdout_position, "target"]
    )
    frame.loc[same_class_train[1], ["signal", "segment", "region", "target"]] = frame.loc[
        same_class_train[0], ["signal", "segment", "region", "target"]
    ].to_numpy()
    frame.loc[same_class_cross[1], ["signal", "segment", "region", "target"]] = frame.loc[
        same_class_cross[0], ["signal", "segment", "region", "target"]
    ].to_numpy()
    changed = frame.copy()
    changed.loc[same_class_cross[1], "signal"] = 999999.0

    original_spec, original_training, original_training_log, _, _ = _partition_cleaning(
        frame,
        split,
        ["drop_exact_duplicates"],
    )
    changed_spec, changed_training, changed_training_log, _, _ = _partition_cleaning(
        changed,
        split,
        ["drop_exact_duplicates"],
    )

    assert original_spec.training_duplicate_row_positions == changed_spec.training_duplicate_row_positions
    assert same_class_train[1] in original_spec.training_duplicate_row_positions
    assert same_class_cross[1] not in original_spec.training_duplicate_row_positions
    assert original_training_log["removed_row_positions"] == changed_training_log["removed_row_positions"]
    pd.testing.assert_frame_equal(original_training, changed_training)


def test_end_to_end_holdout_perturbation_preserves_all_pre_evaluation_training_artifacts(tmp_path: Path):
    base = _frame(100)
    split = freeze_supervised_split(base, "target", "classification")
    base["constant_feature"] = "constant"
    base["numeric_text"] = [str(value) for value in range(len(base))]
    base["null_only_feature"] = None
    train_position = split.train_row_positions[0]
    holdout_position = next(
        position
        for position in split.holdout_row_positions
        if base.loc[position, "target"] == base.loc[train_position, "target"]
    )
    base.loc[holdout_position, ["signal", "segment", "region", "target"]] = base.loc[
        train_position, ["signal", "segment", "region", "target"]
    ].to_numpy()
    perturbed = base.copy()
    holdout = list(split.holdout_row_positions)
    perturbed.loc[holdout, "constant_feature"] = [f"changed-{index}" for index in range(len(holdout))]
    perturbed.loc[holdout, "numeric_text"] = "not-numeric"
    perturbed.loc[holdout, "null_only_feature"] = [f"filled-{index}" for index in range(len(holdout))]
    perturbed.loc[holdout_position, "signal"] = 999999.0

    original_path = tmp_path / "original.csv"
    perturbed_path = tmp_path / "perturbed.csv"
    base.to_csv(original_path, index=False)
    perturbed.to_csv(perturbed_path, index=False)
    original_result = run_analysis(
        original_path,
        "Classify target from the measured features.",
        target_column="target",
        output_dir=tmp_path / "original_run",
        offline=True,
    )
    perturbed_result = run_analysis(
        perturbed_path,
        "Classify target from the measured features.",
        target_column="target",
        output_dir=tmp_path / "perturbed_run",
        offline=True,
    )

    original_dir = Path(original_result["run_dir"])
    perturbed_dir = Path(perturbed_result["run_dir"])
    original_decision, original_planning, _ = _read_run(original_dir)
    perturbed_decision, perturbed_planning, _ = _read_run(perturbed_dir)
    original_cleaning = json.loads((original_dir / "cleaning.json").read_text(encoding="utf-8"))
    perturbed_cleaning = json.loads((perturbed_dir / "cleaning.json").read_text(encoding="utf-8"))
    original_eda = json.loads((original_dir / "eda.json").read_text(encoding="utf-8"))
    perturbed_eda = json.loads((perturbed_dir / "eda.json").read_text(encoding="utf-8"))

    assert original_planning == perturbed_planning
    assert original_decision["deterministic_recommendation"] == perturbed_decision["deterministic_recommendation"]
    assert original_decision["validation"]["selected_method"] == perturbed_decision["validation"]["selected_method"]
    assert original_decision["validation"]["approved_preprocessing"] == perturbed_decision["validation"]["approved_preprocessing"]
    assert original_cleaning["fitted_specification"] == perturbed_cleaning["fitted_specification"]
    assert original_cleaning["training_only_evidence"] == perturbed_cleaning["training_only_evidence"]
    assert original_cleaning["applied_transformations"]["training"] == perturbed_cleaning["applied_transformations"]["training"]
    assert original_eda["computed"] == perturbed_eda["computed"]
    assert original_eda["data_scope"] == perturbed_eda["data_scope"] == "training_partition_only"
