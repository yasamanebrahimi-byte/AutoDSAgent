from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.deterministic import deterministic_recommendation, eda_summary, profile_dataframe
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
    assert evidence["policy_version"] == "3"
    assert set(evidence["method_scores"]) == {
        "linear",
        "regularized_linear",
        "tree_ensemble",
        "boosted_tree",
    }
    assert evidence["diagnostics"]["rows"] == split.as_dict()["train_rows"]
    assert "holdout" not in str(evidence).lower()
