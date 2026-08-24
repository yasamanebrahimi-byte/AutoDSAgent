from app.pipeline import _validate_before_training
from app.schemas import AgentPlan, ConflictResolution, DeterministicRecommendation, PreprocessingContract
from app.validation import freeze_supervised_split
import numpy as np
import pandas as pd


class FakeReconciliationAgent:
    def reconcile(self, question, profile, agent_plan, deterministic):
        return ConflictResolution(
            selected_target_column="target",
            selected_task_type="classification",
            selected_method="tree_ensemble",
            checks=["target_exists", "task_matches_target", "method_is_supported"],
            justification="The deterministic recommendation is more conservative for the observed categorical feature and missingness pattern.",
            confidence=0.86,
        )


def test_disagreement_requires_a_recorded_reconciliation():
    agent_plan = AgentPlan(
        target_column="target",
        task_type="classification",
        recommended_method="linear",
        preprocessing=["scale_numeric_features"],
        reasoning="The independent agent favors an interpretable linear decision boundary.",
        confidence=0.7,
    )
    deterministic = DeterministicRecommendation(
        target_column="target",
        task_type="classification",
        recommended_method="tree_ensemble",
        preprocessing=["training_only_imputation"],
        reasoning="The deterministic policy sees categorical structure and recommends a robust tree family.",
        evidence=["categorical_features=2"],
    )
    sources = {}
    result = _validate_before_training(
        FakeReconciliationAgent(),
        {"rows": 100},
        "classify target",
        agent_plan,
        deterministic,
        [],
        sources,
        offline=False,
    )

    assert result["status"] == "disagreement_resolved"
    assert result["selected_method"] == "tree_ensemble"
    assert result["justification"]
    assert sources["reconciliation"] == "openai"


def test_valid_soft_disagreement_can_legally_keep_the_agent_plan():
    frame = pd.DataFrame(
        {
            "signal": np.linspace(-1.0, 1.0, 60),
            "target": ["yes", "no"] * 30,
        }
    )
    split = freeze_supervised_split(frame, "target", "classification")
    agent_plan = AgentPlan(
        target_column="target",
        task_type="classification",
        recommended_method="linear",
        preprocessing=PreprocessingContract(numeric_scaling="standard"),
        reasoning="The independent agent favors a transparent linear baseline for this test case.",
        confidence=0.7,
    )
    deterministic = DeterministicRecommendation(
        target_column="target",
        task_type="classification",
        recommended_method="tree_ensemble",
        preprocessing=PreprocessingContract(),
        reasoning="The deterministic challenger favors a flexible tree family for this test case.",
        evidence=["training-only structural evidence"],
        confidence="low",
    )

    class AgentWins:
        def reconcile(self, question, profile, agent_plan, deterministic):
            del question, profile, deterministic
            return ConflictResolution(
                selected_target_column="target",
                selected_task_type="classification",
                selected_method=agent_plan.recommended_method,
                selected_preprocessing=agent_plan.preprocessing,
                checks=["both_hard_validated", "agent_preserved"],
                justification="Both proposals passed hard validation; the initial agent plan remains methodologically defensible for this evidence.",
                confidence=0.75,
            )

    result = _validate_before_training(
        AgentWins(),
        {"rows": len(frame)},
        "classify target",
        agent_plan,
        deterministic,
        [],
        {},
        offline=False,
        dataframe=frame,
        split=split,
        row_positions=list(range(len(frame))),
        established_target="target",
        established_task="classification",
    )

    assert result["hard_validation"]["status"] == "passed"
    assert result["hard_validation"]["initial_hard_invalid"] is False
    assert result["soft_challenge"]["status"] == "disagreement"
    assert result["soft_challenge"]["deterministic_confidence"] == "low"
    assert result["final"]["recommended_method"] == "linear"
    assert result["final"]["selected_source"] == "agent"
