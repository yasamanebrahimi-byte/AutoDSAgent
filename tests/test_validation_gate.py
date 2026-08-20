from app.pipeline import _validate_before_training
from app.schemas import AgentPlan, ConflictResolution, DeterministicRecommendation


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
