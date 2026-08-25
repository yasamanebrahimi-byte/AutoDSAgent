import pytest
from pydantic import ValidationError

from app.llm import OpenAIAgents
from app.schemas import ModelingPlan


def test_modeling_prompt_documents_the_valid_pair_for_one_hot_encoding(monkeypatch):
    captured = {}

    def fake_structured(self, schema_name, schema, instructions, payload):
        captured["instructions"] = instructions
        return ModelingPlan(
            recommended_method="tree_ensemble",
            reasoning="The plan uses a supported model family and executable preprocessing.",
            confidence=0.7,
        )

    monkeypatch.setattr(OpenAIAgents, "_structured", fake_structured)
    OpenAIAgents(api_key="test").modeling_plan(
        {"columns": []},
        "classify target",
        "target",
        "classification",
    )

    assert "one_hot with" in captured["instructions"]
    assert "categorical_unknown_handling='ignore'" in captured["instructions"]


def test_modeling_plan_rejects_the_failed_one_hot_unknown_handling_pair():
    with pytest.raises(ValidationError, match="one_hot encoding requires"):
        ModelingPlan.model_validate(
            {
                "recommended_method": "tree_ensemble",
                "preprocessing": {
                    "categorical_encoding": "one_hot",
                    "categorical_unknown_handling": "use_encoded_value",
                },
                "reasoning": "The plan uses a supported model family and executable preprocessing.",
                "confidence": 0.7,
            }
        )
