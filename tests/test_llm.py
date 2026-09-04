import pytest
from pydantic import ValidationError

from app.llm import GenerationSettingsError, OpenAIAgents
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


def _mock_response():
    return type(
        "Response",
        (),
        {
            "output_parsed": ModelingPlan(
                recommended_method="tree_ensemble",
                reasoning="A supported plan for this classification task.",
                confidence=0.7,
            ),
            "output_text": "",
            "id": "resp_test",
            "model": "gpt-4.1-mini-2026-01-01",
        },
    )()


def test_supported_generation_setting_reaches_responses_request(monkeypatch):
    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return _mock_response()

    agents = OpenAIAgents(
        api_key="test",
        model="gpt-4.1-mini",
        generation_settings={"temperature": 0.2},
    )
    agents._client = type("Client", (), {"responses": Responses()})()
    agents.modeling_plan({"columns": []}, "classify target", "target", "classification")

    assert captured["temperature"] == 0.2
    assert "top_p" not in captured
    assert agents.last_request_provenance["generation_settings_sent"] == {"temperature": 0.2}


def test_null_generation_setting_uses_provider_default(monkeypatch):
    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return _mock_response()

    agents = OpenAIAgents(
        api_key="test",
        model="gpt-4.1-mini",
        generation_settings={"temperature": None},
    )
    agents._client = type("Client", (), {"responses": Responses()})()
    agents.modeling_plan({"columns": []}, "classify target", "target", "classification")

    assert "temperature" not in captured
    assert agents.last_request_provenance["provider_default_settings"] == ["temperature"]


def test_unsupported_generation_setting_fails_before_api_request():
    with pytest.raises(GenerationSettingsError, match="seed.*not supported"):
        OpenAIAgents(
            api_key="test",
            model="gpt-4.1-mini",
            generation_settings={"seed": 17},
        )
