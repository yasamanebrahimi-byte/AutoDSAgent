from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.llm import (
    GeminiAgents,
    GenerationSettingsError,
    LLMUnavailable,
    OpenAIAgents,
    build_agents,
)
from app.schemas import ModelingPlan, ModelingResolution, PreprocessingContract
from evaluation.ablation import _health_row, ablation_presets


class _GeminiResponse:
    def __init__(self, parsed=None, text: str = "", model_version: str = "gemini-3.8-flash"):
        self.parsed = parsed
        self.text = text
        self.response_id = "gemini-response-test"
        self.model_version = model_version
        self.usage_metadata = {
            "prompt_token_count": 31,
            "candidates_token_count": 17,
            "thoughts_token_count": 9,
            "total_token_count": 57,
        }


class _GeminiModels:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


class _GeminiClient:
    def __init__(self, response):
        self.models = _GeminiModels(response)


def _plan() -> ModelingPlan:
    return ModelingPlan(
        recommended_method="tree_ensemble",
        preprocessing=PreprocessingContract(),
        reasoning="A valid synthetic smoke-test plan.",
        confidence=0.8,
    )


def test_provider_factory_selects_openai_and_google_and_rejects_unknown():
    assert isinstance(build_agents(provider="openai", model="gpt-4.1-mini"), OpenAIAgents)
    assert isinstance(build_agents(provider="google", model="gemini-3.8-flash"), GeminiAgents)
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_agents(provider="gemini", model="gemini-3.8-flash")


def test_gemini_credentials_prefer_gemini_then_google(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
    assert GeminiAgents(model="gemini-3.8-flash", respect_environment_model=False).api_key == "google-secret"

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    assert GeminiAgents(model="gemini-3.8-flash", respect_environment_model=False).api_key == "gemini-secret"

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    agent = GeminiAgents(model="gemini-3.8-flash", respect_environment_model=False)
    assert agent.available is False
    with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
        agent.modeling_plan({}, "classify target", "target", "classification")


def test_gemini_structured_output_uses_shared_schema_and_records_provenance():
    plan = _plan()
    client = _GeminiClient(_GeminiResponse(parsed=plan))
    agent = GeminiAgents(
        api_key="test-secret",
        model="gemini-3.8-flash",
        generation_settings={
            "temperature": None,
            "top_p": None,
            "seed": None,
            "reasoning_effort": "medium",
        },
        respect_environment_model=False,
    )
    agent._client = client

    result = agent.modeling_plan({}, "classify target", "target", "classification")

    assert result == plan
    request = client.models.requests[0]
    assert request["model"] == "gemini-3.8-flash"
    assert json.loads(request["contents"])["question"] == "classify target"
    assert request["config"]["system_instruction"].startswith(
        "You are the independent post-formulation modeling agent."
    )
    assert request["config"]["response_mime_type"] == "application/json"
    assert request["config"]["response_schema"] is ModelingPlan
    assert request["config"]["thinking_config"] == {"thinking_level": "medium"}
    assert "temperature" not in request["config"]
    assert "top_p" not in request["config"]
    assert "seed" not in request["config"]
    assert agent.last_request_provenance == {
        "provider": "google",
        "endpoint": "models.generate_content",
        "api_surface": "google.genai.Client.models.generate_content",
        "model_requested": "gemini-3.8-flash",
        "generation_settings_requested": {
            "temperature": None,
            "top_p": None,
            "seed": None,
            "reasoning_effort": "medium",
        },
        "generation_settings_sent": {"thinking_config": {"thinking_level": "medium"}},
        "provider_default_settings": ["seed", "temperature", "top_p"],
        "reasoning_effort_mapping": {
            "reasoning_effort": "medium",
            "thinking_level": "medium",
        },
        "wall_clock_seconds": agent.last_request_provenance["wall_clock_seconds"],
        "input_tokens": 31,
        "output_tokens": 17,
        "thought_tokens": 9,
        "total_tokens": 57,
        "response_metadata": {
            "response_id": "gemini-response-test",
            "model_version": "gemini-3.8-flash",
        },
        "model_effective": "gemini-3.8-flash",
    }
    assert agent.assert_effective_model(expected_model="gemini-3.8-flash") == "gemini-3.8-flash"


def test_gemini_malformed_structured_output_fails_without_coercion():
    client = _GeminiClient(_GeminiResponse(parsed=None, text='{"recommended_method":"not-a-method"}'))
    agent = GeminiAgents(
        api_key="test-secret",
        model="gemini-3.8-flash",
        respect_environment_model=False,
    )
    agent._client = client
    with pytest.raises(LLMUnavailable, match="invalid structured output"):
        agent.modeling_plan({}, "classify target", "target", "classification")


def test_gemini_missing_structured_output_uses_existing_error_contract():
    client = _GeminiClient(_GeminiResponse(parsed=None, text=""))
    agent = GeminiAgents(
        api_key="test-secret",
        model="gemini-3.8-flash",
        respect_environment_model=False,
    )
    agent._client = client

    with pytest.raises(LLMUnavailable, match="no structured output"):
        agent.modeling_plan({}, "classify target", "target", "classification")


def test_gemini_provider_errors_redact_direct_secret():
    class FailingModels:
        def generate_content(self, **kwargs):
            del kwargs
            raise RuntimeError("request used test-secret")

    agent = GeminiAgents(
        api_key="test-secret",
        model="gemini-3.8-flash",
        respect_environment_model=False,
    )
    agent._client = type("Client", (), {"models": FailingModels()})()
    with pytest.raises(LLMUnavailable) as exc_info:
        agent.modeling_plan({}, "classify target", "target", "classification")
    assert "test-secret" not in str(exc_info.value)


def test_gemini_generation_settings_validate_provider_specific_support():
    with pytest.raises(GenerationSettingsError, match="minimal thinking level"):
        GeminiAgents(
            api_key="test",
            model="gemini-3.8-flash",
            generation_settings={"reasoning_effort": "minimal"},
            respect_environment_model=False,
        )


def test_gemini_effective_numeric_revision_is_accepted_but_other_model_is_not():
    agent = GeminiAgents(api_key="test", model="gemini-3.8-flash", respect_environment_model=False)
    agent.last_request_provenance = {"model_effective": "gemini-3.8-flash-001"}
    assert agent.assert_effective_model(expected_model="gemini-3.8-flash") == "gemini-3.8-flash-001"
    agent.last_request_provenance = {"model_effective": "gemini-3.5-flash-lite"}
    with pytest.raises(LLMUnavailable, match="model mismatch"):
        agent.assert_effective_model(expected_model="gemini-3.8-flash")


def test_shared_reconciler_schema_remains_modeling_resolution():
    resolution = ModelingResolution(
        selected_method="tree_ensemble",
        selected_preprocessing=PreprocessingContract(),
        checks=["valid"],
        justification="The synthetic smoke-test comparison selected the valid proposal.",
        confidence=0.8,
    )
    client = _GeminiClient(_GeminiResponse(parsed=resolution))
    agent = GeminiAgents(api_key="test", model="gemini-3.8-flash", respect_environment_model=False)
    agent._client = client
    result = agent.reconcile_modeling(
        "classify target",
        {"reconciliation_mode": "blinded_evidence_comparison", "proposal_a": {}, "proposal_b": {}},
        _plan(),
        {"_blinded_reconciliation_payload": {"proposal_a": {}, "proposal_b": {}}},
    )
    assert result == resolution
    assert client.models.requests[0]["config"]["response_schema"] is ModelingResolution


def test_replication_manifest_is_provider_valid_and_preserves_matrix_definition():
    from evaluation.confirmatory import validate_confirmatory_preflight

    manifest_path = Path(__file__).parents[1] / "evaluation/configs/paper_cross_provider_replication_v1.json"
    metadata = validate_confirmatory_preflight(manifest_path)
    assert [item["provider"] for item in metadata["model_conditions"]] == ["google", "google"]
    assert metadata["llm_repetition_ids"] == ["rep_001", "rep_002", "rep_003"]
    assert metadata["generation_settings"]["reasoning_effort"] == "medium"


def test_provider_aware_cache_identity_separates_openai_and_google():
    from evaluation.benchmarks import BenchmarkCase
    from evaluation.runner import _proposal_cache_key

    case = BenchmarkCase("task", None, "q", "classification", "test")
    common = dict(
        case=case,
        perturbation_id="clean",
        split_seed=42,
        llm_repetition=0,
        model="same-model",
        model_condition_id="same-condition",
        prompt_schema_version="p",
        training_profile={},
    )
    assert _proposal_cache_key(**common, provider="openai") != _proposal_cache_key(
        **common, provider="google"
    )


def test_openai_only_metrics_require_openai_provider_identity():
    from evaluation.metrics import summarize_trials

    openai_row = {
        "provider": "openai",
        "agent_source": "openai",
        "trial_status": "completed",
        "requested_live_trial": True,
    }
    mismatched_row = {
        "provider": "google",
        "agent_source": "openai",
        "trial_status": "completed",
        "requested_live_trial": True,
    }

    summary = summarize_trials([openai_row, mismatched_row])

    assert summary["successful_openai_trials"] == 1
    assert summary["openai_only"]["requested_live_trials"] == 1


def test_successful_google_runner_call_is_live_not_fallback(tmp_path, monkeypatch):
    import pandas as pd

    from evaluation.benchmarks import BenchmarkCase
    import evaluation.runner as runner

    class FakeGoogleAgents:
        provider = "google"
        api_key_env_vars = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        available = True

        def __init__(self, *, model, **kwargs):
            del kwargs
            self.model = model
            self.api_key = "test"
            self.last_request_provenance = None

        def modeling_plan(self, *args, **kwargs):
            del args, kwargs
            self.last_request_provenance = {
                "provider": "google",
                "model_requested": self.model,
                "model_effective": self.model,
                "input_tokens": 10,
                "output_tokens": 5,
            }
            return _plan()

        def assert_effective_model(self, *, expected_model):
            assert expected_model == self.model
            return self.model

    monkeypatch.setattr(runner, "build_agents", lambda **kwargs: FakeGoogleAgents(**kwargs))
    frame = pd.DataFrame(
        {
            "x": list(range(30)),
            "target": ["yes" if index % 2 else "no" for index in range(30)],
        }
    )
    case = BenchmarkCase(
        "google_live_fixture",
        "target",
        "Classify target from x.",
        "classification",
        "synthetic",
        dataframe=frame,
    )
    result = runner.run_evaluation(
        tmp_path / "google-live",
        cases=[case],
        model="gemini-3.8-flash",
        provider="google",
        gate_mode="llm_only",
        require_live=True,
    )
    trial = result["trials"][0]
    assert trial["agent_source"] == "google"
    assert trial["api_status"] == "live_succeeded"
    assert trial["fallback_row"] is False
    assert result["summary"]["successful_initial_live_calls"] == 1
    assert result["summary"]["planner_live_success"] == 1
    assert result["summary"]["strict_live_valid"] is True


def test_failed_google_runner_call_uses_explicit_offline_fallback(tmp_path, monkeypatch):
    import pandas as pd

    from evaluation.benchmarks import BenchmarkCase
    import evaluation.runner as runner

    class FailingGoogleAgents:
        provider = "google"
        api_key_env_vars = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        available = True

        def __init__(self, *, model, **kwargs):
            del kwargs
            self.model = model
            self.api_key = "private-gemini-token"
            self.last_request_provenance = None

        def modeling_plan(self, *args, **kwargs):
            del args, kwargs
            raise LLMUnavailable("Gemini request failed: private-gemini-token")

    monkeypatch.setattr(runner, "build_agents", lambda **kwargs: FailingGoogleAgents(**kwargs))
    frame = pd.DataFrame(
        {
            "x": list(range(30)),
            "target": ["yes" if index % 2 else "no" for index in range(30)],
        }
    )
    case = BenchmarkCase(
        "google_fallback_fixture",
        "target",
        "Classify target from x.",
        "classification",
        "synthetic",
        dataframe=frame,
    )
    result = runner.run_evaluation(
        tmp_path / "google-fallback",
        cases=[case],
        model="gemini-3.8-flash",
        provider="google",
        gate_mode="llm_only",
        require_live=False,
    )
    trial = result["trials"][0]
    assert trial["agent_source"] == "offline_fallback"
    assert trial["fallback_row"] is True
    assert trial["api_status"] == "offline"
    assert "private-gemini-token" not in trial["agent_request_error"]
    assert result["summary"]["successful_initial_live_calls"] == 0

    strict = runner.run_evaluation(
        tmp_path / "google-strict-failure",
        cases=[case],
        model="gemini-3.8-flash",
        provider="google",
        gate_mode="llm_only",
        require_live=True,
    )
    strict_trial = strict["trials"][0]
    assert strict_trial["agent_request_status"] == "failed"
    assert strict_trial["live_request_failed"] is True
    assert strict["summary"]["live_request_failed_trials"] == 1
    api_usage = _health_row(
        "llm_only", strict, ablation_presets()["llm_only"]
    )["api_usage"]
    assert api_usage["failed_initial_live_calls"] == 1
    assert api_usage["failed_initial_openai_calls"] == 0
