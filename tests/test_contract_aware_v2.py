from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.llm import OpenAIAgents, PROMPT_SCHEMA_VERSION
from app.schemas import ModelingPlan
from app.validation import build_execution_contract, execution_contract_digest
from evaluation.benchmarks import BenchmarkCase
from evaluation.runner import _proposal_cache_key


def _frame(rows: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric": list(range(rows)),
            "category": ["a", "b", "c"] * (rows // 3),
            "target": ["yes", "no"] * (rows // 2),
        }
    )


def test_execution_contract_is_training_only_and_explicitly_covers_boosted_trees():
    contract = build_execution_contract(_frame(), "target", "classification")

    assert contract["schema_version"] == "execution-contract-v1"
    assert contract["scope"] == "frozen_training_partition_only"
    assert contract["holdout_used"] is False
    assert set(contract["model_families"]) == {
        "linear", "regularized_linear", "tree_ensemble", "boosted_tree"
    }
    boosted = contract["model_families"]["boosted_tree"]
    assert boosted["compatible_contract_for_observed_schema"]["categorical_encoding"] == "ordinal"
    assert any(
        "boosted_tree with usable categorical features requires ordinal encoding" in rule
        for rule in contract["hard_constraints"]["conditional_rules"]
    )
    assert "recommended_method" not in json.dumps(contract)
    assert "holdout_metrics" not in json.dumps(contract)
    assert "score" not in json.dumps(contract).lower()


def test_execution_contract_reports_one_hot_feasibility_from_training_rows_only():
    rows = 80
    data = {f"cat_{index}": pd.Categorical(list(range(rows))) for index in range(50)}
    data["target"] = ["yes", "no"] * (rows // 2)
    contract = build_execution_contract(pd.DataFrame(data), "target", "classification")

    feasibility = contract["dataset_feasibility"]
    assert feasibility["has_categorical_features"] is True
    assert feasibility["estimated_one_hot_features"] > feasibility["max_one_hot_features"]
    assert feasibility["one_hot_feasible"] is False
    assert contract["model_families"]["boosted_tree"]["executable_for_dataset"] is True


def test_contract_aware_planner_payload_hides_recommender_and_holdout(monkeypatch):
    captured: dict = {}

    def fake_structured(self, schema_name, schema, instructions, payload):
        captured.update(payload)
        return ModelingPlan(
            recommended_method="tree_ensemble",
            reasoning="The contract permits this supported executable family.",
            confidence=0.5,
        )

    monkeypatch.setattr(OpenAIAgents, "_structured", fake_structured)
    contract = build_execution_contract(_frame(), "target", "classification")
    OpenAIAgents(api_key="test").modeling_plan(
        {"column_details": []},
        "Classify target.",
        "target",
        "classification",
        execution_contract=contract,
        prompt_schema_version=PROMPT_SCHEMA_VERSION,
    )

    assert captured["execution_contract"] == contract
    assert "deterministic_recommendation" not in captured
    assert "holdout_metrics" not in json.dumps(captured["execution_contract"]).lower()
    assert "empirical_probe" not in json.dumps(captured["execution_contract"]).lower()


def test_contract_digest_is_cache_identity():
    case = BenchmarkCase(
        name="cache-contract",
        dataframe=_frame(),
        target_column="target",
        question="Classify target.",
        expected_task_type="classification",
        dataset_source="test",
    )
    profile = {"rows": 10, "column_details": []}
    first = build_execution_contract(_frame(), "target", "classification")
    second = json.loads(json.dumps(first))
    second["dataset_feasibility"]["estimated_one_hot_features"] += 1
    assert execution_contract_digest(first) != execution_contract_digest(second)
    key_a = _proposal_cache_key(
        case=case, perturbation_id="clean", split_seed=42, llm_repetition=0,
        model="gpt-5.6-luna", prompt_schema_version=PROMPT_SCHEMA_VERSION,
        training_profile=profile, execution_contract=first,
    )
    key_b = _proposal_cache_key(
        case=case, perturbation_id="clean", split_seed=42, llm_repetition=0,
        model="gpt-5.6-luna", prompt_schema_version=PROMPT_SCHEMA_VERSION,
        training_profile=profile, execution_contract=second,
    )
    assert key_a != key_b


def test_v2_manifest_starts_as_draft_and_v1_is_untouched():
    root = Path(__file__).parents[1] / "evaluation" / "configs"
    v1 = json.loads((root / "paper_confirmatory_v1.json").read_text(encoding="utf-8"))
    v2 = json.loads((root / "paper_confirmatory_v2.json").read_text(encoding="utf-8"))
    assert v1["experiment_config_version"] == "paper-confirmatory-v1"
    assert v2["experiment_config_version"] == "paper-confirmatory-v2"
    assert v2["status"] in {"draft", "frozen"}
    assert v2["prompts"]["planner_schema_version"] == PROMPT_SCHEMA_VERSION


def test_api_usage_instrumentation_preserves_nulls_and_reported_counts():
    class Usage:
        input_tokens = 17
        output_tokens = 9

    class Response:
        output_parsed = ModelingPlan(
            recommended_method="linear",
            reasoning="The supported linear family is executable under the contract.",
            confidence=0.5,
        )
        output_text = ""
        id = "response-test"
        model = "gpt-5.6-luna"
        usage = Usage()

    class Responses:
        def parse(self, **kwargs):
            return Response()

    agents = OpenAIAgents(api_key="test", model="gpt-5.6-luna")
    agents._client = type("Client", (), {"responses": Responses()})()
    agents.modeling_plan({"column_details": []}, "Classify target.", "target", "classification")

    assert agents.last_request_provenance["input_tokens"] == 17
    assert agents.last_request_provenance["output_tokens"] == 9
    assert agents.last_request_provenance["wall_clock_seconds"] is not None
