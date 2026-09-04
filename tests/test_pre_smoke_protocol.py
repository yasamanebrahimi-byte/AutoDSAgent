from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.llm import LLMUnavailable, OpenAIAgents
from app.schemas import ModelingPlan, PreprocessingContract
from app.validation import freeze_supervised_split, training_profile_frame, validate_training_plan
from evaluation.ablation import ablation_presets
from evaluation.confirmatory import (
    CONFIRMATORY_GENERATION_SETTINGS,
    CONFIRMATORY_REPETITION_IDS,
    CONFIRMATORY_SPLIT_SEEDS,
    model_conditions,
    validate_confirmatory_preflight,
)
from evaluation.runner import _proposal_cache_key, run_evaluation
from evaluation.benchmarks import BenchmarkCase


MANIFEST_PATH = Path(__file__).parents[1] / "evaluation" / "configs" / "paper_confirmatory_v1.json"


def _case() -> BenchmarkCase:
    frame = pd.DataFrame(
        {
            "signal": list(range(60)),
            "noise": [index % 7 for index in range(60)],
            "target": ["yes" if index % 2 else "no" for index in range(60)],
        }
    )
    return BenchmarkCase(
        name="pre_smoke_fixture",
        dataframe=frame,
        target_column="target",
        question="Classify target from the supplied features.",
        expected_task_type="classification",
        dataset_source="in-memory test fixture",
    )


def test_draft_preflight_has_exact_matrix_and_never_freezes_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    result = validate_confirmatory_preflight(manifest)

    assert manifest["status"] == "draft"
    assert manifest["expected_experiment_code_sha256"] is None
    assert manifest["source_git_commit"] is None
    assert [item["condition_id"] for item in model_conditions(manifest)] == [
        "gpt5_mini_2025_08_07",
        "gpt54_mini_2026_03_17",
        "gpt54_2026_03_05",
    ]
    assert all(item["llm_repetitions"] == 3 for item in result["model_conditions"])
    assert all(item["llm_repetition_ids"] == list(CONFIRMATORY_REPETITION_IDS) for item in result["model_conditions"])
    assert result["split_seeds"] == list(CONFIRMATORY_SPLIT_SEEDS)
    assert result["generation_settings"] == CONFIRMATORY_GENERATION_SETTINGS
    assert ablation_presets()["llm_with_diagnostics"].analysis_role == "secondary"


def test_draft_preflight_rejects_condition_drift():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["model_conditions"][1]["planner_model"] = "another-model"
    with pytest.raises(ValueError, match="design preflight"):
        validate_confirmatory_preflight(manifest)


def test_training_only_validation_records_zero_holdout_rows():
    case = _case()
    frame = case.load()
    split = freeze_supervised_split(frame, "target", "classification", random_state=42)
    training = training_profile_frame(
        frame,
        "target",
        "classification",
        test_size=0.2,
        random_state=42,
        split=split,
    )
    result = validate_training_plan(
        training,
        "target",
        "classification",
        "linear",
        preprocessing=PreprocessingContract(numeric_scaling="standard"),
        split=split,
        training_only=True,
    )
    assert result.status == "passed"
    assert result.split["scope"] == "training_only"
    assert result.split["holdout_rows_seen"] == 0
    assert result.split["valid_rows"] == len(training)
    assert result.split["train_rows"] + result.split["holdout_rows"] == len(training)


def test_diagnostics_mode_has_a_distinct_proposal_cache_namespace():
    case = _case()
    profile = {"rows": 10, "column_details": []}
    ordinary = _proposal_cache_key(
        case=case,
        perturbation_id="clean",
        split_seed=42,
        llm_repetition=0,
        model="model",
        prompt_schema_version="prompt",
        training_profile=profile,
    )
    diagnostics = _proposal_cache_key(
        case=case,
        perturbation_id="clean",
        split_seed=42,
        llm_repetition=0,
        model="model",
        prompt_schema_version="prompt",
        evidence_mode="training_only_structural_diagnostics",
        planner_diagnostics={"structural_complexity": "moderate"},
        training_profile=profile,
    )
    assert ordinary != diagnostics


def test_diagnostics_ablation_exposes_training_only_payload_to_initial_planner(tmp_path: Path):
    captured: list[dict] = []

    def factory(context: dict) -> ModelingPlan:
        captured.append(context)
        return ModelingPlan(
            recommended_method="linear",
            preprocessing=PreprocessingContract(numeric_scaling="standard"),
            reasoning="A complete training-only plan.",
            confidence=0.7,
        )

    result = run_evaluation(
        tmp_path / "diagnostics",
        cases=[_case()],
        ablation_spec=ablation_presets()["llm_with_diagnostics"],
        modeling_plan_factory=factory,
    )
    assert captured
    diagnostics = captured[0]["training_only_structural_diagnostics"]
    assert diagnostics is not None
    assert captured[0]["planner_evidence_mode"] == "training_only_structural_diagnostics"
    assert "holdout" not in json.dumps(diagnostics).lower()
    trial = result["trials"][0]
    assert trial["planner_structural_diagnostics_exposed"] is True
    assert trial["challenger_enabled"] is False
    assert trial["probe_enabled"] is False


def test_exact_model_provenance_and_reasoning_request_are_preserved():
    requested = "gpt-5.4-mini-2026-03-17"
    captured: dict = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Response",
                (),
                {
                    "output_parsed": ModelingPlan(
                        recommended_method="linear",
                        reasoning="A supported training-only plan.",
                        confidence=0.7,
                    ),
                    "output_text": "",
                    "id": "response-test",
                    "model": requested,
                },
            )()

    agents = OpenAIAgents(
        api_key="test",
        model=requested,
        respect_environment_model=False,
        generation_settings={
            "reasoning_effort": "medium",
            "temperature": None,
            "top_p": None,
            "seed": None,
        },
    )
    agents._client = type("Client", (), {"responses": Responses()})()
    agents.modeling_plan({"columns": []}, "classify target", "target", "classification")

    assert captured["model"] == requested
    assert captured["reasoning"] == {"effort": "medium"}
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "seed" not in captured
    assert agents.last_request_provenance["model_requested"] == requested
    assert agents.last_request_provenance["model_effective"] == requested
    assert agents.assert_effective_model(expected_model=requested) == requested
    with pytest.raises(LLMUnavailable, match="model mismatch"):
        agents.assert_effective_model(expected_model="gpt-5-mini-2025-08-07")
