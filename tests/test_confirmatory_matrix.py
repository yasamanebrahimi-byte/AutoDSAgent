from __future__ import annotations

from evaluation.confirmatory import (
    expand_confirmatory_matrix,
    model_conditions,
    repetition_ids,
    runtime_manifest_values,
    validate_confirmatory_manifest,
)
from evaluation.runner import _proposal_cache_key
from evaluation.benchmarks import BenchmarkCase


def _manifest():
    return {
        "model_conditions": [
            {"condition_id": "a", "planner_model": "planner-a", "reconciler_model": "planner-a", "llm_repetitions": 2},
            {"condition_id": "b", "planner_model": "planner-b", "reconciler_model": "reconciler-b", "llm_repetitions": 2},
        ],
        "splits_and_repetitions": {"llm_repetitions": 2, "llm_repetition_ids": ["r1", "r2"]},
        "ablations": {"primary": ["llm_only", "full"]},
    }


def test_confirmatory_matrix_expands_conditions_repetitions_and_ablations():
    rows = expand_confirmatory_matrix(_manifest(), dataset_ids=["task-1"])
    assert len(rows) == 8
    assert {(row["condition_id"], row["llm_repetition_id"], row["ablation"]) for row in rows}.__len__() == 8


def test_repetition_ids_are_explicit_and_conditions_are_normalized():
    assert repetition_ids(_manifest()) == ["r1", "r2"]
    assert [row["condition_id"] for row in model_conditions(_manifest())] == ["a", "b"]


def test_proposal_cache_identity_separates_models_and_repetitions():
    case = BenchmarkCase("task", None, "q", "classification", "test")
    common = dict(case=case, perturbation_id="clean", split_seed=42, model="same", prompt_schema_version="p", training_profile={})
    assert _proposal_cache_key(**common, llm_repetition=1, model_condition_id="a") != _proposal_cache_key(**common, llm_repetition=1, model_condition_id="b")
    assert _proposal_cache_key(**common, llm_repetition=1, llm_repetition_id="r1") != _proposal_cache_key(**common, llm_repetition=2, llm_repetition_id="r2")


def test_runtime_matrix_projection_can_detect_condition_set_drift():
    values = runtime_manifest_values(
        experiment_name="e", planner_model="planner-a", reconciler_model="planner-a",
        split_seeds=[42], llm_repetitions=2, holdout_fraction=.2, selected_ablations=["full"],
        deterministic_policy_version="d", empirical_probe_policy_version="p",
        planner_prompt_schema_version="p", reconciler_prompt_schema_version="r",
        candidate_model_families=[], classification_neutral_tolerance=.02,
        regression_neutral_tolerance=.02, benchmark_manifest_version="b",
        strict_live_required=True, bootstrap_settings={}, experiment_config_version="v",
        model_conditions=model_conditions(_manifest()), llm_repetition_ids=["r1", "r2"],
    )
    assert len(values["model_conditions"]) == 2
