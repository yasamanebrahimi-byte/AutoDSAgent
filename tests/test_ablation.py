from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_moons

from app.deterministic import deterministic_recommendation, profile_dataframe
from app.deterministic_policy import DeterministicPolicy
from app.schemas import ModelingPlan, PreprocessingContract
from app.soft_challenge import decide_soft_challenge
from evaluation.ablation import PRIMARY_ABLATION_NAMES, ablation_presets, run_ablation_study
from evaluation.benchmarks import BenchmarkCase
from evaluation.runner import _proposal_cache_key, run_evaluation
from evaluation.confirmatory import (
    experiment_code_sha256,
    validate_confirmatory_completeness,
)


def _case() -> BenchmarkCase:
    frame = pd.DataFrame(
        {
            "signal": [float(index) for index in range(48)],
            "noise": [float(index % 5) for index in range(48)],
            "target": ["yes" if index % 2 else "no" for index in range(48)],
        }
    )
    return BenchmarkCase(
        name="ablation_fixture",
        dataframe=frame,
        target_column="target",
        question="Classify target from the supplied features.",
        expected_task_type="classification",
        dataset_source="in-memory test fixture",
    )


def _plan(context: dict) -> ModelingPlan:
    return ModelingPlan(
        recommended_method="linear",
        preprocessing=PreprocessingContract(numeric_scaling="standard"),
        reasoning="The paired test proposal is a complete training-only linear baseline.",
        confidence=0.7,
    )


def test_named_presets_are_explicit_and_versioned():
    presets = ablation_presets()
    assert set(PRIMARY_ABLATION_NAMES) == {
        "llm_only",
        "hard_validation_only",
        "deterministic_only",
        "always_reconcile",
        "probe_direct",
        "full",
    }
    assert presets["selective_calibrated"].interaction_diagnostics is False
    assert presets["interaction_boundary_aware"].interaction_diagnostics is True
    assert presets["empirical_probe"].empirical_probe is True
    assert presets["probe_first"].decision_mode == "probe_direct"
    assert presets["probe_first"].empirical_probe is True
    assert presets["full"].schema_version == "modeling-gate-ablation-v1"


def test_primary_ablation_effective_specs_are_unique():
    specs = ablation_presets()
    keys = []
    for name in PRIMARY_ABLATION_NAMES:
        values = specs[name].as_dict()
        values.pop("name")
        keys.append(json.dumps(values, sort_keys=True))
    assert len(keys) == len(set(keys))


def test_default_ablation_set_is_the_primary_four(tmp_path: Path):
    result = run_ablation_study(tmp_path / "default", cases=[_case()])

    assert tuple(result["summary"]["selected_ablations"]) == PRIMARY_ABLATION_NAMES


def test_high_confidence_only_ignores_calibration_reliability():
    kwargs = {
        "agent_method": "linear",
        "deterministic_method": "tree_ensemble",
        "deterministic_confidence": "high",
        "score_margin": 30.0,
        "diagnostics": {"sample_to_feature_ratio": 20, "effective_features_estimate": 5},
        "task_type": "classification",
        "calibration_artifact": {"regimes": {}},
        "strategy": "high_confidence_only",
    }
    assert decide_soft_challenge(**kwargs).decision == "challenge"
    assert decide_soft_challenge(**{**kwargs, "deterministic_confidence": "medium"}).decision == "abstain"
    assert decide_soft_challenge(**{**kwargs, "deterministic_confidence": "low"}).decision == "abstain"


def test_interaction_diagnostic_off_removes_its_score_contribution():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(96, 3))
    frame = pd.DataFrame(features, columns=["x0", "x1", "x2"])
    frame["target"] = features[:, 0] * features[:, 1]
    disabled = replace(DeterministicPolicy(), enable_regression_interaction_diagnostics=False)
    recommendation = deterministic_recommendation(
        frame,
        "Estimate target",
        target_hint="target",
        task_type="regression",
        policy=disabled,
    )
    signals = recommendation.diagnostics.interaction_signals
    assert signals.interaction_applicable is False
    assert signals.diagnostic_reason == "disabled_by_ablation"
    assert all(
        contribution.factor != "interaction"
        for assessment in recommendation.method_assessments.values()
        for contribution in assessment.contributions
    )


def test_boundary_diagnostic_off_is_explicit():
    features, target = make_moons(n_samples=120, noise=0.15, random_state=41)
    frame = pd.DataFrame(features, columns=["x0", "x1"])
    frame["target"] = target
    disabled = replace(DeterministicPolicy(), enable_classification_boundary_diagnostics=False)
    recommendation = deterministic_recommendation(
        frame,
        "Classify target",
        target_hint="target",
        task_type="classification",
        policy=disabled,
    )
    boundary = recommendation.diagnostics.classification_boundary_signals
    assert boundary.boundary_complexity_applicable is False
    assert boundary.boundary_diagnostic_reason == "disabled_by_ablation"


def test_proposal_cache_key_separates_repetition_and_split():
    case = _case()
    profile = profile_dataframe(case.load().iloc[:32])
    first = _proposal_cache_key(
        case=case,
        perturbation_id="clean",
        split_seed=42,
        llm_repetition=0,
        model="test-model",
        prompt_schema_version="test-v1",
        training_profile=profile,
    )
    repetition = _proposal_cache_key(
        case=case,
        perturbation_id="clean",
        split_seed=42,
        llm_repetition=1,
        model="test-model",
        prompt_schema_version="test-v1",
        training_profile=profile,
    )
    split = _proposal_cache_key(
        case=case,
        perturbation_id="clean",
        split_seed=123,
        llm_repetition=0,
        model="test-model",
        prompt_schema_version="test-v1",
        training_profile=profile,
    )
    assert len({first, repetition, split}) == 3


def test_same_initial_proposal_is_reused_across_ablation_presets(tmp_path: Path, monkeypatch):
    import evaluation.runner as runner

    monkeypatch.setattr(
        runner,
        "evaluate_empirical_reference",
        lambda *args, **kwargs: {
            "best_method": "linear",
            "best_primary_mean": 0.8,
            "candidate_metrics": {
                method: {"status": "evaluated", "primary_mean": 0.8}
                for method in ("linear", "regularized_linear", "tree_ensemble", "boosted_tree")
            },
        },
    )
    monkeypatch.setattr(runner, "evaluate_plan_cv", lambda *args, **kwargs: {"primary_mean": 0.8})
    monkeypatch.setattr(
        runner,
        "evaluate_holdout_plan",
        lambda *args, **kwargs: {"holdout_metrics": {}, "validation": {"split": {"contract": {}}}},
    )
    calls: list[dict] = []

    def factory(context):
        calls.append(context)
        return _plan(context)

    result = run_ablation_study(
        tmp_path / "paired",
        cases=[_case()],
        split_seeds=[42, 123],
        repetitions=2,
        ablations=["llm_only", "selective_calibrated"],
        modeling_plan_factory=factory,
    )
    assert len(calls) == 4  # two split seeds x two LLM repetitions, not x ablations
    rows = result["summary"]["summaries"]
    assert rows["llm_only"]["trial_count"] == 4
    assert rows["selective_calibrated"]["trial_count"] == 4
    cached = [
        json.loads(line)
        for line in (tmp_path / "paired" / "selective_calibrated" / "trials.jsonl").read_text().splitlines()
    ]
    assert all(row["initial_proposal_cache_hit"] is True for row in cached)
    assert all(row["initial_modeling_call_made"] is False for row in cached)


def test_deterministic_only_does_not_call_initial_factory(tmp_path: Path):
    def forbidden(_context):
        raise AssertionError("deterministic_only must not request an initial modeling plan")

    result = run_evaluation(
        tmp_path / "deterministic",
        cases=[_case()],
        gate_mode="deterministic_only",
        modeling_plan_factory=forbidden,
        offline=False,
    )
    assert result["trials"][0]["agent_source"] == "deterministic_only"
    assert result["trials"][0]["initial_modeling_call_made"] is False


def test_strict_live_records_failure_without_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run_evaluation(
        tmp_path / "strict",
        cases=[_case()],
        require_live=True,
        offline=False,
    )
    trial = result["trials"][0]
    assert trial["trial_status"] == "failed"
    assert trial["agent_source"] == "failed"
    assert trial["agent_initial"] is None
    assert trial.get("fallback_row") is False


def test_proposal_cache_contains_no_credentials(tmp_path: Path):
    output = tmp_path / "offline"
    run_evaluation(output, cases=[_case()], offline=True, proposal_cache_path=output / "proposal_cache.jsonl")
    content = (output / "proposal_cache.jsonl").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in content
    assert "api_key" not in content.lower()
    assert "Authorization" not in content


def test_confirmatory_orchestrator_executes_complete_multi_model_matrix(tmp_path: Path, monkeypatch):
    """Exercise the paper-level loop with a no-API, two-condition fixture."""
    import evaluation.ablation as ablation
    import evaluation.metrics as metrics
    import evaluation.runner as runner

    original_cluster_bootstrap_ci = metrics.cluster_bootstrap_ci
    monkeypatch.setattr(
        metrics,
        "cluster_bootstrap_ci",
        lambda data, statistic, cluster_col, **kwargs: original_cluster_bootstrap_ci(
            data, statistic, cluster_col, n_bootstrap=20, **kwargs
        ),
    )

    for module in (runner,):
        monkeypatch.setattr(
            module,
            "evaluate_empirical_reference",
            lambda *args, **kwargs: {
                "best_method": "linear",
                "best_primary_mean": 0.8,
                "candidate_metrics": {
                    method: {"status": "evaluated", "primary_mean": 0.8}
                    for method in ("linear", "regularized_linear", "tree_ensemble", "boosted_tree")
                },
            },
        )
        monkeypatch.setattr(module, "evaluate_plan_cv", lambda *args, **kwargs: {"primary_mean": 0.8})
        monkeypatch.setattr(
            module,
            "evaluate_holdout_plan",
            lambda *args, **kwargs: {"holdout_metrics": {}, "validation": {"split": {"contract": {}}}},
        )
    manifest = json.loads(
        (Path(__file__).parents[1] / "evaluation/configs/paper_confirmatory_v1.json").read_text()
    )
    manifest.update({"status": "frozen", "expected_experiment_code_sha256": experiment_code_sha256()})
    manifest["model_conditions"] = [
        {"condition_id": "model_a", "planner_model": "planner-a", "reconciler_model": "reconciler-a", "llm_repetitions": 2, "generation_settings": {"temperature": None}},
        {"condition_id": "model_b", "planner_model": "planner-b", "reconciler_model": "reconciler-b", "llm_repetitions": 2, "generation_settings": {"temperature": None}},
    ]
    manifest["splits_and_repetitions"] = {"split_seeds": [42], "llm_repetitions": 2, "llm_repetition_ids": ["r1", "r2"]}
    manifest["ablations"]["primary"] = ["llm_only", "full"]
    manifest_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metadata = {
        "status": "frozen", "experiment_config_sha256": "fixture", "expected_experiment_code_sha256": experiment_code_sha256(),
        "source_git_commit": None, "benchmark_manifest_matches": True,
    }
    monkeypatch.setattr(ablation, "validate_confirmatory_manifest", lambda *_args: metadata)
    monkeypatch.setattr(runner, "validate_confirmatory_manifest", lambda *_args: metadata)
    calls: list[dict] = []

    def factory(context):
        calls.append(context)
        return _plan(context)

    result = run_ablation_study(
        tmp_path / "matrix",
        cases=[_case(), replace(_case(), name="ablation_fixture_2")],
        suite="external",
        offline=True,
        confirmatory_config_path=manifest_path,
        ablations=["llm_only"],  # frozen manifest must win over this runtime flag
        modeling_plan_factory=factory,
    )
    assert set(result["summary"]["by_model_condition"]) == {"model_a", "model_b"}
    assert result["summary"]["model_condition_reporting"]["combined_summary_role"].startswith(
        "descriptive audit total"
    )
    assert result["summary"]["analysis_summaries"]["secondary"] == {}
    assert len(result["summary"]["summaries"]["llm_only"]["by_dataset"]) == 2
    # The combined summary is condition-aware; inspect persisted trial rows from both ablations.
    persisted = []
    for name in ("llm_only", "full"):
        path = tmp_path / "matrix" / name
        for condition_dir in (path / "model_a", path / "model_b"):
            persisted.extend(json.loads(line) for line in (condition_dir / "trials.jsonl").read_text().splitlines())
    assert len(persisted) == 16
    assert len({row["trial_id"] for row in persisted}) == 16
    assert {row["model_condition_id"] for row in persisted} == {"model_a", "model_b"}
    assert {row["llm_repetition_id"] for row in persisted} == {"r1", "r2"}
    assert len(calls) == 8  # condition x repetition x dataset, never x ablation
    assert len({row["initial_proposal_cache_key"] for row in persisted}) == 8
    assert all(row["initial_proposal_cache_hit"] for row in persisted if row["ablation_name"] == "full")
    assert result["summary"]["confirmatory_matrix"]["complete"] is True

    expected = []
    for condition_id in ("model_a", "model_b"):
        for repetition_id in ("r1", "r2"):
            for case_name in ("ablation_fixture", "ablation_fixture_2"):
                for ablation_name in ("llm_only", "full"):
                    expected.append({"model_condition_id": condition_id, "llm_repetition_id": repetition_id, "benchmark_case": case_name, "perturbation_id": "clean", "split_seed": 42, "ablation_name": ablation_name})
    validate_confirmatory_completeness(expected, persisted)
    with pytest.raises(ValueError, match="incomplete"):
        validate_confirmatory_completeness(expected, persisted[:-1])
