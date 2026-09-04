from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.benchmarks import BENCHMARK_SUITE_VERSION, BenchmarkCase, BenchmarkRole, default_benchmark_cases
from evaluation.policy_calibration import (
    aggregate_candidate_records,
    calibration_cases,
    policy_candidates,
    run_policy_calibration,
    select_policy_candidate,
)
from evaluation.policy_evaluation import run_policy_evaluation
from app.deterministic_policy import DeterministicPolicy


def _development_case(name: str = "calibration_fixture") -> BenchmarkCase:
    rng = np.random.default_rng(5)
    signal = rng.normal(size=72)
    return BenchmarkCase(
        name=name,
        dataframe=pd.DataFrame(
            {
                "signal": signal,
                "noise": rng.normal(size=len(signal)),
                "target": np.where(signal > 0, "yes", "no"),
            }
        ),
        target_column="target",
        question="Classify target from the supplied features.",
        expected_task_type="classification",
        dataset_source="deterministic test fixture",
        role=BenchmarkRole.POLICY_DEVELOPMENT,
    )


def _final_case(name: str = "final_fixture") -> BenchmarkCase:
    case = _development_case(name)
    return BenchmarkCase(
        **{
            **case.__dict__,
            "role": BenchmarkRole.FINAL_EVALUATION,
        }
    )


def test_default_benchmark_roles_are_fixed_and_non_overlapping():
    cases = default_benchmark_cases()
    development = [case for case in cases if case.role is BenchmarkRole.POLICY_DEVELOPMENT]
    final = [case for case in cases if case.role is BenchmarkRole.FINAL_EVALUATION]

    assert BENCHMARK_SUITE_VERSION == "2"
    assert len(cases) >= 12
    assert len(development) >= 8
    assert len(final) >= 4
    assert not ({case.name for case in development} & {case.name for case in final})
    assert all(case.as_dict()["role"] in {role.value for role in BenchmarkRole} for case in cases)
    assert [case.name for case in cases] == [case.name for case in default_benchmark_cases()]


def test_policy_configuration_exposes_thresholds_and_score_contributions():
    policy = DeterministicPolicy()
    serialized = asdict(policy)
    assert serialized["structural_complexity_high_threshold"] == 0.60
    assert "compatibility_points" in serialized
    assert policy.compatibility_point("nonlinearity", 17, "boosted_tree") == 17

    point_table = dict(policy.compatibility_points)
    method_points = dict(point_table["nonlinearity:17"])
    method_points["boosted_tree"] = 12
    point_table["nonlinearity:17"] = tuple(method_points.items())
    changed = replace(policy, compatibility_points=tuple(point_table.items()))
    assert changed.compatibility_point("nonlinearity", 17, "boosted_tree") == 12


def test_calibration_and_final_evaluation_reject_wrong_roles():
    with pytest.raises(ValueError, match="final-evaluation cases"):
        calibration_cases([_development_case(), _final_case()])
    with pytest.raises(ValueError, match="policy-development cases"):
        from evaluation.policy_calibration import final_evaluation_cases

        final_evaluation_cases([_development_case()])


def test_selection_rule_prefers_expected_candidate_and_retains_when_not_better():
    candidates = policy_candidates()
    aggregates = {
        "current": {
            "policy_candidate": "current",
            "policy_complexity": 0,
            "utility": 0.0,
            "catastrophic_introduced_count": 0,
            "harmful_intervention_rate": 0.20,
            "catastrophic_prevented_count": 1,
            "intervention_precision": 0.70,
            "challenge_recall": 0.70,
            "median_regret_reduction": 0.20,
            "unnecessary_intervention_rate": 0.25,
            "exact_reference_match_rate": 0.70,
        },
        "nonlinear_sensitive": {
            "policy_candidate": "nonlinear_sensitive",
            "policy_complexity": 3,
            "utility": 1.0,
            "catastrophic_introduced_count": 0,
            "harmful_intervention_rate": 0.10,
            "catastrophic_prevented_count": 2,
            "intervention_precision": 0.80,
            "challenge_recall": 0.80,
            "median_regret_reduction": 0.30,
            "unnecessary_intervention_rate": 0.10,
            "exact_reference_match_rate": 0.80,
        },
        "high_dimensional_sensitive": {
            "policy_candidate": "high_dimensional_sensitive",
            "policy_complexity": 3,
            "utility": 0.5,
            "catastrophic_introduced_count": 0,
            "harmful_intervention_rate": 0.15,
            "catastrophic_prevented_count": 1,
            "intervention_precision": 0.75,
            "challenge_recall": 0.75,
            "median_regret_reduction": 0.25,
            "unnecessary_intervention_rate": 0.15,
            "exact_reference_match_rate": 0.75,
        },
        "missingness_sensitive": {
            "policy_candidate": "missingness_sensitive",
            "policy_complexity": 3,
            "utility": 0.2,
            "catastrophic_introduced_count": 0,
            "harmful_intervention_rate": 0.18,
            "catastrophic_prevented_count": 1,
            "intervention_precision": 0.72,
            "challenge_recall": 0.72,
            "median_regret_reduction": 0.22,
            "unnecessary_intervention_rate": 0.18,
            "exact_reference_match_rate": 0.72,
        },
    }
    selected = select_policy_candidate(aggregates, candidates)
    assert selected["selected_candidate"] == "nonlinear_sensitive"
    assert selected["recommendation"] == "promote"

    for aggregate in aggregates.values():
        aggregate["utility"] = 0.0
        aggregate["harmful_intervention_rate"] = 0.20
        aggregate["catastrophic_prevented_count"] = 1
        aggregate["intervention_precision"] = 0.70
        aggregate["challenge_recall"] = 0.70
        aggregate["median_regret_reduction"] = 0.20
        aggregate["unnecessary_intervention_rate"] = 0.25
        aggregate["exact_reference_match_rate"] = 0.70
    retained = select_policy_candidate(aggregates, candidates)
    assert retained["recommendation"] == "retain_current"


def test_aggregate_metrics_average_seeds_within_dataset_first():
    candidate = policy_candidates()[0]
    records = [
        {
            "policy_candidate": "current",
            "dataset_id": "a",
            "seed": 1,
            "normalized_regret": 0.0,
            "exact_reference_match": True,
            "catastrophic_regret": False,
            "top2_compatibility_success": True,
            "deterministic_selected_method": "linear",
        },
        {
            "policy_candidate": "current",
            "dataset_id": "a",
            "seed": 2,
            "normalized_regret": 0.2,
            "exact_reference_match": False,
            "catastrophic_regret": True,
            "top2_compatibility_success": True,
            "deterministic_selected_method": "tree_ensemble",
        },
        {
            "policy_candidate": "current",
            "dataset_id": "b",
            "seed": 1,
            "normalized_regret": 0.4,
            "exact_reference_match": False,
            "catastrophic_regret": True,
            "top2_compatibility_success": False,
            "deterministic_selected_method": "boosted_tree",
        },
    ]
    aggregate = aggregate_candidate_records(records, candidate)
    assert aggregate["dataset_count"] == 2
    assert aggregate["mean_normalized_regret"] == pytest.approx((0.1 + 0.4) / 2)
    assert aggregate["policy_stability"]["datasets_with_seed_selection_variation"] == 1


def test_policy_calibration_smoke_writes_training_only_artifacts(tmp_path: Path):
    result = run_policy_calibration(
        tmp_path / "calibration",
        cases=[_development_case()],
        seeds=(11,),
        candidates=policy_candidates()[:2],
    )
    assert result["evaluation_role"] == "policy_development"
    assert result["dataset_count"] == 1
    assert result["recommendation"] in {"retain_current", "promote"}
    assert all(record["holdout_used"] is False for record in result["raw_records"])
    artifact = json.loads((tmp_path / "calibration" / "policy_calibration.json").read_text())
    assert artifact["selection_rule"]
    assert artifact["failure_cases"] is not None
    assert (tmp_path / "calibration" / "policy_calibration_report.md").is_file()


def test_final_policy_evaluation_is_frozen_and_reports_holdout(tmp_path: Path):
    result = run_policy_evaluation(
        tmp_path / "final",
        cases=[_final_case()],
        seeds=(11,),
    )
    assert result["evaluation_role"] == "final_evaluation"
    assert result["policy_version"] == "4"
    assert all(record["holdout_used"] is True for record in result["raw_records"])
    assert all(record["holdout_evaluation"] is not None for record in result["raw_records"])
    assert (tmp_path / "final" / "policy_evaluation.json").is_file()


def test_runtime_app_has_no_calibration_import_boundary():
    root = Path(__file__).resolve().parents[1] / "app"
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "evaluation.policy_calibration" not in source
        assert "evaluation.policy_evaluation" not in source
        assert "from evaluation" not in source
