import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.schemas import AgentPlan, ConflictResolution, PreprocessingContract
from app.validation import freeze_supervised_split, training_profile_frame
from evaluation.benchmarks import BenchmarkCase, default_benchmark_cases
from evaluation.empirical_reference import evaluate_empirical_reference
from evaluation.metrics import regret, summarize_trials
from evaluation.perturbations import default_perturbations
from evaluation.runner import run_evaluation


def _classification_frame(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    signal = rng.normal(size=rows)
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=rows),
            "target": np.where(signal > 0, "yes", "no"),
        }
    )


def _regression_frame(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    signal = rng.normal(size=rows)
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=rows),
            "target": 2.0 * signal + rng.normal(scale=0.2, size=rows),
        }
    )


def _case(frame: pd.DataFrame, name: str = "fixture", task: str = "classification") -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        dataframe_loader=lambda frame=frame: frame.copy(),
        target_column="target",
        question="Predict target from the supplied features.",
        expected_task_type=task,
        dataset_source="in-memory test fixture",
    )


def _plan(method: str = "linear", preprocessing: PreprocessingContract | None = None) -> AgentPlan:
    return AgentPlan(
        target_column="target",
        task_type="classification",
        recommended_method=method,
        preprocessing=preprocessing or PreprocessingContract(numeric_scaling="standard"),
        reasoning="The test agent selected a complete supported plan for the benchmark fixture.",
        confidence=0.7,
    )


def _deterministic_resolution(context, agent, deterministic):
    del context, agent
    return ConflictResolution(
        selected_target_column=deterministic.target_column,
        selected_task_type=deterministic.task_type,
        selected_method=deterministic.recommended_method,
        selected_preprocessing=deterministic.preprocessing,
        checks=["deterministic_plan_checked"],
        justification="The deterministic recommendation satisfies the supported schema and safety contract.",
        confidence=0.8,
    )


def _read_trials(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "trials.jsonl").read_text().splitlines()]


def test_empirical_reference_evaluates_all_supported_families_training_only():
    frame = _classification_frame(72)
    split = freeze_supervised_split(frame, "target", "classification", random_state=13)
    training = training_profile_frame(
        frame,
        "target",
        "classification",
        test_size=0.2,
        random_state=13,
        split=split,
    )
    profile = {"column_details": [
        {"name": column, "dtype": str(training[column].dtype), "semantic_type": "numeric", "missing": 0, "infinity": 0, "unique": training[column].nunique()}
        for column in ("signal", "noise")
    ] + [{"name": "target", "dtype": "object", "semantic_type": "categorical", "missing": 0, "unique": 2}]} 
    result = evaluate_empirical_reference(
        training, "target", "classification", profile, random_state=13
    )
    assert result["status"] == "evaluated"
    assert set(result["candidate_metrics"]) == {
        "linear",
        "regularized_linear",
        "tree_ensemble",
        "boosted_tree",
    }
    assert all(item["status"] == "evaluated" for item in result["candidate_metrics"].values())
    assert result["holdout_used"] is False


def test_empirical_reference_ranking_is_unchanged_by_holdout_only_changes():
    frame = _classification_frame(72)
    split = freeze_supervised_split(frame, "target", "classification", random_state=13)
    perturbed = frame.copy()
    perturbed.loc[list(split.holdout_row_positions), "signal"] = 999999.0
    left = training_profile_frame(frame, "target", "classification", test_size=0.2, random_state=13, split=split)
    right = training_profile_frame(perturbed, "target", "classification", test_size=0.2, random_state=13, split=split)
    profile = {
        "column_details": [
            {"name": column, "dtype": str(left[column].dtype), "semantic_type": "numeric", "missing": 0, "infinity": 0, "unique": left[column].nunique()}
            for column in ("signal", "noise")
        ] + [{"name": "target", "dtype": "object", "semantic_type": "categorical", "missing": 0, "unique": 2}]
    }
    first = evaluate_empirical_reference(left, "target", "classification", profile, random_state=13)
    second = evaluate_empirical_reference(right, "target", "classification", profile, random_state=13)
    assert first["ranking"] == second["ranking"]
    assert first["best_primary_mean"] == second["best_primary_mean"]
    assert first["candidate_metrics"] == second["candidate_metrics"]


def test_invalid_agent_proposal_is_persisted_and_not_fitted(tmp_path: Path):
    frame = _classification_frame(60)
    frame.loc[:3, "signal"] = np.nan
    case = _case(frame, "invalid_agent")

    def invalid_plan(context):
        del context
        return _plan(preprocessing=PreprocessingContract(numeric_imputation="none", numeric_scaling="standard"))

    result = run_evaluation(
        tmp_path / "evaluation",
        cases=[case],
        offline=False,
        agent_plan_factory=invalid_plan,
    )
    trial = _read_trials(tmp_path / "evaluation")[0]
    assert trial["agent_source"] == "mock"
    assert trial["agent_initial_valid"] is False
    assert "numeric_missing_values_are_handled" in {
        failure["code"] for failure in trial["agent_initial_validation_failures"]
    }
    assert trial["agent_initial_cv_metric"] is None
    assert trial["agent_initial_holdout_metrics"] == {}
    assert trial["unsafe_plan_intercepted"] is True
    assert trial["hard_validation"]["intervention_required"] is True
    assert trial["hard_validation"]["initial_proposal"]["status"] == "failed"
    assert trial["final_hard_invalid"] is False
    assert result["summary"]["agent_initial_invalid_count"] == 1


def test_disagreement_and_reconciliation_fields_are_recorded(tmp_path: Path):
    case = _case(_classification_frame(), "method_disagreement")

    def agent_plan(context):
        del context
        return _plan(
            method="tree_ensemble",
            preprocessing=PreprocessingContract(numeric_scaling="standard"),
        )

    run_evaluation(
        tmp_path / "evaluation",
        cases=[case],
        agent_plan_factory=agent_plan,
        reconciliation_factory=_deterministic_resolution,
    )
    trial = _read_trials(tmp_path / "evaluation")[0]
    assert trial["method_disagreement"] is True
    assert trial["preprocessing_disagreement"] is False
    assert trial["reconciliation_invoked"] is True
    assert trial["reconciliation_status"] == "succeeded"
    assert trial["reconciliation_method_source"] == "deterministic"
    assert trial["hard_validation"]["status"] == "passed"
    assert trial["soft_challenge"]["status"] == "disagreement"


def test_valid_soft_challenge_can_side_with_agent(tmp_path: Path):
    case = _case(_classification_frame(), "agent_wins_soft_challenge")

    def agent_plan(context):
        del context
        return _plan(method="tree_ensemble")

    def agent_resolution(context, agent, deterministic):
        del context, deterministic
        return ConflictResolution(
            selected_target_column=agent.target_column,
            selected_task_type=agent.task_type,
            selected_method=agent.recommended_method,
            selected_preprocessing=agent.preprocessing,
            checks=["both_hard_validated", "agent_preserved"],
            justification="Both proposals passed hard safety validation and the initial agent plan remains a defensible choice for the observed evidence.",
            confidence=0.75,
        )

    run_evaluation(
        tmp_path / "evaluation",
        cases=[case],
        agent_plan_factory=agent_plan,
        reconciliation_factory=agent_resolution,
    )
    trial = _read_trials(tmp_path / "evaluation")[0]
    assert trial["soft_challenge"]["status"] == "disagreement"
    assert trial["hard_validation"]["status"] == "passed"
    assert trial["reconciliation_method_source"] == "agent"
    assert trial["final_method"] == trial["agent_initial_method"]


def test_preprocessing_only_disagreement_is_material(tmp_path: Path):
    case = _case(_classification_frame(), "preprocessing_disagreement")

    def agent_plan(context):
        del context
        return _plan(method="linear", preprocessing=PreprocessingContract(numeric_scaling="none"))

    run_evaluation(
        tmp_path / "evaluation",
        cases=[case],
        agent_plan_factory=agent_plan,
        reconciliation_factory=_deterministic_resolution,
    )
    trial = _read_trials(tmp_path / "evaluation")[0]
    assert trial["method_disagreement"] is False
    assert trial["preprocessing_disagreement"] is True
    assert trial["reconciliation_invoked"] is True


def test_regret_formulas_for_both_tasks():
    assert regret("classification", 0.80, 0.75) == pytest.approx(0.05)
    assert regret("regression", 2.0, 2.5) == pytest.approx(0.5)


def test_aggregation_reports_rates_regrets_and_paired_outcomes():
    records = [
        {
            "task_type": "classification",
            "perturbation_id": "clean",
            "agent_initial_valid": True,
            "agreement_status": "disagreement",
            "reconciliation_invoked": True,
            "reconciliation_status": "succeeded",
            "unsafe_plan_intercepted": False,
            "final_valid": True,
            "agent_initial_method": "linear",
            "final_method": "tree_ensemble",
            "empirical_best_method": "tree_ensemble",
            "agent_normalized_regret": 0.05,
            "gated_normalized_regret": 0.0,
            "method_disagreement": True,
        },
        {
            "task_type": "classification",
            "perturbation_id": "target_copy_leakage",
            "agent_initial_valid": False,
            "agreement_status": "agreement",
            "reconciliation_invoked": False,
            "reconciliation_status": "not_invoked",
            "unsafe_plan_intercepted": True,
            "final_valid": False,
            "agent_initial_method": "linear",
            "final_method": None,
            "empirical_best_method": None,
            "agent_normalized_regret": None,
            "gated_normalized_regret": None,
            "method_disagreement": False,
        },
    ]
    summary = summarize_trials(records)
    assert summary["trial_count"] == 2
    assert summary["agent_initial_validity_rate"] == 0.5
    assert summary["unsafe_plan_interception_count"] == 1
    assert summary["gating_outcome_counts"]["gated_better_count"] == 1
    assert summary["gating_outcome_counts"]["tie_count"] == 0
    assert summary["agent_empirical_reference_match_rate"] == 0.0
    assert summary["gated_empirical_reference_match_rate"] == 1.0


def test_offline_and_mock_sources_are_never_labeled_openai(tmp_path: Path):
    case = _case(_classification_frame(48), "source_labels")
    result = run_evaluation(tmp_path / "evaluation", cases=[case], offline=True)
    trial = _read_trials(tmp_path / "evaluation")[0]
    assert trial["agent_source"] == "offline_fallback"
    assert "openai" not in result["summary"]["source_counts"]


def test_offline_wine_smoke_writes_required_artifacts(tmp_path: Path):
    wine = next(case for case in default_benchmark_cases() if case.name == "wine")
    output_dir = tmp_path / "wine-evaluation"

    result = run_evaluation(output_dir, cases=[wine], offline=True)

    assert result["summary"]["trial_count"] == 1
    assert result["summary"]["valid_trial_count"] == 1
    for filename in ("config.json", "trials.jsonl", "summary.json", "summary.md"):
        assert (output_dir / filename).is_file()
    assert "not a universal optimum" in (output_dir / "summary.md").read_text(encoding="utf-8")


def test_perturbations_are_seeded_and_repeatable():
    case = _case(_classification_frame(), "perturbation_fixture")
    frame = case.load()

    for perturbation in default_perturbations():
        if not perturbation.applies(case):
            continue
        first, first_changes = perturbation.apply(frame, 101, case)
        second, second_changes = perturbation.apply(frame, 101, case)
        pd.testing.assert_frame_equal(first, second)
        assert first_changes == second_changes


def test_target_copy_perturbation_is_intercepted(tmp_path: Path):
    case = _case(_classification_frame(48), "unsafe_perturbation")

    result = run_evaluation(
        tmp_path / "evaluation",
        cases=[case],
        offline=True,
        include_perturbations=True,
    )
    trials = _read_trials(tmp_path / "evaluation")
    target_copy = next(trial for trial in trials if trial["perturbation_id"] == "target_copy_leakage")

    assert target_copy["final_valid"] is False
    assert target_copy["unsafe_plan_intercepted"] is True
    assert "no_direct_target_copy_features" in target_copy["validation_failure_codes"]
    assert result["summary"]["validation_interception_rate"] == 1.0


def test_offline_deterministic_fields_are_reproducible(tmp_path: Path):
    case = _case(_classification_frame(48), "reproducibility")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    run_evaluation(first_dir, cases=[case], offline=True, seed=33)
    run_evaluation(second_dir, cases=[case], offline=True, seed=33)
    first = _read_trials(first_dir)[0]
    second = _read_trials(second_dir)[0]
    for field in (
        "split_contract",
        "deterministic_recommendation",
        "agreement_status",
        "candidate_cv_metrics",
        "empirical_best_method",
        "agent_regret",
        "gated_regret",
    ):
        assert first[field] == second[field]
