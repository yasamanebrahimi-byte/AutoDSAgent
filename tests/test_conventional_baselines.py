from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.conventional_baselines import (
    BASELINE_ANALYSIS_ROLE,
    MissingHistoricalFields,
    analyze_result_directory,
    derive_baseline_trials,
    select_all_four_cv,
    select_pairwise_cv_always,
    summarize_baseline_trials,
)
import evaluation.conventional_baselines as conventional_baselines
import evaluation.prospective as prospective
from evaluation.external_benchmarks import (
    PROSPECTIVE_PANEL_MANIFEST_SCHEMA_VERSION,
    freeze_prospective_panel_manifest,
    prospective_benchmark_cases,
    prospective_panel_content_sha256,
    validate_prospective_panel_manifest,
)
from app.validation import freeze_supervised_split


def _contract(*, scaling: str = "standard") -> dict[str, object]:
    return {
        "numeric_imputation": "median",
        "categorical_imputation": "none",
        "numeric_scaling": scaling,
        "categorical_encoding": "none",
        "categorical_unknown_handling": "ignore",
        "identifier_handling": "exclude",
        "high_cardinality_handling": "exclude",
        "unsupported_text_handling": "exclude",
        "datetime_handling": "exclude",
        "infinity_handling": "replace_with_missing",
        "fit_inside_pipeline": True,
    }


def _probe(a: float, b: float, *, evidence: str = "tie") -> dict[str, object]:
    return {
        "status": "completed",
        "cv_folds": 3,
        "fit_count": 6,
        "evidence_strength": evidence,
        "winner": "tie",
        "proposal_a": {"model_family": "linear", "mean_score": a, "preprocessing": _contract()},
        "proposal_b": {"model_family": "tree_ensemble", "mean_score": b, "preprocessing": _contract(scaling="none")},
    }


def _candidate_metrics(values: dict[str, float], *, task_type: str = "classification") -> dict[str, object]:
    metric = "macro_f1" if task_type == "classification" else "rmse"
    return {
        method: {
            "status": "evaluated",
            "primary_metric": metric,
            "primary_mean": value,
            "primary_std": 0.01,
            "cv_folds": 3,
            "validation": {"approved_preprocessing": _contract()},
        }
        for method, value in values.items()
    }


def _row(*, task_type: str = "classification", probe: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "trial_id": "condition:task:clean:split42:rep_001:llm0:probe_direct",
        "benchmark_case": "task",
        "dataset_source": "fixture",
        "task_type": task_type,
        "model_condition_id": "condition",
        "llm_repetition_id": "rep_001",
        "trial": 0,
        "split_seed": 42,
        "test_size": 0.2,
        "agent_initial_target": "target",
        "agent_initial_method": "linear",
        "agent_initial_preprocessing": _contract(),
        "agent_initial_valid": True,
        "deterministic_method": "tree_ensemble",
        "deterministic_preprocessing": _contract(scaling="none"),
        "deterministic_valid": True,
        "proposal_a_source": "agent",
        "proposal_b_source": "deterministic",
        "empirical_probe": probe,
        "candidate_cv_metrics": _candidate_metrics(
            {"linear": 0.70, "regularized_linear": 0.71, "tree_ensemble": 0.75, "boosted_tree": 0.73}
            if task_type == "classification"
            else {"linear": 10.0, "regularized_linear": 9.0, "tree_ensemble": 8.0, "boosted_tree": 8.5},
            task_type=task_type,
        ),
        "initial_holdout_metric": 0.70 if task_type == "classification" else 10.0,
        "final_method": "tree_ensemble",
        "final_preprocessing": _contract(scaling="none"),
        "final_holdout_metric": 0.75 if task_type == "classification" else 8.0,
        "ablation_name": "probe_direct",
        "gate_mode": "probe_direct",
        "trial_status": "completed",
    }


def test_pairwise_classification_uses_higher_raw_mean_even_when_probe_says_tie():
    result = select_pairwise_cv_always(
        "classification", "linear", "tree_ensemble", _probe(0.70, 0.71),
        challenger_valid=True,
        proposal_a_source="agent", proposal_b_source="deterministic",
    )
    assert result["selected_baseline_family"] == "tree_ensemble"
    assert result["raw_mean_cv_winner"] == "B"
    assert result["evidence_strength_ignored"] is True


def test_pairwise_regression_uses_lower_raw_mean():
    result = select_pairwise_cv_always(
        "regression", "linear", "tree_ensemble", _probe(10.0, 8.0),
        challenger_valid=True,
        proposal_a_source="agent", proposal_b_source="deterministic",
    )
    assert result["selected_baseline_family"] == "tree_ensemble"
    assert result["selection_rule"] == "lower_mean_cv_rmse"


def test_pairwise_true_numerical_tie_and_agreement_preserve_incumbent():
    tied = select_pairwise_cv_always(
        "classification", "linear", "tree_ensemble", _probe(0.7, 0.7),
        challenger_valid=True,
        proposal_a_source="agent", proposal_b_source="deterministic",
    )
    assert tied["selected_baseline_family"] == "linear"
    assert tied["selection_source"] == "initial_llm_tie"
    agreed = select_pairwise_cv_always("classification", "linear", "linear", None)
    assert agreed["selected_baseline_family"] == "linear"
    assert agreed["selection_source"] == "initial_llm_incumbent"


def test_pairwise_hard_invalid_initial_preserves_repair_semantics_without_probe():
    repaired = select_pairwise_cv_always(
        "classification", "linear", "tree_ensemble", None,
        initial_valid=False, challenger_valid=True,
        challenger_preprocessing=_contract(scaling="none"),
    )
    assert repaired["selected_baseline_family"] == "tree_ensemble"
    assert repaired["selection_source"] == "hard_validation_repair"
    unresolved = select_pairwise_cv_always(
        "classification", "linear", "tree_ensemble", None,
        initial_valid=False, challenger_valid=False,
    )
    assert unresolved["status"] == "not_evaluable"


def test_pairwise_requires_raw_proposal_means_for_actionable_disagreement():
    with pytest.raises(MissingHistoricalFields, match="raw mean-CV"):
        select_pairwise_cv_always(
            "classification", "linear", "tree_ensemble", {"winner": "tie"}, challenger_valid=True
        )


def test_all_four_uses_all_eligible_families_and_existing_tie_break():
    metrics = _candidate_metrics(
        {"linear": 0.80, "regularized_linear": 0.80, "tree_ensemble": 0.70, "boosted_tree": 0.60}
    )
    selected = select_all_four_cv("classification", metrics)
    assert selected["selected_baseline_family"] == "linear"
    assert selected["ranking"][:2] == ["linear", "regularized_linear"]
    assert selected["distinct_candidate_families_cv_evaluated"] == 4
    assert selected["counterfactual_selection_fit_count"] == 12


def test_all_four_regression_direction_and_selected_canonical_preprocessing():
    metrics = _candidate_metrics(
        {"linear": 10.0, "regularized_linear": 9.0, "tree_ensemble": 8.0, "boosted_tree": 8.5},
        task_type="regression",
    )
    metrics["tree_ensemble"]["validation"] = {"approved_preprocessing": _contract(scaling="none")}
    selected = select_all_four_cv("regression", metrics)
    assert selected["selected_baseline_family"] == "tree_ensemble"
    assert selected["selected_preprocessing"]["numeric_scaling"] == "none"


def test_derived_rows_have_provenance_coverage_and_no_reconciler_calls():
    rows = derive_baseline_trials([_row(probe=_probe(0.70, 0.71))], source_run="fixture-run")
    pair = next(row for row in rows if row["baseline_name"] == "pairwise_cv_always")
    all4 = next(row for row in rows if row["baseline_name"] == "all_four_cv")
    assert pair["analysis_role"] == BASELINE_ANALYSIS_ROLE
    assert pair["selected_baseline_family"] == "tree_ensemble"
    assert pair["reconciler_invoked"] is False
    assert pair["reconciler_llm_call_count"] == 0
    assert all4["selected_baseline_family"] == "tree_ensemble"
    assert all4["two_candidate_coverage"] is True
    assert all4["challenger_incremental_hit"] is True


def test_derived_selection_does_not_read_holdout_for_pairwise_decision():
    row = _row(probe=_probe(0.70, 0.71))
    row["initial_holdout_metric"] = 0.99
    row["final_holdout_metric"] = 0.01
    derived = derive_baseline_trials([row], baselines=["pairwise_cv_always"])[0]
    assert derived["selected_baseline_family"] == "tree_ensemble"
    assert derived["selection_cv_values"]["proposal_b_mean_score"] == pytest.approx(0.71)


def test_cached_and_recomputed_reference_decisions_are_equivalent():
    row = _row(probe=_probe(0.70, 0.71))
    cached = derive_baseline_trials([row], baselines=["all_four_cv"])[0]
    row_without_candidate = copy.deepcopy(row)
    row_without_candidate["candidate_cv_metrics"] = {}
    with pytest.raises(MissingHistoricalFields):
        derive_baseline_trials([row_without_candidate], baselines=["all_four_cv"], strict=True)
    assert cached["selected_baseline_family"] == "tree_ensemble"


def test_recomputed_reference_reuses_same_selection_rule_as_cached(monkeypatch):
    row = _row(probe=_probe(0.70, 0.71))
    cached = select_all_four_cv("classification", row["candidate_cv_metrics"])
    frame = pd.DataFrame({"feature": list(range(40)), "target": [0, 1] * 20})
    split = freeze_supervised_split(frame, "target", "classification", test_size=0.2, random_state=42)
    row_without_candidate = copy.deepcopy(row)
    row_without_candidate["candidate_cv_metrics"] = {}
    row_without_candidate["split_contract"] = split.as_dict()
    row_without_candidate["agent_initial_target"] = "target"
    monkeypatch.setattr(
        conventional_baselines,
        "evaluate_empirical_reference",
        lambda *args, **kwargs: {"candidate_metrics": row["candidate_cv_metrics"]},
    )
    recomputed, reused = conventional_baselines._reference_for_row(
        row_without_candidate,
        frame_loader={"task": frame},
        recompute_missing=True,
        reference_cache={},
        frame_cache={},
    )
    assert reused is False
    assert recomputed["selected_baseline_family"] == cached["selected_baseline_family"]
    assert recomputed["selected_preprocessing"] == cached["selected_preprocessing"]


def test_fit_accounting_and_summary_keep_dataset_as_statistical_unit():
    first = _row(probe=_probe(0.70, 0.71))
    second = copy.deepcopy(first)
    second["trial_id"] = "condition:task:clean:split42:rep_002:llm1:probe_direct"
    second["llm_repetition_id"] = "rep_002"
    derived = derive_baseline_trials([first, second], source_run="fixture")
    summary = summarize_baseline_trials(derived, bootstrap_replicates=20)
    all4 = summary["baseline_summaries"]["all_four_cv"]
    assert all4["counterfactual_selection_fit_count_total"] == 12
    assert summary["candidate_set_coverage"]["dataset_count"] == 1
    assert summary["independent_statistical_unit"] == "dataset/task"


def _arm_row(
    arm: str,
    repetition: str,
    *,
    condition: str = "condition-a",
    split_seed: int = 42,
) -> dict[str, object]:
    row = _row(probe=_probe(0.70, 0.71))
    row.update({
        "trial_id": f"{condition}:task:clean:split{split_seed}:{repetition}:{arm}",
        "llm_repetition_id": repetition,
        "model_condition_id": condition,
        "provider": "openai",
        "planner_model": "planner-fixture",
        "split_seed": split_seed,
        "ablation_name": arm,
        "gate_mode": arm,
    })
    if arm == "llm_only":
        row.update({
            "empirical_probe": None,
            "deterministic_method": None,
            "deterministic_preprocessing": None,
            "deterministic_valid": None,
            "final_method": "linear",
            "final_preprocessing": _contract(),
            "final_holdout_metric": 0.70,
        })
    elif arm == "full":
        row.update({
            "final_method": "tree_ensemble",
            "final_preprocessing": _contract(),
            "final_holdout_metric": 0.80,
        })
    return row


def _write_arm_tree(root: Path, *, include_direct: bool = True, conditions: tuple[str, ...] = ("condition-a",)) -> None:
    arms = ["llm_only", "probe_direct", "full"] if include_direct else ["llm_only", "full"]
    for arm in arms:
        for condition in conditions:
            directory = root / arm / condition
            directory.mkdir(parents=True, exist_ok=True)
            config = {
                "ablation_name": arm,
                "model_condition_id": condition,
                "provider": "openai",
                "planner_model": "planner-fixture",
                "reconciler_model": "planner-fixture",
                "evaluation_id": "fixture-tree",
            }
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            rows = [_arm_row(arm, "rep_001", condition=condition), _arm_row(arm, "rep_002", condition=condition)]
            (directory / "trials.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )


def test_realistic_cross_ablation_tree_is_grouped_before_derivation(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    _write_arm_tree(source)
    result = analyze_result_directory(source, output, bootstrap_replicates=10)
    assert len(result["trials"]) == 2 * 4
    for baseline in ("llm_only", "pairwise_cv_always", "probe_direct", "all_four_cv"):
        baseline_rows = [row for row in result["trials"] if row["baseline_name"] == baseline]
        assert len(baseline_rows) == 2
        assert {row["llm_repetition_id"] for row in baseline_rows} == {"rep_001", "rep_002"}
    logical_keys = {tuple(row["logical_trial_key"]) for row in result["trials"]}
    assert len(logical_keys) == 2
    pair = next(row for row in result["trials"] if row["baseline_name"] == "pairwise_cv_always")
    assert pair["source_arm_for_baseline"] == "probe_direct"
    assert pair["source_arm_roles"]["llm_only_holdout"]["source_ablation"] == "llm_only"
    assert pair["source_arm_roles"]["empirical_probe"]["source_ablation"] == "probe_direct"
    assert pair["source_arm_roles"]["probe_direct_selective"]["source_ablation"] == "probe_direct"
    assert pair["source_arm_roles"]["full_all_four_empirical_reference"]["source_ablation"] == "full"
    assert pair["selected_baseline_family"] == "tree_ensemble"
    assert result["summary"]["incomplete_logical_trial_count"] == 0
    summary_json = json.loads((output / "baseline_summary.json").read_text(encoding="utf-8"))
    assert "condition-a" in summary_json["comparisons_by_model_condition"]
    assert "condition-a" in (output / "baseline_summary.csv").read_text(encoding="utf-8")
    assert "condition-a" in (output / "baseline_summary.md").read_text(encoding="utf-8")


def test_cross_ablation_missing_companion_is_explicit(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    _write_arm_tree(source, include_direct=False)
    result = analyze_result_directory(source, output, bootstrap_replicates=5)
    direct_rows = [row for row in result["trials"] if row["baseline_name"] == "probe_direct"]
    assert len(direct_rows) == 2
    assert all(row["baseline_status"] == "missing_artifact" for row in direct_rows)
    assert any("probe_direct_selective" in role for role in result["summary"]["missing_companion_diagnostics"][0]["missing_source_roles"])
    assert "missing companion source role" in " ".join(direct_rows[0]["missing_fields"])


def test_all_four_cost_is_deduplicated_by_dataset_and_split_not_repetition():
    first = _row(probe=_probe(0.70, 0.71))
    second = copy.deepcopy(first)
    second.update({"trial_id": "condition:task:clean:split43:rep_001:llm0", "split_seed": 43, "llm_repetition_id": "rep_001"})
    derived = derive_baseline_trials([first, second])
    summary = summarize_baseline_trials(derived, bootstrap_replicates=5)
    assert summary["baseline_summaries"]["all_four_cv"]["counterfactual_selection_fit_count_total"] == 24


def test_compute_accounting_separates_cached_analysis_from_deployed_policy_cost():
    rows = derive_baseline_trials([_row(probe=_probe(0.70, 0.71))])
    all4 = next(row for row in rows if row["baseline_name"] == "all_four_cv")
    pair = next(row for row in rows if row["baseline_name"] == "pairwise_cv_always")
    llm = next(row for row in rows if row["baseline_name"] == "llm_only")
    assert all4["analysis_reused_cached_scores"] is True
    assert all4["actual_analysis_fit_count"] == 0
    assert all4["counterfactual_selection_fit_count"] == 12
    assert all4["counterfactual_planner_llm_call_count"] == 0
    assert all4["actual_analysis_llm_call_count"] == 0
    assert pair["counterfactual_selection_fit_count"] == 6
    assert pair["counterfactual_planner_llm_call_count"] == 1
    assert pair["counterfactual_probe_invocation_count"] == 1
    assert llm["counterfactual_planner_llm_call_count"] == 1


def test_comparisons_are_primary_by_model_condition_and_keep_combined_audit_only():
    first = _row(probe=_probe(0.70, 0.71))
    second = _row(probe=_probe(0.70, 0.71))
    for fixture in (first, second):
        fixture["candidate_cv_metrics"]["tree_ensemble"]["validation"] = {"approved_preprocessing": _contract(scaling="none")}
        fixture["final_preprocessing"] = _contract(scaling="none")
    rows = derive_baseline_trials([
        first,
        {**second, "model_condition_id": "condition-b", "trial_id": "condition-b:task:clean:split42:rep_001"},
    ])
    summary = summarize_baseline_trials(rows, bootstrap_replicates=5)
    assert set(summary["comparisons_by_model_condition"]) == {"condition", "condition-b"}
    for condition in ("condition", "condition-b"):
        comparison = summary["comparisons_by_model_condition"][condition]["llm_only_vs_all_four_cv"]
        assert comparison["independent_dataset_count"] == 1
    assert summary["descriptive_combined_condition_comparisons"]["scope"] == "descriptive_audit_only_pooled_model_conditions"


def test_pairwise_preprocessing_fallback_follows_semantic_source_when_proposals_are_reversed():
    reversed_probe = {
        "status": "completed",
        "cv_folds": 3,
        "proposal_a": {"model_family": "tree_ensemble", "mean_score": 0.71},
        "proposal_b": {"model_family": "linear", "mean_score": 0.70},
    }
    result = select_pairwise_cv_always(
        "classification",
        "linear",
        "tree_ensemble",
        reversed_probe,
        challenger_valid=True,
        initial_preprocessing=_contract(),
        challenger_preprocessing=_contract(scaling="none"),
    )
    assert result["selected_baseline_family"] == "tree_ensemble"
    assert result["selected_preprocessing"]["numeric_scaling"] == "none"


def test_missing_challenger_validity_is_not_treated_as_valid():
    row = _row(probe=_probe(0.70, 0.71))
    row.pop("deterministic_valid")
    derived = derive_baseline_trials([row], baselines=["pairwise_cv_always"])[0]
    assert derived["challenger_hard_valid"] is None
    assert derived["baseline_status"] == "missing_artifact"
    assert any("validity/actionability" in field for field in derived["missing_fields"])


def test_analyze_writes_new_outputs_and_does_not_overwrite_source(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    source.mkdir()
    row = _row(probe=_probe(0.70, 0.71))
    (source / "config.json").write_text(json.dumps({"confirmatory_mode": True}), encoding="utf-8")
    (source / "trials.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    before = (source / "trials.jsonl").read_bytes()
    result = analyze_result_directory(source, output, bootstrap_replicates=10)
    assert set(result["paths"]) == {
        "baseline_trials.jsonl", "baseline_summary.json", "baseline_summary.csv", "baseline_summary.md"
    }
    assert all(path.is_file() for path in output.iterdir())
    assert (source / "trials.jsonl").read_bytes() == before
    assert json.loads((output / "baseline_summary.json").read_text())["analysis_role"] == BASELINE_ANALYSIS_ROLE


def test_prospective_panel_validation_requires_explicit_selection_metadata_and_hash():
    manifest = {
        "manifest_schema_version": PROSPECTIVE_PANEL_MANIFEST_SCHEMA_VERSION,
        "panel_id": "panel-v2",
        "status": "frozen",
        "analysis_role": "prospective_generalization",
        "panel_selection": {
            "sampling_frame": "OpenML tasks declared before outcomes",
            "selection_seed": 12345,
            "inclusion_criteria": ["supervised tabular task"],
            "exclusion_criteria": ["duplicate dataset version"],
        },
        "tasks": [{
            "task_id": 123,
            "dataset_id": 456,
            "dataset_name": "fixture",
            "dataset_version": "1",
            "task_type": "classification",
            "target": "label",
            "expected_rows": 100,
            "expected_features": 4,
            "expected_classes": 2,
            "provenance": {"source": "fixture"},
            "selection_metadata": {"draw": 1},
        }],
    }
    manifest["content_sha256"] = prospective_panel_content_sha256(manifest)
    result = validate_prospective_panel_manifest(manifest, require_frozen=True)
    assert result["content_sha256"] == manifest["content_sha256"]
    draft = copy.deepcopy(manifest)
    draft["status"] = "draft"
    draft.pop("content_sha256")
    assert freeze_prospective_panel_manifest(draft)["status"] == "frozen"
    cases = prospective_benchmark_cases(manifest)
    assert cases[0].openml_task_id == 123
    assert cases[0].role.value == "external_evaluation"
    changed = copy.deepcopy(manifest)
    changed["tasks"][0]["dataset_version"] = "2"
    with pytest.raises(ValueError, match="content_sha256"):
        validate_prospective_panel_manifest(changed, require_frozen=True)


def test_missing_result_trial_files_fail_explicitly(tmp_path: Path):
    with pytest.raises(MissingHistoricalFields, match="trials.jsonl"):
        analyze_result_directory(tmp_path, tmp_path.parent / "missing_result_out", bootstrap_replicates=5)


def test_prospective_wrapper_stamps_role_manifest_and_split_seeds(tmp_path: Path, monkeypatch):
    manifest = {
        "manifest_schema_version": PROSPECTIVE_PANEL_MANIFEST_SCHEMA_VERSION,
        "panel_id": "panel-wrapper",
        "status": "frozen",
        "analysis_role": "prospective_generalization",
        "panel_selection": {
            "sampling_frame": "offline fixture",
            "selection_seed": 9,
            "inclusion_criteria": ["fixture"],
            "exclusion_criteria": [],
        },
        "tasks": [{
            "task_id": 123,
            "dataset_id": 456,
            "dataset_name": "fixture",
            "dataset_version": "1",
            "task_type": "classification",
            "target": "label",
            "expected_rows": 100,
            "expected_features": 4,
            "expected_classes": 2,
            "provenance": {"source": "fixture"},
            "selection_metadata": {"draw": 1},
        }],
    }
    manifest["content_sha256"] = prospective_panel_content_sha256(manifest)
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_run(output, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps({"suite": "local"}), encoding="utf-8")
        (output / "ablation_summary.json").write_text(json.dumps({}), encoding="utf-8")
        (output / "trials.jsonl").write_text(json.dumps({"trial_id": "fixture"}) + "\n", encoding="utf-8")
        return {"output_dir": str(output), "config": {}, "summary": {}, "trials": []}

    monkeypatch.setattr(prospective, "run_ablation_study", fake_run)
    result = prospective.run_prospective_experiment(
        tmp_path / "out",
        panel_path,
        split_seeds=[101, 202],
        offline=True,
    )
    assert result["config"]["analysis_role"] == "prospective_generalization"
    assert result["config"]["prospective_split_seeds"] == [101, 202]
    trial = json.loads((tmp_path / "out" / "trials.jsonl").read_text().strip())
    assert trial["analysis_role"] == "prospective_generalization"
    assert trial["prospective_split_seeds"] == [101, 202]
