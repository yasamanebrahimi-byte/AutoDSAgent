from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evaluation.mlsys_panel import PanelBuildOptions, PanelEligibility, build_draft_panel
from evaluation.mlsys_prospective import MLSYS_TRIAL_SCHEMA_VERSION
from evaluation.validate_mlsys_run import MlsysValidationError, validate_mlsys_run


def _record(task_id: int, task_type: str, *, name: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "task_id": task_id,
        "dataset_id": task_id + 1000,
        "dataset_name": name or f"fixture-{task_id}",
        "dataset_version": "1",
        "task_type": task_type,
        "target": "target",
        "rows": 200,
        "features": 5,
    }
    if task_type == "classification":
        value["classes"] = 2
    return value


def test_panel_builder_is_deterministic_draft_and_audits_rejections():
    records = [
        _record(9001, "classification"),
        _record(9002, "classification"),
        _record(9003, "classification", name="adult"),
        _record(9101, "regression"),
        _record(9102, "regression"),
        _record(9103, "regression", name="too-wide"),
    ]
    options = PanelBuildOptions(
        classification_count=2,
        regression_count=2,
        selection_seed=17,
        eligibility=PanelEligibility(max_features=10),
    )
    first = build_draft_panel(task_records=records, options=options)
    second = build_draft_panel(task_records=records, options=options)
    assert first == second
    assert first["status"] == "draft"
    assert first["panel_selection"]["selected_counts"] == {"classification": 2, "regression": 2}
    assert len(first["tasks"]) == 4
    assert any("historical_dataset_name" in row["rejection_reasons"] for row in first["rejected_tasks"])
    assert first["content_sha256"]
    assert first["canonical_manifest_sha256"]
    assert all("rejection_reasons" in row for row in first["rejected_tasks"])


def _panel(run_dir: Path) -> str:
    panel_hash = "panel-fixture-hash"
    (run_dir / "prospective_task_panel_manifest.json").write_text(
        json.dumps({"status": "frozen", "content_sha256": panel_hash}), encoding="utf-8"
    )
    return panel_hash


def _candidate_records():
    return {
        family: {
            "family": family,
            "status": "evaluated",
            "cv_mean": 0.7,
            "preprocessing_contract": {"fit_inside_pipeline": True, "numeric_imputation": "median"},
            "metric": "macro_f1",
            "cv_folds": 3,
        }
        for family in ("linear", "regularized_linear", "tree_ensemble", "boosted_tree")
    }


def _synthetic_rows(panel_hash: str, *, include_pairwise: bool = True, include_four: bool = True):
    rows = []
    for arm in ("llm_only", "probe_direct", "full"):
        row = {
            "trial_id": f"{arm}-trial",
            "logical_trial_id": "logical-001",
            "trial_schema_version": MLSYS_TRIAL_SCHEMA_VERSION,
            "trial_status": "completed",
            "source_arm": arm,
            "ablation_name": arm,
            "model_condition_id": "fixture",
            "provider": "mock",
            "planner_model": "fixture-model",
            "benchmark_case": "fixture-task",
            "task_id": 77,
            "dataset_id": 88,
            "dataset_version": "1",
            "task_type": "classification",
            "split_seed": 42,
            "split_random_state": 42,
            "llm_repetition_id": "rep_001",
            "trial": 0,
            "agent_initial": {
                "method": "linear",
                "preprocessing": {"fit_inside_pipeline": True, "numeric_imputation": "median"},
            },
            "agent_initial_method": "linear",
            "agent_initial_preprocessing": {"fit_inside_pipeline": True, "numeric_imputation": "median"},
            "agent_initial_valid": True,
            "final_holdout_metric": 0.8,
            "panel_hash": panel_hash,
        }
        rows.append(row)
    if include_pairwise:
        rows[1]["pairwise_cv_evidence"] = {
            "candidate_a": {"model_family": "linear", "raw_mean_cv_score": 0.70, "preprocessing_contract": {"fit_inside_pipeline": True}},
            "candidate_b": {"model_family": "tree_ensemble", "raw_mean_cv_score": 0.75, "preprocessing_contract": {"fit_inside_pipeline": True}},
            "metric": "macro_f1",
            "higher_is_better": True,
            "raw_mean_difference": -0.05,
            "thresholded_decision": "abstain",
            "probe_status": "completed",
        }
    if include_four:
        rows[2]["four_family_cv_records"] = _candidate_records()
    return rows


def _write_run(tmp_path: Path, rows):
    tmp_path.mkdir(exist_ok=True)
    panel_hash = _panel(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"run_status": "complete", "panel_hash": panel_hash}), encoding="utf-8"
    )
    (tmp_path / "full" ).mkdir(exist_ok=True)
    (tmp_path / "full" / "trials.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return panel_hash


def test_complete_synthetic_run_is_ready(tmp_path: Path):
    panel_hash = _write_run(tmp_path, _synthetic_rows("panel-fixture-hash"))
    report = validate_mlsys_run(tmp_path, strict=True)
    assert report["ready_for_baseline_derivation"] is True
    assert report["missing_source_arms"] == 0
    assert report["missing_pairwise_evidence"] == 0
    assert report["incomplete_four_family_references"] == 0
    assert panel_hash == report["panel_hash"]


def test_missing_pairwise_evidence_fails_strict_validation(tmp_path: Path):
    _write_run(tmp_path, _synthetic_rows("panel-fixture-hash", include_pairwise=False))
    with pytest.raises(MlsysValidationError, match="Missing pairwise evidence: 1"):
        validate_mlsys_run(tmp_path, strict=True)
    assert json.loads((tmp_path / "mlsys_validation_report.json").read_text())["missing_pairwise_evidence"] == 1


def test_missing_all_four_candidate_fails_strict_validation(tmp_path: Path):
    rows = _synthetic_rows("panel-fixture-hash")
    del rows[2]["four_family_cv_records"]["boosted_tree"]
    _write_run(tmp_path, rows)
    with pytest.raises(MlsysValidationError, match="Incomplete four-family references: 1"):
        validate_mlsys_run(tmp_path, strict=True)


def test_conflicting_duplicate_trial_id_is_reported(tmp_path: Path):
    rows = _synthetic_rows("panel-fixture-hash")
    duplicate = copy.deepcopy(rows[0])
    duplicate["trial_id"] = rows[1]["trial_id"]
    duplicate["final_holdout_metric"] = 0.1
    _write_run(tmp_path, rows + [duplicate])
    report = validate_mlsys_run(tmp_path)
    assert report["conflicting_trial_ids"] == 1
    assert report["ready_for_baseline_derivation"] is False


@pytest.mark.parametrize("missing_arm", ["llm_only", "probe_direct", "full"])
def test_each_missing_source_arm_is_reported(tmp_path: Path, missing_arm: str):
    rows = [row for row in _synthetic_rows("panel-fixture-hash") if row["source_arm"] != missing_arm]
    _write_run(tmp_path, rows)
    report = validate_mlsys_run(tmp_path)
    assert report["missing_source_arms"] == 1
    assert report["ready_for_baseline_derivation"] is False


def test_hash_and_split_mismatches_are_reported(tmp_path: Path):
    rows = _synthetic_rows("panel-fixture-hash")
    rows[1]["panel_hash"] = "different-panel"
    rows[2]["split_seed"] = 99
    _write_run(tmp_path, rows)
    report = validate_mlsys_run(tmp_path)
    assert report["hash_mismatches"] == 1
    assert report["split_identity_mismatches"] == 1
