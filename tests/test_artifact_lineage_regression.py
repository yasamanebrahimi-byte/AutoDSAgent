from __future__ import annotations

import pytest

from app.backend.schemas.eda import EDARequest
from app.backend.schemas.modeling import ModelingRequest
from app.backend.services.eda_service import EDAService
from app.backend.services.modeling_service import ModelingService
from app.backend.services.report_service import ReportService
from app.backend.services.run_manager import RunManager
from app.tools.artifact_lineage import (
    file_sha256,
    fingerprint_payload,
    write_artifact_lineage,
)
from app.tools.file_utils import load_json, save_json
from app.workflows.workflow_state import create_initial_workflow_state

from tests.report_test_utils import create_report_run


def test_target_change_makes_old_modeling_artifacts_not_current(tmp_path):
    manager, paths, run_id = create_report_run(tmp_path, include_modeling=True)
    state_path = paths.logs / "workflow_state.json"
    state = load_json(state_path)
    state["generation_id"] = "new-target-generation"
    state["target_column"] = "revenue"
    state["task_type"] = "regression"
    state["analysis_input"] = {
        "dataset_used": "raw",
        "path": "input/raw_data.csv",
        "fingerprint": state["source_fingerprint"],
        "source_fingerprint": state["source_fingerprint"],
        "selection_reason": "target_changed",
    }
    save_json(state_path, state)

    modeling_service = ModelingService(manager)
    with pytest.raises(ValueError, match="stale"):
        modeling_service.load_modeling_summary(run_id)

    report_service = ReportService(manager)
    loaded = report_service.load_source_artifacts(run_id)
    assert "modeling_summary" not in loaded.artifacts
    assert "intermediate/modeling_summary.json" in loaded.missing

    response = report_service.generate_reports(run_id)
    assert "intermediate/modeling_summary.json" in response.metadata.source_artifacts_missing
    assert "intermediate/modeling_summary.json" not in response.metadata.source_artifacts_used


def test_cleaning_change_invalidates_old_modeling_artifacts(tmp_path):
    manager, paths, run_id = create_report_run(tmp_path, include_modeling=True)
    state_path = paths.logs / "workflow_state.json"
    state = load_json(state_path)

    cleaned_path = paths.intermediate / "cleaned_data.csv"
    cleaned_path.write_text(
        "customer_id,age,income,segment,churn\n"
        "1,31,60000,A,yes\n"
        "2,36,70000,B,no\n"
        "3,36,80000,A,yes\n"
        "4,41,90000,C,no\n",
        encoding="utf-8",
    )
    new_cleaned_fingerprint = file_sha256(cleaned_path)
    _write_lineage(
        cleaned_path,
        paths.root,
        run_id,
        "cleaned_data",
        state["generation_id"],
        state["source_fingerprint"],
        "churn",
        None,
        {"source_data": state["source_fingerprint"], "cleaning_plan": "changed-plan"},
    )
    state["analysis_input"] = {
        "dataset_used": "cleaned",
        "path": "intermediate/cleaned_data.csv",
        "fingerprint": new_cleaned_fingerprint,
        "source_fingerprint": state["source_fingerprint"],
        "selection_reason": "cleaning_reapplied",
    }
    save_json(state_path, state)

    loaded = ReportService(manager).load_source_artifacts(run_id)
    assert "modeling_summary" not in loaded.artifacts
    assert "evaluation_summary" not in loaded.artifacts
    assert "model_results" not in loaded.artifacts
    assert "upstream fingerprint" in " ".join(loaded.warnings)


def test_cleaning_skipped_uses_raw_even_when_old_cleaned_data_exists(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "skipped-cleaning-test"
    paths = manager.create_run(run_id)
    raw_path = paths.input / "raw_data.csv"
    raw_path.write_text(
        "age,income,churn\n"
        "30,60000,yes\n"
        "35,70000,no\n"
        "40,80000,yes\n"
        "45,90000,no\n",
        encoding="utf-8",
    )
    source_fingerprint = file_sha256(raw_path)
    cleaned_path = paths.intermediate / "cleaned_data.csv"
    cleaned_path.write_text(
        "age,income,churn\n"
        "300,60000,yes\n"
        "350,70000,no\n"
        "400,80000,yes\n"
        "450,90000,no\n",
        encoding="utf-8",
    )
    _write_lineage(
        cleaned_path,
        paths.root,
        run_id,
        "cleaned_data",
        "old-generation",
        source_fingerprint,
        "churn",
        None,
        {"source_data": source_fingerprint},
    )
    state = create_initial_workflow_state(
        run_id=run_id,
        target_column="churn",
        generation_id="new-generation",
        source_fingerprint=source_fingerprint,
        require_cleaning_approval=False,
        require_modeling_approval=False,
    )
    state["steps"]["cleaning"]["status"] = "skipped"
    state["analysis_input"] = {
        "dataset_used": "raw",
        "path": "input/raw_data.csv",
        "fingerprint": source_fingerprint,
        "source_fingerprint": source_fingerprint,
        "selection_reason": "cleaning_rejected",
    }
    save_json(paths.logs / "workflow_state.json", state)

    response = EDAService(manager).generate_eda(run_id, EDARequest(target_column="churn"))

    assert response.summary.dataset_used == "raw"
    assert response.summary.dataset_path == "input/raw_data.csv"


def test_source_data_change_makes_derived_artifacts_unavailable(tmp_path):
    manager, paths, run_id = create_report_run(tmp_path, include_modeling=True)
    paths.input.joinpath("raw_data.csv").write_text(
        "customer_id,age,income,segment,churn\n"
        "10,55,120000,D,no\n"
        "11,60,130000,D,yes\n",
        encoding="utf-8",
    )
    state_path = paths.logs / "workflow_state.json"
    state = load_json(state_path)
    state["generation_id"] = "source-change-generation"
    state["source_fingerprint"] = file_sha256(paths.input / "raw_data.csv")
    state["analysis_input"] = {
        "dataset_used": "raw",
        "path": "input/raw_data.csv",
        "fingerprint": state["source_fingerprint"],
        "source_fingerprint": state["source_fingerprint"],
        "selection_reason": "source_changed",
    }
    save_json(state_path, state)

    loaded = ReportService(manager).load_source_artifacts(run_id)

    assert "metadata" not in loaded.artifacts
    assert "profile" not in loaded.artifacts
    assert "modeling_summary" not in loaded.artifacts
    assert any("source_fingerprint" in warning for warning in loaded.warnings)


def test_failed_modeling_rerun_does_not_restore_old_success(tmp_path, monkeypatch):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "failed-rerun-test"
    paths = manager.create_run(run_id)
    raw_path = paths.input / "raw_data.csv"
    _write_regression_csv(raw_path)
    source_fingerprint = file_sha256(raw_path)
    cleaned_path = paths.intermediate / "cleaned_data.csv"
    _write_regression_csv(cleaned_path)
    cleaned_fingerprint = file_sha256(cleaned_path)
    generation_id = "failed-rerun-generation"
    _write_lineage(
        cleaned_path,
        paths.root,
        run_id,
        "cleaned_data",
        generation_id,
        source_fingerprint,
        "sale_price",
        None,
        {"source_data": source_fingerprint},
    )
    state = create_initial_workflow_state(
        run_id=run_id,
        target_column="sale_price",
        task_type="regression",
        generation_id=generation_id,
        source_fingerprint=source_fingerprint,
        require_cleaning_approval=False,
        require_modeling_approval=False,
    )
    state["analysis_input"] = {
        "dataset_used": "cleaned",
        "path": "intermediate/cleaned_data.csv",
        "fingerprint": cleaned_fingerprint,
        "source_fingerprint": source_fingerprint,
        "selection_reason": "cleaning_completed",
    }
    save_json(paths.logs / "workflow_state.json", state)

    old_summary_path = paths.intermediate / "modeling_summary.json"
    save_json(old_summary_path, {"run_id": run_id, "target_column": "old_target"})
    _write_lineage(
        old_summary_path,
        paths.root,
        run_id,
        "modeling_summary",
        generation_id,
        source_fingerprint,
        "sale_price",
        "regression",
        {"source_data": source_fingerprint, "cleaned": cleaned_fingerprint},
    )

    from app.backend.services import modeling_service as modeling_service_module

    def fail_training(prepared, random_state=42):
        raise RuntimeError("forced training failure")

    monkeypatch.setattr(modeling_service_module, "train_models", fail_training)

    with pytest.raises(RuntimeError, match="forced training failure"):
        ModelingService(manager).train_and_evaluate(
            run_id,
            ModelingRequest(target_column="sale_price", task_type="regression"),
        )

    assert not old_summary_path.exists()
    assert not (paths.intermediate / "modeling_summary.json.lineage.json").exists()


def test_report_generation_uses_only_current_lineage(tmp_path):
    manager, paths, run_id = create_report_run(tmp_path, include_modeling=True)
    state_path = paths.logs / "workflow_state.json"
    state = load_json(state_path)
    state["generation_id"] = "report-current-generation"
    state["target_column"] = "revenue"
    state["task_type"] = "regression"
    state["analysis_input"] = {
        "dataset_used": "raw",
        "path": "input/raw_data.csv",
        "fingerprint": state["source_fingerprint"],
        "source_fingerprint": state["source_fingerprint"],
        "selection_reason": "report_current_fixture",
    }
    save_json(state_path, state)

    response = ReportService(manager).generate_reports(run_id)

    assert "intermediate/modeling_summary.json" in response.metadata.source_artifacts_missing
    assert "intermediate/modeling_summary.json" not in response.metadata.source_artifacts_used
    assert "modeling_summary" not in response.metadata.source_artifact_lineage


def _write_lineage(
    artifact_path,
    run_root,
    run_id: str,
    artifact_type: str,
    generation_id: str | None,
    source_fingerprint: str,
    target_column: str | None,
    task_type: str | None,
    upstream_fingerprints: dict,
) -> None:
    config = {
        "artifact_type": artifact_type,
        "source_fingerprint": source_fingerprint,
        "target_column": target_column,
        "task_type": task_type,
        "upstream_fingerprints": upstream_fingerprints,
    }
    write_artifact_lineage(
        artifact_path,
        run_root=run_root,
        run_id=run_id,
        artifact_type=artifact_type,
        generation_id=generation_id,
        source_fingerprint=source_fingerprint,
        target_column=target_column,
        task_type=task_type,
        config_fingerprint=fingerprint_payload(config),
        upstream_fingerprints=upstream_fingerprints,
        relevant_config=config,
    )


def _write_regression_csv(path) -> None:
    rows = ["customer_id,feature_a,feature_b,region,sale_price"]
    for index in range(1, 49):
        feature_a = ((index * 7) % 17) + (index % 3) * 0.25
        feature_b = index % 5
        region = ["north", "south", "west"][index % 3]
        sale_price = 1000 + feature_a * 30 + feature_b * 12 + (index % 4)
        rows.append(f"{index},{feature_a:.2f},{feature_b},{region},{sale_price:.2f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
