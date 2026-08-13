from app.backend.schemas.workflow import WorkflowApprovalRequest, WorkflowStartRequest

from tests.workflow_test_utils import create_workflow_service


def test_workflow_runs_core_steps_and_skips_modeling_without_target(tmp_path):
    service, _, paths, run_id = create_workflow_service(tmp_path)

    state = service.start_workflow(
        run_id,
        WorkflowStartRequest(
            require_cleaning_approval=False,
            require_modeling_approval=False,
        ),
    )

    assert state["status"] == "completed"
    assert state["steps"]["profile"]["status"] == "completed"
    assert state["steps"]["cleaning_plan"]["status"] == "completed"
    assert state["steps"]["cleaning"]["status"] == "completed"
    assert state["steps"]["eda"]["status"] == "completed"
    assert state["steps"]["modeling"]["status"] == "skipped"
    assert state["steps"]["report"]["status"] == "completed"
    assert state["steps"]["modeling"]["outputs"]["skip_reason"] == (
        "Modeling was skipped because no target column was provided."
    )
    assert state["artifacts"]["final_report"] == "reports/final_report.md"
    assert state["artifacts"]["report_metadata"] == "intermediate/report_metadata.json"
    assert (paths.reports / "final_report.md").exists()
    assert (paths.reports / "executive_summary.md").exists()
    assert (paths.reports / "technical_summary.md").exists()
    assert (paths.reports / "limitations.md").exists()
    assert (paths.reports / "report_index.json").exists()
    assert (paths.intermediate / "report_metadata.json").exists()
    assert (paths.logs / "workflow_state.json").exists()
    assert (paths.logs / "agent_trace.json").exists()

    event_types = [event["event_type"] for event in service.get_trace(run_id)]
    assert "step_started" in event_types
    assert "step_completed" in event_types
    assert "step_skipped" in event_types
    assert any(event["step"] == "report" for event in service.get_trace(run_id))


def test_modeling_runs_after_modeling_approval_is_granted(tmp_path):
    service, _, paths, run_id = create_workflow_service(tmp_path)

    state = service.start_workflow(
        run_id,
        WorkflowStartRequest(
            target_column="target",
            require_cleaning_approval=False,
            require_modeling_approval=True,
        ),
    )

    assert state["status"] == "waiting_for_approval"
    assert state["current_step"] == "modeling"
    assert state["steps"]["modeling"]["approval_status"] == "pending"

    state = service.apply_approval(
        run_id,
        WorkflowApprovalRequest(step="modeling", action="approve"),
    )

    assert state["status"] == "completed"
    assert state["steps"]["modeling"]["status"] == "completed"
    assert state["steps"]["report"]["status"] == "completed"
    assert state["steps"]["modeling"]["approval_status"] == "approved"
    assert state["artifacts"]["best_model"] == "models/best_model.pkl"
    assert state["artifacts"]["final_report"] == "reports/final_report.md"
    assert (paths.models / "best_model.pkl").exists()
    assert (paths.intermediate / "evaluation_summary.json").exists()
    assert (paths.reports / "final_report.md").exists()
