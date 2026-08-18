from app.backend.schemas.workflow import WorkflowStartRequest

from tests.workflow_test_utils import create_workflow_service


def test_workflow_includes_report_step_after_modeling_and_updates_state(tmp_path):
    service, _, paths, run_id = create_workflow_service(tmp_path)

    state = service.start_workflow(
        run_id,
        WorkflowStartRequest(
            target_column=None,
            require_cleaning_approval=False,
            require_modeling_approval=False,
        ),
    )

    assert list(state["steps"]) == [
        "profile",
        "cleaning_plan",
        "cleaning",
        "eda",
        "modeling",
        "report",
    ]
    assert state["steps"]["modeling"]["status"] == "skipped"
    assert state["steps"]["report"]["status"] == "completed"
    assert state["artifacts"]["final_report"] == "reports/final_report.md"
    assert state["artifacts"]["report_index"] == "reports/report_index.json"
    assert (paths.reports / "final_report.md").exists()
    assert (paths.intermediate / "report_metadata.json").exists()

    trace = service.get_trace(run_id)
    modeling_skip_index = next(
        index
        for index, event in enumerate(trace)
        if event["step"] == "modeling" and event["event_type"] == "step_skipped"
    )
    report_start_index = next(
        index
        for index, event in enumerate(trace)
        if event["step"] == "report" and event["event_type"] == "step_started"
    )
    assert modeling_skip_index < report_start_index
    assert any(
        event["step"] == "report" and event["event_type"] == "step_completed"
        for event in trace
    )


def test_completed_targeted_workflow_report_uses_terminal_target_aware_state(tmp_path):
    service, _, paths, run_id = create_workflow_service(
        tmp_path,
        run_id="targeted-report-state",
    )

    state = service.start_workflow(
        run_id,
        WorkflowStartRequest(
            target_column="target",
            require_cleaning_approval=False,
            require_modeling_approval=False,
        ),
    )
    report = (paths.reports / "final_report.md").read_text(encoding="utf-8")

    assert state["status"] == "completed"
    assert state["steps"]["report"]["status"] == "completed"
    assert "| Workflow status | completed |" in report
    assert "| Selected target | target |" in report
    assert "Target column: `target`" in report
    assert "No target column has been selected yet." not in report
