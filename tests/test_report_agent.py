from app.agents.report_agent import ReportAgent
from app.backend.services.report_service import ReportService
from app.workflows.workflow_state import create_initial_workflow_state

from tests.report_test_utils import create_report_run


def test_report_agent_updates_state_with_report_artifact_paths(tmp_path):
    manager, paths, run_id = create_report_run(tmp_path, include_modeling=False)
    agent = ReportAgent(ReportService(manager))
    state = create_initial_workflow_state(
        run_id=run_id,
        require_cleaning_approval=False,
        require_modeling_approval=False,
    )

    updated = agent.run(state)

    assert updated["artifacts"]["final_report"] == "reports/final_report.md"
    assert updated["artifacts"]["executive_summary"] == "reports/executive_summary.md"
    assert updated["artifacts"]["technical_summary"] == "reports/technical_summary.md"
    assert updated["artifacts"]["limitations_report"] == "reports/limitations.md"
    assert updated["artifacts"]["report_metadata"] == "intermediate/report_metadata.json"
    assert updated["artifacts"]["report_index"] == "reports/report_index.json"
    assert updated["steps"]["report"]["outputs"]["report_status"] == "partial"
    assert "modeling_methodology" in updated["steps"]["report"]["outputs"]["sections_skipped"]
    assert (paths.reports / "final_report.md").exists()
    assert (paths.intermediate / "report_metadata.json").exists()
