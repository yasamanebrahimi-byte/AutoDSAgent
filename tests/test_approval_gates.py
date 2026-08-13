from app.backend.schemas.workflow import WorkflowApprovalRequest, WorkflowStartRequest
from app.tools.approval import cleaning_approval_decision, modeling_approval_decision

from tests.workflow_test_utils import create_workflow_service


def test_approval_decision_helpers_detect_cleaning_and_modeling_gates():
    cleaning_decision = cleaning_approval_decision(
        {
            "columns_recommended_for_dropping": ["constant"],
            "duplicate_row_handling": {"apply": True},
            "missing_value_strategies": [
                {"column": "age", "apply": True},
                {"column": "segment", "apply": True},
            ],
            "warnings_requiring_review": ["Review high-cardinality columns."],
        }
    )
    assert cleaning_decision["required"] is True
    assert len(cleaning_decision["reasons"]) >= 3

    modeling_decision = modeling_approval_decision(
        {"target_column": "target", "warnings": []},
        metadata={"rows": 25},
        require_approval=True,
    )
    assert modeling_decision["required"] is True
    assert "target" in modeling_decision["details"]["target_column"]


def test_cleaning_approval_is_triggered_and_approval_continues(tmp_path):
    service, _, _, run_id = create_workflow_service(tmp_path)

    state = service.start_workflow(
        run_id,
        WorkflowStartRequest(
            require_cleaning_approval=True,
            require_modeling_approval=False,
        ),
    )

    assert state["status"] == "waiting_for_approval"
    assert state["current_step"] == "cleaning"
    assert state["steps"]["cleaning"]["status"] == "waiting_for_approval"
    assert state["steps"]["cleaning"]["approval_status"] == "pending"
    assert state["steps"]["cleaning"]["approval_reason"]

    state = service.apply_approval(
        run_id,
        WorkflowApprovalRequest(step="cleaning", action="approve"),
    )

    assert state["status"] == "completed"
    assert state["steps"]["cleaning"]["status"] == "completed"
    assert state["steps"]["cleaning"]["approval_status"] == "approved"
    assert any(
        event["event_type"] == "approval_granted"
        for event in service.get_trace(run_id)
    )


def test_rejection_skips_cleaning_and_prevents_cleaned_data_creation(tmp_path):
    service, _, paths, run_id = create_workflow_service(tmp_path)
    state = service.start_workflow(
        run_id,
        WorkflowStartRequest(
            require_cleaning_approval=True,
            require_modeling_approval=False,
        ),
    )

    state = service.apply_approval(
        run_id,
        WorkflowApprovalRequest(step="cleaning", action="reject"),
    )

    assert state["status"] == "completed"
    assert state["steps"]["cleaning"]["status"] == "skipped"
    assert state["steps"]["cleaning"]["approval_status"] == "rejected"
    assert state["steps"]["eda"]["status"] == "completed"
    assert not (paths.intermediate / "cleaned_data.csv").exists()
    assert any(
        event["event_type"] == "approval_rejected"
        for event in service.get_trace(run_id)
    )
