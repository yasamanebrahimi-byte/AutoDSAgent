from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.routes import workflow as workflow_route
from app.backend.services.workflow_service import WorkflowService

from tests.workflow_test_utils import create_workflow_service


def test_workflow_api_start_state_trace_and_retry_routes(tmp_path, monkeypatch):
    service, manager, _, run_id = create_workflow_service(tmp_path)
    monkeypatch.setattr(workflow_route, "workflow_service", WorkflowService(manager))
    client = TestClient(app)

    start_response = client.post(
        f"/runs/{run_id}/workflow/start",
        json={
            "target_column": None,
            "task_type": None,
            "require_cleaning_approval": False,
            "require_modeling_approval": False,
        },
    )
    assert start_response.status_code == 200
    assert start_response.json()["status"] == "completed"

    state_response = client.get(f"/runs/{run_id}/workflow/state")
    assert state_response.status_code == 200
    assert state_response.json()["run_id"] == run_id

    trace_response = client.get(f"/runs/{run_id}/workflow/trace")
    assert trace_response.status_code == 200
    assert trace_response.json()

    retry_response = client.post(
        f"/runs/{run_id}/workflow/retry",
        json={"step": "eda"},
    )
    assert retry_response.status_code == 400

    assert service.get_trace(run_id)
