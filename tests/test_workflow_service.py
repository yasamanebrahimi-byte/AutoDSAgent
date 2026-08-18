import time
from threading import Event

from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.routes import workflow as workflow_route
from app.backend.services.workflow_job_manager import WorkflowJobManager
from app.backend.services.workflow_service import WorkflowService

from tests.workflow_test_utils import create_workflow_service


def test_workflow_api_start_state_trace_and_retry_routes(tmp_path, monkeypatch):
    service, manager, _, run_id = create_workflow_service(tmp_path)
    monkeypatch.setattr(workflow_route, "workflow_service", WorkflowService(manager))
    job_manager = WorkflowJobManager(max_workers=1)
    monkeypatch.setattr(workflow_route, "workflow_job_manager", job_manager)
    client = TestClient(app)

    try:
        start_response = client.post(
            f"/runs/{run_id}/workflow/start",
            json={
                "target_column": None,
                "task_type": None,
                "require_cleaning_approval": False,
                "require_modeling_approval": False,
            },
        )
        assert start_response.status_code == 202
        assert start_response.headers["location"] == start_response.json()["status_url"]

        job = _wait_for_job(client, start_response.json()["status_url"])
        assert job["status"] == "completed"
    finally:
        job_manager.shutdown()

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


def test_workflow_start_returns_before_background_execution_finishes(tmp_path, monkeypatch):
    _, manager, _, run_id = create_workflow_service(tmp_path, run_id="background-job")
    service = WorkflowService(manager)
    execution_started = Event()
    release_execution = Event()

    def delayed_execution(state):
        execution_started.set()
        assert release_execution.wait(timeout=5)
        return state

    monkeypatch.setattr(service, "run_initialized_workflow", delayed_execution)
    monkeypatch.setattr(workflow_route, "workflow_service", service)
    job_manager = WorkflowJobManager(max_workers=1)
    monkeypatch.setattr(workflow_route, "workflow_job_manager", job_manager)
    client = TestClient(app)

    started_at = time.monotonic()
    response = client.post(
        f"/runs/{run_id}/workflow/start",
        json={
            "require_cleaning_approval": False,
            "require_modeling_approval": False,
        },
    )
    elapsed = time.monotonic() - started_at

    try:
        assert response.status_code == 202
        assert elapsed < 1
        assert execution_started.wait(timeout=1)
        assert client.get(response.json()["status_url"]).json()["status"] == "running"
        assert client.get(f"/runs/{run_id}/workflow/state").status_code == 200
        release_execution.set()
        assert _wait_for_job(client, response.json()["status_url"])["status"] == "completed"
    finally:
        release_execution.set()
        job_manager.shutdown()


def _wait_for_job(
    client: TestClient,
    status_url: str,
    timeout: float = 20,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(status_url)
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"Workflow job did not finish within {timeout} seconds.")
