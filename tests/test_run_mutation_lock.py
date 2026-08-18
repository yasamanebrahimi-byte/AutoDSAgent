import time
from threading import Event

import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.routes import profile as profile_route
from app.backend.routes import workflow as workflow_route
from app.backend.services.profiling_service import ProfilingService
from app.backend.services.run_manager import RunManager
from app.backend.services.workflow_job_manager import WorkflowJobManager
from app.backend.services.workflow_service import WorkflowService
from app.tools.run_lock import RunMutationConflictError, acquire_run_mutation_lock

from tests.workflow_test_utils import create_workflow_service


def test_run_mutation_lock_is_exclusive_and_reusable(tmp_path):
    manager = RunManager(tmp_path)
    manager.create_run("locked-run")

    with acquire_run_mutation_lock(manager, "locked-run"):
        with pytest.raises(RunMutationConflictError, match="already being modified"):
            with acquire_run_mutation_lock(manager, "locked-run"):
                pass

    with acquire_run_mutation_lock(manager, "locked-run"):
        pass


def test_running_workflow_rejects_concurrent_workflow_and_manual_mutations(
    tmp_path,
    monkeypatch,
):
    _, manager, _, run_id = create_workflow_service(tmp_path, run_id="concurrent-run")
    service = WorkflowService(manager)
    execution_started = Event()
    release_execution = Event()

    def delayed_execution(state):
        execution_started.set()
        assert release_execution.wait(timeout=5)
        return state

    monkeypatch.setattr(service, "run_initialized_workflow", delayed_execution)
    monkeypatch.setattr(workflow_route, "workflow_service", service)
    monkeypatch.setattr(profile_route, "profiling_service", ProfilingService(manager))
    job_manager = WorkflowJobManager(max_workers=2)
    monkeypatch.setattr(workflow_route, "workflow_job_manager", job_manager)
    client = TestClient(app)
    payload = {
        "require_cleaning_approval": False,
        "require_modeling_approval": False,
    }

    first_job = client.post(f"/runs/{run_id}/workflow/start", json=payload).json()
    try:
        assert execution_started.wait(timeout=1)

        profile_response = client.post(f"/runs/{run_id}/profile")
        assert profile_response.status_code == 409
        assert profile_response.headers["retry-after"] == "1"
        assert "already being modified" in profile_response.json()["detail"]

        second_job = client.post(f"/runs/{run_id}/workflow/start", json=payload).json()
        failed_job = _wait_for_job(client, second_job["status_url"])
        assert failed_job["status"] == "failed"
        assert "already being modified" in failed_job["error"]

        release_execution.set()
        assert _wait_for_job(client, first_job["status_url"])["status"] == "completed"
        assert client.post(f"/runs/{run_id}/profile").status_code == 200
    finally:
        release_execution.set()
        job_manager.shutdown()


def _wait_for_job(client: TestClient, status_url: str, timeout: float = 5) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(status_url).json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"Workflow job did not finish within {timeout} seconds.")
