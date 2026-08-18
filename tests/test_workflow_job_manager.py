from threading import Event

import pytest

from app.backend.services.workflow_job_manager import WorkflowJobManager


def test_job_manager_records_background_failure():
    manager = WorkflowJobManager(max_workers=1)
    finished = Event()

    def fail():
        try:
            raise RuntimeError("worker failed")
        finally:
            finished.set()

    job = manager.submit("run-one", fail)
    try:
        assert finished.wait(timeout=1)
        completed = manager.get("run-one", job["job_id"])
        assert completed["status"] == "failed"
        assert completed["error"] == "worker failed"
        assert completed["completed_at"] is not None
    finally:
        manager.shutdown()


def test_job_manager_scopes_job_ids_to_their_run():
    manager = WorkflowJobManager(max_workers=1)
    job = manager.submit("run-one", lambda: {"status": "completed"})
    try:
        with pytest.raises(KeyError):
            manager.get("run-two", job["job_id"])
    finally:
        manager.shutdown()
