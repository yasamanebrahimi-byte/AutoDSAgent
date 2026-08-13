import pytest

from app.agents.eda_agent import EDAAgent
from app.agents.profiler_agent import ProfilerAgent
from app.backend.schemas.workflow import WorkflowRetryRequest, WorkflowStartRequest

from tests.workflow_test_utils import create_workflow_service


def test_failed_step_records_error_and_retry_succeeds(tmp_path, monkeypatch):
    original_run = EDAAgent.run
    calls = {"count": 0}

    def flaky_run(self, state):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("forced EDA failure")
        return original_run(self, state)

    monkeypatch.setattr(EDAAgent, "run", flaky_run)
    service, _, _, run_id = create_workflow_service(tmp_path)

    state = service.start_workflow(
        run_id,
        WorkflowStartRequest(
            require_cleaning_approval=False,
            require_modeling_approval=False,
        ),
    )

    assert state["status"] == "failed"
    assert state["steps"]["eda"]["status"] == "failed"
    assert state["steps"]["eda"]["attempts"] == 1
    assert state["steps"]["eda"]["error"] == "forced EDA failure"

    state = service.retry_step(run_id, WorkflowRetryRequest(step="eda"))

    assert state["status"] == "completed"
    assert state["steps"]["eda"]["status"] == "completed"
    assert state["steps"]["eda"]["attempts"] == 2
    assert any(
        event["event_type"] == "step_retried"
        for event in service.get_trace(run_id)
    )


def test_retry_is_blocked_after_max_attempts(tmp_path, monkeypatch):
    def always_fail(self, state):
        raise RuntimeError("profile is unavailable")

    monkeypatch.setattr(ProfilerAgent, "run", always_fail)
    service, _, _, run_id = create_workflow_service(tmp_path)

    state = service.start_workflow(
        run_id,
        WorkflowStartRequest(
            require_cleaning_approval=False,
            require_modeling_approval=False,
        ),
    )
    assert state["steps"]["profile"]["attempts"] == 1

    state = service.retry_step(run_id, WorkflowRetryRequest(step="profile"))
    assert state["steps"]["profile"]["attempts"] == 2
    assert state["steps"]["profile"]["status"] == "failed"

    with pytest.raises(ValueError, match="no retry attempts remaining"):
        service.retry_step(run_id, WorkflowRetryRequest(step="profile"))
