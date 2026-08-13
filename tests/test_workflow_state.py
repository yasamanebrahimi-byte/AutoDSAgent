import pytest

from app.workflows.workflow_state import (
    create_initial_workflow_state,
    load_workflow_state,
    mark_step_completed,
    relative_to_run,
    save_workflow_state,
    set_artifact,
)


def test_initial_workflow_state_is_created_correctly():
    state = create_initial_workflow_state(
        run_id="state-test",
        target_column="target",
        task_type="classification",
        require_cleaning_approval=True,
        require_modeling_approval=True,
    )

    assert state["run_id"] == "state-test"
    assert state["status"] == "pending"
    assert state["target_column"] == "target"
    assert state["task_type"] == "classification"
    assert state["current_step"] == "profile"
    assert list(state["steps"]) == [
        "profile",
        "cleaning_plan",
        "cleaning",
        "eda",
        "modeling",
        "report",
    ]
    assert state["steps"]["profile"]["status"] == "pending"
    assert state["steps"]["cleaning"]["requires_approval"] is True
    assert state["steps"]["cleaning"]["approval_status"] == "pending"
    assert state["steps"]["modeling"]["max_attempts"] == 1
    assert state["steps"]["report"]["max_attempts"] == 2
    assert state["artifacts"]["profile"] is None
    assert state["artifacts"]["final_report"] is None


def test_workflow_state_is_saved_loaded_and_updates_artifacts(tmp_path):
    state = create_initial_workflow_state(run_id="state-save-test")
    set_artifact(state, "profile", "intermediate/profile.json")
    mark_step_completed(state, "profile", {"rows": 10})

    path = tmp_path / "logs" / "workflow_state.json"
    save_workflow_state(path, state)
    loaded = load_workflow_state(path)

    assert loaded["artifacts"]["profile"] == "intermediate/profile.json"
    assert loaded["steps"]["profile"]["status"] == "completed"
    assert loaded["steps"]["profile"]["outputs"]["rows"] == 10
    assert loaded["updated_at"]


def test_relative_to_run_rejects_artifacts_outside_run_root(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    artifact = run_root / "intermediate" / "profile.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")

    assert relative_to_run(artifact, run_root) == "intermediate/profile.json"
    with pytest.raises(ValueError):
        relative_to_run(tmp_path / "outside.json", run_root)
