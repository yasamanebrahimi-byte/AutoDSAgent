"""Structured workflow state helpers for orchestration."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.tools.file_utils import ensure_directory, load_json, save_json
from app.tools.artifact_lineage import new_generation_id
from app.workflows.workflow_steps import (
    CANONICAL_WORKFLOW_STEPS,
    CLEANING_STEP,
    EDA_STEP,
    MODELING_STEP,
    PROFILE_STEP,
    STEP_DEFINITIONS,
)


WORKFLOW_STATE_FILENAME = "workflow_state.json"

WORKFLOW_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
    "waiting_for_approval",
}
STEP_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
    "skipped",
    "waiting_for_approval",
}
APPROVAL_STATUSES = {"not_required", "pending", "approved", "rejected"}
TERMINAL_STEP_STATUSES = {"completed", "skipped"}


def utc_now_iso() -> str:
    """Return a timezone-aware timestamp suitable for JSON artifacts."""

    return datetime.now(timezone.utc).isoformat()


def create_initial_workflow_state(
    run_id: str,
    target_column: str | None = None,
    task_type: str | None = None,
    require_cleaning_approval: bool = True,
    require_modeling_approval: bool = True,
    generation_id: str | None = None,
    source_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable workflow state payload."""

    timestamp = utc_now_iso()
    selected_generation_id = generation_id or new_generation_id()
    return {
        "workflow_version": "week6",
        "run_id": run_id,
        "generation_id": selected_generation_id,
        "source_fingerprint": source_fingerprint,
        "status": "pending",
        "target_column": target_column,
        "task_type": task_type,
        "analysis_input": {
            "dataset_used": "raw",
            "path": "input/raw_data.csv",
            "fingerprint": source_fingerprint,
            "source_fingerprint": source_fingerprint,
            "selection_reason": "workflow_started",
        },
        "current_step": PROFILE_STEP,
        "steps": {
            step_name: _initial_step_state(
                step_name=step_name,
                requires_approval=(
                    (step_name == CLEANING_STEP and require_cleaning_approval)
                    or (step_name == MODELING_STEP and require_modeling_approval)
                ),
            )
            for step_name in CANONICAL_WORKFLOW_STEPS
        },
        "artifacts": {
            "metadata": "intermediate/metadata.json",
            "profile": None,
            "cleaning_plan": None,
            "cleaned_data": None,
            "cleaning_summary": None,
            "eda_summary": None,
            "eda_findings": None,
            "eda_report": None,
            "plots": [],
            "modeling_summary": None,
            "evaluation_summary": None,
            "model_results": None,
            "baseline_model": None,
            "best_model": None,
            "final_report": None,
            "executive_summary": None,
            "technical_summary": None,
            "limitations_report": None,
            "report_metadata": None,
            "report_index": None,
        },
        "artifact_lineage": {},
        "config_fingerprints": {},
        "approval_settings": {
            "require_cleaning_approval": require_cleaning_approval,
            "require_modeling_approval": require_modeling_approval,
        },
        "warnings": [],
        "errors": [],
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def state_path_for_logs_dir(logs_dir: str | Path) -> Path:
    """Return the workflow state path for a run logs directory."""

    return Path(logs_dir) / WORKFLOW_STATE_FILENAME


def save_workflow_state(path: str | Path, state: dict[str, Any]) -> Path:
    """Persist workflow state and refresh the updated timestamp."""

    payload = deepcopy(state)
    payload["updated_at"] = utc_now_iso()
    saved_path = save_json(path, payload)
    state["updated_at"] = payload["updated_at"]
    return saved_path


def load_workflow_state(path: str | Path) -> dict[str, Any]:
    """Load workflow state from disk."""

    return load_json(path)


def mark_workflow_running(state: dict[str, Any], current_step: str | None = None) -> None:
    """Mark the overall workflow running."""

    state["status"] = "running"
    if current_step is not None:
        state["current_step"] = current_step
    touch_state(state)


def mark_workflow_completed(state: dict[str, Any]) -> None:
    """Mark the overall workflow completed."""

    state["status"] = "completed"
    state["current_step"] = None
    touch_state(state)


def mark_workflow_failed(state: dict[str, Any], step: str, error: str) -> None:
    """Mark the overall workflow failed with a structured error."""

    state["status"] = "failed"
    state["current_step"] = step
    state.setdefault("errors", []).append(
        {
            "step": step,
            "message": error,
            "timestamp": utc_now_iso(),
        }
    )
    touch_state(state)


def mark_workflow_waiting(state: dict[str, Any], step: str) -> None:
    """Mark the workflow as waiting for human approval."""

    state["status"] = "waiting_for_approval"
    state["current_step"] = step
    touch_state(state)


def mark_step_running(state: dict[str, Any], step: str) -> None:
    """Mark a step running and increment its attempt count."""

    step_state = get_step_state(state, step)
    step_state["status"] = "running"
    step_state["started_at"] = utc_now_iso()
    step_state["completed_at"] = None
    step_state["attempts"] = int(step_state.get("attempts", 0)) + 1
    step_state["error"] = None
    state["current_step"] = step
    state["status"] = "running"
    touch_state(state)


def mark_step_completed(
    state: dict[str, Any],
    step: str,
    outputs: dict[str, Any] | None = None,
) -> None:
    """Mark a step completed and merge optional outputs."""

    step_state = get_step_state(state, step)
    step_state["status"] = "completed"
    step_state["completed_at"] = utc_now_iso()
    step_state["error"] = None
    if outputs:
        step_state.setdefault("outputs", {}).update(outputs)
    touch_state(state)


def mark_step_failed(state: dict[str, Any], step: str, error: str) -> None:
    """Mark a step failed and record the step error."""

    step_state = get_step_state(state, step)
    step_state["status"] = "failed"
    step_state["completed_at"] = utc_now_iso()
    step_state["error"] = error
    mark_workflow_failed(state, step, error)


def mark_step_skipped(
    state: dict[str, Any],
    step: str,
    reason: str,
    outputs: dict[str, Any] | None = None,
) -> None:
    """Mark a step skipped with an explicit reason."""

    step_state = get_step_state(state, step)
    step_state["status"] = "skipped"
    step_state["completed_at"] = utc_now_iso()
    step_state["error"] = None
    step_state.setdefault("outputs", {})["skip_reason"] = reason
    if outputs:
        step_state["outputs"].update(outputs)
    touch_state(state)


def mark_step_waiting_for_approval(
    state: dict[str, Any],
    step: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Mark a step and workflow as waiting for human approval."""

    step_state = get_step_state(state, step)
    step_state["status"] = "waiting_for_approval"
    step_state["requires_approval"] = True
    step_state["approval_status"] = "pending"
    step_state["approval_reason"] = reason
    step_state["approval_details"] = details or {}
    mark_workflow_waiting(state, step)


def set_step_approval(
    state: dict[str, Any],
    step: str,
    approval_status: str,
    reason: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Update approval metadata for a step."""

    if approval_status not in APPROVAL_STATUSES:
        raise ValueError(f"Unknown approval status: {approval_status}")
    step_state = get_step_state(state, step)
    step_state["approval_status"] = approval_status
    if approval_status == "not_required":
        step_state["requires_approval"] = False
    if reason is not None:
        step_state["approval_reason"] = reason
    if details is not None:
        step_state["approval_details"] = details
    touch_state(state)


def reset_step_for_retry(state: dict[str, Any], step: str) -> None:
    """Reset a failed step so it can be attempted again."""

    step_state = get_step_state(state, step)
    attempts = int(step_state.get("attempts", 0))
    max_attempts = int(step_state.get("max_attempts", 1))
    if attempts >= max_attempts:
        raise ValueError(f"Step '{step}' has no retry attempts remaining.")
    if step_state.get("status") != "failed":
        raise ValueError(f"Step '{step}' is not failed and cannot be retried.")

    step_state["status"] = "pending"
    step_state["started_at"] = None
    step_state["completed_at"] = None
    step_state["error"] = None
    state["status"] = "running"
    state["current_step"] = step
    touch_state(state)


def get_step_state(state: dict[str, Any], step: str) -> dict[str, Any]:
    """Return mutable step state or raise a clear error."""

    steps = state.setdefault("steps", {})
    if step not in steps:
        raise ValueError(f"Unknown workflow step: {step}")
    return steps[step]


def next_pending_step(state: dict[str, Any]) -> str | None:
    """Return the first pending step in canonical order."""

    for step in CANONICAL_WORKFLOW_STEPS:
        if get_step_state(state, step).get("status") == "pending":
            return step
    return None


def all_steps_terminal(state: dict[str, Any]) -> bool:
    """Return whether all steps are completed or skipped."""

    return all(
        get_step_state(state, step).get("status") in TERMINAL_STEP_STATUSES
        for step in CANONICAL_WORKFLOW_STEPS
    )


def set_artifact(state: dict[str, Any], key: str, value: Any) -> None:
    """Set a workflow artifact pointer."""

    state.setdefault("artifacts", {})[key] = value
    touch_state(state)


def set_artifact_lineage(
    state: dict[str, Any],
    key: str,
    metadata: dict[str, Any] | None,
) -> None:
    """Record current lineage metadata for a workflow artifact."""

    if not metadata:
        return
    state.setdefault("artifact_lineage", {})[key] = metadata
    artifact_type = metadata.get("artifact_type") or key
    config_fingerprint = metadata.get("config_fingerprint")
    if config_fingerprint:
        state.setdefault("config_fingerprints", {})[artifact_type] = config_fingerprint
    touch_state(state)


def set_analysis_input(
    state: dict[str, Any],
    *,
    dataset_used: str,
    path: str,
    fingerprint: str,
    source_fingerprint: str | None = None,
    selection_reason: str,
) -> None:
    """Set the explicit dataset input for EDA/modeling/report descendants."""

    state["analysis_input"] = {
        "dataset_used": dataset_used,
        "path": path,
        "fingerprint": fingerprint,
        "source_fingerprint": source_fingerprint or state.get("source_fingerprint"),
        "selection_reason": selection_reason,
    }
    touch_state(state)


def relative_to_run(path: str | Path, run_root: str | Path) -> str:
    """Return a stable path relative to the run root."""

    resolved_path = Path(path).resolve()
    resolved_root = Path(run_root).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        raise ValueError("Artifact paths must resolve inside the run directory.")


def ensure_workflow_log_dir(logs_dir: str | Path) -> Path:
    """Create the workflow logs directory if needed."""

    return ensure_directory(logs_dir)


def touch_state(state: dict[str, Any]) -> None:
    """Refresh the state updated timestamp in memory."""

    state["updated_at"] = utc_now_iso()


def _initial_step_state(step_name: str, requires_approval: bool) -> dict[str, Any]:
    definition = STEP_DEFINITIONS[step_name]
    return {
        "status": "pending",
        "started_at": None,
        "completed_at": None,
        "attempts": 0,
        "max_attempts": definition.max_attempts,
        "requires_approval": requires_approval,
        "approval_status": "pending" if requires_approval else "not_required",
        "approval_reason": None,
        "approval_details": {},
        "error": None,
        "outputs": {},
    }
