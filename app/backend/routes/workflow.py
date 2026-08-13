"""Automated workflow orchestration endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.backend.schemas.workflow import (
    WorkflowApprovalRequest,
    WorkflowRetryRequest,
    WorkflowStartRequest,
    WorkflowState,
    WorkflowTraceEvent,
)
from app.backend.services.workflow_service import WorkflowService


router = APIRouter(tags=["workflow"])
workflow_service = WorkflowService()


@router.post("/runs/{run_id}/workflow/start", response_model=WorkflowState)
def start_workflow(
    run_id: str,
    request: WorkflowStartRequest,
) -> WorkflowState:
    """Start or restart an automated workflow for an existing run."""

    try:
        state = workflow_service.start_workflow(run_id, request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WorkflowState(**state)


@router.get("/runs/{run_id}/workflow/state", response_model=WorkflowState)
def get_workflow_state(run_id: str) -> WorkflowState:
    """Return current workflow state for a run."""

    try:
        state = workflow_service.get_state(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow state for run '{run_id}' was not found.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WorkflowState(**state)


@router.get("/runs/{run_id}/workflow/trace", response_model=list[WorkflowTraceEvent])
def get_workflow_trace(run_id: str) -> list[WorkflowTraceEvent]:
    """Return ordered workflow trace events for a run."""

    try:
        return [
            WorkflowTraceEvent(**event)
            for event in workflow_service.get_trace(run_id)
        ]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/runs/{run_id}/workflow/approve", response_model=WorkflowState)
def apply_workflow_approval(
    run_id: str,
    request: WorkflowApprovalRequest,
) -> WorkflowState:
    """Approve or reject a waiting human approval gate."""

    try:
        state = workflow_service.apply_approval(run_id, request)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow state for run '{run_id}' was not found.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WorkflowState(**state)


@router.post("/runs/{run_id}/workflow/retry", response_model=WorkflowState)
def retry_workflow_step(
    run_id: str,
    request: WorkflowRetryRequest,
) -> WorkflowState:
    """Retry a failed workflow step when attempts remain."""

    try:
        state = workflow_service.retry_step(run_id, request)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow state for run '{run_id}' was not found.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WorkflowState(**state)


@router.post("/runs/{run_id}/workflow/reset")
def reset_workflow(run_id: str) -> dict[str, str]:
    """Reset workflow logs for a run without deleting data artifacts."""

    try:
        return workflow_service.reset_workflow(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
