"""Automated workflow orchestration endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from app.backend.schemas.workflow import (
    WorkflowApprovalRequest,
    WorkflowJob,
    WorkflowRetryRequest,
    WorkflowStartRequest,
    WorkflowState,
    WorkflowTraceEvent,
)
from app.backend.services.workflow_job_manager import WorkflowJobManager
from app.backend.services.workflow_service import WorkflowService
from app.tools.run_lock import RunMutationConflictError


router = APIRouter(tags=["workflow"])
workflow_service = WorkflowService()
workflow_job_manager = WorkflowJobManager()


@router.post(
    "/runs/{run_id}/workflow/start",
    response_model=WorkflowJob,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_workflow(
    run_id: str,
    request: WorkflowStartRequest,
    response: Response,
) -> WorkflowJob:
    """Queue an automated workflow and return immediately with a poll URL."""

    try:
        workflow_service.validate_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    selected_service = workflow_service
    job = workflow_job_manager.submit(
        run_id,
        lambda: selected_service.start_workflow(run_id, request),
    )
    response.headers["Location"] = str(job["status_url"])
    return WorkflowJob(**job)


@router.get(
    "/runs/{run_id}/workflow/jobs/{job_id}",
    response_model=WorkflowJob,
)
def get_workflow_job(run_id: str, job_id: str) -> WorkflowJob:
    """Return background workflow execution status for polling clients."""

    try:
        job = workflow_job_manager.get(run_id, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Workflow job was not found.") from exc
    return WorkflowJob(**job)


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
    except RunMutationConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
            headers={"Retry-After": "1"},
        ) from exc
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
    except RunMutationConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
            headers={"Retry-After": "1"},
        ) from exc
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
    except RunMutationConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
            headers={"Retry-After": "1"},
        ) from exc
