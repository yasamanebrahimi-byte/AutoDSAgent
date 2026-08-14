"""Workflow orchestration API schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    """Overall workflow status values."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    waiting_for_approval = "waiting_for_approval"


class StepStatus(str, Enum):
    """Step-level workflow status values."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"
    waiting_for_approval = "waiting_for_approval"


class ApprovalStatus(str, Enum):
    """Approval status for human-gated workflow steps."""

    not_required = "not_required"
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class ApprovalAction(str, Enum):
    """Supported approval actions."""

    approve = "approve"
    reject = "reject"


class WorkflowStartRequest(BaseModel):
    """Options for starting an automated workflow."""

    target_column: str | None = None
    task_type: Literal["regression", "classification"] | None = None
    require_cleaning_approval: bool = True
    require_modeling_approval: bool = True


class WorkflowApprovalRequest(BaseModel):
    """Approval action for a waiting workflow step."""

    step: Literal["cleaning", "modeling"]
    action: ApprovalAction


class WorkflowRetryRequest(BaseModel):
    """Retry request for a failed workflow step."""

    step: Literal["profile", "cleaning_plan", "cleaning", "eda", "modeling", "report"]


class WorkflowStepState(BaseModel):
    """State for one workflow step."""

    status: StepStatus
    started_at: str | None = None
    completed_at: str | None = None
    attempts: int = 0
    max_attempts: int = 1
    requires_approval: bool = False
    approval_status: ApprovalStatus = ApprovalStatus.not_required
    approval_reason: str | None = None
    approval_details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Full workflow state returned by orchestration endpoints."""

    workflow_version: str = "week6"
    run_id: str
    generation_id: str | None = None
    source_fingerprint: str | None = None
    status: WorkflowStatus
    target_column: str | None = None
    task_type: Literal["regression", "classification"] | None = None
    analysis_input: dict[str, Any] = Field(default_factory=dict)
    current_step: str | None = None
    steps: dict[str, WorkflowStepState]
    artifacts: dict[str, Any] = Field(default_factory=dict)
    artifact_lineage: dict[str, Any] = Field(default_factory=dict)
    config_fingerprints: dict[str, str] = Field(default_factory=dict)
    approval_settings: dict[str, bool] = Field(default_factory=dict)
    warnings: list[Any] = Field(default_factory=list)
    errors: list[Any] = Field(default_factory=list)
    created_at: str
    updated_at: str


class WorkflowTraceEvent(BaseModel):
    """One auditable agent trace event."""

    timestamp: str
    run_id: str
    agent: str
    step: str | None = None
    event_type: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class WorkflowResponse(BaseModel):
    """Wrapper reserved for future workflow metadata."""

    state: WorkflowState
