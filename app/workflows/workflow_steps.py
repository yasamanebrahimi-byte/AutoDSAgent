"""Canonical step definitions for the deterministic analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass


PROFILE_STEP = "profile"
CLEANING_PLAN_STEP = "cleaning_plan"
CLEANING_STEP = "cleaning"
EDA_STEP = "eda"
MODELING_STEP = "modeling"
REPORT_STEP = "report"

CANONICAL_WORKFLOW_STEPS = (
    PROFILE_STEP,
    CLEANING_PLAN_STEP,
    CLEANING_STEP,
    EDA_STEP,
    MODELING_STEP,
    REPORT_STEP,
)


@dataclass(frozen=True)
class WorkflowStepDefinition:
    """Static metadata for one workflow step."""

    name: str
    label: str
    required: bool = True
    max_attempts: int = 2
    can_require_approval: bool = False


STEP_DEFINITIONS: dict[str, WorkflowStepDefinition] = {
    PROFILE_STEP: WorkflowStepDefinition(
        name=PROFILE_STEP,
        label="Profile dataset",
        required=True,
        max_attempts=2,
    ),
    CLEANING_PLAN_STEP: WorkflowStepDefinition(
        name=CLEANING_PLAN_STEP,
        label="Generate cleaning plan",
        required=True,
        max_attempts=2,
    ),
    CLEANING_STEP: WorkflowStepDefinition(
        name=CLEANING_STEP,
        label="Apply safe cleaning",
        required=True,
        max_attempts=2,
        can_require_approval=True,
    ),
    EDA_STEP: WorkflowStepDefinition(
        name=EDA_STEP,
        label="Generate exploratory analysis",
        required=True,
        max_attempts=2,
    ),
    MODELING_STEP: WorkflowStepDefinition(
        name=MODELING_STEP,
        label="Train and evaluate models",
        required=False,
        max_attempts=1,
        can_require_approval=True,
    ),
    REPORT_STEP: WorkflowStepDefinition(
        name=REPORT_STEP,
        label="Generate final reports",
        required=True,
        max_attempts=2,
    ),
}


def is_valid_workflow_step(step: str) -> bool:
    """Return whether a step name is part of the canonical workflow."""

    return step in STEP_DEFINITIONS


def get_step_definition(step: str) -> WorkflowStepDefinition:
    """Return a step definition or raise a clear error."""

    if step not in STEP_DEFINITIONS:
        raise ValueError(f"Unknown workflow step: {step}")
    return STEP_DEFINITIONS[step]
