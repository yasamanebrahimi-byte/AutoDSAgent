"""Small, explicit schemas shared by the agents and the deterministic engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TaskType = Literal["classification", "regression"]
Method = Literal["linear", "regularized_linear", "tree_ensemble", "boosted_tree"]
CleaningAction = Literal[
    "trim_strings",
    "drop_exact_duplicates",
    "drop_all_null_columns",
    "drop_constant_features",
    "drop_rows_missing_target",
    "coerce_numeric_strings",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentPlan(StrictModel):
    target_column: str = Field(description="The outcome column to predict.")
    task_type: TaskType
    recommended_method: Method
    preprocessing: list[str] = Field(min_length=1, max_length=8)
    reasoning: str = Field(min_length=20, max_length=1200)
    confidence: float = Field(ge=0, le=1)


class DeterministicRecommendation(StrictModel):
    target_column: str
    task_type: TaskType
    recommended_method: Method
    preprocessing: list[str] = Field(min_length=1, max_length=8)
    reasoning: str = Field(min_length=20, max_length=1200)
    evidence: list[str] = Field(min_length=1, max_length=8)


class ConflictResolution(StrictModel):
    selected_target_column: str
    selected_task_type: TaskType
    selected_method: Method
    checks: list[str] = Field(min_length=1, max_length=8)
    justification: str = Field(min_length=20, max_length=1600)
    confidence: float = Field(ge=0, le=1)


class CleaningPlan(StrictModel):
    actions: list[CleaningAction] = Field(max_length=6)
    reasoning: str = Field(min_length=10, max_length=1000)


class ReportDraft(StrictModel):
    executive_summary: str = Field(min_length=30, max_length=1800)
    key_findings: list[str] = Field(min_length=1, max_length=8)
    modeling_interpretation: str = Field(min_length=20, max_length=1600)
    limitations: list[str] = Field(min_length=1, max_length=8)
    next_steps: list[str] = Field(min_length=1, max_length=8)

