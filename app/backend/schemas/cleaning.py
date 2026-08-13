"""Cleaning plan and cleaning summary API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CleaningAction(BaseModel):
    """One recommended or applied cleaning action."""

    action_type: str
    column: str | None = None
    strategy: str
    reason: str
    apply: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class CleaningRequest(BaseModel):
    """Optional cleaning context supplied by API or workflow callers."""

    target_column: str | None = None


class CleaningPlan(BaseModel):
    """Conservative cleaning plan for one run."""

    run_id: str
    target_column: str | None = None
    duplicate_row_handling: CleaningAction
    missing_value_strategies: list[CleaningAction] = Field(default_factory=list)
    columns_recommended_for_dropping: list[str] = Field(default_factory=list)
    columns_recommended_for_keeping: list[str] = Field(default_factory=list)
    type_conversion_recommendations: list[CleaningAction] = Field(default_factory=list)
    encoding_recommendations: list[CleaningAction] = Field(default_factory=list)
    warnings_requiring_review: list[str] = Field(default_factory=list)
    actions: list[CleaningAction] = Field(default_factory=list)
    created_at: str


class CleaningSummary(BaseModel):
    """Summary of safe cleaning that was applied."""

    run_id: str
    target_column: str | None = None
    original_shape: list[int]
    cleaned_shape: list[int]
    duplicate_rows_removed: int
    columns_dropped: list[str] = Field(default_factory=list)
    missing_values_before: int
    missing_values_after: int
    imputation_strategies_used: dict[str, str] = Field(default_factory=dict)
    type_conversions_applied: dict[str, str] = Field(default_factory=dict)
    target_missing_values_before: int | None = None
    target_missing_values_after: int | None = None
    target_action: str | None = None
    datetime_parse_failures: dict[str, dict[str, Any]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
