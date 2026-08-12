"""Exploratory data analysis API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


DatasetUsed = Literal["cleaned", "raw"]


class EDARequest(BaseModel):
    """Request options for deterministic EDA generation."""

    target_column: str | None = None
    max_numeric_plots: int = Field(default=10, ge=0, le=25)
    max_categorical_plots: int = Field(default=10, ge=0, le=25)
    max_target_relationship_plots: int = Field(default=5, ge=0, le=15)


class EDAPlotInfo(BaseModel):
    """Information about one generated plot artifact."""

    path: str
    label: str
    category: str


class EDASummary(BaseModel):
    """Structured EDA summary saved for a run."""

    run_id: str
    dataset_used: DatasetUsed
    dataset_path: str
    target_column: str | None = None
    rows: int
    columns: int
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    boolean_columns: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)
    missing_values_remaining: dict[str, int] = Field(default_factory=dict)
    duplicate_rows_remaining: int = 0
    generated_plots: list[EDAPlotInfo] = Field(default_factory=list)
    key_statistics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    created_at: str


class EDAFinding(BaseModel):
    """Optional structured finding model reserved for future richer findings."""

    category: str
    message: str
    column: str | None = None
    related_column: str | None = None
    severity: str = "info"


class EDAFindings(BaseModel):
    """Deterministic findings produced by the EDA service."""

    univariate_findings: list[str] = Field(default_factory=list)
    bivariate_findings: list[str] = Field(default_factory=list)
    target_findings: list[str] = Field(default_factory=list)
    correlation_findings: list[str] = Field(default_factory=list)
    data_quality_notes: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


class EDAResponse(BaseModel):
    """EDA summary plus machine-readable findings."""

    summary: EDASummary
    findings: EDAFindings
