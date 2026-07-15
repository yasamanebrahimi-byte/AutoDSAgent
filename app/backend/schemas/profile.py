"""Dataset profiling API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Severity = Literal["info", "warning", "critical"]
SemanticType = Literal[
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "text",
    "id",
    "unknown",
]


class ValueCount(BaseModel):
    """A value and its observed frequency."""

    value: Any
    count: int


class NumericColumnStats(BaseModel):
    """Summary statistics for numeric columns."""

    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    percentile_25: float | None = None
    percentile_75: float | None = None
    possible_outlier_count: int = 0


class DataQualityIssue(BaseModel):
    """Structured data quality warning."""

    severity: Severity
    issue_type: str
    column: str | None = None
    message: str
    recommendation: str


class ColumnProfile(BaseModel):
    """Profile for one dataset column."""

    column_name: str
    pandas_dtype: str
    semantic_type: SemanticType
    missing_values: int
    missing_percentage: float
    unique_values: int
    unique_percentage: float
    sample_values: list[Any] = Field(default_factory=list)
    is_constant: bool
    is_high_cardinality: bool
    is_id: bool
    is_datetime: bool
    is_numeric: bool
    is_categorical: bool
    is_boolean: bool
    is_text_like: bool
    numeric_stats: NumericColumnStats | None = None
    top_values: list[ValueCount] = Field(default_factory=list)
    average_string_length: float | None = None


class DatasetProfile(BaseModel):
    """Rich dataset profile saved for a run."""

    run_id: str
    rows: int
    columns: int
    total_missing_values: int
    duplicate_rows: int
    memory_usage_bytes: int
    column_type_counts: dict[str, int]
    is_empty: bool
    has_duplicate_rows: bool
    has_missing_values: bool
    column_profiles: list[ColumnProfile]
    data_quality_issues: list[DataQualityIssue] = Field(default_factory=list)
    created_at: str
