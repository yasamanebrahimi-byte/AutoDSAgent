"""Dataset metadata schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Basic metadata generated for an uploaded dataset."""

    run_id: str = Field(..., description="Unique identifier for this analysis run.")
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    duplicate_rows: int
    preview: list[dict[str, Any]]
    created_at: str
