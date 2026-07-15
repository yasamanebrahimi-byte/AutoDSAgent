"""Run-related API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class RunSummary(BaseModel):
    """Lightweight run listing response."""

    run_id: str
    filename: str | None = None
    rows: int | None = None
    columns: int | None = None
    created_at: str | None = None
