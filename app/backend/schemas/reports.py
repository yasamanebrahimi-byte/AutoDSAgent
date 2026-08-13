"""Final report generation API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ReportStatus = Literal["completed", "partial", "failed"]
SupportedReportName = Literal[
    "final_report",
    "executive_summary",
    "technical_summary",
    "limitations",
]


class ReportGenerateRequest(BaseModel):
    """Options for deterministic report generation."""

    include_html: bool = False
    force_regenerate: bool = True


class ReportFileInfo(BaseModel):
    """Information about one generated report artifact."""

    name: str
    path: str
    description: str
    media_type: str = "text/markdown"
    size_bytes: int | None = None


class ReportMetadata(BaseModel):
    """Structured metadata saved after report generation."""

    run_id: str
    report_status: ReportStatus
    reports_generated: list[str] = Field(default_factory=list)
    source_artifacts_used: list[str] = Field(default_factory=list)
    source_artifacts_missing: list[str] = Field(default_factory=list)
    sections_generated: list[str] = Field(default_factory=list)
    sections_skipped: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class ReportIndex(BaseModel):
    """Index of generated report artifacts for one run."""

    run_id: str
    reports: list[ReportFileInfo] = Field(default_factory=list)


class ReportGenerateResponse(BaseModel):
    """Response returned after generating or loading reports."""

    metadata: ReportMetadata
    index: ReportIndex


class ReportContentResponse(BaseModel):
    """Markdown content for one generated report."""

    run_id: str
    report_name: SupportedReportName
    path: str
    media_type: str = "text/markdown"
    content: str
