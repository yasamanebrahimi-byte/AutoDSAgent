"""Final report generation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.backend.schemas.reports import (
    ReportContentResponse,
    ReportGenerateRequest,
    ReportGenerateResponse,
    SupportedReportName,
)
from app.backend.services.report_service import ReportService


router = APIRouter(tags=["reports"])
report_service = ReportService()


@router.post("/runs/{run_id}/reports/generate", response_model=ReportGenerateResponse)
def generate_reports(
    run_id: str,
    request: ReportGenerateRequest | None = None,
) -> ReportGenerateResponse:
    """Generate final report artifacts for a run."""

    try:
        return report_service.generate_reports(
            run_id,
            request or ReportGenerateRequest(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs/{run_id}/reports", response_model=ReportGenerateResponse)
def get_reports(run_id: str) -> ReportGenerateResponse:
    """Return generated report metadata and index."""

    try:
        return report_service.load_reports(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Reports for run '{run_id}' were not found. Generate reports first.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/runs/{run_id}/reports/download/{report_name}",
    response_class=FileResponse,
)
def download_report(
    run_id: str,
    report_name: SupportedReportName,
) -> FileResponse:
    """Return one generated report as a downloadable Markdown file."""

    try:
        path = report_service.resolve_report_file(run_id, report_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Report file was not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(
        path,
        media_type="text/markdown",
        filename=path.name,
    )


@router.get("/runs/{run_id}/reports/{report_name}", response_model=ReportContentResponse)
def get_report_content(
    run_id: str,
    report_name: SupportedReportName,
) -> ReportContentResponse:
    """Return Markdown content for one generated report."""

    try:
        return report_service.get_report_content(run_id, report_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Report file was not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
