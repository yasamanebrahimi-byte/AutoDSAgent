"""Exploratory data analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.backend.schemas.eda import EDAPlotInfo, EDARequest, EDAResponse
from app.backend.routes.run_mutation import locked_run_mutation
from app.backend.services.eda_service import EDAService


router = APIRouter(tags=["eda"])
eda_service = EDAService()


@router.post("/runs/{run_id}/eda", response_model=EDAResponse)
def generate_eda(run_id: str, request: EDARequest | None = None) -> EDAResponse:
    """Generate and save EDA summaries, findings, plots, and a Markdown report."""

    with locked_run_mutation(eda_service.run_manager, run_id):
        try:
            return eda_service.generate_eda(run_id, request or EDARequest())
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Dataset for run '{run_id}' was not found. "
                    "Upload data before generating EDA."
                ),
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs/{run_id}/eda", response_model=EDAResponse)
def get_eda(run_id: str) -> EDAResponse:
    """Return existing EDA summary and findings for a run."""

    try:
        return eda_service.load_eda(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"EDA artifacts for run '{run_id}' were not found. Generate EDA first.",
        ) from exc


@router.get("/runs/{run_id}/plots", response_model=list[EDAPlotInfo])
def list_plots(run_id: str) -> list[EDAPlotInfo]:
    """Return generated plot files for a run."""

    try:
        return eda_service.list_plots(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc


@router.get("/runs/{run_id}/plots/{plot_path:path}")
def get_plot(run_id: str, plot_path: str) -> FileResponse:
    """Return one generated plot image from a run."""

    try:
        resolved_path = eda_service.resolve_plot_path(run_id, plot_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Plot file was not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(resolved_path, media_type="image/png")
