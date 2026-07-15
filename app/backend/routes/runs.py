"""Run metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.backend.schemas.dataset import DatasetMetadata
from app.backend.schemas.run import RunSummary
from app.backend.services.run_manager import RunManager


router = APIRouter(tags=["runs"])
run_manager = RunManager()


@router.get("/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    """Return available runs with lightweight metadata."""

    return [RunSummary(**summary) for summary in run_manager.list_runs()]


@router.get("/runs/{run_id}", response_model=DatasetMetadata)
def get_run(run_id: str) -> DatasetMetadata:
    """Return saved metadata for one run."""

    try:
        metadata = run_manager.load_metadata(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc

    return DatasetMetadata(**metadata)
