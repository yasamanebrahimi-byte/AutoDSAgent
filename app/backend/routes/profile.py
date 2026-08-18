"""Dataset profiling endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.backend.schemas.profile import DatasetProfile
from app.backend.routes.run_mutation import locked_run_mutation
from app.backend.services.profiling_service import ProfilingService


router = APIRouter(tags=["profile"])
profiling_service = ProfilingService()


@router.post("/runs/{run_id}/profile", response_model=DatasetProfile)
def generate_profile(run_id: str) -> DatasetProfile:
    """Generate and save a dataset profile for a run."""

    with locked_run_mutation(profiling_service.run_manager, run_id):
        try:
            return profiling_service.generate_profile(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Raw dataset for run '{run_id}' was not found.",
            ) from exc


@router.get("/runs/{run_id}/profile", response_model=DatasetProfile)
def get_profile(run_id: str) -> DatasetProfile:
    """Return an existing dataset profile for a run."""

    try:
        return profiling_service.load_profile(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Profile for run '{run_id}' was not found. Generate it first.",
        ) from exc
