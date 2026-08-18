"""Cleaning plan and safe cleaning endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.backend.schemas.cleaning import CleaningPlan, CleaningRequest, CleaningSummary
from app.backend.routes.run_mutation import locked_run_mutation
from app.backend.services.cleaning_service import CleaningService


router = APIRouter(tags=["cleaning"])
cleaning_service = CleaningService()


@router.post("/runs/{run_id}/cleaning-plan", response_model=CleaningPlan)
def generate_cleaning_plan(
    run_id: str,
    request: CleaningRequest | None = None,
) -> CleaningPlan:
    """Generate and save a conservative cleaning plan."""

    with locked_run_mutation(cleaning_service.run_manager, run_id):
        try:
            return cleaning_service.generate_cleaning_plan(
                run_id,
                target_column=(request.target_column if request else None),
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Raw dataset for run '{run_id}' was not found.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs/{run_id}/cleaning-plan", response_model=CleaningPlan)
def get_cleaning_plan(run_id: str) -> CleaningPlan:
    """Return an existing cleaning plan."""

    try:
        return cleaning_service.load_cleaning_plan(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Cleaning plan for run '{run_id}' was not found. Generate it first.",
        ) from exc


@router.post("/runs/{run_id}/clean", response_model=CleaningSummary)
def apply_cleaning(
    run_id: str,
    request: CleaningRequest | None = None,
) -> CleaningSummary:
    """Apply safe cleaning and save cleaned artifacts."""

    with locked_run_mutation(cleaning_service.run_manager, run_id):
        try:
            return cleaning_service.apply_cleaning(
                run_id,
                target_column=(request.target_column if request else None),
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Raw dataset for run '{run_id}' was not found.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs/{run_id}/cleaning-summary", response_model=CleaningSummary)
def get_cleaning_summary(run_id: str) -> CleaningSummary:
    """Return an existing cleaning summary."""

    try:
        return cleaning_service.load_cleaning_summary(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Cleaning summary for run '{run_id}' was not found. Apply cleaning first.",
        ) from exc
