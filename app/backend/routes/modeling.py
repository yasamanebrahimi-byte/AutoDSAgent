"""Modeling and evaluation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.backend.schemas.modeling import (
    EvaluationSummary,
    ModelingRequest,
    ModelingResponse,
    ModelingSummary,
    SavedModelInfo,
)
from app.backend.services.evaluation_service import EvaluationService
from app.backend.services.modeling_service import ModelingService


router = APIRouter(tags=["modeling"])
modeling_service = ModelingService()
evaluation_service = EvaluationService(modeling_service.run_manager)


@router.post("/runs/{run_id}/model", response_model=ModelingResponse)
def train_and_evaluate_models(
    run_id: str,
    request: ModelingRequest,
) -> ModelingResponse:
    """Train baseline and candidate models, evaluate them, and save artifacts."""

    try:
        return modeling_service.train_and_evaluate(run_id, request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs/{run_id}/modeling-summary", response_model=ModelingSummary)
def get_modeling_summary(run_id: str) -> ModelingSummary:
    """Return an existing modeling summary for a run."""

    try:
        return modeling_service.load_modeling_summary(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Modeling summary for run '{run_id}' was not found. Train models first.",
        ) from exc


@router.get("/runs/{run_id}/evaluation-summary", response_model=EvaluationSummary)
def get_evaluation_summary(run_id: str) -> EvaluationSummary:
    """Return an existing evaluation summary for a run."""

    try:
        return evaluation_service.load_evaluation_summary(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation summary for run '{run_id}' was not found. Train models first.",
        ) from exc


@router.get("/runs/{run_id}/models", response_model=list[SavedModelInfo])
def list_models(run_id: str) -> list[SavedModelInfo]:
    """Return saved model artifacts and model result files."""

    try:
        return modeling_service.list_saved_models(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.") from exc
