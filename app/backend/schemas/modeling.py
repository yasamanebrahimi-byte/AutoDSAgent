"""Modeling and evaluation API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.backend.config import settings


TaskType = Literal["regression", "classification"]


class ModelingRequest(BaseModel):
    """Request options for a deterministic modeling run."""

    target_column: str
    task_type: TaskType | None = None
    test_size: float = Field(default=settings.default_test_size, gt=0, lt=1)
    random_state: int = Field(default=settings.default_random_seed, ge=0)


class ModelMetricResult(BaseModel):
    """Metrics and status for one attempted model."""

    model_name: str
    role: Literal["baseline", "candidate"]
    status: Literal["succeeded", "failed"]
    metrics: dict[str, float | None] = Field(default_factory=dict)
    primary_metric_value: float | None = None
    error: str | None = None


class EvaluationPlotInfo(BaseModel):
    """Information about one evaluation plot artifact."""

    path: str
    label: str
    category: str


class ModelingSummary(BaseModel):
    """Structured modeling summary saved for one run."""

    run_id: str
    dataset_path: str
    target_column: str
    task_type: TaskType
    rows_used: int
    columns_used: int
    features_used: list[str] = Field(default_factory=list)
    features_excluded: list[str] = Field(default_factory=list)
    excluded_feature_reasons: dict[str, str] = Field(default_factory=dict)
    train_rows: int
    test_rows: int
    models_attempted: list[str] = Field(default_factory=list)
    models_succeeded: list[str] = Field(default_factory=list)
    models_failed: list[str] = Field(default_factory=list)
    best_model_name: str
    baseline_model_name: str
    primary_metric: str
    warnings: list[str] = Field(default_factory=list)
    created_at: str


class EvaluationSummary(BaseModel):
    """Structured evaluation summary saved for one run."""

    run_id: str
    target_column: str
    task_type: TaskType
    primary_metric: str
    best_model_name: str
    baseline_metrics: dict[str, float | None] = Field(default_factory=dict)
    best_model_metrics: dict[str, float | None] = Field(default_factory=dict)
    all_model_metrics: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    baseline_comparison: dict[str, Any] = Field(default_factory=dict)
    generated_plots: list[EvaluationPlotInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str


class ModelingResponse(BaseModel):
    """Response returned after training and evaluating models."""

    modeling_summary: ModelingSummary
    evaluation_summary: EvaluationSummary
    model_results: dict[str, Any] = Field(default_factory=dict)


class SavedModelInfo(BaseModel):
    """Information about one saved model-related artifact."""

    name: str
    path: str
    artifact_type: Literal["model", "results"]
    size_bytes: int
    modified_at: float
