"""Modeling and evaluation API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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
    metrics: dict[str, Any] = Field(default_factory=dict)
    cv_metrics: dict[str, Any] = Field(default_factory=dict)
    holdout_metrics: dict[str, Any] = Field(default_factory=dict)
    primary_metric_value: float | None = None
    fold_count: int | None = None
    selection_metric: str | None = None
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
    actual_test_size: float | None = None
    cv_folds: int | None = None
    cv_strategy: str | None = None
    task_inference_reason: str | None = None
    classification_validation: dict[str, Any] = Field(default_factory=dict)
    models_attempted: list[str] = Field(default_factory=list)
    models_succeeded: list[str] = Field(default_factory=list)
    models_failed: list[str] = Field(default_factory=list)
    best_candidate_name: str | None = None
    best_candidate_metrics: dict[str, Any] = Field(default_factory=dict)
    selected_model_name: str
    selected_model_role: Literal["baseline", "candidate"] | None = None
    baseline_model_name: str | None = None
    candidate_beats_baseline: bool | None = None
    selection_outcome: str | None = None
    best_model_name: str | None = None
    primary_metric: str
    warnings: list[str] = Field(default_factory=list)
    created_at: str

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_best_model_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        legacy_best_model = payload.get("best_model_name")
        selected_model = payload.get("selected_model_name")
        if selected_model is None and legacy_best_model is not None:
            payload["selected_model_name"] = legacy_best_model
        if legacy_best_model is None and payload.get("selected_model_name") is not None:
            payload["best_model_name"] = payload["selected_model_name"]
        if "best_candidate_name" not in payload and legacy_best_model is not None:
            payload["best_candidate_name"] = legacy_best_model
        return payload


class EvaluationSummary(BaseModel):
    """Structured evaluation summary saved for one run."""

    run_id: str
    target_column: str
    task_type: TaskType
    primary_metric: str
    baseline_model_name: str | None = None
    best_candidate_name: str | None = None
    best_candidate_metrics: dict[str, Any] = Field(default_factory=dict)
    selected_model_name: str
    selected_model_role: Literal["baseline", "candidate"] | None = None
    selected_model_cv_metrics: dict[str, Any] = Field(default_factory=dict)
    selected_model_holdout_metrics: dict[str, Any] = Field(default_factory=dict)
    candidate_beats_baseline: bool | None = None
    selection_outcome: str | None = None
    best_model_name: str | None = None
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    best_model_metrics: dict[str, Any] = Field(default_factory=dict)
    all_model_metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    candidate_cv_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cv_model_metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    final_test_metrics: dict[str, Any] = Field(default_factory=dict)
    holdout_metrics: dict[str, Any] = Field(default_factory=dict)
    cv_folds: int | None = None
    cv_strategy: str | None = None
    selection_metric: str | None = None
    selection_direction: Literal["lower", "higher"] | None = None
    selection_tiebreaker: str | None = None
    test_evaluated_model_names: list[str] = Field(default_factory=list)
    baseline_comparison: dict[str, Any] = Field(default_factory=dict)
    generated_plots: list[EvaluationPlotInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_best_model_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        legacy_best_model = payload.get("best_model_name")
        selected_model = payload.get("selected_model_name")
        if selected_model is None and legacy_best_model is not None:
            payload["selected_model_name"] = legacy_best_model
        if legacy_best_model is None and payload.get("selected_model_name") is not None:
            payload["best_model_name"] = payload["selected_model_name"]
        if "best_candidate_name" not in payload and legacy_best_model is not None:
            payload["best_candidate_name"] = legacy_best_model
        if not payload.get("selected_model_holdout_metrics"):
            payload["selected_model_holdout_metrics"] = (
                payload.get("final_test_metrics")
                or payload.get("holdout_metrics")
                or payload.get("best_model_metrics")
                or {}
            )
        if not payload.get("best_model_metrics"):
            payload["best_model_metrics"] = payload.get("selected_model_holdout_metrics") or {}
        return payload


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
