"""Deterministic model training utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from app.tools.preprocessing import PreprocessingResult, TaskType


ModelRole = Literal["baseline", "candidate"]
ModelStatus = Literal["succeeded", "failed"]


@dataclass
class ModelTrainingResult:
    """Result for one attempted model fit."""

    model_name: str
    role: ModelRole
    status: ModelStatus
    estimator: Pipeline | None = None
    error: str | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    primary_metric_value: float | None = None


def train_models(
    prepared: PreprocessingResult,
    random_state: int = 42,
) -> list[ModelTrainingResult]:
    """Train the baseline and candidate models for a prepared dataset."""

    results: list[ModelTrainingResult] = []
    for model_name, role, estimator in _model_specs(prepared.task_type, random_state):
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(prepared.preprocessor)),
                ("model", estimator),
            ]
        )
        try:
            pipeline.fit(prepared.X_train, prepared.y_train)
            results.append(
                ModelTrainingResult(
                    model_name=model_name,
                    role=role,
                    status="succeeded",
                    estimator=pipeline,
                )
            )
        except Exception as exc:
            results.append(
                ModelTrainingResult(
                    model_name=model_name,
                    role=role,
                    status="failed",
                    estimator=None,
                    error=str(exc),
                )
            )

    return results


def serialize_training_results(
    results: list[ModelTrainingResult],
) -> list[dict[str, Any]]:
    """Return JSON-safe training and evaluation result records."""

    serialized: list[dict[str, Any]] = []
    for result in results:
        serialized.append(
            {
                "model_name": result.model_name,
                "role": result.role,
                "status": result.status,
                "metrics": result.metrics,
                "primary_metric_value": result.primary_metric_value,
                "error": result.error,
            }
        )
    return serialized


def _model_specs(
    task_type: TaskType,
    random_state: int,
) -> list[tuple[str, ModelRole, object]]:
    if task_type == "regression":
        return [
            ("baseline_median", "baseline", DummyRegressor(strategy="median")),
            ("linear_regression", "candidate", LinearRegression()),
            ("ridge", "candidate", Ridge(random_state=random_state)),
            (
                "random_forest",
                "candidate",
                RandomForestRegressor(
                    n_estimators=25,
                    max_depth=None,
                    min_samples_leaf=1,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
            (
                "hist_gradient_boosting",
                "candidate",
                HistGradientBoostingRegressor(
                    max_iter=30,
                    learning_rate=0.1,
                    random_state=random_state,
                ),
            ),
        ]

    return [
        ("baseline_most_frequent", "baseline", DummyClassifier(strategy="most_frequent")),
        (
            "logistic_regression",
            "candidate",
            LogisticRegression(max_iter=500, random_state=random_state),
        ),
        (
            "random_forest",
            "candidate",
            RandomForestClassifier(
                n_estimators=25,
                max_depth=None,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        (
            "hist_gradient_boosting",
            "candidate",
            HistGradientBoostingClassifier(
                max_iter=30,
                learning_rate=0.1,
                random_state=random_state,
            ),
        ),
    ]
