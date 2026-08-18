"""Deterministic model training utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from app.tools.preprocessing import PreprocessingResult, TaskType, build_preprocessor


ModelRole = Literal["baseline", "candidate"]
ModelStatus = Literal["succeeded", "failed"]
DENSE_ONLY_MODELS = {"hist_gradient_boosting"}


@dataclass
class ModelTrainingResult:
    """Result for one attempted model fit."""

    model_name: str
    role: ModelRole
    status: ModelStatus
    estimator: Pipeline | None = None
    error: str | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    cv_metrics: dict[str, float | None] = field(default_factory=dict)
    holdout_metrics: dict[str, Any] = field(default_factory=dict)
    holdout_predictions: Any = None
    holdout_probabilities: Any = None
    primary_metric_value: float | None = None
    fold_count: int | None = None
    selection_metric: str | None = None


def train_models(
    prepared: PreprocessingResult,
    random_state: int = 42,
) -> list[ModelTrainingResult]:
    """Cross-validate and fit models using only the training partition."""

    results: list[ModelTrainingResult] = []
    for model_name, role, estimator in _model_specs(prepared.task_type, random_state):
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _preprocessor_for_model(prepared, model_name)),
                ("model", estimator),
            ]
        )
        try:
            cv_metrics = _cross_validate_pipeline(pipeline, prepared, random_state)
            pipeline.fit(prepared.X_train, prepared.y_train)
            primary_metric = _primary_metric(prepared.task_type)
            results.append(
                ModelTrainingResult(
                    model_name=model_name,
                    role=role,
                    status="succeeded",
                    estimator=pipeline,
                    metrics=cv_metrics,
                    cv_metrics=cv_metrics,
                    primary_metric_value=cv_metrics.get(f"cv_{primary_metric}_mean"),
                    fold_count=prepared.cv_folds,
                    selection_metric=primary_metric,
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
                "metrics": _finite_metric_dict(result.metrics),
                "cv_metrics": _finite_metric_dict(result.cv_metrics),
                "holdout_metrics": result.holdout_metrics,
                "primary_metric_value": _finite_or_none(result.primary_metric_value),
                "fold_count": result.fold_count,
                "selection_metric": result.selection_metric,
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


def _preprocessor_for_model(
    prepared: PreprocessingResult,
    model_name: str,
):
    if model_name not in DENSE_ONLY_MODELS:
        return clone(prepared.preprocessor)

    return build_preprocessor(
        numeric_features=prepared.numeric_features,
        categorical_features=prepared.categorical_features,
        boolean_features=prepared.boolean_features,
        sparse_output=False,
    )


def _cross_validate_pipeline(
    pipeline: Pipeline,
    prepared: PreprocessingResult,
    random_state: int,
) -> dict[str, float | None]:
    splitter = _cv_splitter(prepared, random_state)
    scoring = _scoring(prepared.task_type)
    scores = cross_validate(
        pipeline,
        prepared.X_train,
        prepared.y_train,
        cv=splitter,
        scoring=scoring,
        error_score="raise",
        n_jobs=None,
    )
    return _summarize_cv_scores(scores, prepared.task_type)


def _cv_splitter(prepared: PreprocessingResult, random_state: int) -> KFold | StratifiedKFold:
    if prepared.task_type == "classification":
        return StratifiedKFold(
            n_splits=prepared.cv_folds,
            shuffle=True,
            random_state=random_state,
        )
    return KFold(
        n_splits=prepared.cv_folds,
        shuffle=True,
        random_state=random_state,
    )


def _scoring(task_type: TaskType) -> dict[str, object]:
    if task_type == "regression":
        return {
            "rmse": _negative_rmse_scorer,
            "mae": _negative_mae_scorer,
            "r2": _r2_scorer,
        }

    return {
        "macro_f1": _macro_f1_scorer,
        "weighted_f1": _weighted_f1_scorer,
        "balanced_accuracy": _balanced_accuracy_scorer,
    }


def _summarize_cv_scores(
    scores: dict[str, np.ndarray],
    task_type: TaskType,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    lower_is_better_metrics = {"rmse", "mae"} if task_type == "regression" else set()

    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        metric_name = key.removeprefix("test_")
        metric_values = np.asarray(values, dtype=float)
        if metric_name in lower_is_better_metrics:
            metric_values = -metric_values
        metrics[f"cv_{metric_name}_mean"] = _finite_or_none(float(np.mean(metric_values)))
        metrics[f"cv_{metric_name}_std"] = _finite_or_none(float(np.std(metric_values)))

    return metrics


def _primary_metric(task_type: TaskType) -> str:
    return "rmse" if task_type == "regression" else "macro_f1"


def _rmse(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _negative_rmse_scorer(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    return -_rmse(y, estimator.predict(X))


def _negative_mae_scorer(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    return -float(mean_absolute_error(y, estimator.predict(X)))


def _r2_scorer(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    return float(estimator.score(X, y))


def _macro_f1_scorer(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    return float(f1_score(y, estimator.predict(X), average="macro", zero_division=0))


def _weighted_f1_scorer(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    return float(f1_score(y, estimator.predict(X), average="weighted", zero_division=0))


def _balanced_accuracy_scorer(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    return float(balanced_accuracy_score(y, estimator.predict(X)))


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if np.isfinite(numeric_value) else None


def _finite_metric_dict(
    metrics: dict[str, float | None],
) -> dict[str, float | None]:
    return {key: _finite_or_none(value) for key, value in metrics.items()}
