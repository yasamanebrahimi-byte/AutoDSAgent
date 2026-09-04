"""Evaluation-only post-hoc comparison of the supported model families.

This module is intentionally outside ``app.pipeline``.  It is called only
after the runtime gate has made its decision and never participates in that
decision.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from app.modeling import _estimator, _metrics
from app.preprocessing import build_preprocessor, requirements_from_records
from app.schemas import Method, PreprocessingContract, TaskType
from app.validation import (
    FrozenSplit,
    ValidationResult,
    modeling_arrays,
    validate_training_plan,
    validated_row_positions,
)


SUPPORTED_METHOD_ORDER: tuple[Method, ...] = (
    "linear",
    "regularized_linear",
    "tree_ensemble",
    "boosted_tree",
)
PRIMARY_METRICS: dict[str, str] = {
    "classification": "macro_f1",
    "regression": "rmse",
}


def primary_metric(task_type: str) -> str:
    try:
        return PRIMARY_METRICS[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task type for evaluation: {task_type!r}") from exc


def training_only_contract(
    training_profile: dict[str, Any], target_column: str, task_type: str, method: str
) -> PreprocessingContract:
    return training_only_requirements(training_profile, target_column, task_type, method).expected_contract


def training_only_requirements(
    training_profile: dict[str, Any], target_column: str, task_type: str, method: str
):
    records = [
        record
        for record in training_profile.get("column_details", [])
        if str(record.get("name")) != str(target_column)
    ]
    return requirements_from_records(records, task_type, method)


def _scoring(task_type: TaskType) -> dict[str, str]:
    if task_type == "classification":
        return {
            "macro_f1": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
            "accuracy": "accuracy",
        }
    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }


def _normalise_cv_scores(task_type: str, scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        name = key.removeprefix("test_")
        numeric = np.asarray(values, dtype=float)
        if task_type == "regression" and name in {"rmse", "mae"}:
            numeric = -numeric
        metrics[name] = {
            "mean": float(numeric.mean()),
            "std": float(numeric.std()),
            "folds": [float(value) for value in numeric],
        }
    return metrics


def _validated_training_data(
    training_frame: pd.DataFrame,
    target_column: str,
    task_type: str,
    method: str,
    preprocessing: PreprocessingContract,
    random_state: int,
) -> tuple[ValidationResult, pd.DataFrame, pd.Series]:
    validation = validate_training_plan(
        training_frame,
        target_column,
        task_type,
        method,
        random_state=random_state,
        preprocessing=preprocessing,
    )
    if validation.status != "passed":
        return validation, pd.DataFrame(), pd.Series(dtype=float)
    features, target = modeling_arrays(training_frame, validation)
    return validation, features, target


def evaluate_plan_cv(
    training_frame: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    method: Method,
    preprocessing: PreprocessingContract,
    *,
    random_state: int,
) -> dict[str, Any]:
    """Evaluate one already-specified plan on the frozen training partition."""

    validation, features, target = _validated_training_data(
        training_frame,
        target_column,
        task_type,
        method,
        preprocessing,
        random_state,
    )
    if validation.status != "passed":
        return {
            "status": "invalid",
            "primary_metric": primary_metric(task_type),
            "primary_mean": None,
            "primary_std": None,
            "metrics": {},
            "validation": validation.as_dict(),
        }

    numeric_features = [
        column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_features = [column for column in features.columns if column not in numeric_features]
    preprocessor = build_preprocessor(preprocessing, numeric_features, categorical_features, method)
    estimator = _estimator(task_type, method, random_state)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    cv_folds = int(validation.split.get("cv_folds", 0))
    if task_type == "classification":
        splitter: Any = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    try:
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=splitter,
            scoring=_scoring(task_type),
            error_score="raise",
        )
    except Exception as exc:
        return {
            "status": "fit_failed",
            "primary_metric": primary_metric(task_type),
            "primary_mean": None,
            "primary_std": None,
            "metrics": {},
            "validation": validation.as_dict(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    metrics = _normalise_cv_scores(task_type, scores)
    primary = metrics[primary_metric(task_type)]
    return {
        "status": "evaluated",
        "primary_metric": primary_metric(task_type),
        "primary_mean": primary["mean"],
        "primary_std": primary["std"],
        "metrics": metrics,
        "cv_folds": cv_folds,
        "cv_strategy": "stratified_kfold" if task_type == "classification" else "kfold",
        "data_used": "frozen_training_partition_only",
        "holdout_used": False,
        "validation": validation.as_dict(),
    }


def evaluate_empirical_reference(
    training_frame: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    training_profile: dict[str, Any],
    *,
    random_state: int,
) -> dict[str, Any]:
    """Fit every eligible supported family with CV on training rows only."""

    candidates: dict[str, dict[str, Any]] = {}
    for method in SUPPORTED_METHOD_ORDER:
        contract = training_only_contract(training_profile, target_column, task_type, method)
        candidates[method] = evaluate_plan_cv(
            training_frame,
            target_column,
            task_type,
            method,
            contract,
            random_state=random_state,
        )

    metric = primary_metric(task_type)
    eligible = [
        (method, result)
        for method, result in candidates.items()
        if result.get("status") == "evaluated" and result.get("primary_mean") is not None
    ]
    if task_type == "classification":
        ranking = sorted(eligible, key=lambda item: (-float(item[1]["primary_mean"]), item[0]))
    else:
        ranking = sorted(eligible, key=lambda item: (float(item[1]["primary_mean"]), item[0]))
    ranking_names = [method for method, _ in ranking]
    return {
        "status": "evaluated" if ranking else "unavailable",
        "primary_metric": metric,
        "selection_rule": (
            "highest_mean_cv_macro_f1" if task_type == "classification" else "lowest_mean_cv_rmse"
        ),
        "best_method": ranking_names[0] if ranking_names else None,
        "best_primary_mean": ranking[0][1]["primary_mean"] if ranking else None,
        "best_primary_std": ranking[0][1]["primary_std"] if ranking else None,
        "ranking": ranking_names,
        "candidate_metrics": candidates,
        "data_used": "frozen_training_partition_only",
        "holdout_used": False,
    }


def evaluate_holdout_plan(
    dataframe: pd.DataFrame,
    split: FrozenSplit,
    target_column: str,
    task_type: TaskType,
    method: Method,
    preprocessing: PreprocessingContract,
    *,
    random_state: int,
    row_positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit a frozen plan on train rows and score the holdout once."""

    validation = validate_training_plan(
        dataframe,
        target_column,
        task_type,
        method,
        random_state=random_state,
        preprocessing=preprocessing,
        split=split,
        row_positions=row_positions,
    )
    if validation.status != "passed":
        return {"status": "invalid", "holdout_metrics": {}, "validation": validation.as_dict()}
    features, target = modeling_arrays(dataframe, validation)
    valid_positions = validated_row_positions(dataframe, validation, row_positions)
    train_mask = np.isin(valid_positions, np.asarray(split.train_row_positions, dtype=int))
    holdout_mask = np.isin(valid_positions, np.asarray(split.holdout_row_positions, dtype=int))
    X_train = features.loc[train_mask].copy()
    X_holdout = features.loc[holdout_mask].copy()
    y_train = target.loc[train_mask].copy()
    y_holdout = target.loc[holdout_mask].copy()
    numeric_features = [
        column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_features = [column for column in features.columns if column not in numeric_features]
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(preprocessing, numeric_features, categorical_features, method)),
            ("model", _estimator(task_type, method, random_state)),
        ]
    )
    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_holdout)
    except Exception as exc:
        return {
            "status": "fit_failed",
            "holdout_metrics": {},
            "validation": validation.as_dict(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "evaluated",
        "holdout_metrics": _metrics(task_type, y_holdout, predictions),
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_holdout)),
        "holdout_used": "final_evaluation_only",
        "validation": validation.as_dict(),
    }
