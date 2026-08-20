"""Training-only preprocessing and one approved-model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.deterministic import is_identifier, semantic_type
from app.schemas import Method, TaskType


def fit_selected_model(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    method: Method,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    frame = dataframe.copy()
    target = frame.pop(target_column)
    if task_type == "regression":
        target = pd.to_numeric(target, errors="coerce")
    else:
        target = target.astype(str)
    valid = target.notna()
    frame = frame.loc[valid].reset_index(drop=True)
    target = target.loc[valid].reset_index(drop=True)
    if task_type == "classification" and target.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    feature_names = list(frame.columns)
    usable_features = [
        column
        for column in feature_names
        if not is_identifier(str(column), frame[column])
        and semantic_type(frame[column]) not in {"text", "datetime", "unknown"}
        and not (
            semantic_type(frame[column]) in {"categorical", "boolean"}
            and frame[column].nunique(dropna=True) > 80
        )
    ]
    if not usable_features:
        raise ValueError("No usable feature columns remain after schema safeguards.")
    frame = frame[usable_features]
    numeric_features = [
        column for column in usable_features if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_features = [column for column in usable_features if column not in numeric_features]
    dense = method == "boosted_tree"

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    stratify = (
        target
        if task_type == "classification" and target.value_counts().min() >= 2
        else None
    )
    X_train, X_test, y_train, y_test = train_test_split(
        frame,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    if task_type == "classification":
        min_class_count = int(y_train.value_counts().min())
        cv_folds = min(5, min_class_count)
        if cv_folds < 2:
            raise ValueError("Each training class needs at least two rows for validation.")
        splitter: Any = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        scoring = {
            "macro_f1": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
            "accuracy": "accuracy",
        }
    else:
        cv_folds = min(5, len(y_train))
        if cv_folds < 2:
            raise ValueError("At least two training rows are required for validation.")
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }

    estimator = _estimator(task_type, method, random_state)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    scores = cross_validate(
        pipeline, X_train, y_train, cv=splitter, scoring=scoring, error_score="raise"
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    holdout = _metrics(task_type, y_test, predictions)
    cv_metrics: dict[str, float] = {}
    for name, values in scores.items():
        if not name.startswith("test_"):
            continue
        metric = name.removeprefix("test_")
        values = np.asarray(values, dtype=float)
        if task_type == "regression" and metric in {"rmse", "mae"}:
            values = -values
        cv_metrics[f"cv_{metric}_mean"] = float(values.mean())
        cv_metrics[f"cv_{metric}_std"] = float(values.std())

    baseline_predictions = _baseline_predictions(task_type, y_train, len(y_test))
    baseline_metrics = _metrics(task_type, y_test, baseline_predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "selected_model.joblib"
    joblib.dump(pipeline, model_path)
    return {
        "target_column": target_column,
        "task_type": task_type,
        "selected_method": method,
        "selected_model": _model_name(task_type, method),
        "features_used": usable_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_folds": cv_folds,
        "cv_strategy": "stratified_kfold" if task_type == "classification" else "kfold",
        "cv_metrics": cv_metrics,
        "holdout_metrics": holdout,
        "baseline_metrics": baseline_metrics,
        "model_path": str(model_path),
    }


def _estimator(task_type: TaskType, method: Method, random_state: int) -> Any:
    if task_type == "classification":
        return {
            "linear": LogisticRegression(max_iter=1000, random_state=random_state),
            "regularized_linear": LogisticRegression(
                C=0.5, max_iter=1000, random_state=random_state
            ),
            "tree_ensemble": RandomForestClassifier(
                n_estimators=80, random_state=random_state, n_jobs=-1
            ),
            "boosted_tree": HistGradientBoostingClassifier(
                max_iter=80, random_state=random_state
            ),
        }[method]
    return {
        "linear": LinearRegression(),
        "regularized_linear": Ridge(alpha=1.0),
        "tree_ensemble": RandomForestRegressor(
            n_estimators=80, random_state=random_state, n_jobs=-1
        ),
        "boosted_tree": HistGradientBoostingRegressor(
            max_iter=80, random_state=random_state
        ),
    }[method]


def _model_name(task_type: TaskType, method: Method) -> str:
    names = {
        "linear": "logistic_regression" if task_type == "classification" else "linear_regression",
        "regularized_linear": "regularized_logistic"
        if task_type == "classification"
        else "ridge",
        "tree_ensemble": "random_forest",
        "boosted_tree": "hist_gradient_boosting",
    }
    return names[method]


def _baseline_predictions(task_type: TaskType, y_train: pd.Series, count: int) -> np.ndarray:
    if task_type == "classification":
        return np.repeat(y_train.mode().iloc[0], count)
    return np.repeat(float(y_train.median()), count)


def _metrics(task_type: TaskType, y_true: pd.Series, predictions: Any) -> dict[str, float]:
    if task_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(y_true, predictions, average="weighted", zero_division=0)
            ),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        }
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }
