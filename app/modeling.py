"""Training-only preprocessing and one approved-model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.schemas import Method, TaskType
from app.validation import modeling_arrays, validate_training_plan


def fit_selected_model(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    method: Method,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    validation = validate_training_plan(
        dataframe,
        target_column,
        task_type,
        method,
        test_size=test_size,
        random_state=random_state,
        feature_columns=feature_columns,
    )
    validation.raise_if_failed()
    frame, target = modeling_arrays(dataframe, validation)
    usable_features = validation.features_used
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
        categorical_encoder: Any = (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            if dense
            else OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        )
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", categorical_encoder),
                    ]
                ),
                categorical_features,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    stratify = target if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        frame,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    if task_type == "classification":
        cv_folds = int(validation.split["cv_folds"])
        splitter: Any = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        scoring = {
            "macro_f1": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
            "accuracy": "accuracy",
        }
    else:
        cv_folds = int(validation.split["cv_folds"])
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
        "excluded_features": validation.excluded_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_rows_removed": validation.target_rows_removed,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_folds": cv_folds,
        "cv_strategy": "stratified_kfold" if task_type == "classification" else "kfold",
        "cv_metrics": cv_metrics,
        "holdout_metrics": holdout,
        "baseline_metrics": baseline_metrics,
        "model_path": str(model_path),
        "validation": validation.as_dict(),
        "split_evidence": validation.split,
        "preprocessing_policy": {
            "fit_inside_pipeline": True,
            "numeric_infinity_policy": "converted_to_missing_before_imputation",
            "categorical_encoding": "ordinal_for_boosted_tree" if dense else "one_hot",
            "holdout_used_for": "final_evaluation_only",
        },
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
