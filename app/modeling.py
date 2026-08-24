"""Training-only preprocessing and one approved-model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
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

from app.preprocessing import build_preprocessor
from app.schemas import Method, PreprocessingContract, TaskType
from app.validation import (
    FrozenSplit,
    modeling_arrays,
    validated_row_positions,
    validate_training_plan,
)


def fit_selected_model(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    method: Method,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_columns: Sequence[str] | None = None,
    preprocessing: PreprocessingContract | dict[str, Any] | list[str] | None = None,
    split: FrozenSplit | None = None,
    row_positions: Sequence[int] | None = None,
    evidence_dataframe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    validation = validate_training_plan(
        dataframe,
        target_column,
        task_type,
        method,
        test_size=test_size,
        random_state=random_state,
        feature_columns=feature_columns,
        preprocessing=preprocessing,
        split=split,
        row_positions=row_positions,
        evidence_dataframe=evidence_dataframe,
    )
    validation.raise_if_failed()
    frame, target = modeling_arrays(dataframe, validation)
    usable_features = validation.features_used
    numeric_features = [
        column for column in usable_features if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_features = [column for column in usable_features if column not in numeric_features]
    approved_preprocessing = validation.preprocessing_contract
    if approved_preprocessing is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("Validated training plan did not produce a preprocessing contract.")
    preprocessor = build_preprocessor(
        approved_preprocessing,
        numeric_features,
        categorical_features,
        method,
    )
    if split is None:
        stratify = target if task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            frame,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    else:
        current_positions = validated_row_positions(dataframe, validation, row_positions)
        train_mask = np.isin(current_positions, np.asarray(split.train_row_positions, dtype=int))
        holdout_mask = np.isin(current_positions, np.asarray(split.holdout_row_positions, dtype=int))
        if not np.all(train_mask | holdout_mask) or np.any(train_mask & holdout_mask):
            raise ValueError("The validated rows do not resolve to the frozen train/holdout membership.")
        X_train = frame.loc[train_mask].copy()
        X_test = frame.loc[holdout_mask].copy()
        y_train = target.loc[train_mask].copy()
        y_test = target.loc[holdout_mask].copy()
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
        "approved_preprocessing": approved_preprocessing.model_dump(mode="json"),
        "executed_preprocessing": {
            "contract": approved_preprocessing.model_dump(mode="json"),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "fit_inside_pipeline": approved_preprocessing.fit_inside_pipeline,
            "holdout_used_for": "final_evaluation_only",
            "pipeline_components": _pipeline_components(
                approved_preprocessing,
                numeric_features,
                categorical_features,
            ),
        },
    }


def _estimator(task_type: TaskType, method: Method, random_state: int) -> Any:
    if task_type == "classification":
        return {
            # ``linear`` is intentionally unregularized.  sklearn's default
            # LogisticRegression penalty is L2, so the old C=1 vs C=0.5
            # distinction did not accurately represent two model families.
            "linear": LogisticRegression(
                penalty=None, solver="lbfgs", max_iter=1000, random_state=random_state
            ),
            "regularized_linear": LogisticRegression(
                # l1_ratio=0 expresses the L2 branch in sklearn's newer API
                # while remaining an accepted no-op compatibility parameter
                # on the project's older supported sklearn versions.
                C=0.5, l1_ratio=0.0, max_iter=1000, random_state=random_state
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


def _pipeline_components(
    contract: PreprocessingContract,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> dict[str, Any]:
    """Return a concise, stable description of what the fitted pipeline runs."""

    numeric_steps: list[str] = []
    if numeric_features and contract.numeric_imputation == "median":
        numeric_steps.append("median_imputation")
    if numeric_features and contract.numeric_scaling == "standard":
        numeric_steps.append("standard_scaling")
    categorical_steps: list[str] = []
    if categorical_features and contract.categorical_imputation == "most_frequent":
        categorical_steps.append("most_frequent_imputation")
    if categorical_features and contract.categorical_encoding != "none":
        categorical_steps.append(contract.categorical_encoding)
    return {
        "numeric": numeric_steps,
        "categorical": categorical_steps,
        "unknown_category_handling": contract.categorical_unknown_handling,
        "remainder": "drop",
        "infinity_handling_before_pipeline": contract.infinity_handling,
    }


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
