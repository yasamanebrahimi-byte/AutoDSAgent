"""Preprocessing utilities for deterministic tabular modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.tools.schema_inference import (
    infer_identifier,
    infer_semantic_type,
    is_boolean_like,
)


TaskType = Literal["regression", "classification"]

VALID_TASK_TYPES: set[str] = {"regression", "classification"}
MIN_TARGET_NON_NULL_VALUES = 5
MIN_CLASS_OBSERVATIONS_FOR_HOLDOUT_AND_CV = 3
DEFAULT_CV_FOLDS = 5
MAX_TEXT_TARGET_CLASSES = 50
MAX_CATEGORICAL_FEATURE_LEVELS = 50
MAX_TOTAL_ONE_HOT_LEVELS = 500


@dataclass(frozen=True)
class TaskInferenceResult:
    """Task decision with a compact reason for reports and debugging."""

    task_type: TaskType
    reason: str
    validation: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreprocessingResult:
    """Prepared train/test data and reusable preprocessing metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    task_type: TaskType
    target_column: str
    rows_used: int
    columns_used: int
    train_rows: int
    test_rows: int
    features_used: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    boolean_features: list[str]
    features_excluded: list[str]
    excluded_feature_reasons: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    task_inference_reason: str | None = None
    classification_validation: dict[str, object] = field(default_factory=dict)
    cv_folds: int = DEFAULT_CV_FOLDS
    cv_strategy: str = "kfold"
    actual_test_size: float = 0.2


def infer_task_type(
    dataframe: pd.DataFrame,
    target_column: str,
    requested_task_type: TaskType | str | None = None,
) -> TaskType:
    """Infer or validate whether a modeling target is regression or classification."""

    return infer_task_type_with_reason(
        dataframe=dataframe,
        target_column=target_column,
        requested_task_type=requested_task_type,
    ).task_type


def infer_task_type_with_reason(
    dataframe: pd.DataFrame,
    target_column: str,
    requested_task_type: TaskType | str | None = None,
    test_size: float = 0.2,
) -> TaskInferenceResult:
    """Infer or validate the task type and explain the decision."""

    _validate_target_exists(dataframe, target_column)
    target = dataframe[target_column]
    non_null = target.dropna()

    if len(non_null) < MIN_TARGET_NON_NULL_VALUES:
        raise ValueError(
            f"Target column '{target_column}' must have at least "
            f"{MIN_TARGET_NON_NULL_VALUES} non-null values."
        )

    unique_values = int(non_null.nunique(dropna=True))
    if unique_values <= 1:
        raise ValueError(f"Target column '{target_column}' is constant and cannot be modeled.")

    if infer_identifier(non_null, target_column, context="target"):
        raise ValueError(
            f"Target column '{target_column}' appears to be an identifier. "
            "Choose an outcome column instead."
        )

    if requested_task_type is not None:
        task_type = str(requested_task_type).lower()
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(
                "task_type must be either 'regression', 'classification', or null."
            )
        if task_type == "regression":
            numeric_target = pd.to_numeric(non_null, errors="coerce")
            if float(numeric_target.notna().mean()) < 0.95:
                raise ValueError(
                    f"Target column '{target_column}' cannot be used for regression "
                    "because it is not numeric."
                )
            return TaskInferenceResult(
                task_type="regression",
                reason="explicit regression request with numeric target",
            )

        validation = validate_classification_target(
            non_null,
            target_column=target_column,
            test_size=test_size,
            allow_numeric_continuous=False,
        )
        return TaskInferenceResult(
            task_type="classification",
            reason="explicit classification request validated against target structure",
            validation=validation,
        )

    if is_bool_dtype(target) or is_boolean_like(non_null):
        validation = validate_classification_target(
            non_null,
            target_column=target_column,
            test_size=test_size,
            allow_numeric_continuous=True,
        )
        return TaskInferenceResult(
            task_type="classification",
            reason="boolean target -> classification",
            validation=validation,
        )

    numeric_target = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = float(numeric_target.notna().mean())
    if is_numeric_dtype(target) or numeric_ratio >= 0.95:
        if _is_low_cardinality_discrete_numeric_target(numeric_target, len(non_null)):
            validation = validate_classification_target(
                non_null,
                target_column=target_column,
                test_size=test_size,
                allow_numeric_continuous=True,
            )
            return TaskInferenceResult(
                task_type="classification",
                reason="low-cardinality discrete numeric target -> classification",
                validation=validation,
            )

        return TaskInferenceResult(
            task_type="regression",
            reason="continuous numeric target -> regression",
        )

    unique_ratio = unique_values / max(len(non_null), 1)
    if unique_values <= 20 and (unique_values <= 5 or unique_ratio <= 0.5):
        validation = validate_classification_target(
            non_null,
            target_column=target_column,
            test_size=test_size,
            allow_numeric_continuous=True,
        )
        return TaskInferenceResult(
            task_type="classification",
            reason="low-cardinality categorical string target -> classification",
            validation=validation,
        )

    raise ValueError(
        f"Target column '{target_column}' has too many unique text values for the current "
        "classification workflow. Text modeling is future work."
    )


def validate_classification_target(
    series: pd.Series,
    target_column: str,
    test_size: float = 0.2,
    allow_numeric_continuous: bool = False,
) -> dict[str, object]:
    """Validate that a target can support classification selection and holdout testing."""

    non_null = series.dropna()
    row_count = int(len(non_null))
    if row_count < MIN_TARGET_NON_NULL_VALUES:
        raise ValueError(
            f"Target column '{target_column}' must have at least "
            f"{MIN_TARGET_NON_NULL_VALUES} non-null values."
        )

    class_counts = non_null.astype(str).value_counts(dropna=False)
    class_count = int(len(class_counts))
    if class_count <= 1:
        raise ValueError(f"Target column '{target_column}' is constant and cannot be modeled.")

    unique_ratio = class_count / max(row_count, 1)
    min_class_count = int(class_counts.min())
    if infer_identifier(non_null, target_column, context="target"):
        raise ValueError(
            f"Target column '{target_column}' appears to be an identifier. "
            "Choose an outcome column instead."
        )

    numeric_target = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = float(numeric_target.notna().mean())
    if (
        not allow_numeric_continuous
        and numeric_ratio >= 0.95
        and not is_boolean_like(non_null)
        and not _is_low_cardinality_discrete_numeric_target(numeric_target, row_count)
    ):
        raise ValueError(
            f"Target column '{target_column}' looks continuous/high-cardinality, "
            "so it is not a valid classification target. Use regression or choose "
            "a discrete outcome."
        )

    if class_count > MAX_TEXT_TARGET_CLASSES or (
        class_count > 20 and unique_ratio > 0.5
    ):
        raise ValueError(
            f"Target column '{target_column}' has {class_count} classes "
            f"({unique_ratio:.1%} unique values), which is too high-cardinality "
            "for reliable classification."
        )

    if min_class_count < MIN_CLASS_OBSERVATIONS_FOR_HOLDOUT_AND_CV:
        raise ValueError(
            f"Target column '{target_column}' has a rare class with only "
            f"{min_class_count} observation(s). Classification needs at least "
            f"{MIN_CLASS_OBSERVATIONS_FOR_HOLDOUT_AND_CV} observations per class "
            "so one holdout row and at least two training rows remain for CV."
        )

    split_plan = _classification_split_plan(non_null.astype(str), float(test_size))
    return {
        "class_count": class_count,
        "min_class_count": min_class_count,
        "unique_ratio": float(unique_ratio),
        "requested_test_size": float(test_size),
        "planned_test_rows": int(split_plan["test_rows"]),
        "planned_train_rows": int(row_count - int(split_plan["test_rows"])),
    }


def prepare_modeling_data(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: TaskType | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessingResult:
    """Validate a dataset, build features, and return leakage-safe train/test splits."""

    if not 0 < float(test_size) < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    inference = infer_task_type_with_reason(
        dataframe=dataframe,
        target_column=target_column,
        requested_task_type=task_type,
        test_size=float(test_size),
    )
    inferred_task_type = inference.task_type
    working = dataframe.copy()

    if inferred_task_type == "regression":
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")

    working = working.loc[working[target_column].notna()].copy()
    _validate_rows_for_split(len(working), float(test_size))

    if int(working[target_column].nunique(dropna=True)) <= 1:
        raise ValueError(
            f"Target column '{target_column}' is constant after removing missing values."
        )

    if inferred_task_type == "regression":
        y = pd.to_numeric(working[target_column], errors="coerce")
        split_test_size: float | int = float(test_size)
        split_warnings: list[str] = []
    else:
        y = working[target_column].astype(str)
        split_plan = _classification_split_plan(y, float(test_size))
        split_test_size = int(split_plan["test_rows"])
        split_warnings = list(split_plan["warnings"])

    try:
        train_frame, test_frame, y_train, y_test = train_test_split(
            working,
            y,
            test_size=split_test_size,
            random_state=int(random_state),
            stratify=y if inferred_task_type == "classification" else None,
        )
    except ValueError as exc:
        if inferred_task_type == "classification":
            raise ValueError(
                "The classification target cannot be split into stratified train/test "
                f"partitions for the requested test_size: {exc}"
            ) from exc
        raise

    # Feature metadata is inferred after splitting so holdout-only values cannot
    # decide which preprocessing columns or levels are learned.
    X_train, X_test, feature_metadata = _build_feature_frames_from_training_partition(
        train_dataframe=train_frame,
        test_dataframe=test_frame,
        target_column=target_column,
    )
    if X_train.empty or not feature_metadata["features_used"]:
        raise ValueError("No usable feature columns remain after preprocessing exclusions.")

    preprocessor = build_preprocessor(
        numeric_features=feature_metadata["numeric_features"],
        categorical_features=feature_metadata["categorical_features"],
        boolean_features=feature_metadata["boolean_features"],
    )

    cv_folds, cv_strategy, cv_warnings = _cv_metadata(
        y_train=y_train,
        task_type=inferred_task_type,
    )
    warnings = (
        list(feature_metadata["warnings"])
        + list(inference.warnings)
        + split_warnings
        + cv_warnings
    )

    return PreprocessingResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        task_type=inferred_task_type,
        target_column=target_column,
        rows_used=int(len(working)),
        columns_used=len(feature_metadata["features_used"]),
        train_rows=int(len(X_train)),
        test_rows=int(len(X_test)),
        features_used=feature_metadata["features_used"],
        numeric_features=feature_metadata["numeric_features"],
        categorical_features=feature_metadata["categorical_features"],
        boolean_features=feature_metadata["boolean_features"],
        features_excluded=feature_metadata["features_excluded"],
        excluded_feature_reasons=feature_metadata["excluded_feature_reasons"],
        warnings=_deduplicate(warnings),
        task_inference_reason=inference.reason,
        classification_validation=inference.validation,
        cv_folds=cv_folds,
        cv_strategy=cv_strategy,
        actual_test_size=float(len(X_test) / len(working)),
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    boolean_features: list[str],
    sparse_output: bool = True,
) -> ColumnTransformer:
    """Build a reusable sklearn ColumnTransformer for tabular features."""

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if boolean_features:
        boolean_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)),
            ]
        )
        transformers.append(("boolean", boolean_pipeline, boolean_features))

    if not transformers:
        raise ValueError("At least one usable feature column is required.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return transformed feature names after a ColumnTransformer has been fit."""

    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        return []
    return [str(name) for name in names]


def _build_feature_frames_from_training_partition(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train_working = train_dataframe.copy()
    test_working = test_dataframe.copy()
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    boolean_features: list[str] = []
    features_excluded: list[str] = []
    excluded_feature_reasons: dict[str, str] = {}
    warnings: list[str] = []
    estimated_one_hot_levels = 0

    def exclude(column: str, reason: str) -> None:
        if column not in excluded_feature_reasons:
            features_excluded.append(column)
            excluded_feature_reasons[column] = reason

    for column in list(train_dataframe.columns):
        column_name = str(column)

        if column_name == target_column:
            exclude(column_name, "target column")
            continue

        series = train_working[column_name]
        non_null_count = int(series.notna().sum())
        if non_null_count == 0:
            exclude(column_name, "all values are missing")
            continue

        if int(series.nunique(dropna=True)) <= 1:
            exclude(column_name, "constant feature")
            continue

        if infer_identifier(series, column_name, context="feature"):
            exclude(column_name, "likely identifier")
            continue

        semantic_type = infer_semantic_type(series, column_name)
        if semantic_type == "id" and _looks_like_free_text(series):
            semantic_type = "text"
        if semantic_type == "numeric":
            train_working[column_name] = pd.to_numeric(series, errors="coerce")
            test_working[column_name] = pd.to_numeric(
                test_working[column_name],
                errors="coerce",
            )
            numeric_features.append(column_name)
        elif semantic_type == "boolean":
            boolean_features.append(column_name)
        elif semantic_type == "categorical":
            level_count = int(series.nunique(dropna=True))
            if _is_high_cardinality_categorical(series):
                exclude(column_name, "high-cardinality categorical feature")
                continue
            if estimated_one_hot_levels + level_count > MAX_TOTAL_ONE_HOT_LEVELS:
                exclude(column_name, "one-hot feature budget exceeded")
                continue
            estimated_one_hot_levels += level_count
            categorical_features.append(column_name)
        elif semantic_type == "datetime":
            created_columns = _expand_datetime_feature(
                train_working,
                test_working,
                column_name,
            )
            if created_columns:
                exclude(column_name, "expanded into simple datetime features")
                numeric_features.extend(created_columns)
            else:
                exclude(column_name, "datetime values could not be parsed consistently")
        elif semantic_type == "text":
            exclude(column_name, "free-text modeling is future work")
        else:
            exclude(column_name, f"unsupported type: {semantic_type}")

    features_used = numeric_features + categorical_features + boolean_features
    X_train = train_working[features_used].copy()
    X_test = test_working[features_used].copy()

    for column in numeric_features:
        X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
        X_test[column] = pd.to_numeric(X_test[column], errors="coerce")

    for column in categorical_features + boolean_features:
        _stringify_non_missing_values(X_train, column)
        _stringify_non_missing_values(X_test, column)

    excluded_reasons = set(excluded_feature_reasons.values())
    if "free-text modeling is future work" in excluded_reasons:
        warnings.append("Free-text columns were excluded from the current modeling workflow.")
    if "likely identifier" in excluded_reasons:
        warnings.append("Likely ID columns were excluded from modeling.")
    if any("cardinality" in reason or "one-hot" in reason for reason in excluded_reasons):
        warnings.append(
            "High-cardinality categorical columns were excluded to avoid excessive one-hot expansion."
        )

    return X_train, X_test, {
        "features_used": features_used,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "boolean_features": boolean_features,
        "features_excluded": features_excluded,
        "excluded_feature_reasons": excluded_feature_reasons,
        "warnings": warnings,
    }


def _stringify_non_missing_values(dataframe: pd.DataFrame, column: str) -> None:
    missing_mask = dataframe[column].isna()
    dataframe[column] = dataframe[column].astype("object")
    dataframe.loc[~missing_mask, column] = dataframe.loc[~missing_mask, column].astype(str)


def _expand_datetime_feature(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    column: str,
) -> list[str]:
    parsed_train = pd.to_datetime(train_dataframe[column], errors="coerce", format="mixed")
    if int(parsed_train.notna().sum()) < 2:
        return []
    parsed_test = pd.to_datetime(test_dataframe[column], errors="coerce", format="mixed")

    components = {
        "year": (parsed_train.dt.year, parsed_test.dt.year),
        "month": (parsed_train.dt.month, parsed_test.dt.month),
        "day": (parsed_train.dt.day, parsed_test.dt.day),
        "dayofweek": (parsed_train.dt.dayofweek, parsed_test.dt.dayofweek),
    }
    created_columns: list[str] = []
    for suffix, (train_values, test_values) in components.items():
        feature_name = _unique_feature_name(train_dataframe, f"{column}__{suffix}")
        train_dataframe[feature_name] = train_values.astype("float64")
        test_dataframe[feature_name] = test_values.astype("float64")
        created_columns.append(feature_name)

    return created_columns


def _validate_target_exists(dataframe: pd.DataFrame, target_column: str) -> None:
    if not target_column or not str(target_column).strip():
        raise ValueError("A target column is required for modeling.")
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")


def _validate_rows_for_split(row_count: int, test_size: float) -> None:
    if row_count < MIN_TARGET_NON_NULL_VALUES:
        raise ValueError(
            f"At least {MIN_TARGET_NON_NULL_VALUES} rows with non-null target values "
            "are required for modeling."
        )

    test_rows = int(math.ceil(row_count * test_size))
    train_rows = row_count - test_rows
    if train_rows < 2 or test_rows < 1:
        raise ValueError(
            "The dataset is too small for the requested train/test split. "
            "Use more rows or a smaller test_size."
        )


def _classification_split_plan(y: pd.Series, test_size: float) -> dict[str, object]:
    class_counts = y.astype(str).value_counts(dropna=False)
    row_count = int(len(y))
    class_count = int(len(class_counts))
    requested_test_rows = int(math.ceil(row_count * test_size))
    min_test_rows = class_count
    max_test_rows = int(sum(max(int(count) - 2, 0) for count in class_counts))

    if max_test_rows < min_test_rows:
        raise ValueError(
            "The classification target cannot support both a stratified holdout set "
            "and stratified cross-validation. Each class needs at least three rows."
        )

    test_rows = min(max(requested_test_rows, min_test_rows), max_test_rows)
    warnings: list[str] = []
    if test_rows != requested_test_rows:
        warnings.append(
            "Classification test_size was adjusted so every class can appear in the "
            "holdout set while at least two training rows per class remain for CV."
        )

    train_rows = row_count - test_rows
    if train_rows < class_count * 2:
        raise ValueError(
            "The classification train split would not leave enough rows per class "
            "for cross-validation."
        )

    return {"test_rows": int(test_rows), "warnings": warnings}


def _cv_metadata(y_train: pd.Series, task_type: TaskType) -> tuple[int, str, list[str]]:
    warnings: list[str] = []
    if task_type == "regression":
        folds = min(DEFAULT_CV_FOLDS, int(len(y_train)))
        if folds < 2:
            raise ValueError("At least two training rows are required for regression CV.")
        if folds < DEFAULT_CV_FOLDS:
            warnings.append(f"Regression CV was reduced to {folds} folds for the training size.")
        return folds, "kfold", warnings

    class_counts = y_train.astype(str).value_counts(dropna=False)
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    folds = min(DEFAULT_CV_FOLDS, min_class_count)
    if folds < 2:
        raise ValueError(
            "The classification train split has a class with fewer than two rows, "
            "so stratified cross-validation is not reliable."
        )
    if folds < DEFAULT_CV_FOLDS:
        warnings.append(
            f"Classification CV was reduced to {folds} folds because of rare classes."
        )
    return folds, "stratified_kfold", warnings


def _is_low_cardinality_discrete_numeric_target(
    numeric_target: pd.Series,
    row_count: int,
) -> bool:
    numeric = pd.to_numeric(numeric_target, errors="coerce").dropna()
    if numeric.empty:
        return False
    unique_values = int(numeric.nunique(dropna=True))
    unique_ratio = unique_values / max(row_count, 1)
    if not _is_integer_like(numeric):
        return False
    return unique_values <= 10 or (unique_values <= 20 and unique_ratio <= 0.02)


def _is_high_cardinality_categorical(series: pd.Series) -> bool:
    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return False
    unique_values = int(series.nunique(dropna=True))
    unique_ratio = unique_values / non_null_count
    return unique_values > MAX_CATEGORICAL_FEATURE_LEVELS and unique_ratio > 0.5


def _is_integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    return bool((numeric % 1 == 0).all())


def _looks_like_free_text(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False

    average_length = float(non_null.str.len().mean())
    return average_length >= 40 or (
        average_length >= 20 and _unique_ratio(series) >= 0.5
    )


def _unique_ratio(series: pd.Series) -> float:
    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return 0.0
    return float(series.nunique(dropna=True) / non_null_count)


def _unique_feature_name(dataframe: pd.DataFrame, base_name: str) -> str:
    if base_name not in dataframe.columns:
        return base_name

    index = 2
    while f"{base_name}_{index}" in dataframe.columns:
        index += 1
    return f"{base_name}_{index}"


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduplicated: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduplicated.append(value)
    return deduplicated
