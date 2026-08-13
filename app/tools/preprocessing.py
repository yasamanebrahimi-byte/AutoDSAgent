"""Preprocessing utilities for deterministic tabular modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.tools.schema_inference import infer_semantic_type


TaskType = Literal["regression", "classification"]

VALID_TASK_TYPES: set[str] = {"regression", "classification"}
MIN_TARGET_NON_NULL_VALUES = 5
MAX_TEXT_TARGET_CLASSES = 50


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


def infer_task_type(
    dataframe: pd.DataFrame,
    target_column: str,
    requested_task_type: TaskType | str | None = None,
) -> TaskType:
    """Infer or validate whether a modeling target is regression or classification."""

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

    if _is_likely_id_target(non_null, target_column):
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
        return task_type  # type: ignore[return-value]

    if is_bool_dtype(target) or _is_boolean_like(non_null):
        return "classification"

    numeric_target = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = float(numeric_target.notna().mean())
    if is_numeric_dtype(target) or numeric_ratio >= 0.95:
        numeric_unique_values = int(numeric_target.nunique(dropna=True))
        if numeric_unique_values <= _low_cardinality_limit(len(non_null)):
            return "classification"
        return "regression"

    unique_ratio = unique_values / max(len(non_null), 1)
    if unique_values <= MAX_TEXT_TARGET_CLASSES and (
        unique_values <= 20 or unique_ratio <= 0.5
    ):
        return "classification"

    raise ValueError(
        f"Target column '{target_column}' has too many unique text values for the current "
        "classification workflow. Text modeling is future work."
    )


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

    inferred_task_type = infer_task_type(dataframe, target_column, task_type)
    working = dataframe.copy()

    if inferred_task_type == "regression":
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")

    working = working.loc[working[target_column].notna()].copy()
    _validate_rows_for_split(len(working), float(test_size))

    if int(working[target_column].nunique(dropna=True)) <= 1:
        raise ValueError(
            f"Target column '{target_column}' is constant after removing missing values."
        )

    X, feature_metadata = _build_feature_frame(working, target_column)
    if X.empty or not feature_metadata["features_used"]:
        raise ValueError("No usable feature columns remain after preprocessing exclusions.")

    if inferred_task_type == "regression":
        y = pd.to_numeric(working[target_column], errors="coerce")
    else:
        y = working[target_column].astype(str)

    preprocessor = build_preprocessor(
        numeric_features=feature_metadata["numeric_features"],
        categorical_features=feature_metadata["categorical_features"],
        boolean_features=feature_metadata["boolean_features"],
    )

    stratify_target = (
        _stratify_target(y, float(test_size))
        if inferred_task_type == "classification"
        else None
    )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=stratify_target,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=None,
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
        warnings=feature_metadata["warnings"],
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    boolean_features: list[str],
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
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if boolean_features:
        boolean_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("boolean", boolean_pipeline, boolean_features))

    if not transformers:
        raise ValueError("At least one usable feature column is required.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return transformed feature names after a ColumnTransformer has been fit."""

    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        return []
    return [str(name) for name in names]


def _build_feature_frame(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    working = dataframe.copy()
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    boolean_features: list[str] = []
    features_excluded: list[str] = []
    excluded_feature_reasons: dict[str, str] = {}
    warnings: list[str] = []

    def exclude(column: str, reason: str) -> None:
        if column not in excluded_feature_reasons:
            features_excluded.append(column)
            excluded_feature_reasons[column] = reason

    for column in list(dataframe.columns):
        column_name = str(column)

        if column_name == target_column:
            exclude(column_name, "target column")
            continue

        series = working[column_name]
        non_null_count = int(series.notna().sum())
        if non_null_count == 0:
            exclude(column_name, "all values are missing")
            continue

        if int(series.nunique(dropna=True)) <= 1:
            exclude(column_name, "constant feature")
            continue

        if _is_likely_id_feature(series, column_name):
            exclude(column_name, "likely identifier")
            continue

        semantic_type = infer_semantic_type(series, column_name)
        if semantic_type == "id" and _looks_like_free_text(series):
            semantic_type = "text"
        if semantic_type == "numeric":
            working[column_name] = pd.to_numeric(series, errors="coerce")
            numeric_features.append(column_name)
        elif semantic_type == "boolean":
            boolean_features.append(column_name)
        elif semantic_type == "categorical":
            categorical_features.append(column_name)
        elif semantic_type == "datetime":
            created_columns = _expand_datetime_feature(working, column_name)
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
    X = working[features_used].copy()

    for column in numeric_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    for column in categorical_features + boolean_features:
        missing_mask = X[column].isna()
        X[column] = X[column].astype("object")
        X.loc[~missing_mask, column] = X.loc[~missing_mask, column].astype(str)

    if any(reason == "free-text modeling is future work" for reason in excluded_feature_reasons.values()):
        warnings.append("Free-text columns were excluded from the current modeling workflow.")
    if any(reason == "likely identifier" for reason in excluded_feature_reasons.values()):
        warnings.append("Likely ID columns were excluded from modeling.")

    return X, {
        "features_used": features_used,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "boolean_features": boolean_features,
        "features_excluded": features_excluded,
        "excluded_feature_reasons": excluded_feature_reasons,
        "warnings": warnings,
    }


def _expand_datetime_feature(dataframe: pd.DataFrame, column: str) -> list[str]:
    parsed = pd.to_datetime(dataframe[column], errors="coerce", format="mixed")
    if int(parsed.notna().sum()) < 2:
        return []

    components = {
        "year": parsed.dt.year,
        "month": parsed.dt.month,
        "day": parsed.dt.day,
        "dayofweek": parsed.dt.dayofweek,
    }
    created_columns: list[str] = []
    for suffix, values in components.items():
        feature_name = _unique_feature_name(dataframe, f"{column}__{suffix}")
        dataframe[feature_name] = values.astype("float64")
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


def _stratify_target(y: pd.Series, test_size: float) -> pd.Series | None:
    class_counts = y.value_counts(dropna=False)
    if class_counts.empty or int(class_counts.min()) < 2:
        return None

    row_count = len(y)
    class_count = len(class_counts)
    test_rows = int(math.ceil(row_count * test_size))
    train_rows = row_count - test_rows
    if test_rows < class_count or train_rows < class_count:
        return None

    return y


def _is_likely_id_target(series: pd.Series, column_name: str) -> bool:
    if _is_id_like_name(column_name):
        return True

    if is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return (
            len(numeric) >= 20
            and _unique_ratio(numeric) >= 0.98
            and is_integer_dtype(numeric)
            and (numeric.is_monotonic_increasing or numeric.is_monotonic_decreasing)
        )

    return len(series) >= 20 and _unique_ratio(series) >= 0.98


def _is_likely_id_feature(series: pd.Series, column_name: str) -> bool:
    if _is_id_like_name(column_name):
        return True

    non_null = series.dropna()
    if len(non_null) < 20 or _unique_ratio(series) < 0.98:
        return False

    if is_numeric_dtype(non_null):
        numeric = pd.to_numeric(non_null, errors="coerce")
        if is_integer_dtype(numeric) and (
            numeric.is_monotonic_increasing or numeric.is_monotonic_decreasing
        ):
            return True
        return False

    average_length = float(non_null.astype(str).str.len().mean())
    if average_length >= 40:
        return False

    return True


def _is_boolean_like(series: pd.Series) -> bool:
    normalized_values = {str(value).strip().lower() for value in series.dropna().unique()}
    return len(normalized_values) == 2 and normalized_values.issubset(
        {"0", "1", "false", "true", "f", "t", "n", "no", "y", "yes"}
    )


def _looks_like_free_text(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False

    average_length = float(non_null.str.len().mean())
    return average_length >= 40 or (
        average_length >= 20 and _unique_ratio(series) >= 0.5
    )


def _low_cardinality_limit(row_count: int) -> int:
    return max(10, int(row_count * 0.05))


def _unique_ratio(series: pd.Series) -> float:
    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return 0.0
    return float(series.nunique(dropna=True) / non_null_count)


def _is_id_like_name(column_name: str) -> bool:
    normalized = _normalize_column_name(column_name)
    return (
        normalized == "id"
        or normalized.endswith("_id")
        or normalized.startswith("id_")
        or "_id_" in normalized
        or "identifier" in normalized
        or "uuid" in normalized
        or "guid" in normalized
    )


def _normalize_column_name(column_name: str) -> str:
    normalized = column_name.strip().lower()
    for character in (" ", "-", ".", "/"):
        normalized = normalized.replace(character, "_")
    return normalized


def _unique_feature_name(dataframe: pd.DataFrame, base_name: str) -> str:
    if base_name not in dataframe.columns:
        return base_name

    index = 2
    while f"{base_name}_{index}" in dataframe.columns:
        index += 1
    return f"{base_name}_{index}"
