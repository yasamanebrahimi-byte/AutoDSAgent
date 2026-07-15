"""Schema inference utilities for tabular datasets."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


SemanticType = Literal[
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "text",
    "id",
    "unknown",
]

BOOLEAN_TOKENS = {
    "0",
    "1",
    "false",
    "true",
    "f",
    "t",
    "n",
    "no",
    "y",
    "yes",
}


def infer_column_dtypes(dataframe: pd.DataFrame) -> dict[str, str]:
    """Return pandas dtype names keyed by column name."""

    return {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}


def infer_schema(dataframe: pd.DataFrame) -> dict[str, SemanticType]:
    """Infer semantic types for every column in a DataFrame."""

    return {
        str(column): infer_semantic_type(dataframe[column], str(column))
        for column in dataframe.columns
    }


def infer_semantic_type(series: pd.Series, column_name: str) -> SemanticType:
    """Infer an explainable semantic type for one column."""

    non_null = series.dropna()

    if non_null.empty:
        return "unknown"

    if is_likely_id_column(series, column_name):
        return "id"

    if is_boolean_like(series):
        return "boolean"

    if is_datetime_like(series):
        return "datetime"

    if is_numeric_dtype(series):
        return "numeric"

    if is_text_like(series):
        return "text"

    if is_categorical_like(series):
        return "categorical"

    return "unknown"


def is_likely_id_column(series: pd.Series, column_name: str) -> bool:
    """Return whether a column looks like an identifier."""

    normalized_name = _normalize_column_name(column_name)
    if (
        normalized_name == "id"
        or normalized_name.endswith("_id")
        or normalized_name.startswith("id_")
        or "_id_" in normalized_name
        or "identifier" in normalized_name
    ):
        return True

    non_null_count = int(series.notna().sum())
    if non_null_count < 20:
        return False

    unique_ratio = _unique_ratio(series)
    return unique_ratio >= 0.98


def is_boolean_like(series: pd.Series) -> bool:
    """Return whether a column looks boolean."""

    if is_bool_dtype(series):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_values = non_null.drop_duplicates()
    if len(unique_values) != 2:
        return False

    normalized_values = {str(value).strip().lower() for value in unique_values}
    if normalized_values.issubset(BOOLEAN_TOKENS):
        return True

    return not is_numeric_dtype(series)


def is_datetime_like(series: pd.Series, parse_threshold: float = 0.8) -> bool:
    """Return whether most non-null values can be parsed as datetimes."""

    if is_datetime64_any_dtype(series):
        return True

    if is_numeric_dtype(series):
        return False

    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False

    non_null = series.dropna()
    if len(non_null) < 2:
        return False

    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= parse_threshold


def is_text_like(series: pd.Series, min_average_length: int = 40) -> bool:
    """Return whether an object/string column looks like free text."""

    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False

    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False

    average_length = float(non_null.str.len().mean())
    unique_ratio = _unique_ratio(series)

    return average_length >= min_average_length or (
        average_length >= 20 and unique_ratio >= 0.5
    )


def is_categorical_like(series: pd.Series) -> bool:
    """Return whether a column looks categorical."""

    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False

    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return False

    unique_count = int(series.nunique(dropna=True))
    unique_ratio = unique_count / non_null_count

    return unique_count <= 20 or unique_ratio <= 0.2


def _unique_ratio(series: pd.Series) -> float:
    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return 0.0
    return float(series.nunique(dropna=True) / non_null_count)


def _normalize_column_name(column_name: str) -> str:
    normalized = column_name.strip().lower()
    for character in (" ", "-", ".", "/"):
        normalized = normalized.replace(character, "_")
    return normalized
