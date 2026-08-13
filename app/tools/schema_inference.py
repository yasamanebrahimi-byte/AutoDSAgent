"""Schema inference utilities for tabular datasets."""

from __future__ import annotations

import re
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
IdentifierContext = Literal["feature", "target", "profile"]

BOOLEAN_TOKEN_PAIRS = {
    frozenset({"true", "false"}),
    frozenset({"t", "f"}),
    frozenset({"yes", "no"}),
    frozenset({"y", "n"}),
    frozenset({"1", "0"}),
    frozenset({"on", "off"}),
}
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
CODE_PATTERN = re.compile(
    r"^(?:[A-Za-z]{1,12}[-_ ]?\d{2,}[A-Za-z0-9_-]*|\d{2,}[-_ ]?[A-Za-z]{1,12}[A-Za-z0-9_-]*)$"
)


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

    if infer_identifier(series, column_name, context="profile"):
        return "id"

    if is_boolean_like(series):
        return "boolean"

    if is_datetime_like(series):
        return "datetime"

    if is_numeric_dtype(series):
        return "numeric"

    if is_numeric_like_string_feature(series, column_name):
        return "numeric"

    if is_text_like(series):
        return "text"

    if is_categorical_like(series):
        return "categorical"

    return "unknown"


def infer_identifier(
    series: pd.Series,
    column_name: str,
    context: IdentifierContext = "feature",
) -> bool:
    """Return whether a column has identifier evidence in the given context.

    Feature/profile inference can remove obvious identifiers aggressively. Target
    inference is intentionally narrower so continuous regression outcomes are
    not rejected merely because every value is unique.
    """

    non_null = series.dropna()
    if non_null.empty:
        return False

    if _looks_like_uuid_values(non_null):
        return True

    has_identifier_name = _is_identifier_name(column_name)
    has_code_name = _is_code_name(column_name)
    if has_identifier_name:
        return not _contains_non_integer_numeric_values(non_null)

    non_null_count = int(series.notna().sum())
    if non_null_count < 20:
        return False

    unique_ratio = _unique_ratio(series)
    if unique_ratio < 0.98:
        return False

    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(non_null) else 0.0
    if is_numeric_dtype(non_null) or numeric_ratio >= 0.98:
        numeric = numeric.dropna()
        if numeric.empty or not _is_integer_like_numeric(numeric):
            return False
        if context == "target":
            return has_code_name and bool(
                numeric.is_monotonic_increasing or numeric.is_monotonic_decreasing
            )
        return has_code_name or bool(
            numeric.is_monotonic_increasing or numeric.is_monotonic_decreasing
        )

    string_values = non_null.astype(str).str.strip()
    average_length = float(string_values.str.len().mean())
    if average_length >= 40:
        return False

    if context == "target":
        return has_code_name and _looks_like_fixed_format_codes(string_values)

    return has_code_name and _looks_like_fixed_format_codes(string_values)


def is_likely_id_column(series: pd.Series, column_name: str) -> bool:
    """Return whether a column has explicit identifier evidence.

    High cardinality alone is deliberately not enough: continuous numeric
    measurements, regression targets, latitudes, probabilities, and random
    floats are often unique without being identifiers.
    """

    return infer_identifier(series, column_name, context="profile")


def is_numeric_like_string_feature(
    series: pd.Series,
    column_name: str,
    parse_threshold: float = 0.8,
) -> bool:
    """Return whether an object/string feature should be modeled as numeric."""

    if is_numeric_dtype(series):
        return False
    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False
    if infer_identifier(series, column_name, context="feature"):
        return False
    if is_boolean_like(series):
        return False

    non_null = series.dropna()
    if len(non_null) < 3:
        return False

    text_values = non_null.astype(str).str.strip()
    numeric = pd.to_numeric(text_values, errors="coerce")
    if float(numeric.notna().mean()) < parse_threshold:
        return False
    if int(numeric.nunique(dropna=True)) <= 1:
        return False
    if _looks_like_categorical_numeric_codes(text_values, numeric, column_name):
        return False

    return True


def _looks_like_categorical_numeric_codes(
    values: pd.Series,
    numeric: pd.Series,
    column_name: str,
) -> bool:
    if not _is_postal_or_code_name(column_name):
        return False

    numeric_non_null = numeric.dropna()
    if numeric_non_null.empty or not _is_integer_like_numeric(numeric_non_null):
        return False

    integer_like_values = values[numeric.notna()]
    stripped = integer_like_values.str.replace(r"^[+-]", "", regex=True)
    if not stripped.str.fullmatch(r"\d+").all():
        return False

    lengths = stripped.str.len()
    return int(lengths.nunique(dropna=True)) <= 2 and float(lengths.mean()) >= 2


def is_boolean_like(series: pd.Series) -> bool:
    """Return whether a column looks boolean."""

    if is_bool_dtype(series):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    normalized_values = frozenset(
        _normalize_boolean_token(value) for value in non_null.unique()
    )
    if len(normalized_values) != 2:
        return False
    return normalized_values in BOOLEAN_TOKEN_PAIRS


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

    return True


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


def _normalize_boolean_token(value: object) -> str:
    token = str(value).strip().lower()
    if token in {"1.0", "0.0"}:
        return token.split(".", maxsplit=1)[0]
    return token


def _is_identifier_name(column_name: str) -> bool:
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


def _is_code_name(column_name: str) -> bool:
    normalized = _normalize_column_name(column_name)
    parts = {part for part in normalized.split("_") if part}
    return bool(
        parts
        & {
            "id",
            "identifier",
            "uuid",
            "guid",
            "key",
            "code",
            "number",
            "num",
        }
    )


def _is_postal_or_code_name(column_name: str) -> bool:
    normalized = _normalize_column_name(column_name)
    parts = {part for part in normalized.split("_") if part}
    return bool(
        parts
        & {
            "zip",
            "zipcode",
            "postal",
            "postcode",
            "code",
            "fips",
        }
    )


def _looks_like_uuid_values(series: pd.Series) -> bool:
    values = series.astype(str).str.strip()
    if values.empty:
        return False
    if len(values) < 3:
        return False
    parsed_ratio = float(values.map(lambda value: bool(UUID_PATTERN.match(value))).mean())
    return parsed_ratio >= 0.95 and _unique_ratio(series) >= 0.95


def _contains_non_integer_numeric_values(series: pd.Series) -> bool:
    if not is_numeric_dtype(series):
        return False

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    return not _is_integer_like_numeric(numeric)


def _is_integer_like_numeric(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    return bool((numeric % 1 == 0).all())


def _looks_like_fixed_format_codes(values: pd.Series) -> bool:
    if values.empty:
        return False
    matched_ratio = float(values.map(lambda value: bool(CODE_PATTERN.match(value))).mean())
    if matched_ratio < 0.95:
        return False
    lengths = values.str.len()
    return int(lengths.nunique(dropna=True)) <= max(3, int(len(values) * 0.1))
