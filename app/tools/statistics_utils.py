"""Small statistical helpers for deterministic EDA."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.tools.file_utils import json_safe


def numeric_non_null(series: pd.Series) -> pd.Series:
    """Return a numeric version of a series with null and non-numeric values removed."""

    return pd.to_numeric(series, errors="coerce").dropna()


def safe_float(value: Any, digits: int = 4) -> float | None:
    """Convert a value to a finite rounded float when possible."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric_value):
        return None
    return round(numeric_value, digits)


def iqr_bounds(series: pd.Series) -> tuple[float | None, float | None]:
    """Return lower and upper IQR outlier bounds for a numeric series."""

    numeric = numeric_non_null(series)
    if numeric.empty:
        return None, None

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    if not math.isfinite(float(iqr)) or iqr == 0:
        return None, None

    return float(q1 - (1.5 * iqr)), float(q3 + (1.5 * iqr))


def iqr_outlier_count(series: pd.Series) -> int:
    """Count possible outliers using the standard 1.5x IQR rule."""

    numeric = numeric_non_null(series)
    lower_bound, upper_bound = iqr_bounds(numeric)
    if lower_bound is None or upper_bound is None:
        return 0

    return int(((numeric < lower_bound) | (numeric > upper_bound)).sum())


def calculate_skewness(series: pd.Series) -> float | None:
    """Calculate skewness for a numeric series when enough variation exists."""

    numeric = numeric_non_null(series)
    if len(numeric) < 3 or numeric.nunique(dropna=True) < 2:
        return None

    return safe_float(numeric.skew())


def safe_correlation(series_a: pd.Series, series_b: pd.Series) -> float | None:
    """Calculate a pairwise Pearson correlation without raising on messy data."""

    values = pd.DataFrame(
        {
            "a": pd.to_numeric(series_a, errors="coerce"),
            "b": pd.to_numeric(series_b, errors="coerce"),
        }
    ).dropna()

    if len(values) < 2:
        return None

    if values["a"].nunique(dropna=True) < 2 or values["b"].nunique(dropna=True) < 2:
        return None

    return safe_float(values["a"].corr(values["b"]))


def format_percentage(part: int | float, total: int | float, digits: int = 1) -> str:
    """Format a ratio as a percentage while handling empty denominators."""

    if not total:
        return "0.0%"
    return f"{(float(part) / float(total)) * 100:.{digits}f}%"


def to_json_safe(value: Any) -> Any:
    """Convert pandas and numpy values into JSON-safe Python values."""

    return json_safe(value)
