"""Dataset profiling service."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.backend.schemas.profile import (
    ColumnProfile,
    DatasetProfile,
    NumericColumnStats,
    ValueCount,
)
from app.backend.services.run_manager import RunManager
from app.tools.data_loader import load_csv
from app.tools.data_quality import detect_data_quality_issues
from app.tools.file_utils import json_safe, load_json, save_json
from app.tools.schema_inference import infer_schema


class ProfilingService:
    """Generate and load deterministic dataset profiles."""

    def __init__(self, run_manager: RunManager | None = None) -> None:
        self.run_manager = run_manager or RunManager()

    def raw_data_path(self, run_id: str) -> Path:
        """Return the preserved raw dataset path for a run."""

        return self.run_manager.get_paths(run_id).input / "raw_data.csv"

    def profile_path(self, run_id: str) -> Path:
        """Return the profile artifact path for a run."""

        return self.run_manager.get_paths(run_id).intermediate / "profile.json"

    def generate_profile(self, run_id: str) -> DatasetProfile:
        """Generate, save, and return a rich dataset profile."""

        raw_path = self.raw_data_path(run_id)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        dataframe = load_csv(raw_path)
        profile = self.profile_dataframe(dataframe, run_id=run_id)
        save_json(self.profile_path(run_id), profile.model_dump(mode="json"))
        return profile

    def load_profile(self, run_id: str) -> DatasetProfile:
        """Load a saved profile for a run."""

        path = self.profile_path(run_id)
        if not path.exists():
            raise FileNotFoundError(path)
        return DatasetProfile(**load_json(path))

    def profile_dataframe(self, dataframe: pd.DataFrame, run_id: str) -> DatasetProfile:
        """Create a profile object from an in-memory DataFrame."""

        rows, columns = dataframe.shape
        semantic_schema = infer_schema(dataframe)
        column_profiles = [
            _profile_column(dataframe[column], str(column), semantic_schema[str(column)], rows)
            for column in dataframe.columns
        ]
        column_type_counts = _count_semantic_types(column_profiles)
        quality_issues = detect_data_quality_issues(
            dataframe=dataframe,
            column_profiles=[
                column_profile.model_dump(mode="json")
                for column_profile in column_profiles
            ],
        )

        return DatasetProfile(
            run_id=run_id,
            rows=int(rows),
            columns=int(columns),
            total_missing_values=int(dataframe.isna().sum().sum()),
            duplicate_rows=int(dataframe.duplicated().sum()),
            memory_usage_bytes=int(dataframe.memory_usage(deep=True).sum()),
            column_type_counts=column_type_counts,
            is_empty=bool(dataframe.empty),
            has_duplicate_rows=bool(dataframe.duplicated().any()),
            has_missing_values=bool(dataframe.isna().any().any()),
            column_profiles=column_profiles,
            data_quality_issues=quality_issues,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def _profile_column(
    series: pd.Series,
    column_name: str,
    semantic_type: str,
    total_rows: int,
) -> ColumnProfile:
    missing_values = int(series.isna().sum())
    non_null_count = int(series.notna().sum())
    unique_values = int(series.nunique(dropna=True))
    unique_ratio = unique_values / non_null_count if non_null_count else 0.0
    unique_percentage = (unique_values / total_rows * 100) if total_rows else 0.0
    missing_percentage = (missing_values / total_rows * 100) if total_rows else 0.0
    is_high_cardinality = _is_high_cardinality(unique_values, unique_ratio, semantic_type)

    return ColumnProfile(
        column_name=column_name,
        pandas_dtype=str(series.dtype),
        semantic_type=semantic_type,
        missing_values=missing_values,
        missing_percentage=round(missing_percentage, 4),
        unique_values=unique_values,
        unique_percentage=round(unique_percentage, 4),
        sample_values=_sample_values(series),
        is_constant=bool(unique_values <= 1),
        is_high_cardinality=is_high_cardinality,
        is_id=semantic_type == "id",
        is_datetime=semantic_type == "datetime",
        is_numeric=semantic_type == "numeric",
        is_categorical=semantic_type == "categorical",
        is_boolean=semantic_type == "boolean",
        is_text_like=semantic_type == "text",
        numeric_stats=_numeric_stats(series) if semantic_type == "numeric" else None,
        top_values=_top_values(series) if semantic_type in {"categorical", "text"} else [],
        average_string_length=_average_string_length(series)
        if semantic_type == "text"
        else None,
    )


def _numeric_stats(series: pd.Series) -> NumericColumnStats:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return NumericColumnStats()

    percentile_25 = numeric.quantile(0.25)
    percentile_75 = numeric.quantile(0.75)
    iqr = percentile_75 - percentile_25
    if iqr == 0:
        outlier_count = 0
    else:
        lower_bound = percentile_25 - (1.5 * iqr)
        upper_bound = percentile_75 + (1.5 * iqr)
        outlier_count = int(((numeric < lower_bound) | (numeric > upper_bound)).sum())

    return NumericColumnStats(
        mean=_safe_float(numeric.mean()),
        median=_safe_float(numeric.median()),
        standard_deviation=_safe_float(numeric.std()),
        minimum=_safe_float(numeric.min()),
        maximum=_safe_float(numeric.max()),
        percentile_25=_safe_float(percentile_25),
        percentile_75=_safe_float(percentile_75),
        possible_outlier_count=outlier_count,
    )


def _top_values(series: pd.Series, limit: int = 10) -> list[ValueCount]:
    counts = series.value_counts(dropna=False).head(limit)
    return [
        ValueCount(value=json_safe(value), count=int(count))
        for value, count in counts.items()
    ]


def _sample_values(series: pd.Series, limit: int = 5) -> list[Any]:
    samples = series.dropna().drop_duplicates().head(limit).tolist()
    return json_safe(samples)


def _average_string_length(series: pd.Series) -> float | None:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None
    return round(float(non_null.str.len().mean()), 4)


def _is_high_cardinality(
    unique_values: int,
    unique_ratio: float,
    semantic_type: str,
) -> bool:
    if semantic_type in {"numeric", "datetime", "boolean", "unknown"}:
        return False
    return unique_values >= 50 or (unique_values >= 20 and unique_ratio >= 0.8)


def _count_semantic_types(column_profiles: list[ColumnProfile]) -> dict[str, int]:
    counts = {
        "numeric": 0,
        "categorical": 0,
        "boolean": 0,
        "datetime": 0,
        "text": 0,
        "id": 0,
        "unknown": 0,
    }
    for profile in column_profiles:
        counts[profile.semantic_type] = counts.get(profile.semantic_type, 0) + 1
    return counts


def _safe_float(value: Any) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric_value):
        return None
    return numeric_value
