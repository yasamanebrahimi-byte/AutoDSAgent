"""Dataset validation, loading, and metadata generation."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.backend.schemas.dataset import DatasetMetadata
from app.tools.data_loader import load_csv as load_csv_from_path
from app.tools.schema_inference import infer_column_dtypes


def validate_csv_filename(filename: str | None) -> None:
    """Validate that an uploaded filename looks like a CSV file."""

    if not filename:
        raise ValueError("A CSV file is required.")

    if Path(filename).suffix.lower() != ".csv":
        raise ValueError("Only .csv files are supported.")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV from disk into a DataFrame."""

    return load_csv_from_path(path)


def generate_dataset_metadata(
    dataframe: pd.DataFrame,
    filename: str,
    run_id: str,
    preview_rows: int = 5,
) -> DatasetMetadata:
    """Create JSON-safe metadata for a DataFrame."""

    return DatasetMetadata(
        run_id=run_id,
        filename=filename,
        rows=int(dataframe.shape[0]),
        columns=int(dataframe.shape[1]),
        column_names=[str(column) for column in dataframe.columns],
        dtypes=infer_column_dtypes(dataframe),
        missing_values={
            str(column): int(count)
            for column, count in dataframe.isna().sum().items()
        },
        duplicate_rows=int(dataframe.duplicated().sum()),
        preview=_json_safe(dataframe.head(preview_rows).to_dict(orient="records")),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _json_safe(value: Any) -> Any:
    """Convert pandas and numpy scalar values into JSON-safe Python objects."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if value is None:
        return None

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        return _json_safe(value.item())

    if isinstance(value, (str, int, bool)):
        return value

    return str(value)
