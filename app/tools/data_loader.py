"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_csv(path: str | Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV file from disk."""

    return pd.read_csv(Path(path), **read_csv_kwargs)
