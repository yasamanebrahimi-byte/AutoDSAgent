"""Small filesystem helpers used across the project."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save a dictionary as pretty JSON."""

    json_path = Path(path)
    ensure_directory(json_path.parent)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return json_path


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def json_safe(value: Any) -> Any:
    """Convert pandas and numpy values into JSON-safe Python values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if value is None:
        return None

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "isoformat") and not isinstance(value, (str, bytes)):
        return value.isoformat()

    if hasattr(value, "item"):
        return json_safe(value.item())

    if isinstance(value, (str, int, bool)):
        return value

    return str(value)
