"""Small filesystem helpers used across the project."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: Any) -> Path:
    """Save a JSON payload using an atomic same-directory replace."""

    json_path = Path(path)
    ensure_directory(json_path.parent)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=json_path.parent,
            prefix=f".{json_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, json_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return json_path


def write_text_atomic(path: str | Path, content: str, encoding: str = "utf-8") -> Path:
    """Write text using an atomic same-directory replace."""

    text_path = Path(path)
    ensure_directory(text_path.parent)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding=encoding,
            dir=text_path.parent,
            prefix=f".{text_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, text_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return text_path


def load_json(path: str | Path) -> Any:
    """Load a JSON payload from disk."""

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
