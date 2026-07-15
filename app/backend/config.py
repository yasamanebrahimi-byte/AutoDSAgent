"""Application settings for the AutoDS Agent backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Small settings container that can later move to pydantic-settings."""

    project_root: Path = PROJECT_ROOT
    runs_dir: Path = _resolve_project_path(os.getenv("AUTODS_RUNS_DIR", "runs"))
    backend_url: str = os.getenv("AUTODS_BACKEND_URL", "http://localhost:8000")
    max_upload_size_mb: int = int(os.getenv("AUTODS_MAX_UPLOAD_SIZE_MB", "100"))


settings = Settings()
