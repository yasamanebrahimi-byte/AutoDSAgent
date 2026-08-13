"""Application settings for AutoDS Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Environment-driven settings for local, Docker, and test usage."""

    project_root: Path
    runs_dir: Path
    backend_url: str
    environment: str
    log_level: str
    max_upload_size_mb: int
    default_test_size: float
    default_random_seed: int
    mlflow_enabled: bool
    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    def public_status(self) -> dict[str, str | bool | int | float]:
        """Return non-secret configuration values suitable for API exposure."""

        return {
            "environment": self.environment,
            "project_root": ".",
            "runs_dir": _display_path(self.runs_dir, self.project_root),
            "backend_url": self.backend_url,
            "max_upload_size_mb": self.max_upload_size_mb,
            "default_test_size": self.default_test_size,
            "default_random_seed": self.default_random_seed,
            "log_level": self.log_level,
            "mlflow_enabled": self.mlflow_enabled,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "mlflow_experiment_name": self.mlflow_experiment_name,
        }


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Load settings from environment variables."""

    env = os.environ if environ is None else environ
    project_root = _resolve_project_root(env.get("AUTODS_PROJECT_ROOT"))

    return Settings(
        project_root=project_root,
        runs_dir=_resolve_project_path(
            env.get("AUTODS_RUNS_DIR", "runs"),
            project_root=project_root,
        ),
        backend_url=env.get("AUTODS_BACKEND_URL", "http://localhost:8000").rstrip("/"),
        environment=env.get("AUTODS_ENV", "local"),
        log_level=env.get("AUTODS_LOG_LEVEL", "INFO").upper(),
        max_upload_size_mb=_env_int(env, "AUTODS_MAX_UPLOAD_SIZE_MB", 100),
        default_test_size=_env_float(env, "AUTODS_DEFAULT_TEST_SIZE", 0.2),
        default_random_seed=_env_int(env, "AUTODS_DEFAULT_RANDOM_SEED", 42),
        mlflow_enabled=_env_bool(env, "AUTODS_ENABLE_MLFLOW", False),
        mlflow_tracking_uri=env.get("AUTODS_MLFLOW_TRACKING_URI", "http://localhost:5000"),
        mlflow_experiment_name=env.get(
            "AUTODS_MLFLOW_EXPERIMENT_NAME",
            "AutoDS-Agent",
        ),
    )


def _resolve_project_root(value: str | None) -> Path:
    if not value:
        return PROJECT_ROOT
    return Path(value).expanduser().resolve()


def _resolve_project_path(value: str | Path, project_root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((project_root or PROJECT_ROOT) / path).resolve()


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, str(default)))
    except ValueError:
        return default


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, str(default)))
    except ValueError:
        return default


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.name


settings = load_settings()
