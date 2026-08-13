"""Run folder management for uploaded datasets."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.backend.config import settings
from app.tools.file_utils import ensure_directory, load_json, save_json


RUN_SUBDIRECTORIES = ("input", "intermediate", "models", "plots", "reports", "logs")


@dataclass(frozen=True)
class RunPaths:
    """Filesystem paths for one analysis run."""

    root: Path
    input: Path
    intermediate: Path
    models: Path
    plots: Path
    reports: Path
    logs: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "input": self.input,
            "intermediate": self.intermediate,
            "models": self.models,
            "plots": self.plots,
            "reports": self.reports,
            "logs": self.logs,
        }


class RunManager:
    """Create and inspect analysis run folders."""

    def __init__(self, runs_dir: str | Path | None = None) -> None:
        self.runs_dir = Path(runs_dir or settings.runs_dir).resolve()
        ensure_directory(self.runs_dir)

    def generate_run_id(self) -> str:
        """Generate a unique, readable run ID."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}_{uuid4().hex[:8]}"

    def create_run(self, run_id: str | None = None) -> RunPaths:
        """Create a new run folder with the standard subdirectories."""

        selected_run_id = run_id or self.generate_run_id()
        root = self._resolve_run_path(selected_run_id)

        if root.exists():
            raise FileExistsError(f"Run already exists: {selected_run_id}")

        paths = self.get_paths(selected_run_id)
        for directory in paths.as_dict().values():
            ensure_directory(directory)

        return paths

    def get_paths(self, run_id: str) -> RunPaths:
        """Return standard paths for a run without requiring files to exist."""

        root = self._resolve_run_path(run_id)
        return RunPaths(
            root=root,
            input=root / "input",
            intermediate=root / "intermediate",
            models=root / "models",
            plots=root / "plots",
            reports=root / "reports",
            logs=root / "logs",
        )

    def metadata_path(self, run_id: str) -> Path:
        """Return the metadata JSON path for a run."""

        return self.get_paths(run_id).intermediate / "metadata.json"

    def save_metadata(self, run_id: str, metadata: dict[str, Any]) -> Path:
        """Persist metadata for a run."""

        path = self.metadata_path(run_id)
        save_json(path, metadata)
        return path

    def load_metadata(self, run_id: str) -> dict[str, Any]:
        """Load metadata for a run, raising FileNotFoundError if absent."""

        path = self.metadata_path(run_id)
        if not path.exists():
            raise FileNotFoundError(path)
        return load_json(path)

    def list_runs(self) -> list[dict[str, Any]]:
        """List run folders, enriching summaries with saved metadata when present."""

        if not self.runs_dir.exists():
            return []

        summaries: list[dict[str, Any]] = []
        run_paths = sorted(
            (path for path in self.runs_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        for run_path in run_paths:
            metadata_path = run_path / "intermediate" / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = load_json(metadata_path)

            summaries.append(
                {
                    "run_id": run_path.name,
                    "filename": metadata.get("filename"),
                    "rows": metadata.get("rows"),
                    "columns": metadata.get("columns"),
                    "created_at": metadata.get("created_at"),
                }
            )

        return summaries

    def delete_run(self, run_id: str) -> None:
        """Remove a run directory after a failed transactional operation."""

        root = self._resolve_run_path(run_id)
        if root.exists():
            shutil.rmtree(root)

    def _resolve_run_path(self, run_id: str) -> Path:
        """Resolve a run path and keep it inside the configured runs directory."""

        if not run_id or not run_id.strip():
            raise ValueError("run_id is required.")

        candidate = (self.runs_dir / run_id).resolve()
        if self.runs_dir not in candidate.parents:
            raise ValueError("run_id must resolve inside the runs directory.")

        return candidate
