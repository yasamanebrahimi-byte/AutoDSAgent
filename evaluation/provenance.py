"""Deterministic software and experiment provenance for audit artifacts."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

from evaluation.confirmatory import experiment_code_sha256, manifest_sha256, repository_commit


_MATERIAL_PACKAGES = (
    "autods-agent",
    "numpy",
    "pandas",
    "scikit-learn",
    "openai",
    "openml",
    "pydantic",
    "joblib",
    "matplotlib",
)


def _package_versions() -> dict[str, str | None]:
    return {
        package: importlib.metadata.version(package)
        if _distribution_exists(package)
        else None
        for package in _MATERIAL_PACKAGES
    }


def _distribution_exists(package: str) -> bool:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def environment_provenance(
    *,
    repository_root: str | Path | None = None,
    manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Return JSON-safe, secret-free provenance with stable field semantics."""

    payload: dict[str, Any] = {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "provenance_schema_version": "environment-provenance-v1",
    }
    if repository_root is not None or manifest is not None:
        payload["experiment_code_sha256"] = experiment_code_sha256(repository_root)
        payload["source_git_commit"] = repository_commit()
    if manifest is not None:
        payload["confirmatory_manifest_sha256"] = manifest_sha256(manifest)
    return payload
