"""Model artifact persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from app.tools.file_utils import ensure_directory, save_json
from app.tools.modeling import ModelTrainingResult


def save_model_artifacts(
    models_dir: str | Path,
    results: list[ModelTrainingResult],
    selected_result: ModelTrainingResult,
    model_results_payload: dict[str, Any],
) -> dict[str, Path]:
    """Save baseline, selected model, and model-results artifacts."""

    output_dir = ensure_directory(models_dir)
    baseline_result = next(
        (
            result
            for result in results
            if result.role == "baseline"
            and result.status == "succeeded"
            and result.estimator is not None
        ),
        None,
    )

    if selected_result.estimator is None:
        raise RuntimeError("The selected model artifact is unavailable.")

    baseline_path = output_dir / "baseline_model.pkl"
    selected_path = output_dir / "selected_model.pkl"
    best_path = output_dir / "best_model.pkl"
    model_results_path = output_dir / "model_results.json"

    if baseline_result is not None:
        joblib.dump(baseline_result.estimator, baseline_path)
    joblib.dump(selected_result.estimator, selected_path)
    joblib.dump(selected_result.estimator, best_path)
    save_json(model_results_path, model_results_payload)

    paths = {
        "selected_model_path": selected_path,
        "best_model_path": best_path,
        "model_results_path": model_results_path,
    }
    if baseline_result is not None:
        paths["baseline_model_path"] = baseline_path
    return paths


def list_model_artifacts(models_dir: str | Path, run_root: str | Path) -> list[dict[str, Any]]:
    """List saved model artifacts for an existing run."""

    directory = Path(models_dir)
    root = Path(run_root)
    if not directory.exists():
        return []

    artifacts: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".pkl", ".json"}:
            continue

        artifacts.append(
            {
                "name": path.name,
                "path": path.relative_to(root).as_posix(),
                "artifact_type": "model" if path.suffix.lower() == ".pkl" else "results",
                "size_bytes": int(path.stat().st_size),
                "modified_at": path.stat().st_mtime,
            }
        )

    return artifacts
