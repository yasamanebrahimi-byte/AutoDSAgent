"""Optional MLflow logging for model evaluation runs."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Mapping

from app.backend.config import Settings, settings
from app.tools.app_logging import get_logger, log_event


MAX_MODEL_ARTIFACT_BYTES = 50 * 1024 * 1024


class MLflowLogger:
    """Log AutoDS modeling artifacts to MLflow when enabled."""

    def __init__(
        self,
        active_settings: Settings | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = active_settings or settings
        self.logger = logger or get_logger(__name__)

    @property
    def enabled(self) -> bool:
        """Return whether MLflow logging is enabled."""

        return bool(self.settings.mlflow_enabled)

    def log_modeling_run(
        self,
        run_id: str,
        request: Any,
        modeling_summary: Mapping[str, Any],
        evaluation_summary: Mapping[str, Any],
        model_results: Mapping[str, Any],
        run_root: str | Path,
    ) -> list[str]:
        """Log one AutoDS modeling run to MLflow and return non-fatal warnings."""

        if not self.enabled:
            return []

        run_root_path = Path(run_root)
        try:
            mlflow = importlib.import_module("mlflow")
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.settings.mlflow_experiment_name)

            tags = format_run_tags(run_id, modeling_summary)
            parent_params = format_run_parameters(request, modeling_summary)
            final_test_metrics = (
                evaluation_summary.get("final_test_metrics")
                or evaluation_summary.get("holdout_metrics")
                or evaluation_summary.get("selected_model_holdout_metrics")
                or evaluation_summary.get("best_model_metrics", {})
            )
            parent_metrics = prefix_metrics(
                final_test_metrics,
                prefix="selected",
            )

            with mlflow.start_run(run_name=run_id, tags=tags):
                if parent_params:
                    mlflow.log_params(parent_params)
                if parent_metrics:
                    mlflow.log_metrics(parent_metrics)

                for artifact_path in artifact_paths_for_mlflow(run_root_path):
                    _log_artifact_if_reasonable(mlflow, artifact_path)

                for model_result in model_results.get("results", []):
                    model_name = str(model_result.get("model_name") or "model")
                    with mlflow.start_run(
                        run_name=model_name,
                        nested=True,
                        tags={**tags, "model_name": model_name},
                    ):
                        params = {
                            "model_name": model_name,
                            "role": model_result.get("role"),
                            "status": model_result.get("status"),
                            "primary_metric": model_results.get("primary_metric"),
                        }
                        mlflow.log_params(_drop_none(params))
                        metrics = format_metric_dict(
                            model_result.get("cv_metrics")
                            or model_result.get("metrics", {})
                        )
                        if metrics:
                            mlflow.log_metrics(metrics)
                        if model_result.get("error"):
                            mlflow.set_tag("model_error", str(model_result["error"])[:500])

            log_event(
                self.logger,
                logging.INFO,
                "MLflow logging completed.",
                run_id=run_id,
                tracking_uri=self.settings.mlflow_tracking_uri,
                experiment=self.settings.mlflow_experiment_name,
            )
            return []
        except Exception as exc:
            warning = f"MLflow logging failed and was skipped: {exc}"
            log_event(
                self.logger,
                logging.WARNING,
                warning,
                run_id=run_id,
                tracking_uri=self.settings.mlflow_tracking_uri,
            )
            return [warning]


def format_run_tags(run_id: str, modeling_summary: Mapping[str, Any]) -> dict[str, str]:
    """Return MLflow tags for an AutoDS modeling run."""

    return {
        "run_id": run_id,
        "target_column": str(modeling_summary.get("target_column") or ""),
        "task_type": str(modeling_summary.get("task_type") or ""),
        "dataset_path": str(modeling_summary.get("dataset_path") or ""),
        "autods_stage": "modeling",
        "project": "autods-agent",
    }


def format_run_parameters(request: Any, modeling_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return MLflow-safe parent-run parameters."""

    params = {
        "target_column": modeling_summary.get("target_column"),
        "task_type": modeling_summary.get("task_type"),
        "test_size": getattr(request, "test_size", None),
        "random_state": getattr(request, "random_state", None),
        "rows_used": modeling_summary.get("rows_used"),
        "columns_used": modeling_summary.get("columns_used"),
        "num_features_used": len(modeling_summary.get("features_used") or []),
        "num_features_excluded": len(modeling_summary.get("features_excluded") or []),
        "best_candidate_name": modeling_summary.get("best_candidate_name"),
        "selected_model_name": (
            modeling_summary.get("selected_model_name")
            or modeling_summary.get("best_model_name")
        ),
        "best_model_name": (
            modeling_summary.get("selected_model_name")
            or modeling_summary.get("best_model_name")
        ),
        "baseline_model_name": modeling_summary.get("baseline_model_name"),
        "candidate_beats_baseline": modeling_summary.get("candidate_beats_baseline"),
        "primary_metric": modeling_summary.get("primary_metric"),
    }
    return _drop_none(params)


def format_metric_dict(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Return finite numeric metrics suitable for MLflow."""

    formatted: dict[str, float] = {}
    for metric_name, value in metrics.items():
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric != numeric:
            continue
        formatted[str(metric_name).lower()] = numeric
    return formatted


def prefix_metrics(metrics: Mapping[str, Any], prefix: str) -> dict[str, float]:
    """Prefix metrics for parent-run summary logging."""

    return {
        f"{prefix}_{metric_name}": value
        for metric_name, value in format_metric_dict(metrics).items()
    }


def artifact_paths_for_mlflow(run_root: Path) -> list[Path]:
    """Return relevant local artifacts to log to MLflow when present."""

    candidates = [
        run_root / "intermediate" / "modeling_summary.json",
        run_root / "intermediate" / "evaluation_summary.json",
        run_root / "models" / "model_results.json",
        run_root / "models" / "selected_model.pkl",
        run_root / "models" / "best_model.pkl",
        run_root / "reports" / "final_report.md",
    ]
    evaluation_dir = run_root / "plots" / "evaluation"
    if evaluation_dir.exists():
        candidates.extend(sorted(evaluation_dir.glob("*.png")))
    return [path for path in candidates if path.exists() and path.is_file()]


def _log_artifact_if_reasonable(mlflow: Any, path: Path) -> None:
    if path.suffix == ".pkl" and path.stat().st_size > MAX_MODEL_ARTIFACT_BYTES:
        return
    mlflow.log_artifact(str(path))


def _drop_none(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
