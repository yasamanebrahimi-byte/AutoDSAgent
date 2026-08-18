"""Modeling service for training and evaluation."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.backend.schemas.modeling import (
    ModelingRequest,
    ModelingResponse,
    ModelingSummary,
    SavedModelInfo,
)
from app.backend.services.evaluation_service import EvaluationService
from app.backend.services.run_manager import RunManager
from app.tools.artifact_lineage import (
    fingerprint_payload,
    invalidate_downstream_artifacts,
    lineage_context,
    select_analysis_input,
    validate_artifact_for_state,
    write_artifact_lineage,
)
from app.tools.app_logging import get_logger, log_event
from app.tools.data_loader import load_csv
from app.tools.evaluation import PRIMARY_METRIC_BY_TASK
from app.tools.file_utils import load_json, save_json
from app.tools.mlflow_logger import MLflowLogger
from app.tools.model_persistence import list_model_artifacts, save_model_artifacts
from app.tools.modeling import ModelTrainingResult, train_models
from app.tools.preprocessing import prepare_modeling_data
from app.workflows.workflow_steps import MODELING_STEP


class ModelingService:
    """Train baseline and candidate models for cleaned tabular datasets."""

    def __init__(
        self,
        run_manager: RunManager | None = None,
        evaluation_service: EvaluationService | None = None,
        mlflow_logger: MLflowLogger | None = None,
    ) -> None:
        self.run_manager = run_manager or RunManager()
        self.evaluation_service = evaluation_service or EvaluationService(self.run_manager)
        self.mlflow_logger = mlflow_logger or MLflowLogger()
        self.logger = get_logger(__name__)

    def modeling_summary_path(self, run_id: str) -> Path:
        """Return the modeling summary artifact path."""

        return self.run_manager.get_paths(run_id).intermediate / "modeling_summary.json"

    def model_results_path(self, run_id: str) -> Path:
        """Return the model results artifact path."""

        return self.run_manager.get_paths(run_id).models / "model_results.json"

    def baseline_model_path(self, run_id: str) -> Path:
        """Return the baseline model artifact path."""

        return self.run_manager.get_paths(run_id).models / "baseline_model.pkl"

    def best_model_path(self, run_id: str) -> Path:
        """Return the legacy selected-model alias artifact path."""

        return self.run_manager.get_paths(run_id).models / "best_model.pkl"

    def selected_model_path(self, run_id: str) -> Path:
        """Return the selected model artifact path."""

        return self.run_manager.get_paths(run_id).models / "selected_model.pkl"

    def train_and_evaluate(
        self,
        run_id: str,
        request: ModelingRequest,
    ) -> ModelingResponse:
        """Train, evaluate, save artifacts, and return modeling outputs."""

        paths = self.run_manager.get_paths(run_id)
        if not paths.root.exists():
            raise FileNotFoundError(paths.root)

        invalidate_downstream_artifacts(self.run_manager, run_id, MODELING_STEP)
        dataset_selection = select_analysis_input(
            self.run_manager,
            run_id,
            target_column=request.target_column,
            require_cleaned=True,
        )
        dataset_path = dataset_selection.path

        log_event(
            self.logger,
            logging.INFO,
            "Modeling run started.",
            run_id=run_id,
            target_column=request.target_column,
            task_type=request.task_type or "auto",
        )

        dataframe = load_csv(dataset_path)
        prepared = prepare_modeling_data(
            dataframe=dataframe,
            target_column=request.target_column,
            task_type=request.task_type,
            test_size=request.test_size,
            random_state=request.random_state,
        )
        training_results = train_models(prepared, random_state=request.random_state)

        evaluation_result = self.evaluation_service.evaluate_and_save(
            run_id=run_id,
            prepared=prepared,
            training_results=training_results,
            warnings=prepared.warnings,
        )

        artifact_paths = self._artifact_paths_payload(paths.root)
        model_results_payload = dict(evaluation_result.model_results_payload)
        model_results_payload["artifact_paths"] = artifact_paths
        saved_model_artifacts = save_model_artifacts(
            models_dir=paths.models,
            results=training_results,
            selected_result=evaluation_result.selected_result,
            model_results_payload=model_results_payload,
        )

        modeling_summary = self._build_modeling_summary(
            run_id=run_id,
            dataset_path=dataset_path.relative_to(paths.root).as_posix(),
            prepared=prepared,
            training_results=training_results,
            selected_result=evaluation_result.selected_result,
            best_candidate_result=evaluation_result.best_candidate_result,
            baseline_result=evaluation_result.baseline_result,
            candidate_beats_baseline=(
                evaluation_result.summary.candidate_beats_baseline
            ),
            selection_outcome=evaluation_result.summary.selection_outcome,
        )
        modeling_summary_path = save_json(
            self.modeling_summary_path(run_id),
            modeling_summary.model_dump(mode="json"),
        )
        context = lineage_context(self.run_manager, run_id)
        config_payload = {
            "artifact_family": "modeling",
            "source_fingerprint": dataset_selection.source_fingerprint,
            "analysis_input": dataset_selection.dataset_used,
            "analysis_input_fingerprint": dataset_selection.fingerprint,
            "target_column": request.target_column,
            "requested_task_type": request.task_type,
            "effective_task_type": modeling_summary.task_type,
            "test_size": request.test_size,
            "random_state": request.random_state,
        }
        config_fingerprint = fingerprint_payload(config_payload)
        upstream_key = dataset_selection.dataset_used
        upstream_fingerprints = {
            "source_data": dataset_selection.source_fingerprint,
            upstream_key: dataset_selection.fingerprint,
        }
        artifacts_for_lineage = [
            (modeling_summary_path, "modeling_summary"),
            (
                self.evaluation_service.evaluation_summary_path(run_id),
                "evaluation_summary",
            ),
            (saved_model_artifacts["model_results_path"], "model_results"),
            (saved_model_artifacts["selected_model_path"], "selected_model"),
            (saved_model_artifacts["best_model_path"], "best_model"),
        ]
        if "baseline_model_path" in saved_model_artifacts:
            artifacts_for_lineage.append(
                (saved_model_artifacts["baseline_model_path"], "baseline_model")
            )

        for artifact_path, artifact_type in artifacts_for_lineage:
            write_artifact_lineage(
                artifact_path,
                run_root=paths.root,
                run_id=run_id,
                artifact_type=artifact_type,
                generation_id=context["generation_id"],
                source_fingerprint=dataset_selection.source_fingerprint,
                target_column=modeling_summary.target_column,
                task_type=modeling_summary.task_type,
                config_fingerprint=config_fingerprint,
                upstream_fingerprints=upstream_fingerprints,
                relevant_config=config_payload,
            )

        try:
            mlflow_warnings = self.mlflow_logger.log_modeling_run(
                run_id=run_id,
                request=request,
                modeling_summary=modeling_summary.model_dump(mode="json"),
                evaluation_summary=evaluation_result.summary.model_dump(mode="json"),
                model_results=model_results_payload,
                run_root=paths.root,
            )
        except Exception as exc:
            mlflow_warnings = [f"MLflow logging failed and was skipped: {exc}"]
            log_event(
                self.logger,
                logging.WARNING,
                mlflow_warnings[0],
                run_id=run_id,
            )
        if mlflow_warnings:
            evaluation_result.summary.warnings.extend(mlflow_warnings)
            save_json(
                self.evaluation_service.evaluation_summary_path(run_id),
                evaluation_result.summary.model_dump(mode="json"),
            )

        log_event(
            self.logger,
            logging.INFO,
            "Modeling run completed.",
            run_id=run_id,
            target_column=modeling_summary.target_column,
            task_type=modeling_summary.task_type,
            selected_model=modeling_summary.selected_model_name,
        )

        return ModelingResponse(
            modeling_summary=modeling_summary,
            evaluation_summary=evaluation_result.summary,
            model_results=model_results_payload,
        )

    def load_modeling_summary(self, run_id: str) -> ModelingSummary:
        """Load an existing modeling summary."""

        path = self.modeling_summary_path(run_id)
        if not path.exists():
            raise FileNotFoundError(path)
        state = lineage_context(self.run_manager, run_id)["state"]
        validation = validate_artifact_for_state(
            path,
            artifact_type="modeling_summary",
            state=state,
        )
        if not validation.is_current:
            raise ValueError(f"Modeling summary artifact is stale: {validation.reason}.")
        return ModelingSummary(**load_json(path))

    def load_model_results(self, run_id: str) -> dict[str, Any]:
        """Load saved model comparison details."""

        path = self.model_results_path(run_id)
        if not path.exists():
            raise FileNotFoundError(path)
        state = lineage_context(self.run_manager, run_id)["state"]
        validation = validate_artifact_for_state(
            path,
            artifact_type="model_results",
            state=state,
        )
        if not validation.is_current:
            raise ValueError(f"Model results artifact is stale: {validation.reason}.")
        return load_json(path)

    def list_saved_models(self, run_id: str) -> list[SavedModelInfo]:
        """Return saved model artifact metadata for one run."""

        paths = self.run_manager.get_paths(run_id)
        if not paths.root.exists():
            raise FileNotFoundError(paths.root)
        state = lineage_context(self.run_manager, run_id)["state"]
        expected_artifacts = {
            "baseline_model.pkl": "baseline_model",
            "selected_model.pkl": "selected_model",
            "best_model.pkl": "best_model",
            "model_results.json": "model_results",
        }
        return [
            SavedModelInfo(**artifact)
            for artifact in list_model_artifacts(paths.models, paths.root)
            if (
                expected_artifacts.get(artifact["name"]) is None
                or validate_artifact_for_state(
                    paths.root / artifact["path"],
                    artifact_type=expected_artifacts[artifact["name"]],
                    state=state,
                ).is_current
            )
        ]

    def _build_modeling_summary(
        self,
        run_id: str,
        dataset_path: str,
        prepared,
        training_results: list[ModelTrainingResult],
        selected_result: ModelTrainingResult,
        best_candidate_result: ModelTrainingResult | None,
        baseline_result: ModelTrainingResult | None,
        candidate_beats_baseline: bool | None,
        selection_outcome: str | None,
    ) -> ModelingSummary:
        models_attempted = [result.model_name for result in training_results]
        models_succeeded = [
            result.model_name for result in training_results if result.status == "succeeded"
        ]
        models_failed = [
            result.model_name for result in training_results if result.status == "failed"
        ]

        return ModelingSummary(
            run_id=run_id,
            dataset_path=dataset_path,
            target_column=prepared.target_column,
            task_type=prepared.task_type,
            rows_used=prepared.rows_used,
            columns_used=prepared.columns_used,
            features_used=prepared.features_used,
            features_excluded=prepared.features_excluded,
            excluded_feature_reasons=prepared.excluded_feature_reasons,
            train_rows=prepared.train_rows,
            test_rows=prepared.test_rows,
            actual_test_size=prepared.actual_test_size,
            cv_folds=prepared.cv_folds,
            cv_strategy=prepared.cv_strategy,
            task_inference_reason=prepared.task_inference_reason,
            classification_validation=prepared.classification_validation,
            models_attempted=models_attempted,
            models_succeeded=models_succeeded,
            models_failed=models_failed,
            best_candidate_name=(
                best_candidate_result.model_name if best_candidate_result else None
            ),
            best_candidate_metrics=(
                _finite_metric_mapping(best_candidate_result.cv_metrics)
                if best_candidate_result
                else {}
            ),
            selected_model_name=selected_result.model_name,
            selected_model_role=selected_result.role,
            baseline_model_name=baseline_result.model_name if baseline_result else None,
            candidate_beats_baseline=candidate_beats_baseline,
            selection_outcome=selection_outcome,
            best_model_name=selected_result.model_name,
            primary_metric=PRIMARY_METRIC_BY_TASK[prepared.task_type],
            warnings=prepared.warnings,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _artifact_paths_payload(self, run_root: Path) -> dict[str, str]:
        models_dir = run_root / "models"
        return {
            "baseline_model_path": (models_dir / "baseline_model.pkl")
            .relative_to(run_root)
            .as_posix(),
            "selected_model_path": (models_dir / "selected_model.pkl")
            .relative_to(run_root)
            .as_posix(),
            "best_model_path": (models_dir / "best_model.pkl")
            .relative_to(run_root)
            .as_posix(),
            "model_results_path": (models_dir / "model_results.json")
            .relative_to(run_root)
            .as_posix(),
        }


def _finite_metric_mapping(metrics: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            cleaned[key] = None
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            cleaned[key] = value
            continue
        cleaned[key] = numeric_value if math.isfinite(numeric_value) else None
    return cleaned
