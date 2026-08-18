"""Model evaluation service."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.backend.schemas.modeling import EvaluationSummary
from app.backend.services.run_manager import RunManager
from app.tools.artifact_lineage import lineage_context, validate_artifact_for_state
from app.tools.evaluation import (
    PRIMARY_METRIC_BY_TASK,
    SELECTION_DIRECTION_BY_TASK,
    SELECTION_TIEBREAKER,
    baseline_comparison,
    create_evaluation_plots,
    evaluate_holdout_model,
    refit_model_on_training_partition,
    select_model_results,
)
from app.tools.file_utils import load_json, save_json
from app.tools.modeling import ModelTrainingResult, serialize_training_results
from app.tools.preprocessing import PreprocessingResult


@dataclass
class EvaluationServiceResult:
    """Evaluation outputs needed by the modeling service."""

    summary: EvaluationSummary
    model_results_payload: dict[str, Any]
    selected_result: ModelTrainingResult
    best_candidate_result: ModelTrainingResult | None
    baseline_result: ModelTrainingResult | None

    @property
    def best_result(self) -> ModelTrainingResult:
        """Backward-compatible alias for the selected overall model."""

        return self.selected_result


class EvaluationService:
    """Evaluate trained models and save evaluation artifacts."""

    def __init__(self, run_manager: RunManager | None = None) -> None:
        self.run_manager = run_manager or RunManager()

    def evaluation_summary_path(self, run_id: str) -> Path:
        """Return the evaluation summary artifact path."""

        return self.run_manager.get_paths(run_id).intermediate / "evaluation_summary.json"

    def evaluate_and_save(
        self,
        run_id: str,
        prepared: PreprocessingResult,
        training_results: list[ModelTrainingResult],
        warnings: list[str] | None = None,
    ) -> EvaluationServiceResult:
        """Select from CV results, evaluate the selected model, and save artifacts."""

        paths = self.run_manager.get_paths(run_id)
        if not paths.root.exists():
            raise FileNotFoundError(paths.root)

        selection = select_model_results(training_results, prepared.task_type)
        selected_result = refit_model_on_training_partition(
            selection.selected_result,
            prepared,
        )
        selected_result = evaluate_holdout_model(selected_result, prepared)
        baseline_result = selection.baseline_result
        best_candidate_result = selection.best_candidate_result
        primary_metric = PRIMARY_METRIC_BY_TASK[prepared.task_type]

        generated_plots = create_evaluation_plots(
            results=training_results,
            prepared=prepared,
            selected_result=selected_result,
            run_root=paths.root,
            plots_dir=paths.plots,
        )

        all_model_metrics = {
            result.model_name: _finite_metric_mapping(result.cv_metrics)
            for result in training_results
            if result.status == "succeeded"
        }
        candidate_cv_results = {
            result.model_name: _finite_metric_mapping(result.cv_metrics)
            for result in training_results
            if result.status == "succeeded" and result.role == "candidate"
        }
        model_failures = [
            f"{result.model_name}: {result.error}"
            for result in training_results
            if result.status == "failed" and result.error
        ]
        evaluation_warnings = list(warnings or [])
        if model_failures:
            evaluation_warnings.append(
                f"{len(model_failures)} model(s) failed and were excluded from model selection."
            )
        invalid_metric_models = [
            result.model_name
            for result in training_results
            if result.status == "succeeded" and not _has_valid_primary_metric(result)
        ]
        if invalid_metric_models:
            evaluation_warnings.append(
                "Model(s) with missing or non-finite selection metrics were excluded "
                f"from model selection: {', '.join(invalid_metric_models)}."
            )

        created_at = datetime.now(timezone.utc).isoformat()
        selection_direction = SELECTION_DIRECTION_BY_TASK[prepared.task_type]
        baseline_metrics = (
            _finite_metric_mapping(baseline_result.cv_metrics)
            if baseline_result is not None and baseline_result.status == "succeeded"
            else {}
        )
        best_candidate_metrics = (
            _finite_metric_mapping(best_candidate_result.cv_metrics)
            if best_candidate_result is not None
            else {}
        )
        selected_model_cv_metrics = _finite_metric_mapping(selected_result.cv_metrics)
        selected_model_holdout_metrics = selected_result.holdout_metrics
        summary_payload = {
            "run_id": run_id,
            "target_column": prepared.target_column,
            "task_type": prepared.task_type,
            "primary_metric": primary_metric,
            "baseline_model_name": baseline_result.model_name if baseline_result else None,
            "baseline_metrics": baseline_metrics,
            "best_candidate_name": (
                best_candidate_result.model_name if best_candidate_result else None
            ),
            "best_candidate_metrics": best_candidate_metrics,
            "selected_model_name": selected_result.model_name,
            "selected_model_role": selected_result.role,
            "selected_model_cv_metrics": selected_model_cv_metrics,
            "selected_model_holdout_metrics": selected_model_holdout_metrics,
            "candidate_beats_baseline": selection.candidate_beats_baseline,
            "selection_outcome": selection.selection_outcome,
            "best_model_name": selected_result.model_name,
            "best_model_metrics": selected_model_holdout_metrics,
            "all_model_metrics": all_model_metrics,
            "candidate_cv_results": candidate_cv_results,
            "cv_model_metrics": all_model_metrics,
            "final_test_metrics": selected_model_holdout_metrics,
            "holdout_metrics": selected_model_holdout_metrics,
            "cv_folds": prepared.cv_folds,
            "cv_strategy": prepared.cv_strategy,
            "selection_metric": primary_metric,
            "selection_direction": selection_direction,
            "selection_tiebreaker": SELECTION_TIEBREAKER,
            "test_evaluated_model_names": [selected_result.model_name],
            "baseline_comparison": baseline_comparison(
                baseline_result,
                prepared.task_type,
                best_candidate_result=best_candidate_result,
                selected_result=selected_result,
            ),
            "generated_plots": generated_plots,
            "warnings": evaluation_warnings,
            "created_at": created_at,
        }
        summary = EvaluationSummary(**summary_payload)
        save_json(self.evaluation_summary_path(run_id), summary.model_dump(mode="json"))

        model_results_payload = {
            "run_id": run_id,
            "target_column": prepared.target_column,
            "task_type": prepared.task_type,
            "primary_metric": primary_metric,
            "selection_direction": selection_direction,
            "selection_tiebreaker": SELECTION_TIEBREAKER,
            "baseline_model_name": baseline_result.model_name if baseline_result else None,
            "best_candidate_name": (
                best_candidate_result.model_name if best_candidate_result else None
            ),
            "best_candidate_metrics": best_candidate_metrics,
            "selected_model_name": selected_result.model_name,
            "selected_model_role": selected_result.role,
            "selected_model_cv_metrics": selected_model_cv_metrics,
            "selected_model_holdout_metrics": selected_model_holdout_metrics,
            "candidate_beats_baseline": selection.candidate_beats_baseline,
            "selection_outcome": selection.selection_outcome,
            "best_model_name": selected_result.model_name,
            "cv_folds": prepared.cv_folds,
            "cv_strategy": prepared.cv_strategy,
            "selection_metric": primary_metric,
            "candidate_cv_results": candidate_cv_results,
            "cv_model_metrics": all_model_metrics,
            "final_test_metrics": selected_model_holdout_metrics,
            "holdout_metrics": selected_model_holdout_metrics,
            "test_evaluated_model_names": [selected_result.model_name],
            "results": serialize_training_results(training_results),
            "failed_models": [
                {
                    "model_name": result.model_name,
                    "role": result.role,
                    "error": result.error,
                }
                for result in training_results
                if result.status == "failed"
            ],
            "created_at": created_at,
        }

        return EvaluationServiceResult(
            summary=summary,
            model_results_payload=model_results_payload,
            selected_result=selected_result,
            best_candidate_result=best_candidate_result,
            baseline_result=baseline_result,
        )

    def load_evaluation_summary(self, run_id: str) -> EvaluationSummary:
        """Load an existing evaluation summary."""

        path = self.evaluation_summary_path(run_id)
        if not path.exists():
            raise FileNotFoundError(path)
        state = lineage_context(self.run_manager, run_id)["state"]
        validation = validate_artifact_for_state(
            path,
            artifact_type="evaluation_summary",
            state=state,
        )
        if not validation.is_current:
            raise ValueError(f"Evaluation summary artifact is stale: {validation.reason}.")
        return EvaluationSummary(**load_json(path))


def _has_valid_primary_metric(result: ModelTrainingResult) -> bool:
    if result.primary_metric_value is None:
        return False
    try:
        return math.isfinite(float(result.primary_metric_value))
    except (TypeError, ValueError):
        return False


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
