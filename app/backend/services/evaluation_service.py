"""Model evaluation service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.backend.schemas.modeling import EvaluationSummary
from app.backend.services.run_manager import RunManager
from app.tools.evaluation import (
    PRIMARY_METRIC_BY_TASK,
    SELECTION_DIRECTION_BY_TASK,
    SELECTION_TIEBREAKER,
    baseline_comparison,
    create_evaluation_plots,
    evaluate_holdout_model,
    refit_model_on_training_partition,
    select_best_model,
)
from app.tools.file_utils import load_json, save_json
from app.tools.modeling import ModelTrainingResult, serialize_training_results
from app.tools.preprocessing import PreprocessingResult


@dataclass
class EvaluationServiceResult:
    """Evaluation outputs needed by the modeling service."""

    summary: EvaluationSummary
    model_results_payload: dict[str, Any]
    best_result: ModelTrainingResult
    baseline_result: ModelTrainingResult | None


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
        """Select from CV results, evaluate the best model, and save artifacts."""

        paths = self.run_manager.get_paths(run_id)
        if not paths.root.exists():
            raise FileNotFoundError(paths.root)

        best_result = select_best_model(training_results, prepared.task_type)
        best_result = refit_model_on_training_partition(best_result, prepared)
        best_result = evaluate_holdout_model(best_result, prepared)
        baseline_result = next(
            (result for result in training_results if result.role == "baseline"),
            None,
        )
        primary_metric = PRIMARY_METRIC_BY_TASK[prepared.task_type]

        generated_plots = create_evaluation_plots(
            results=training_results,
            prepared=prepared,
            best_result=best_result,
            run_root=paths.root,
            plots_dir=paths.plots,
        )

        all_model_metrics = {
            result.model_name: result.cv_metrics
            for result in training_results
            if result.status == "succeeded"
        }
        candidate_cv_results = {
            result.model_name: result.cv_metrics
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
                f"{len(model_failures)} model(s) failed and were excluded from best-model selection."
            )

        created_at = datetime.now(timezone.utc).isoformat()
        selection_direction = SELECTION_DIRECTION_BY_TASK[prepared.task_type]
        summary_payload = {
            "run_id": run_id,
            "target_column": prepared.target_column,
            "task_type": prepared.task_type,
            "primary_metric": primary_metric,
            "best_model_name": best_result.model_name,
            "baseline_metrics": baseline_result.cv_metrics if baseline_result else {},
            "best_model_metrics": best_result.holdout_metrics,
            "all_model_metrics": all_model_metrics,
            "candidate_cv_results": candidate_cv_results,
            "cv_model_metrics": all_model_metrics,
            "final_test_metrics": best_result.holdout_metrics,
            "holdout_metrics": best_result.holdout_metrics,
            "cv_folds": prepared.cv_folds,
            "cv_strategy": prepared.cv_strategy,
            "selection_metric": primary_metric,
            "selection_direction": selection_direction,
            "selection_tiebreaker": SELECTION_TIEBREAKER,
            "test_evaluated_model_names": [best_result.model_name],
            "baseline_comparison": baseline_comparison(
                baseline_result,
                best_result,
                prepared.task_type,
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
            "best_model_name": best_result.model_name,
            "cv_folds": prepared.cv_folds,
            "cv_strategy": prepared.cv_strategy,
            "selection_metric": primary_metric,
            "candidate_cv_results": candidate_cv_results,
            "cv_model_metrics": all_model_metrics,
            "final_test_metrics": best_result.holdout_metrics,
            "holdout_metrics": best_result.holdout_metrics,
            "test_evaluated_model_names": [best_result.model_name],
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
            best_result=best_result,
            baseline_result=baseline_result,
        )

    def load_evaluation_summary(self, run_id: str) -> EvaluationSummary:
        """Load an existing evaluation summary."""

        path = self.evaluation_summary_path(run_id)
        if not path.exists():
            raise FileNotFoundError(path)
        return EvaluationSummary(**load_json(path))
