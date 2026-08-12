"""Modeling agent boundary."""

from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.backend.schemas.modeling import ModelingRequest
from app.backend.services.modeling_service import ModelingService


class ModelingAgent(BaseAgent):
    """Agent boundary for model training and model comparison."""

    name = "ModelingAgent"

    def __init__(self, modeling_service: ModelingService | None = None) -> None:
        self.modeling_service = modeling_service or ModelingService()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Train and evaluate deterministic models when a target is available."""

        run_id = self._require_run_id(state)
        target_column = state.get("target_column")
        if not target_column:
            updated_state = self._copy_state(state)
            updated_state["steps"]["modeling"]["outputs"].update(
                {"skip_reason": "No target column was provided."}
            )
            return updated_state

        request = ModelingRequest(
            target_column=str(target_column),
            task_type=state.get("task_type"),
            test_size=state.get("test_size", 0.2),
            random_state=state.get("random_state", 42),
        )
        response = self.modeling_service.train_and_evaluate(run_id, request)

        paths = self.modeling_service.run_manager.get_paths(run_id)
        updated_state = self._copy_state(state)
        updated_state["task_type"] = response.modeling_summary.task_type
        self._set_artifact(
            updated_state,
            "modeling_summary",
            self.modeling_service.modeling_summary_path(run_id),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "evaluation_summary",
            self.modeling_service.evaluation_service.evaluation_summary_path(run_id),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "baseline_model",
            self.modeling_service.baseline_model_path(run_id),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "best_model",
            self.modeling_service.best_model_path(run_id),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "model_results",
            self.modeling_service.model_results_path(run_id),
            paths.root,
        )
        updated_state["steps"]["modeling"]["outputs"].update(
            {
                "target_column": response.modeling_summary.target_column,
                "task_type": response.modeling_summary.task_type,
                "best_model_name": response.modeling_summary.best_model_name,
                "primary_metric": response.modeling_summary.primary_metric,
                "models_succeeded": list(response.modeling_summary.models_succeeded),
                "models_failed": list(response.modeling_summary.models_failed),
            }
        )
        return updated_state
