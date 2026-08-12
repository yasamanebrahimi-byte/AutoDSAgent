"""Modeling agent boundary."""

from __future__ import annotations

from app.backend.schemas.modeling import ModelingRequest
from app.backend.services.modeling_service import ModelingService

class ModelingAgent:
    """Agent boundary for model training and model comparison."""

    def __init__(self, modeling_service: ModelingService | None = None) -> None:
        self.modeling_service = modeling_service or ModelingService()

    def run(self, state: dict) -> dict:
        """
        Future orchestration entry point.

        For Week 4, this delegates to deterministic modeling service logic and updates
        the shared state with artifact paths for downstream agents.
        """

        run_id = state.get("run_id")
        target_column = state.get("target_column")
        if not run_id:
            raise ValueError("ModelingAgent requires `run_id` in state.")
        if not target_column:
            raise ValueError("ModelingAgent requires `target_column` in state.")

        request = ModelingRequest(
            target_column=str(target_column),
            task_type=state.get("task_type"),
            test_size=state.get("test_size", 0.2),
            random_state=state.get("random_state", 42),
        )
        response = self.modeling_service.train_and_evaluate(str(run_id), request)

        updated_state = dict(state)
        updated_state["modeling_summary_path"] = str(
            self.modeling_service.modeling_summary_path(str(run_id))
        )
        updated_state["evaluation_summary_path"] = str(
            self.modeling_service.evaluation_service.evaluation_summary_path(str(run_id))
        )
        updated_state["baseline_model_path"] = str(
            self.modeling_service.baseline_model_path(str(run_id))
        )
        updated_state["best_model_path"] = str(
            self.modeling_service.best_model_path(str(run_id))
        )
        updated_state["model_results_path"] = str(
            self.modeling_service.model_results_path(str(run_id))
        )
        updated_state["evaluation_plots"] = [
            plot.path for plot in response.evaluation_summary.generated_plots
        ]
        updated_state["modeling_summary"] = response.modeling_summary.model_dump(mode="json")
        updated_state["evaluation_summary"] = response.evaluation_summary.model_dump(
            mode="json"
        )
        return updated_state
