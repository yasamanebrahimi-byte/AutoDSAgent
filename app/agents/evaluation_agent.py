"""Evaluation agent boundary."""

from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.backend.services.evaluation_service import EvaluationService


class EvaluationAgent(BaseAgent):
    """Agent boundary for deterministic model evaluation summaries."""

    name = "EvaluationAgent"

    def __init__(self, evaluation_service: EvaluationService | None = None) -> None:
        self.evaluation_service = evaluation_service or EvaluationService()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Load a saved evaluation summary into workflow state."""

        run_id = self._require_run_id(state)
        paths = self.evaluation_service.run_manager.get_paths(run_id)

        summary = self.evaluation_service.load_evaluation_summary(run_id)
        updated_state = self._copy_state(state)
        self._set_artifact(
            updated_state,
            "evaluation_summary",
            self.evaluation_service.evaluation_summary_path(run_id),
            paths.root,
        )
        updated_state["steps"]["modeling"]["outputs"].update(
            {
                "evaluation_best_candidate_name": summary.best_candidate_name,
                "evaluation_selected_model_name": summary.selected_model_name,
                "evaluation_best_model_name": summary.selected_model_name,
                "evaluation_primary_metric": summary.primary_metric,
            }
        )
        return updated_state
