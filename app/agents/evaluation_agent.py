"""Evaluation agent boundary."""

from __future__ import annotations

from app.backend.services.evaluation_service import EvaluationService

class EvaluationAgent:
    """Agent boundary for deterministic model evaluation summaries."""

    def __init__(self, evaluation_service: EvaluationService | None = None) -> None:
        self.evaluation_service = evaluation_service or EvaluationService()

    def run(self, state: dict) -> dict:
        """
        Future orchestration entry point.

        For Week 4, modeling evaluation is produced by the deterministic modeling
        service. This agent loads the saved evaluation summary into shared state.
        """

        run_id = state.get("run_id")
        if not run_id:
            raise ValueError("EvaluationAgent requires `run_id` in state.")

        summary = self.evaluation_service.load_evaluation_summary(str(run_id))
        updated_state = dict(state)
        updated_state["evaluation_summary_path"] = str(
            self.evaluation_service.evaluation_summary_path(str(run_id))
        )
        updated_state["evaluation_summary"] = summary.model_dump(mode="json")
        updated_state["evaluation_plots"] = [
            plot.path for plot in summary.generated_plots
        ]
        return updated_state
