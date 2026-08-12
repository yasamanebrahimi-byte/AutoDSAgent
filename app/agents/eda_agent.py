"""Exploratory data analysis agent boundary."""

from __future__ import annotations

from app.backend.schemas.eda import EDARequest
from app.backend.services.eda_service import EDAService


class EDAAgent:
    """Agent boundary for exploratory data analysis and visualization."""

    def __init__(self, eda_service: EDAService | None = None) -> None:
        self.eda_service = eda_service or EDAService()

    def run(self, state: dict) -> dict:
        """
        Future orchestration entry point.

        For Week 3, this delegates to deterministic EDA service logic and updates
        the shared state with artifact paths for downstream agents.
        """

        run_id = state.get("run_id")
        if not run_id:
            raise ValueError("EDAAgent requires `run_id` in state.")

        request = EDARequest(
            target_column=state.get("target_column"),
            max_numeric_plots=state.get("max_numeric_plots", 10),
            max_categorical_plots=state.get("max_categorical_plots", 10),
            max_target_relationship_plots=state.get("max_target_relationship_plots", 5),
        )
        response = self.eda_service.generate_eda(run_id=str(run_id), request=request)

        updated_state = dict(state)
        updated_state["eda_summary_path"] = str(self.eda_service.eda_summary_path(str(run_id)))
        updated_state["eda_findings_path"] = str(self.eda_service.eda_findings_path(str(run_id)))
        updated_state["eda_report_path"] = str(self.eda_service.eda_report_path(str(run_id)))
        updated_state["eda_generated_plots"] = [
            plot.path for plot in response.summary.generated_plots
        ]
        updated_state["eda_summary"] = response.summary.model_dump(mode="json")
        updated_state["eda_findings"] = response.findings.model_dump(mode="json")
        return updated_state
