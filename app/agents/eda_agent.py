"""Exploratory data analysis agent boundary."""

from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.backend.schemas.eda import EDARequest
from app.backend.services.eda_service import EDAService


class EDAAgent(BaseAgent):
    """Agent boundary for exploratory data analysis and visualization."""

    name = "EDAAgent"

    def __init__(self, eda_service: EDAService | None = None) -> None:
        self.eda_service = eda_service or EDAService()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate deterministic EDA artifacts and update workflow state."""

        run_id = self._require_run_id(state)
        paths = self.eda_service.run_manager.get_paths(run_id)

        request = EDARequest(
            target_column=state.get("target_column"),
            max_numeric_plots=state.get("max_numeric_plots", 10),
            max_categorical_plots=state.get("max_categorical_plots", 10),
            max_target_relationship_plots=state.get("max_target_relationship_plots", 5),
        )
        response = self.eda_service.generate_eda(run_id=run_id, request=request)

        updated_state = self._copy_state(state)
        self._set_artifact(
            updated_state,
            "eda_summary",
            self.eda_service.eda_summary_path(run_id),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "eda_findings",
            self.eda_service.eda_findings_path(run_id),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "eda_report",
            self.eda_service.eda_report_path(run_id),
            paths.root,
        )
        updated_state["artifacts"]["plots"] = [
            plot.path for plot in response.summary.generated_plots
        ]
        updated_state["steps"]["eda"]["outputs"].update(
            {
                "dataset_used": response.summary.dataset_used,
                "target_column": response.summary.target_column,
                "generated_plot_count": len(response.summary.generated_plots),
                "warning_count": len(response.summary.warnings),
            }
        )
        return updated_state
