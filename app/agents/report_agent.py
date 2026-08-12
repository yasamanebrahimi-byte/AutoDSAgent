"""Final report generation agent boundary."""

from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.backend.schemas.reports import ReportGenerateRequest
from app.backend.services.report_service import ReportService


class ReportAgent(BaseAgent):
    """Agent boundary for deterministic final report generation."""

    name = "ReportAgent"

    def __init__(self, report_service: ReportService | None = None) -> None:
        self.report_service = report_service or ReportService()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate deterministic report artifacts and update workflow state."""

        run_id = self._require_run_id(state)
        paths = self.report_service.run_manager.get_paths(run_id)

        response = self.report_service.generate_reports(
            run_id,
            ReportGenerateRequest(
                include_html=bool(state.get("include_html_report", False)),
                force_regenerate=True,
            ),
        )

        updated_state = self._copy_state(state)
        self._set_artifact(
            updated_state,
            "final_report",
            self.report_service.report_path(run_id, "final_report"),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "executive_summary",
            self.report_service.report_path(run_id, "executive_summary"),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "technical_summary",
            self.report_service.report_path(run_id, "technical_summary"),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "limitations_report",
            self.report_service.report_path(run_id, "limitations"),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "report_metadata",
            self.report_service.report_metadata_path(run_id),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "report_index",
            self.report_service.report_index_path(run_id),
            paths.root,
        )

        updated_state["steps"]["report"]["outputs"].update(
            {
                "report_status": response.metadata.report_status,
                "reports_generated": list(response.metadata.reports_generated),
                "sections_generated": list(response.metadata.sections_generated),
                "sections_skipped": list(response.metadata.sections_skipped),
                "warning_count": len(response.metadata.warnings),
            }
        )
        if response.metadata.warnings:
            updated_state.setdefault("warnings", []).extend(response.metadata.warnings)
        return updated_state
