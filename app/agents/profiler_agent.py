"""Deterministic dataset profiling agent boundary."""

from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.backend.services.profiling_service import ProfilingService


class ProfilerAgent(BaseAgent):
    """Agent wrapper for dataset profiling and data quality analysis."""

    name = "ProfilerAgent"

    def __init__(self, profiling_service: ProfilingService | None = None) -> None:
        self.profiling_service = profiling_service or ProfilingService()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate a profile and attach it to the workflow state."""

        run_id = self._require_run_id(state)
        paths = self.profiling_service.run_manager.get_paths(run_id)
        profile = self.profiling_service.generate_profile(run_id)

        updated_state = self._copy_state(state)
        self._set_artifact(
            updated_state,
            "profile",
            self.profiling_service.profile_path(run_id),
            paths.root,
        )
        updated_state["steps"]["profile"]["outputs"].update(
            {
                "rows": profile.rows,
                "columns": profile.columns,
                "duplicate_rows": profile.duplicate_rows,
                "total_missing_values": profile.total_missing_values,
                "data_quality_issue_count": len(profile.data_quality_issues),
            }
        )
        return updated_state
