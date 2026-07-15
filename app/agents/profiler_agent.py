"""Deterministic dataset profiling agent boundary."""

from __future__ import annotations

from app.backend.services.profiling_service import ProfilingService


class ProfilerAgent:
    """Agent wrapper for dataset profiling and data quality analysis."""

    def __init__(self, profiling_service: ProfilingService | None = None) -> None:
        self.profiling_service = profiling_service or ProfilingService()

    def run(self, state: dict) -> dict:
        """Generate a profile and attach it to the workflow state."""

        run_id = state.get("run_id")
        if not run_id:
            raise ValueError("ProfilerAgent requires state['run_id'].")

        profile = self.profiling_service.generate_profile(str(run_id))
        return {
            **state,
            "profile": profile.model_dump(mode="json"),
            "profile_path": str(self.profiling_service.profile_path(str(run_id))),
        }
