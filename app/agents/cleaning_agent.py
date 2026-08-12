"""Deterministic cleaning agent boundary."""

from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.backend.services.cleaning_service import CleaningService


class CleaningAgent(BaseAgent):
    """Agent wrapper for cleaning planning and safe cleaning execution."""

    name = "CleaningAgent"

    def __init__(self, cleaning_service: CleaningService | None = None) -> None:
        self.cleaning_service = cleaning_service or CleaningService()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the configured cleaning mode."""

        mode = state.get("cleaning_mode", "plan")
        if mode in {"plan", "cleaning_plan"}:
            return self.generate_plan(state)
        if mode in {"apply", "cleaning"}:
            return self.apply_cleaning(state)
        raise ValueError(f"Unknown cleaning mode: {mode}")

    def generate_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate a conservative cleaning plan."""

        run_id = self._require_run_id(state)
        paths = self.cleaning_service.run_manager.get_paths(run_id)
        plan = self.cleaning_service.generate_cleaning_plan(run_id)

        updated_state = self._copy_state(state)
        self._set_artifact(
            updated_state,
            "cleaning_plan",
            self.cleaning_service.cleaning_plan_path(run_id),
            paths.root,
        )
        updated_state["steps"]["cleaning_plan"]["outputs"].update(
            {
                "columns_recommended_for_dropping": list(
                    plan.columns_recommended_for_dropping
                ),
                "duplicate_rows_would_be_removed": bool(
                    plan.duplicate_row_handling.apply
                ),
                "missing_value_strategy_count": len(plan.missing_value_strategies),
                "warnings_requiring_review": list(plan.warnings_requiring_review),
            }
        )
        return updated_state

    def apply_cleaning(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply safe cleaning after any required approval has been granted."""

        run_id = self._require_run_id(state)
        paths = self.cleaning_service.run_manager.get_paths(run_id)
        summary = self.cleaning_service.apply_cleaning(run_id)

        updated_state = self._copy_state(state)
        self._set_artifact(
            updated_state,
            "cleaned_data",
            self.cleaning_service.cleaned_data_path(run_id),
            paths.root,
        )
        self._set_artifact(
            updated_state,
            "cleaning_summary",
            self.cleaning_service.cleaning_summary_path(run_id),
            paths.root,
        )
        updated_state["steps"]["cleaning"]["outputs"].update(
            {
                "original_shape": summary.original_shape,
                "cleaned_shape": summary.cleaned_shape,
                "duplicate_rows_removed": summary.duplicate_rows_removed,
                "columns_dropped": list(summary.columns_dropped),
                "missing_values_before": summary.missing_values_before,
                "missing_values_after": summary.missing_values_after,
            }
        )
        return updated_state
