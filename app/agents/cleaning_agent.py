"""Deterministic cleaning agent boundary."""

from __future__ import annotations

from app.backend.services.cleaning_service import CleaningService


class CleaningAgent:
    """Agent wrapper for cleaning planning and safe cleaning execution."""

    def __init__(self, cleaning_service: CleaningService | None = None) -> None:
        self.cleaning_service = cleaning_service or CleaningService()

    def run(self, state: dict) -> dict:
        """Generate a plan and optionally apply safe cleaning."""

        run_id = state.get("run_id")
        if not run_id:
            raise ValueError("CleaningAgent requires state['run_id'].")

        run_id_text = str(run_id)
        plan = self.cleaning_service.generate_cleaning_plan(run_id_text)
        next_state = {
            **state,
            "cleaning_plan": plan.model_dump(mode="json"),
            "cleaning_plan_path": str(
                self.cleaning_service.cleaning_plan_path(run_id_text)
            ),
        }

        if state.get("apply_cleaning"):
            summary = self.cleaning_service.apply_cleaning(run_id_text)
            next_state.update(
                {
                    "cleaning_summary": summary.model_dump(mode="json"),
                    "cleaning_summary_path": str(
                        self.cleaning_service.cleaning_summary_path(run_id_text)
                    ),
                    "cleaned_data_path": str(
                        self.cleaning_service.cleaned_data_path(run_id_text)
                    ),
                }
            )

        return next_state
