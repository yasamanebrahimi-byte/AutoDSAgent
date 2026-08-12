"""Cleaning plan and safe cleaning execution service."""

from __future__ import annotations

import logging
from pathlib import Path

from app.backend.schemas.cleaning import CleaningPlan, CleaningSummary
from app.backend.services.profiling_service import ProfilingService
from app.backend.services.run_manager import RunManager
from app.tools.app_logging import get_logger, log_event
from app.tools.cleaning import CleaningConfig, apply_safe_cleaning, generate_cleaning_plan_payload
from app.tools.data_loader import load_csv
from app.tools.file_utils import load_json, save_json


class CleaningService:
    """Generate cleaning plans and apply conservative cleaning."""

    def __init__(
        self,
        run_manager: RunManager | None = None,
        profiling_service: ProfilingService | None = None,
        config: CleaningConfig | None = None,
    ) -> None:
        self.run_manager = run_manager or RunManager()
        self.profiling_service = profiling_service or ProfilingService(self.run_manager)
        self.config = config or CleaningConfig()
        self.logger = get_logger(__name__)

    def cleaning_plan_path(self, run_id: str) -> Path:
        """Return the cleaning plan artifact path."""

        return self.run_manager.get_paths(run_id).intermediate / "cleaning_plan.json"

    def cleaned_data_path(self, run_id: str) -> Path:
        """Return the cleaned CSV artifact path."""

        return self.run_manager.get_paths(run_id).intermediate / "cleaned_data.csv"

    def cleaning_summary_path(self, run_id: str) -> Path:
        """Return the cleaning summary artifact path."""

        return self.run_manager.get_paths(run_id).intermediate / "cleaning_summary.json"

    def generate_cleaning_plan(self, run_id: str) -> CleaningPlan:
        """Generate, save, and return a cleaning plan."""

        try:
            profile = self.profiling_service.load_profile(run_id)
        except FileNotFoundError:
            profile = self.profiling_service.generate_profile(run_id)

        payload = generate_cleaning_plan_payload(
            profile=profile.model_dump(mode="json"),
            config=self.config,
        )
        plan = CleaningPlan(**payload)
        save_json(self.cleaning_plan_path(run_id), plan.model_dump(mode="json"))
        log_event(
            self.logger,
            logging.INFO,
            "Cleaning plan generated.",
            run_id=run_id,
            missing_strategies=len(plan.missing_value_strategies),
            review_warnings=len(plan.warnings_requiring_review),
        )
        return plan

    def load_cleaning_plan(self, run_id: str) -> CleaningPlan:
        """Load an existing cleaning plan."""

        path = self.cleaning_plan_path(run_id)
        if not path.exists():
            raise FileNotFoundError(path)
        return CleaningPlan(**load_json(path))

    def apply_cleaning(self, run_id: str) -> CleaningSummary:
        """Apply safe cleaning, save artifacts, and return a summary."""

        try:
            plan = self.load_cleaning_plan(run_id)
        except FileNotFoundError:
            plan = self.generate_cleaning_plan(run_id)

        try:
            profile = self.profiling_service.load_profile(run_id)
        except FileNotFoundError:
            profile = self.profiling_service.generate_profile(run_id)

        raw_path = self.profiling_service.raw_data_path(run_id)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        dataframe = load_csv(raw_path)
        cleaned, summary_payload = apply_safe_cleaning(
            dataframe=dataframe,
            profile=profile.model_dump(mode="json"),
            plan=plan.model_dump(mode="json"),
            config=self.config,
        )

        cleaned.to_csv(self.cleaned_data_path(run_id), index=False)

        summary = CleaningSummary(**summary_payload)
        save_json(self.cleaning_summary_path(run_id), summary.model_dump(mode="json"))
        log_event(
            self.logger,
            logging.INFO,
            "Safe cleaning applied.",
            run_id=run_id,
            duplicates_removed=summary.duplicate_rows_removed,
            missing_before=summary.missing_values_before,
            missing_after=summary.missing_values_after,
        )
        return summary

    def load_cleaning_summary(self, run_id: str) -> CleaningSummary:
        """Load an existing cleaning summary."""

        path = self.cleaning_summary_path(run_id)
        if not path.exists():
            raise FileNotFoundError(path)
        return CleaningSummary(**load_json(path))
