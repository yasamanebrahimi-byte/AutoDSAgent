"""Cleaning plan and safe cleaning execution service."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from app.backend.schemas.cleaning import CleaningPlan, CleaningSummary
from app.backend.services.profiling_service import ProfilingService
from app.backend.services.run_manager import RunManager
from app.tools.artifact_lineage import (
    file_sha256,
    fingerprint_payload,
    invalidate_downstream_artifacts,
    lineage_context,
    load_artifact_lineage,
    validate_artifact_for_state,
    write_artifact_lineage,
)
from app.tools.app_logging import get_logger, log_event
from app.tools.cleaning import CleaningConfig, apply_safe_cleaning, generate_cleaning_plan_payload
from app.tools.data_loader import load_csv
from app.tools.file_utils import load_json, save_json
from app.workflows.workflow_steps import CLEANING_PLAN_STEP, CLEANING_STEP


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

    def generate_cleaning_plan(
        self,
        run_id: str,
        target_column: str | None = None,
    ) -> CleaningPlan:
        """Generate, save, and return a cleaning plan."""

        invalidate_downstream_artifacts(self.run_manager, run_id, CLEANING_PLAN_STEP)
        try:
            profile = self.profiling_service.load_profile(run_id)
        except (FileNotFoundError, ValueError):
            profile = self.profiling_service.generate_profile(run_id)

        payload = generate_cleaning_plan_payload(
            profile=profile.model_dump(mode="json"),
            config=self.config,
            target_column=target_column,
        )
        plan = CleaningPlan(**payload)
        plan_path = save_json(self.cleaning_plan_path(run_id), plan.model_dump(mode="json"))
        context = lineage_context(self.run_manager, run_id)
        profile_lineage = load_artifact_lineage(self.profiling_service.profile_path(run_id))
        source_fingerprint = context["source_fingerprint"]
        config_payload = {
            "artifact_type": "cleaning_plan",
            "source_fingerprint": source_fingerprint,
            "profile_fingerprint": (
                profile_lineage or {}
            ).get("artifact_fingerprint"),
            "target_column": _normalize_optional_target(target_column),
            "cleaning_config": asdict(self.config),
        }
        config_fingerprint = fingerprint_payload(config_payload)
        write_artifact_lineage(
            plan_path,
            run_root=self.run_manager.get_paths(run_id).root,
            run_id=run_id,
            artifact_type="cleaning_plan",
            generation_id=context["generation_id"],
            source_fingerprint=source_fingerprint,
            target_column=_normalize_optional_target(target_column),
            config_fingerprint=config_fingerprint,
            upstream_fingerprints={
                "source_data": source_fingerprint,
                "profile": (profile_lineage or {}).get("artifact_fingerprint"),
            },
            relevant_config=config_payload,
        )
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
        state = lineage_context(self.run_manager, run_id)["state"]
        validation = validate_artifact_for_state(
            path,
            artifact_type="cleaning_plan",
            state=state,
        )
        if not validation.is_current:
            raise ValueError(f"Cleaning plan artifact is stale: {validation.reason}.")
        return CleaningPlan(**load_json(path))

    def apply_cleaning(
        self,
        run_id: str,
        target_column: str | None = None,
    ) -> CleaningSummary:
        """Apply safe cleaning, save artifacts, and return a summary."""

        invalidate_downstream_artifacts(self.run_manager, run_id, CLEANING_STEP)
        requested_target = _normalize_optional_target(target_column)
        try:
            plan = self.load_cleaning_plan(run_id)
        except (FileNotFoundError, ValueError):
            plan = self.generate_cleaning_plan(run_id, target_column=requested_target)

        if _normalize_optional_target(plan.target_column) != requested_target:
            plan = self.generate_cleaning_plan(run_id, target_column=requested_target)

        try:
            profile = self.profiling_service.load_profile(run_id)
        except (FileNotFoundError, ValueError):
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
            target_column=requested_target,
        )

        cleaned_path = self.cleaned_data_path(run_id)
        cleaned.to_csv(cleaned_path, index=False)

        summary = CleaningSummary(**summary_payload)
        summary_path = save_json(
            self.cleaning_summary_path(run_id),
            summary.model_dump(mode="json"),
        )
        context = lineage_context(self.run_manager, run_id)
        source_fingerprint = context["source_fingerprint"]
        plan_lineage = load_artifact_lineage(self.cleaning_plan_path(run_id))
        profile_lineage = load_artifact_lineage(self.profiling_service.profile_path(run_id))
        cleaned_fingerprint = file_sha256(cleaned_path)
        config_payload = {
            "artifact_type": "cleaned_data",
            "source_fingerprint": source_fingerprint,
            "profile_fingerprint": (profile_lineage or {}).get("artifact_fingerprint"),
            "cleaning_plan_fingerprint": (plan_lineage or {}).get("artifact_fingerprint"),
            "target_column": requested_target,
            "cleaning_config": asdict(self.config),
        }
        config_fingerprint = fingerprint_payload(config_payload)
        upstream_fingerprints = {
            "source_data": source_fingerprint,
            "profile": (profile_lineage or {}).get("artifact_fingerprint"),
            "cleaning_plan": (plan_lineage or {}).get("artifact_fingerprint"),
        }
        write_artifact_lineage(
            cleaned_path,
            run_root=self.run_manager.get_paths(run_id).root,
            run_id=run_id,
            artifact_type="cleaned_data",
            generation_id=context["generation_id"],
            source_fingerprint=source_fingerprint,
            target_column=requested_target,
            config_fingerprint=config_fingerprint,
            upstream_fingerprints=upstream_fingerprints,
            relevant_config=config_payload,
        )
        write_artifact_lineage(
            summary_path,
            run_root=self.run_manager.get_paths(run_id).root,
            run_id=run_id,
            artifact_type="cleaning_summary",
            generation_id=context["generation_id"],
            source_fingerprint=source_fingerprint,
            target_column=requested_target,
            config_fingerprint=config_fingerprint,
            upstream_fingerprints={
                **upstream_fingerprints,
                "cleaned_data": cleaned_fingerprint,
            },
            relevant_config=config_payload,
        )
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
        state = lineage_context(self.run_manager, run_id)["state"]
        validation = validate_artifact_for_state(
            path,
            artifact_type="cleaning_summary",
            state=state,
        )
        if not validation.is_current:
            raise ValueError(f"Cleaning summary artifact is stale: {validation.reason}.")
        return CleaningSummary(**load_json(path))


def _normalize_optional_target(target_column: str | None) -> str | None:
    if target_column is None:
        return None
    target = str(target_column).strip()
    return target or None
