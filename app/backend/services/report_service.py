"""Deterministic final report generation service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app.backend.schemas.reports import (
    ReportContentResponse,
    ReportGenerateRequest,
    ReportGenerateResponse,
    ReportIndex,
    ReportMetadata,
)
from app.backend.services.run_manager import RunManager
from app.tools.artifact_lineage import (
    REPORT_ARTIFACT_TYPES,
    fingerprint_payload,
    invalidate_downstream_artifacts,
    lineage_context,
    load_artifact_lineage,
    state_allows_source_artifact,
    validate_artifact_for_state,
    write_artifact_lineage,
)
from app.tools.app_logging import get_logger, log_event
from app.tools.file_utils import load_json
from app.tools.report_builder import (
    ReportDocument,
    build_executive_summary,
    build_final_report,
    build_limitations_report,
    build_report_index,
    build_report_metadata,
    build_technical_summary,
)
from app.tools.report_export import (
    save_html_from_markdown,
    save_json_report,
    save_markdown,
)
from app.workflows.workflow_steps import REPORT_STEP


REPORT_METADATA_FILENAME = "report_metadata.json"
REPORT_INDEX_FILENAME = "report_index.json"

SUPPORTED_REPORTS: dict[str, tuple[str, str]] = {
    "final_report": ("final_report.md", "Full end-to-end analysis report"),
    "executive_summary": ("executive_summary.md", "Short nontechnical summary"),
    "technical_summary": ("technical_summary.md", "Technical methodology summary"),
    "limitations": ("limitations.md", "Limitations and recommended next steps"),
}

SOURCE_ARTIFACTS: dict[str, tuple[str, str]] = {
    "metadata": ("intermediate", "metadata.json"),
    "profile": ("intermediate", "profile.json"),
    "cleaning_plan": ("intermediate", "cleaning_plan.json"),
    "cleaning_summary": ("intermediate", "cleaning_summary.json"),
    "eda_summary": ("intermediate", "eda_summary.json"),
    "eda_findings": ("intermediate", "eda_findings.json"),
    "modeling_summary": ("intermediate", "modeling_summary.json"),
    "evaluation_summary": ("intermediate", "evaluation_summary.json"),
    "model_results": ("models", "model_results.json"),
    "workflow_state": ("logs", "workflow_state.json"),
    "agent_trace": ("logs", "agent_trace.json"),
}

MINIMUM_ARTIFACT_KEYS = {
    "metadata",
    "profile",
    "cleaning_summary",
    "eda_summary",
    "modeling_summary",
    "workflow_state",
}


@dataclass(frozen=True)
class SourceArtifactLoadResult:
    """Loaded source artifacts plus audit details."""

    artifacts: dict[str, Any]
    used: list[str]
    missing: list[str]
    warnings: list[str]
    lineage: dict[str, dict[str, Any]]


class ReportService:
    """Generate, save, and retrieve deterministic final reports."""

    def __init__(self, run_manager: RunManager | None = None) -> None:
        self.run_manager = run_manager or RunManager()
        self.logger = get_logger(__name__)

    def report_metadata_path(self, run_id: str) -> Path:
        """Return the report metadata JSON artifact path."""

        return self.run_manager.get_paths(run_id).intermediate / REPORT_METADATA_FILENAME

    def report_index_path(self, run_id: str) -> Path:
        """Return the report index JSON artifact path."""

        return self.run_manager.get_paths(run_id).reports / REPORT_INDEX_FILENAME

    def report_path(self, run_id: str, report_name: str) -> Path:
        """Return a generated Markdown report path."""

        if report_name not in SUPPORTED_REPORTS:
            raise ValueError(f"Unsupported report name: {report_name}")
        filename, _ = SUPPORTED_REPORTS[report_name]
        return self.run_manager.get_paths(run_id).reports / filename

    def html_report_path(self, run_id: str) -> Path:
        """Return the optional HTML final report path."""

        return self.run_manager.get_paths(run_id).reports / "final_report.html"

    def generate_reports(
        self,
        run_id: str,
        request: ReportGenerateRequest | None = None,
        *,
        workflow_state: Mapping[str, Any] | None = None,
    ) -> ReportGenerateResponse:
        """Generate all deterministic report artifacts for one run."""

        options = request or ReportGenerateRequest()
        self._validate_run(run_id)

        if not options.force_regenerate and self._reports_exist(run_id):
            return self.load_reports(run_id)

        invalidate_downstream_artifacts(self.run_manager, run_id, REPORT_STEP)
        source = self.load_source_artifacts(run_id, workflow_state=workflow_state)
        if not (MINIMUM_ARTIFACT_KEYS & set(source.artifacts)):
            raise ValueError(
                "At least one saved analysis artifact is required to generate a report."
            )

        planned_report_files = self._planned_report_files(run_id, include_html=options.include_html)
        documents = self._build_documents(
            run_id=run_id,
            source=source,
            report_files=planned_report_files,
        )
        saved_report_files = self._save_documents(run_id, documents)

        if options.include_html:
            html_path = save_html_from_markdown(
                self.html_report_path(run_id),
                documents["final_report"].content,
                title="AutoDS Agent Final Analysis Report",
            )
            saved_report_files.append(
                self._report_file_info(
                    run_id=run_id,
                    name="final_report_html",
                    path=html_path,
                    description="HTML export of the full final report",
                    media_type="text/html",
                )
            )

        generated_paths = [
            file_info["path"]
            for file_info in saved_report_files
        ]
        generated_paths.extend(
            [
                self._relative(run_id, self.report_metadata_path(run_id)),
                self._relative(run_id, self.report_index_path(run_id)),
            ]
        )

        context = lineage_context(self.run_manager, run_id)
        source_artifact_fingerprints = {
            key: metadata.get("artifact_fingerprint")
            for key, metadata in source.lineage.items()
            if metadata.get("artifact_fingerprint")
        }
        report_config = {
            "artifact_family": "report",
            "generation_id": context["generation_id"],
            "source_fingerprint": context["source_fingerprint"],
            "target_column": context["target_column"],
            "task_type": context["task_type"],
            "source_artifacts_used": source.used,
            "source_artifact_fingerprints": source_artifact_fingerprints,
        }
        report_config_fingerprint = fingerprint_payload(report_config)
        metadata_payload = build_report_metadata(
            run_id=run_id,
            documents=list(documents.values()),
            reports_generated=generated_paths,
            source_artifacts_used=source.used,
            source_artifacts_missing=source.missing,
            warnings=source.warnings,
        )
        metadata_payload.update(
            {
                "generation_id": context["generation_id"],
                "source_fingerprint": context["source_fingerprint"],
                "target_column": context["target_column"],
                "task_type": context["task_type"],
                "config_fingerprint": report_config_fingerprint,
                "source_artifact_lineage": source.lineage,
            }
        )
        index_payload = build_report_index(run_id, saved_report_files)

        metadata_path = save_json_report(self.report_metadata_path(run_id), metadata_payload)
        index_path = save_json_report(self.report_index_path(run_id), index_payload)
        self._write_report_lineage(
            run_id=run_id,
            report_files=saved_report_files,
            metadata_path=metadata_path,
            index_path=index_path,
            config_fingerprint=report_config_fingerprint,
            config=report_config,
            source_lineage=source.lineage,
        )

        log_event(
            self.logger,
            logging.INFO,
            "Reports generated.",
            run_id=run_id,
            report_status=metadata_payload["report_status"],
            reports=len(index_payload["reports"]),
            missing_sources=len(source.missing),
        )

        return ReportGenerateResponse(
            metadata=ReportMetadata(**metadata_payload),
            index=ReportIndex(**index_payload),
        )

    def load_reports(self, run_id: str) -> ReportGenerateResponse:
        """Load generated report metadata and index."""

        self._validate_run(run_id)
        metadata_path = self.report_metadata_path(run_id)
        index_path = self.report_index_path(run_id)
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        self._ensure_report_current(run_id, metadata_path, "report_metadata")
        self._ensure_report_current(run_id, index_path, "report_index")

        return ReportGenerateResponse(
            metadata=ReportMetadata(**load_json(metadata_path)),
            index=ReportIndex(**load_json(index_path)),
        )

    def get_report_content(self, run_id: str, report_name: str) -> ReportContentResponse:
        """Return Markdown content for a generated report."""

        self._validate_run(run_id)
        path = self.report_path(run_id, report_name)
        if not path.exists():
            raise FileNotFoundError(path)
        self._ensure_report_current(
            run_id,
            path,
            self._report_artifact_type(report_name),
        )

        return ReportContentResponse(
            run_id=run_id,
            report_name=report_name,
            path=self._relative(run_id, path),
            content=path.read_text(encoding="utf-8"),
        )

    def resolve_report_file(self, run_id: str, report_name: str) -> Path:
        """Resolve a generated Markdown report path for download."""

        self._validate_run(run_id)
        path = self.report_path(run_id, report_name)
        if not path.exists():
            raise FileNotFoundError(path)
        self._ensure_report_current(
            run_id,
            path,
            self._report_artifact_type(report_name),
        )
        return path

    def load_source_artifacts(
        self,
        run_id: str,
        *,
        workflow_state: Mapping[str, Any] | None = None,
    ) -> SourceArtifactLoadResult:
        """Load current source artifacts for report generation."""

        self._validate_run(run_id)
        paths = self.run_manager.get_paths(run_id)
        state = (
            dict(workflow_state)
            if workflow_state is not None
            else lineage_context(self.run_manager, run_id)["state"]
        )
        artifacts: dict[str, Any] = {}
        used: list[str] = []
        missing: list[str] = []
        warnings: list[str] = []
        lineage: dict[str, dict[str, Any]] = {}

        for key, (directory_name, filename) in SOURCE_ARTIFACTS.items():
            artifact_path = getattr(paths, directory_name) / filename
            relative_path = artifact_path.relative_to(paths.root).as_posix()
            if key == "workflow_state" and workflow_state is not None:
                artifacts[key] = dict(workflow_state)
                used.append(relative_path)
                continue
            if not artifact_path.exists():
                missing.append(relative_path)
                continue
            step_validation = state_allows_source_artifact(key, state)
            if not step_validation.is_current:
                missing.append(relative_path)
                warnings.append(
                    f"Ignored `{relative_path}` because {step_validation.reason}."
                )
                continue
            if key not in {"workflow_state", "agent_trace"}:
                validation = validate_artifact_for_state(
                    artifact_path,
                    artifact_type=key,
                    state=state,
                )
                if not validation.is_current:
                    missing.append(relative_path)
                    warnings.append(
                        f"Ignored stale `{relative_path}`: {validation.reason}."
                    )
                    continue
                if validation.metadata:
                    lineage[key] = validation.metadata
            try:
                artifacts[key] = load_json(artifact_path)
                used.append(relative_path)
                if key in {"workflow_state", "agent_trace"}:
                    metadata = load_artifact_lineage(artifact_path)
                    if metadata:
                        lineage[key] = metadata
            except ValueError as exc:
                missing.append(relative_path)
                warnings.append(f"Could not load `{relative_path}`: {exc}")

        return SourceArtifactLoadResult(
            artifacts=artifacts,
            used=used,
            missing=missing,
            warnings=warnings,
            lineage=lineage,
        )

    def _build_documents(
        self,
        run_id: str,
        source: SourceArtifactLoadResult,
        report_files: list[dict[str, Any]],
    ) -> dict[str, ReportDocument]:
        final_report = build_final_report(
            run_id=run_id,
            artifacts=source.artifacts,
            source_artifacts_used=source.used,
            source_artifacts_missing=source.missing,
            report_files=report_files,
        )
        executive_summary = build_executive_summary(
            run_id=run_id,
            artifacts=source.artifacts,
            source_artifacts_used=source.used,
            source_artifacts_missing=source.missing,
        )
        technical_summary = build_technical_summary(
            run_id=run_id,
            artifacts=source.artifacts,
            source_artifacts_used=source.used,
            source_artifacts_missing=source.missing,
            report_files=report_files,
        )
        limitations = build_limitations_report(
            run_id=run_id,
            artifacts=source.artifacts,
            source_artifacts_missing=source.missing,
        )
        return {
            "final_report": final_report,
            "executive_summary": executive_summary,
            "technical_summary": technical_summary,
            "limitations": limitations,
        }

    def _save_documents(
        self,
        run_id: str,
        documents: dict[str, ReportDocument],
    ) -> list[dict[str, Any]]:
        saved: list[dict[str, Any]] = []
        for report_name, document in documents.items():
            path = save_markdown(self.report_path(run_id, report_name), document.content)
            saved.append(
                self._report_file_info(
                    run_id=run_id,
                    name=report_name,
                    path=path,
                    description=SUPPORTED_REPORTS[report_name][1],
                    media_type="text/markdown",
                )
            )
        return saved

    def _planned_report_files(self, run_id: str, include_html: bool) -> list[dict[str, Any]]:
        planned = [
            {
                "name": report_name,
                "path": self._relative(run_id, self.report_path(run_id, report_name)),
                "description": description,
                "media_type": "text/markdown",
                "size_bytes": None,
            }
            for report_name, (_, description) in SUPPORTED_REPORTS.items()
        ]
        if include_html:
            planned.append(
                {
                    "name": "final_report_html",
                    "path": self._relative(run_id, self.html_report_path(run_id)),
                    "description": "HTML export of the full final report",
                    "media_type": "text/html",
                    "size_bytes": None,
                }
            )
        return planned

    def _report_file_info(
        self,
        run_id: str,
        name: str,
        path: Path,
        description: str,
        media_type: str,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "path": self._relative(run_id, path),
            "description": description,
            "media_type": media_type,
            "size_bytes": path.stat().st_size if path.exists() else None,
        }

    def _reports_exist(self, run_id: str) -> bool:
        paths = [
            self.report_metadata_path(run_id),
            self.report_index_path(run_id),
            *[
                self.report_path(run_id, report_name)
                for report_name in SUPPORTED_REPORTS
            ],
        ]
        if not all(path.exists() for path in paths):
            return False
        for artifact_path, artifact_type in (
            (self.report_metadata_path(run_id), "report_metadata"),
            (self.report_index_path(run_id), "report_index"),
        ):
            try:
                self._ensure_report_current(run_id, artifact_path, artifact_type)
            except ValueError:
                return False
        return True

    def _write_report_lineage(
        self,
        *,
        run_id: str,
        report_files: list[dict[str, Any]],
        metadata_path: Path,
        index_path: Path,
        config_fingerprint: str,
        config: dict[str, Any],
        source_lineage: dict[str, dict[str, Any]],
    ) -> None:
        paths = self.run_manager.get_paths(run_id)
        context = lineage_context(self.run_manager, run_id)
        upstream_fingerprints = {
            key: metadata.get("artifact_fingerprint")
            for key, metadata in source_lineage.items()
            if metadata.get("artifact_fingerprint")
        }
        for file_info in report_files:
            artifact_path = paths.root / file_info["path"]
            artifact_type = self._report_artifact_type(str(file_info["name"]))
            write_artifact_lineage(
                artifact_path,
                run_root=paths.root,
                run_id=run_id,
                artifact_type=artifact_type,
                generation_id=context["generation_id"],
                source_fingerprint=context["source_fingerprint"],
                target_column=context["target_column"],
                task_type=context["task_type"],
                config_fingerprint=config_fingerprint,
                upstream_fingerprints=upstream_fingerprints,
                relevant_config=config,
            )

        for artifact_path, artifact_type in (
            (metadata_path, "report_metadata"),
            (index_path, "report_index"),
        ):
            write_artifact_lineage(
                artifact_path,
                run_root=paths.root,
                run_id=run_id,
                artifact_type=artifact_type,
                generation_id=context["generation_id"],
                source_fingerprint=context["source_fingerprint"],
                target_column=context["target_column"],
                task_type=context["task_type"],
                config_fingerprint=config_fingerprint,
                upstream_fingerprints=upstream_fingerprints,
                relevant_config=config,
            )

    def _ensure_report_current(
        self,
        run_id: str,
        path: Path,
        artifact_type: str,
    ) -> None:
        state = lineage_context(self.run_manager, run_id)["state"]
        validation = validate_artifact_for_state(
            path,
            artifact_type=artifact_type,
            state=state,
            require_lineage_if_stateful=artifact_type in REPORT_ARTIFACT_TYPES,
        )
        if not validation.is_current:
            raise ValueError(f"Report artifact is stale: {validation.reason}.")

    def _report_artifact_type(self, report_name: str) -> str:
        if report_name == "limitations":
            return "limitations_report"
        return report_name

    def _validate_run(self, run_id: str) -> None:
        paths = self.run_manager.get_paths(run_id)
        if not paths.root.exists():
            raise FileNotFoundError(paths.root)

    def _relative(self, run_id: str, path: Path) -> str:
        paths = self.run_manager.get_paths(run_id)
        return path.relative_to(paths.root).as_posix()
