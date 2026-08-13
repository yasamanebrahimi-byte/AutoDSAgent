"""Deterministic final report generation service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.backend.schemas.reports import (
    ReportContentResponse,
    ReportGenerateRequest,
    ReportGenerateResponse,
    ReportIndex,
    ReportMetadata,
)
from app.backend.services.run_manager import RunManager
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
    ) -> ReportGenerateResponse:
        """Generate all deterministic report artifacts for one run."""

        options = request or ReportGenerateRequest()
        self._validate_run(run_id)

        if not options.force_regenerate and self._reports_exist(run_id):
            return self.load_reports(run_id)

        source = self.load_source_artifacts(run_id)
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

        metadata_payload = build_report_metadata(
            run_id=run_id,
            documents=list(documents.values()),
            reports_generated=generated_paths,
            source_artifacts_used=source.used,
            source_artifacts_missing=source.missing,
            warnings=source.warnings,
        )
        index_payload = build_report_index(run_id, saved_report_files)

        save_json_report(self.report_metadata_path(run_id), metadata_payload)
        save_json_report(self.report_index_path(run_id), index_payload)

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
        return path

    def load_source_artifacts(self, run_id: str) -> SourceArtifactLoadResult:
        """Load any available source artifacts for report generation."""

        self._validate_run(run_id)
        paths = self.run_manager.get_paths(run_id)
        artifacts: dict[str, Any] = {}
        used: list[str] = []
        missing: list[str] = []
        warnings: list[str] = []

        for key, (directory_name, filename) in SOURCE_ARTIFACTS.items():
            artifact_path = getattr(paths, directory_name) / filename
            relative_path = artifact_path.relative_to(paths.root).as_posix()
            if not artifact_path.exists():
                missing.append(relative_path)
                continue
            try:
                artifacts[key] = load_json(artifact_path)
                used.append(relative_path)
            except ValueError as exc:
                missing.append(relative_path)
                warnings.append(f"Could not load `{relative_path}`: {exc}")

        return SourceArtifactLoadResult(
            artifacts=artifacts,
            used=used,
            missing=missing,
            warnings=warnings,
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
        return all(path.exists() for path in paths)

    def _validate_run(self, run_id: str) -> None:
        paths = self.run_manager.get_paths(run_id)
        if not paths.root.exists():
            raise FileNotFoundError(paths.root)

    def _relative(self, run_id: str, path: Path) -> str:
        paths = self.run_manager.get_paths(run_id)
        return path.relative_to(paths.root).as_posix()
