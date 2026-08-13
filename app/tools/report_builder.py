"""Compose deterministic final report artifacts from saved run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.tools.report_sections import (
    ReportSection,
    build_appendix_section,
    build_artifacts_section,
    build_best_model_section,
    build_cleaning_methodology_section,
    build_data_quality_section,
    build_dataset_overview_section,
    build_eda_findings_section,
    build_evaluation_results_section,
    build_executive_summary_section,
    build_limitations_section,
    build_modeling_methodology_section,
    build_next_steps_section,
    build_run_metadata_section,
    build_target_analysis_section,
    build_title_section,
)


@dataclass(frozen=True)
class ReportDocument:
    """Composed Markdown report plus section-generation metadata."""

    name: str
    content: str
    sections_generated: list[str]
    sections_skipped: list[str]
    warnings: list[str]


def build_final_report(
    run_id: str,
    artifacts: Mapping[str, Any],
    source_artifacts_used: Sequence[str],
    source_artifacts_missing: Sequence[str],
    report_files: Sequence[Mapping[str, Any]] | None = None,
) -> ReportDocument:
    """Build the full final analysis report."""

    body_sections = [
        build_run_metadata_section(run_id, artifacts),
        build_executive_summary_section(artifacts),
        build_dataset_overview_section(artifacts),
        build_data_quality_section(artifacts),
        build_cleaning_methodology_section(artifacts),
        build_eda_findings_section(artifacts),
        build_target_analysis_section(artifacts),
        build_modeling_methodology_section(artifacts),
        build_evaluation_results_section(artifacts),
        build_best_model_section(artifacts),
        build_artifacts_section(
            artifacts,
            source_artifacts_used,
            source_artifacts_missing,
            report_files=report_files,
        ),
        build_limitations_section(artifacts, source_artifacts_missing),
        build_next_steps_section(artifacts),
        build_appendix_section(artifacts, source_artifacts_used, source_artifacts_missing),
    ]
    status = (
        "partial"
        if source_artifacts_missing
        else _status_from_sections(body_sections)
    )
    title = build_title_section(
        run_id,
        metadata=_artifact(artifacts, "metadata"),
        report_status=status,
    )
    return _document("final_report", [title, *body_sections])


def build_executive_summary(
    run_id: str,
    artifacts: Mapping[str, Any],
    source_artifacts_used: Sequence[str] | None = None,
    source_artifacts_missing: Sequence[str] | None = None,
) -> ReportDocument:
    """Build the standalone executive summary report."""

    sections = [
        _static_heading(
            "executive_summary_title",
            "# AutoDS Agent Executive Summary",
        ),
        build_run_metadata_section(run_id, artifacts),
        build_executive_summary_section(artifacts),
        build_limitations_section(artifacts, source_artifacts_missing or []),
        build_next_steps_section(artifacts),
    ]
    return _document("executive_summary", sections)


def build_technical_summary(
    run_id: str,
    artifacts: Mapping[str, Any],
    source_artifacts_used: Sequence[str],
    source_artifacts_missing: Sequence[str],
    report_files: Sequence[Mapping[str, Any]] | None = None,
) -> ReportDocument:
    """Build the standalone technical methodology summary."""

    sections = [
        _static_heading(
            "technical_summary_title",
            "# AutoDS Agent Technical Methodology Summary",
        ),
        build_run_metadata_section(run_id, artifacts),
        build_dataset_overview_section(artifacts),
        build_cleaning_methodology_section(artifacts),
        build_eda_findings_section(artifacts),
        build_modeling_methodology_section(artifacts),
        build_evaluation_results_section(artifacts),
        build_artifacts_section(
            artifacts,
            source_artifacts_used,
            source_artifacts_missing,
            report_files=report_files,
        ),
        _technical_guardrails_section(),
    ]
    return _document("technical_summary", sections)


def build_limitations_report(
    run_id: str,
    artifacts: Mapping[str, Any],
    source_artifacts_missing: Sequence[str],
) -> ReportDocument:
    """Build the standalone limitations and next steps report."""

    sections = [
        _static_heading(
            "limitations_title",
            "# AutoDS Agent Limitations And Next Steps",
        ),
        build_run_metadata_section(run_id, artifacts),
        build_limitations_section(artifacts, source_artifacts_missing),
        build_next_steps_section(artifacts),
    ]
    return _document("limitations", sections)


def build_report_metadata(
    run_id: str,
    documents: Sequence[ReportDocument],
    reports_generated: Sequence[str],
    source_artifacts_used: Sequence[str],
    source_artifacts_missing: Sequence[str],
    warnings: Sequence[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build structured report metadata."""

    timestamp = datetime.now(timezone.utc).isoformat()
    sections_generated = _dedupe(
        section
        for document in documents
        for section in document.sections_generated
    )
    sections_skipped = _dedupe(
        section for document in documents for section in document.sections_skipped
    )
    all_warnings = _dedupe(
        [
            *(warnings or []),
            *[
                warning
                for document in documents
                for warning in document.warnings
            ],
        ]
    )
    report_status = "partial" if sections_skipped or source_artifacts_missing else "completed"
    if not reports_generated:
        report_status = "failed"

    return {
        "run_id": run_id,
        "report_status": report_status,
        "reports_generated": list(reports_generated),
        "source_artifacts_used": list(source_artifacts_used),
        "source_artifacts_missing": list(source_artifacts_missing),
        "sections_generated": sections_generated,
        "sections_skipped": sections_skipped,
        "warnings": all_warnings,
        "created_at": created_at or timestamp,
        "updated_at": timestamp,
    }


def build_report_index(
    run_id: str,
    report_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the report index payload."""

    return {
        "run_id": run_id,
        "reports": [dict(report_file) for report_file in report_files],
    }


def _document(name: str, sections: Sequence[ReportSection]) -> ReportDocument:
    generated = []
    skipped = []
    warnings = []
    markdown = []
    for section in sections:
        markdown.append(section.markdown)
        if section.generated:
            generated.append(section.name)
        else:
            skipped.append(section.name)
            if section.skipped_reason:
                warnings.append(f"{section.title} skipped: {section.skipped_reason}")
        warnings.extend(section.warnings)

    return ReportDocument(
        name=name,
        content="\n\n".join(markdown).rstrip() + "\n",
        sections_generated=_dedupe(generated),
        sections_skipped=_dedupe(skipped),
        warnings=_dedupe(warnings),
    )


def _status_from_sections(sections: Sequence[ReportSection]) -> str:
    return "partial" if any(not section.generated for section in sections) else "completed"


def _artifact(artifacts: Mapping[str, Any], key: str) -> Any:
    value = artifacts.get(key)
    return value if value else None


def _static_heading(name: str, markdown: str) -> ReportSection:
    return ReportSection(name=name, title=name, markdown=markdown)


def _technical_guardrails_section() -> ReportSection:
    markdown = "\n".join(
        [
            "## Technical Guardrails",
            "",
            "- Report generation is deterministic and uses only saved artifacts.",
            "- No paid LLM API calls are made by the deterministic report generator.",
            "- Missing source artifacts are tracked instead of silently ignored.",
            "- Modeling metrics are reported only when saved modeling and evaluation artifacts exist.",
            "- Correlation and feature-signal language is framed as association, not causation.",
        ]
    )
    return ReportSection(
        name="technical_guardrails",
        title="Technical Guardrails",
        markdown=markdown,
    )


def _dedupe(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped
