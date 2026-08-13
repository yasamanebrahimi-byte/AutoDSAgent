from app.tools.report_builder import (
    build_executive_summary,
    build_final_report,
    build_limitations_report,
    build_report_metadata,
    build_technical_summary,
)

from tests.report_test_utils import (
    cleaning_plan_payload,
    cleaning_summary_payload,
    eda_findings_payload,
    eda_summary_payload,
    evaluation_summary_payload,
    metadata_payload,
    model_results_payload,
    modeling_summary_payload,
    profile_payload,
)


def test_final_report_includes_sections_for_available_artifacts():
    run_id = "builder-test"
    artifacts = _full_artifacts(run_id)

    report = build_final_report(
        run_id=run_id,
        artifacts=artifacts,
        source_artifacts_used=["intermediate/metadata.json"],
        source_artifacts_missing=[],
    )

    assert "# AutoDS Agent Final Analysis Report" in report.content
    assert "## Dataset Overview" in report.content
    assert "## Cleaning Methodology" in report.content
    assert "## Exploratory Data Analysis" in report.content
    assert "## Modeling Methodology" in report.content
    assert "## Evaluation Results" in report.content
    assert "## Best Model Summary" in report.content
    assert "dataset_overview" in report.sections_generated
    assert "modeling_methodology" in report.sections_generated
    assert "Correlations, feature importance, and model signals do not prove causation" in report.content


def test_final_report_marks_modeling_skipped_when_modeling_artifacts_are_missing():
    run_id = "builder-missing-modeling"
    artifacts = _full_artifacts(run_id)
    artifacts.pop("modeling_summary")
    artifacts.pop("evaluation_summary")
    artifacts.pop("model_results")

    report = build_final_report(
        run_id=run_id,
        artifacts=artifacts,
        source_artifacts_used=["intermediate/metadata.json"],
        source_artifacts_missing=[
            "intermediate/modeling_summary.json",
            "intermediate/evaluation_summary.json",
            "models/model_results.json",
        ],
    )

    assert "Modeling may have been skipped" in report.content
    assert "modeling_methodology" in report.sections_skipped
    assert "evaluation_results" in report.sections_skipped
    assert "best_model" in report.sections_skipped


def test_standalone_reports_and_metadata_are_generated():
    run_id = "builder-standalone"
    artifacts = _full_artifacts(run_id)

    executive = build_executive_summary(run_id, artifacts)
    technical = build_technical_summary(
        run_id,
        artifacts,
        source_artifacts_used=["intermediate/profile.json"],
        source_artifacts_missing=[],
    )
    limitations = build_limitations_report(run_id, artifacts, source_artifacts_missing=[])
    partial = build_final_report(
        run_id,
        artifacts={
            key: value
            for key, value in artifacts.items()
            if key not in {"eda_summary", "eda_findings"}
        },
        source_artifacts_used=["intermediate/profile.json"],
        source_artifacts_missing=[
            "intermediate/eda_summary.json",
            "intermediate/eda_findings.json",
        ],
    )

    metadata = build_report_metadata(
        run_id=run_id,
        documents=[executive, technical, limitations, partial],
        reports_generated=["reports/final_report.md"],
        source_artifacts_used=["intermediate/profile.json"],
        source_artifacts_missing=[
            "intermediate/eda_summary.json",
            "intermediate/eda_findings.json",
        ],
    )

    assert "# AutoDS Agent Executive Summary" in executive.content
    assert "# AutoDS Agent Technical Methodology Summary" in technical.content
    assert "# AutoDS Agent Limitations And Next Steps" in limitations.content
    assert metadata["report_status"] == "partial"
    assert "eda_findings" in metadata["sections_skipped"]


def _full_artifacts(run_id: str) -> dict:
    return {
        "metadata": metadata_payload(run_id),
        "profile": profile_payload(run_id),
        "cleaning_plan": cleaning_plan_payload(run_id),
        "cleaning_summary": cleaning_summary_payload(run_id),
        "eda_summary": eda_summary_payload(run_id),
        "eda_findings": eda_findings_payload(),
        "modeling_summary": modeling_summary_payload(run_id),
        "evaluation_summary": evaluation_summary_payload(run_id),
        "model_results": model_results_payload(run_id),
    }
