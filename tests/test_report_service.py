from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.routes import reports as reports_route
from app.backend.schemas.reports import ReportGenerateRequest
from app.backend.services.report_service import ReportService

from tests.report_test_utils import create_report_run


def test_report_source_loading_tracks_available_and_missing_artifacts(tmp_path):
    manager, _, run_id = create_report_run(
        tmp_path,
        include_eda=False,
        include_modeling=False,
    )
    service = ReportService(manager)

    loaded = service.load_source_artifacts(run_id)

    assert "metadata" in loaded.artifacts
    assert "profile" in loaded.artifacts
    assert "intermediate/metadata.json" in loaded.used
    assert "intermediate/eda_summary.json" in loaded.missing
    assert "intermediate/modeling_summary.json" in loaded.missing


def test_report_service_saves_artifacts_and_retrieves_content(tmp_path):
    manager, paths, run_id = create_report_run(tmp_path)
    service = ReportService(manager)

    response = service.generate_reports(
        run_id,
        ReportGenerateRequest(include_html=True, force_regenerate=True),
    )

    assert response.metadata.report_status == "completed"
    assert (paths.reports / "final_report.md").exists()
    assert (paths.reports / "executive_summary.md").exists()
    assert (paths.reports / "technical_summary.md").exists()
    assert (paths.reports / "limitations.md").exists()
    assert (paths.reports / "report_index.json").exists()
    assert (paths.reports / "final_report.html").exists()
    assert (paths.intermediate / "report_metadata.json").exists()

    content = service.get_report_content(run_id, "final_report")
    assert content.path == "reports/final_report.md"
    assert "AutoDS Agent Final Analysis Report" in content.content

    loaded = service.load_reports(run_id)
    assert loaded.index.reports


def test_report_generation_handles_missing_modeling_artifacts(tmp_path):
    manager, _, run_id = create_report_run(tmp_path, include_modeling=False)
    service = ReportService(manager)

    response = service.generate_reports(run_id)

    assert response.metadata.report_status == "partial"
    assert "modeling_methodology" in response.metadata.sections_skipped
    final_report = service.get_report_content(run_id, "final_report").content
    assert "Modeling may have been skipped" in final_report


def test_report_generation_handles_missing_eda_artifacts_with_basic_inputs(tmp_path):
    manager, _, run_id = create_report_run(
        tmp_path,
        include_eda=False,
        include_modeling=False,
    )
    service = ReportService(manager)

    response = service.generate_reports(run_id)

    assert response.metadata.report_status == "partial"
    assert "eda_findings" in response.metadata.sections_skipped
    assert "dataset_overview" in response.metadata.sections_generated


def test_report_api_generate_index_content_and_download_routes(tmp_path, monkeypatch):
    manager, _, run_id = create_report_run(tmp_path, include_modeling=False)
    monkeypatch.setattr(reports_route, "report_service", ReportService(manager))
    client = TestClient(app)

    generate_response = client.post(
        f"/runs/{run_id}/reports/generate",
        json={"include_html": False, "force_regenerate": True},
    )
    assert generate_response.status_code == 200
    assert generate_response.json()["metadata"]["report_status"] == "partial"

    index_response = client.get(f"/runs/{run_id}/reports")
    assert index_response.status_code == 200
    assert index_response.json()["index"]["reports"]

    content_response = client.get(f"/runs/{run_id}/reports/final_report")
    assert content_response.status_code == 200
    assert "AutoDS Agent Final Analysis Report" in content_response.json()["content"]

    download_response = client.get(f"/runs/{run_id}/reports/download/final_report")
    assert download_response.status_code == 200
    assert download_response.headers["content-type"].startswith("text/markdown")
