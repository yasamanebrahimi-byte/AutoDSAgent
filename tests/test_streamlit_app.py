from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

import requests
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "frontend" / "streamlit_app.py"

METADATA = {
    "run_id": "ui-test-run",
    "filename": "customers.csv",
    "rows": 3,
    "columns": 3,
    "column_names": ["age", "spend", "churn"],
    "dtypes": {"age": "int64", "spend": "float64", "churn": "object"},
    "missing_values": {"age": 0, "spend": 0, "churn": 0},
    "duplicate_rows": 0,
    "preview": [
        {"age": 25, "spend": 120.0, "churn": "no"},
        {"age": 40, "spend": 80.0, "churn": "yes"},
    ],
}

PROFILE = {
    "rows": 3,
    "columns": 3,
    "total_missing_values": 0,
    "duplicate_rows": 0,
    "column_type_counts": {"numeric": 2, "categorical": 1},
    "data_quality_issues": [],
    "column_profiles": [],
}

CLEANING_PLAN = {
    "duplicate_row_handling": {"reason": "No duplicate rows were detected."},
    "missing_value_strategies": [],
    "columns_recommended_for_dropping": [],
    "type_conversion_recommendations": [],
    "warnings_requiring_review": [],
}

REPORTS_RESPONSE = {
    "metadata": {
        "report_status": "completed",
        "reports_generated": [
            "executive_summary",
            "final_report",
            "technical_summary",
            "limitations",
        ],
        "source_artifacts_used": ["workflow_state"],
        "source_artifacts_missing": [],
        "sections_generated": ["overview"],
        "sections_skipped": [],
        "warnings": [],
    },
    "index": {"reports": []},
}


class FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = "" if payload is None else str(payload)

    def json(self):
        return deepcopy(self._payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.upload_failures = 0
        self.job_statuses = ["running", "running", "completed"]
        self.workflow_statuses = ["running", "running", "completed"]
        self.reports_response: dict | None = None

    def post(self, url: str, **kwargs):
        path = urlsplit(url).path
        self.calls.append(("POST", path, kwargs))
        if path == "/upload":
            if self.upload_failures:
                self.upload_failures -= 1
                raise requests.ConnectionError("backend unavailable")
            return FakeResponse(METADATA)
        if path.endswith("/workflow/start"):
            return FakeResponse(
                {
                    "job_id": "job-1",
                    "run_id": METADATA["run_id"],
                    "status": "queued",
                    "submitted_at": "2026-01-01T00:00:00+00:00",
                    "started_at": None,
                    "completed_at": None,
                    "error": None,
                    "status_url": f"/runs/{METADATA['run_id']}/workflow/jobs/job-1",
                    "state_url": f"/runs/{METADATA['run_id']}/workflow/state",
                }
            )
        if path.endswith("/profile"):
            return FakeResponse(PROFILE)
        if path.endswith("/cleaning-plan"):
            return FakeResponse(CLEANING_PLAN)
        return FakeResponse({"detail": "Not found"}, status_code=404)

    def get(self, url: str, **kwargs):
        path = urlsplit(url).path
        self.calls.append(("GET", path, kwargs))
        if path == "/config/status":
            return FakeResponse(
                {
                    "environment": "test",
                    "mlflow_enabled": False,
                    "runs_dir": "runs",
                }
            )
        if "/workflow/jobs/" in path:
            status = self._next(self.job_statuses)
            return FakeResponse(
                {
                    "job_id": "job-1",
                    "run_id": METADATA["run_id"],
                    "status": status,
                    "submitted_at": "2026-01-01T00:00:00+00:00",
                    "started_at": "2026-01-01T00:00:01+00:00",
                    "completed_at": (
                        "2026-01-01T00:00:02+00:00" if status == "completed" else None
                    ),
                    "error": None,
                    "status_url": path,
                    "state_url": f"/runs/{METADATA['run_id']}/workflow/state",
                }
            )
        if path.endswith("/workflow/state"):
            return FakeResponse(self._workflow_state(self._next(self.workflow_statuses)))
        if path.endswith("/workflow/trace"):
            return FakeResponse([])
        if path.endswith("/reports"):
            if self.reports_response is None:
                return FakeResponse({"detail": "Not found"}, status_code=404)
            return FakeResponse(self.reports_response)
        if "/reports/" in path:
            report_name = path.rsplit("/", 1)[-1]
            return FakeResponse({"report_name": report_name, "content": f"# {report_name}"})
        return FakeResponse({"detail": "Not found"}, status_code=404)

    def paths(self, method: str) -> list[str]:
        return [path for call_method, path, _ in self.calls if call_method == method]

    @staticmethod
    def _next(values: list[str]) -> str:
        return values.pop(0) if len(values) > 1 else values[0]

    @staticmethod
    def _workflow_state(status: str) -> dict:
        return {
            "run_id": METADATA["run_id"],
            "status": status,
            "current_step": None if status == "completed" else "eda",
            "target_column": "churn",
            "steps": {},
            "artifacts": {},
        }


def test_upload_creates_an_active_run(monkeypatch):
    backend = _install_backend(monkeypatch)
    at = AppTest.from_file(APP_PATH).run()

    at.file_uploader[0].upload("customers.csv", b"age,spend,churn\n25,120,no\n").run()
    _button(at, "Create Analysis Run").click().run()

    assert not at.exception
    assert at.session_state["metadata"]["run_id"] == METADATA["run_id"]
    upload_call = next(call for call in backend.calls if call[1] == "/upload")
    assert upload_call[2]["files"]["file"][0] == "customers.csv"
    assert upload_call[2]["timeout"] == 60


def test_workflow_start_polls_until_completed(monkeypatch):
    backend = _install_backend(monkeypatch)
    at = _app_with_metadata()

    _button(at, "Start Automated Workflow").click().run()
    assert at.session_state["workflow_job"]["status"] == "running"
    at.run()

    assert not at.exception
    assert at.session_state["workflow_job"]["status"] == "completed"
    assert at.session_state["workflow_state"]["status"] == "completed"
    start_call = next(call for call in backend.calls if call[1].endswith("/workflow/start"))
    assert start_call[2]["timeout"] == 15
    assert len([path for path in backend.paths("GET") if "/workflow/jobs/" in path]) >= 3


def test_advanced_manual_controls_call_profile_and_cleaning_plan(monkeypatch):
    backend = _install_backend(monkeypatch)
    at = _app_with_metadata()

    _button(at, "Generate Dataset Profile").click().run()
    _button(at, "Generate Cleaning Plan").click().run()

    assert not at.exception
    assert at.session_state["profile"] == PROFILE
    assert at.session_state["cleaning_plan"] == CLEANING_PLAN
    assert f"/runs/{METADATA['run_id']}/profile" in backend.paths("POST")
    assert f"/runs/{METADATA['run_id']}/cleaning-plan" in backend.paths("POST")


def test_generated_reports_expose_all_markdown_downloads(monkeypatch):
    backend = _install_backend(monkeypatch)
    backend.reports_response = REPORTS_RESPONSE
    at = _app_with_metadata(reports_response=REPORTS_RESPONSE)

    labels = [button.label for button in at.download_button]
    assert labels == [
        "Download Executive Summary",
        "Download Final Report",
        "Download Technical Summary",
        "Download Limitations",
    ]
    assert all(button.proto.url.endswith(".md") for button in at.download_button)
    assert not at.exception


def test_upload_failure_is_visible_and_retry_recovers(monkeypatch):
    backend = _install_backend(monkeypatch)
    backend.upload_failures = 1
    at = AppTest.from_file(APP_PATH).run()
    at.file_uploader[0].upload("customers.csv", b"age,spend,churn\n25,120,no\n").run()

    _button(at, "Create Analysis Run").click().run()
    assert any("Could not connect to the backend" in error.value for error in at.error)
    assert "metadata" not in at.session_state

    _button(at, "Create Analysis Run").click().run()
    assert at.session_state["metadata"]["run_id"] == METADATA["run_id"]
    assert not at.exception


def _install_backend(monkeypatch) -> FakeBackend:
    backend = FakeBackend()
    monkeypatch.setattr(requests, "post", backend.post)
    monkeypatch.setattr(requests, "get", backend.get)
    return backend


def _app_with_metadata(reports_response: dict | None = None) -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.session_state["metadata"] = deepcopy(METADATA)
    if reports_response is not None:
        at.session_state["reports_response"] = deepcopy(reports_response)
    return at.run()


def _button(at: AppTest, label: str):
    return next(button for button in at.button if button.label == label)
