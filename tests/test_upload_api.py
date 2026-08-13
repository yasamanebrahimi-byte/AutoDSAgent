from fastapi.testclient import TestClient

from app.backend.config import load_settings
from app.backend.main import app
from app.backend.routes import runs as runs_route
from app.backend.routes import upload as upload_route
from app.backend.services.run_manager import RunManager


def test_upload_success_preserves_raw_file_and_lists_run(tmp_path, monkeypatch):
    manager = RunManager(runs_dir=tmp_path)
    monkeypatch.setattr(upload_route, "run_manager", manager)
    monkeypatch.setattr(runs_route, "run_manager", manager)
    client = TestClient(app)

    response = client.post(
        "/upload",
        files={"file": ("sample.csv", b"name,age\nAlice,30\nBob,40\n", "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] == 2
    assert payload["columns"] == 2

    paths = manager.get_paths(payload["run_id"])
    assert (paths.input / "raw_data.csv").read_text(encoding="utf-8") == (
        "name,age\nAlice,30\nBob,40\n"
    )
    assert manager.list_runs()[0]["run_id"] == payload["run_id"]


def test_oversized_upload_is_rejected_without_parsing_or_leaving_run(tmp_path, monkeypatch):
    manager = RunManager(runs_dir=tmp_path)
    monkeypatch.setattr(upload_route, "run_manager", manager)
    monkeypatch.setattr(
        upload_route,
        "settings",
        load_settings(
            {
                "AUTODS_PROJECT_ROOT": str(tmp_path),
                "AUTODS_RUNS_DIR": str(tmp_path),
                "AUTODS_MAX_UPLOAD_SIZE_MB": "1",
            }
        ),
    )
    load_called = False

    def fail_if_called(path):
        nonlocal load_called
        load_called = True
        raise AssertionError(f"oversized upload should not be parsed: {path}")

    monkeypatch.setattr(upload_route, "load_csv", fail_if_called)
    client = TestClient(app)
    oversized_body = b"value\n" + (b"1\n" * (1024 * 1024))

    response = client.post(
        "/upload",
        files={"file": ("big.csv", oversized_body, "text/csv")},
    )

    assert response.status_code == 413
    assert load_called is False
    assert manager.list_runs() == []
    assert [path for path in tmp_path.iterdir() if path.is_dir()] == []


def test_malformed_csv_upload_cleans_up_failed_run(tmp_path, monkeypatch):
    manager = RunManager(runs_dir=tmp_path)
    monkeypatch.setattr(upload_route, "run_manager", manager)
    client = TestClient(app)

    response = client.post(
        "/upload",
        files={"file": ("bad.csv", b'a,b\n"unterminated,2\n', "text/csv")},
    )

    assert response.status_code == 400
    assert manager.list_runs() == []
    assert [path for path in tmp_path.iterdir() if path.is_dir()] == []
