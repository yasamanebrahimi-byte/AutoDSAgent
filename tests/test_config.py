from pathlib import Path

from fastapi.testclient import TestClient

from app.backend.config import load_settings
from app.backend.main import app


def test_config_defaults_load_correctly():
    settings = load_settings({})

    assert settings.environment == "local"
    assert settings.runs_dir == settings.project_root / "runs"
    assert settings.backend_url == "http://localhost:8000"
    assert settings.mlflow_enabled is False
    assert settings.mlflow_tracking_uri == "http://localhost:5000"
    assert settings.mlflow_experiment_name == "AutoDS-Agent"
    assert settings.default_test_size == 0.2
    assert settings.default_random_seed == 42


def test_config_environment_overrides_work(tmp_path):
    settings = load_settings(
        {
            "AUTODS_PROJECT_ROOT": str(tmp_path),
            "AUTODS_RUNS_DIR": "custom_runs",
            "AUTODS_BACKEND_URL": "http://backend:8000/",
            "AUTODS_ENV": "test",
            "AUTODS_LOG_LEVEL": "debug",
            "AUTODS_ENABLE_MLFLOW": "true",
            "AUTODS_MLFLOW_TRACKING_URI": "file:///tmp/mlruns",
            "AUTODS_MLFLOW_EXPERIMENT_NAME": "Demo",
            "AUTODS_DEFAULT_TEST_SIZE": "0.3",
            "AUTODS_DEFAULT_RANDOM_SEED": "123",
        }
    )

    assert settings.project_root == tmp_path.resolve()
    assert settings.runs_dir == (tmp_path / "custom_runs").resolve()
    assert settings.backend_url == "http://backend:8000"
    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.mlflow_enabled is True
    assert settings.mlflow_tracking_uri == "file:///tmp/mlruns"
    assert settings.mlflow_experiment_name == "Demo"
    assert settings.default_test_size == 0.3
    assert settings.default_random_seed == 123


def test_runs_directory_can_be_absolute(tmp_path):
    runs_dir = tmp_path / "runs"
    settings = load_settings({"AUTODS_RUNS_DIR": str(runs_dir)})

    assert settings.runs_dir == runs_dir.resolve()


def test_config_status_endpoint_exposes_non_secret_status():
    client = TestClient(app)

    response = client.get("/config/status")

    assert response.status_code == 200
    payload = response.json()
    assert "environment" in payload
    assert "runs_dir" in payload
    assert "mlflow_enabled" in payload
    assert "mlflow_tracking_uri" in payload
    assert "password" not in payload
    assert payload["project_root"] == "."
    assert not Path(payload["project_root"]).is_absolute()
    assert not Path(payload["runs_dir"]).is_absolute()
