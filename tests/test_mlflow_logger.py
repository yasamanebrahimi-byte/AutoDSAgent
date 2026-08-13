from __future__ import annotations

import importlib
from types import SimpleNamespace

from app.backend.config import load_settings
from app.backend.schemas.modeling import ModelingRequest
from app.backend.services.modeling_service import ModelingService
from app.backend.services.run_manager import RunManager
from app.tools.file_utils import load_json
from app.tools.mlflow_logger import (
    MLflowLogger,
    format_metric_dict,
    format_run_parameters,
)


def test_mlflow_logger_does_nothing_when_disabled(monkeypatch):
    settings = load_settings({"AUTODS_ENABLE_MLFLOW": "false"})

    def fail_import(name):
        raise AssertionError("MLflow should not be imported when disabled.")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    logger = MLflowLogger(settings)

    assert logger.log_modeling_run(
        run_id="disabled",
        request=SimpleNamespace(test_size=0.2, random_state=42),
        modeling_summary={},
        evaluation_summary={},
        model_results={},
        run_root=".",
    ) == []


def test_mlflow_logger_handles_unavailable_mlflow_gracefully(monkeypatch):
    settings = load_settings(
        {
            "AUTODS_ENABLE_MLFLOW": "true",
            "AUTODS_MLFLOW_TRACKING_URI": "http://localhost:5999",
        }
    )

    def missing_mlflow(name):
        if name == "mlflow":
            raise ModuleNotFoundError("No module named 'mlflow'")
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", missing_mlflow)

    logger = MLflowLogger(settings)
    warnings = logger.log_modeling_run(
        run_id="missing",
        request=SimpleNamespace(test_size=0.2, random_state=42),
        modeling_summary={"target_column": "target", "task_type": "classification"},
        evaluation_summary={"best_model_metrics": {"f1": 0.8}},
        model_results={"results": []},
        run_root=".",
    )

    assert warnings
    assert "MLflow logging failed and was skipped" in warnings[0]


def test_mlflow_parameter_and_metric_formatting():
    request = SimpleNamespace(test_size=0.25, random_state=123)
    modeling_summary = {
        "target_column": "churn",
        "task_type": "classification",
        "rows_used": 50,
        "columns_used": 7,
        "features_used": ["a", "b"],
        "features_excluded": ["id"],
        "best_model_name": "logistic_regression",
        "baseline_model_name": "baseline_most_frequent",
        "primary_metric": "f1",
    }

    params = format_run_parameters(request, modeling_summary)
    metrics = format_metric_dict(
        {
            "accuracy": 0.9,
            "precision": "0.85",
            "recall": None,
            "not_numeric": "nope",
        }
    )

    assert params["target_column"] == "churn"
    assert params["num_features_used"] == 2
    assert params["num_features_excluded"] == 1
    assert metrics == {"accuracy": 0.9, "precision": 0.85}


def test_mlflow_logging_failure_does_not_crash_modeling(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "mlflow-failure-modeling"
    paths = manager.create_run(run_id)
    _write_classification_cleaned_data(paths.intermediate / "cleaned_data.csv")

    class ExplodingMLflowLogger:
        def log_modeling_run(self, **kwargs):
            raise RuntimeError("tracking server unavailable")

    service = ModelingService(
        run_manager=manager,
        mlflow_logger=ExplodingMLflowLogger(),
    )

    response = service.train_and_evaluate(
        run_id,
        ModelingRequest(target_column="churn", random_state=42),
    )

    assert response.modeling_summary.best_model_name
    assert any("MLflow logging failed" in warning for warning in response.evaluation_summary.warnings)
    saved_summary = load_json(service.evaluation_service.evaluation_summary_path(run_id))
    assert any("MLflow logging failed" in warning for warning in saved_summary["warnings"])


def _write_classification_cleaned_data(path):
    rows = ["customer_id,feature_a,feature_b,segment,churn"]
    for index in range(1, 65):
        feature_a = ((index * 5) % 19) + (index % 4) * 0.1
        feature_b = index % 6
        segment = ["A", "B", "C", "D"][index % 4]
        churn = "yes" if feature_a > 8.5 or segment == "D" else "no"
        rows.append(f"{index},{feature_a:.2f},{feature_b},{segment},{churn}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
