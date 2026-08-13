import json
from pathlib import Path

import pytest

from app.backend.schemas.modeling import ModelingRequest
from app.backend.services.modeling_service import ModelingService
from app.backend.services.run_manager import RunManager
from app.tools.modeling import ModelTrainingResult


def test_regression_modeling_run_completes_and_saves_artifacts(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "regression-modeling-test"
    paths = manager.create_run(run_id)
    _write_regression_cleaned_data(paths.intermediate / "cleaned_data.csv")

    service = ModelingService(run_manager=manager)
    response = service.train_and_evaluate(
        run_id,
        ModelingRequest(target_column="sale_price", random_state=42),
    )

    assert response.modeling_summary.task_type == "regression"
    assert response.modeling_summary.dataset_path == "intermediate/cleaned_data.csv"
    assert not Path(response.modeling_summary.dataset_path).is_absolute()
    assert response.modeling_summary.baseline_model_name == "baseline_median"
    assert response.modeling_summary.best_model_name
    assert "baseline_median" in response.modeling_summary.models_succeeded
    assert _has_successful_candidate(response.model_results)
    assert service.modeling_summary_path(run_id).exists()
    assert service.evaluation_service.evaluation_summary_path(run_id).exists()
    assert service.baseline_model_path(run_id).exists()
    assert service.best_model_path(run_id).exists()
    assert service.model_results_path(run_id).exists()
    assert (paths.plots / "evaluation" / "model_comparison.png").exists()
    assert (paths.plots / "evaluation" / "predicted_vs_actual.png").exists()
    assert (paths.plots / "evaluation" / "residuals.png").exists()


def test_classification_modeling_run_completes_and_saves_artifacts(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "classification-modeling-test"
    paths = manager.create_run(run_id)
    _write_classification_cleaned_data(paths.intermediate / "cleaned_data.csv")

    service = ModelingService(run_manager=manager)
    response = service.train_and_evaluate(
        run_id,
        ModelingRequest(target_column="churn", random_state=42),
    )

    assert response.modeling_summary.task_type == "classification"
    assert response.modeling_summary.baseline_model_name == "baseline_most_frequent"
    assert response.evaluation_summary.primary_metric == "f1"
    assert "baseline_most_frequent" in response.modeling_summary.models_succeeded
    assert _has_successful_candidate(response.model_results)
    assert (paths.plots / "evaluation" / "model_comparison.png").exists()
    assert (paths.plots / "evaluation" / "confusion_matrix.png").exists()


def test_modeling_requires_cleaned_data(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "missing-cleaned-data-test"
    manager.create_run(run_id)

    service = ModelingService(run_manager=manager)

    with pytest.raises(ValueError, match="Cleaned dataset"):
        service.train_and_evaluate(
            run_id,
            ModelingRequest(target_column="target", random_state=42),
        )


def test_failed_models_are_recorded_without_crashing_run(tmp_path, monkeypatch):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "failed-model-recording-test"
    paths = manager.create_run(run_id)
    _write_regression_cleaned_data(paths.intermediate / "cleaned_data.csv")

    from app.backend.services import modeling_service as modeling_service_module

    original_train_models = modeling_service_module.train_models

    def train_with_failure(prepared, random_state=42):
        results = original_train_models(prepared, random_state=random_state)
        results.append(
            ModelTrainingResult(
                model_name="forced_failure",
                role="candidate",
                status="failed",
                error="intentional test failure",
            )
        )
        return results

    monkeypatch.setattr(modeling_service_module, "train_models", train_with_failure)

    service = ModelingService(run_manager=manager)
    response = service.train_and_evaluate(
        run_id,
        ModelingRequest(target_column="sale_price", random_state=42),
    )

    assert "forced_failure" in response.modeling_summary.models_failed
    assert any(
        result["model_name"] == "forced_failure"
        and result["status"] == "failed"
        for result in response.model_results["results"]
    )
    saved_results = json.loads(service.model_results_path(run_id).read_text(encoding="utf-8"))
    assert any(
        result["model_name"] == "forced_failure"
        for result in saved_results["failed_models"]
    )


def _write_regression_cleaned_data(path):
    rows = ["customer_id,feature_a,feature_b,region,sale_price"]
    for index in range(1, 49):
        feature_a = ((index * 7) % 17) + (index % 3) * 0.25
        feature_b = index % 5
        region = ["north", "south", "west"][index % 3]
        sale_price = 1000 + feature_a * 30 + feature_b * 12 + (index % 4)
        rows.append(
            f"{index},{feature_a:.2f},{feature_b},{region},{sale_price:.2f}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_classification_cleaned_data(path):
    rows = ["customer_id,feature_a,feature_b,segment,churn"]
    for index in range(1, 65):
        feature_a = ((index * 5) % 19) + (index % 4) * 0.1
        feature_b = index % 6
        segment = ["A", "B", "C", "D"][index % 4]
        churn = "yes" if feature_a > 8.5 or segment == "D" else "no"
        rows.append(f"{index},{feature_a:.2f},{feature_b},{segment},{churn}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _has_successful_candidate(model_results: dict) -> bool:
    return any(
        result["role"] == "candidate" and result["status"] == "succeeded"
        for result in model_results["results"]
    )
