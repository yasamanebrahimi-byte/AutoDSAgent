from __future__ import annotations

from pathlib import Path

from scripts.run_full_demo import run_demo


def test_full_regression_demo_runs_end_to_end(tmp_path):
    result = run_demo("regression", runs_dir=tmp_path / "runs")

    assert result.workflow_status == "completed"
    assert result.target_column == "sale_price"
    assert result.task_type == "regression"
    assert result.selected_model_name
    assert result.primary_metric == "rmse"
    assert result.primary_metric_value is not None
    assert_expected_artifacts_exist(result.artifacts)


def test_full_classification_demo_runs_end_to_end(tmp_path):
    result = run_demo("classification", runs_dir=tmp_path / "runs")

    assert result.workflow_status == "completed"
    assert result.target_column == "churn"
    assert result.task_type == "classification"
    assert result.selected_model_name
    assert result.primary_metric == "macro_f1"
    assert result.primary_metric_value is not None
    assert_expected_artifacts_exist(result.artifacts)


def assert_expected_artifacts_exist(artifacts: dict[str, Path]) -> None:
    expected = [
        "raw_data",
        "metadata",
        "profile",
        "cleaning_plan",
        "cleaned_data",
        "cleaning_summary",
        "eda_summary",
        "eda_findings",
        "modeling_summary",
        "evaluation_summary",
        "model_results",
        "baseline_model",
        "selected_model",
        "best_model",
        "final_report",
        "executive_summary",
        "technical_summary",
        "limitations",
        "report_metadata",
        "report_index",
        "workflow_state",
        "agent_trace",
    ]
    for name in expected:
        assert artifacts[name].exists(), f"Missing demo artifact: {name}"
