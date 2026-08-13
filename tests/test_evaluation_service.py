import math

import numpy as np
import pytest

from app.tools.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    create_confusion_matrix_plot,
    create_model_comparison_plot,
    create_predicted_vs_actual_plot,
    create_residuals_plot,
)
from app.tools.modeling import ModelTrainingResult


def test_regression_metrics_are_computed_correctly():
    metrics = compute_regression_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 5.0]),
    )

    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(math.sqrt(4 / 3))
    assert metrics["r2"] == pytest.approx(-1.0)


def test_classification_metrics_are_computed_correctly():
    metrics = compute_classification_metrics(
        np.array(["yes", "no", "yes", "no"]),
        np.array(["yes", "yes", "yes", "no"]),
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(5 / 6)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(0.7333333333)


def test_model_comparison_plot_is_created(tmp_path):
    results = [
        ModelTrainingResult(
            model_name="baseline_median",
            role="baseline",
            status="succeeded",
            metrics={"rmse": 4.0},
            primary_metric_value=4.0,
        ),
        ModelTrainingResult(
            model_name="random_forest",
            role="candidate",
            status="succeeded",
            metrics={"rmse": 2.0},
            primary_metric_value=2.0,
        ),
    ]

    path = create_model_comparison_plot(
        results=results,
        task_type="regression",
        output_dir=tmp_path,
        best_model_name="random_forest",
    )

    assert path is not None
    assert path.exists()


def test_regression_plots_are_created(tmp_path):
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    predicted_path = create_predicted_vs_actual_plot(y_true, y_pred, tmp_path)
    residual_path = create_residuals_plot(y_true, y_pred, tmp_path)

    assert predicted_path is not None
    assert predicted_path.exists()
    assert residual_path is not None
    assert residual_path.exists()


def test_confusion_matrix_plot_is_created(tmp_path):
    path = create_confusion_matrix_plot(
        np.array(["yes", "no", "yes", "no"]),
        np.array(["yes", "yes", "yes", "no"]),
        tmp_path,
    )

    assert path is not None
    assert path.exists()
