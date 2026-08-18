"""Evaluation metrics and plots for deterministic model comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from app.tools.file_utils import ensure_directory
from app.tools.modeling import ModelTrainingResult
from app.tools.preprocessing import (
    PreprocessingResult,
    TaskType,
    get_transformed_feature_names,
)


PRIMARY_METRIC_BY_TASK: dict[TaskType, str] = {
    "regression": "rmse",
    "classification": "macro_f1",
}
SelectionDirection = Literal["lower", "higher"]
SELECTION_DIRECTION_BY_TASK: dict[TaskType, SelectionDirection] = {
    "regression": "lower",
    "classification": "higher",
}
SELECTION_TIEBREAKER = "original_result_order"


@dataclass(frozen=True)
class ModelSelectionResult:
    """Unambiguous CV-based model-selection result."""

    selected_result: ModelTrainingResult
    best_candidate_result: ModelTrainingResult | None
    baseline_result: ModelTrainingResult | None
    selection_metric: str
    selection_direction: SelectionDirection
    selection_tiebreaker: str
    candidate_beats_baseline: bool | None
    selection_outcome: str

    @property
    def best_result(self) -> ModelTrainingResult:
        """Backward-compatible alias for the selected overall model."""

        return self.selected_result


def evaluate_holdout_model(
    result: ModelTrainingResult,
    prepared: PreprocessingResult,
) -> ModelTrainingResult:
    """Evaluate the selected model once on the untouched holdout partition."""

    if result.status != "succeeded" or result.estimator is None:
        raise RuntimeError("The selected model is unavailable for holdout evaluation.")

    try:
        predictions = result.estimator.predict(prepared.X_test)
        result.holdout_predictions = predictions
        if prepared.task_type == "regression":
            metrics = compute_regression_metrics(prepared.y_test, predictions)
        else:
            probabilities, classes = _predict_probabilities(result.estimator, prepared.X_test)
            result.holdout_probabilities = probabilities
            metrics = compute_classification_metrics(
                prepared.y_test,
                predictions,
                probabilities,
                classes,
            )
        result.holdout_metrics = metrics
    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)
        result.holdout_metrics = {}
        raise

    return result


def refit_model_on_training_partition(
    result: ModelTrainingResult,
    prepared: PreprocessingResult,
) -> ModelTrainingResult:
    """Fit the selected estimator on the full training partition before testing."""

    if result.status != "succeeded" or result.estimator is None:
        raise RuntimeError("The selected model is unavailable for final training.")

    try:
        result.estimator.fit(prepared.X_train, prepared.y_train)
        result.holdout_metrics = {}
        result.holdout_predictions = None
        result.holdout_probabilities = None
    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)
        raise

    return result


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float | None]:
    """Compute regression metrics."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else None
    return {
        "mae": _finite_or_none(mae),
        "rmse": _finite_or_none(rmse),
        "r2": _finite_or_none(r2),
    }


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: np.ndarray | None = None,
    classes: np.ndarray | list[object] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics with weighted, macro, and per-class detail."""

    y_true_labels = np.asarray([str(value) for value in y_true])
    y_pred_labels = np.asarray([str(value) for value in y_pred])
    labels = sorted(set(y_true_labels) | set(y_pred_labels), key=str)
    precision_values, recall_values, f1_values, support_values = (
        precision_recall_fscore_support(
            y_true_labels,
            y_pred_labels,
            labels=labels,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
    weighted_precision = _finite_or_none(
        precision_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
    )
    weighted_recall = _finite_or_none(
        recall_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
    )
    weighted_f1 = _finite_or_none(
        f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
    )

    metrics: dict[str, Any] = {
        "accuracy": _finite_or_none(accuracy_score(y_true_labels, y_pred_labels)),
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_f1": _finite_or_none(
            f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": _finite_or_none(
            balanced_accuracy_score(y_true_labels, y_pred_labels)
        ),
        "per_class": {
            label: {
                "precision": _finite_or_none(float(precision_values[index])),
                "recall": _finite_or_none(float(recall_values[index])),
                "f1": _finite_or_none(float(f1_values[index])),
                "support": int(support_values[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_labels": labels,
    }

    if y_proba is not None and len(labels) == 2 and y_proba.ndim == 2:
        model_classes = [str(value) for value in classes] if classes is not None else labels
        positive_label = model_classes[-1] if len(model_classes) == 2 else labels[-1]
        try:
            positive_index = model_classes.index(positive_label)
            y_binary = (y_true_labels == positive_label).astype(int)
            if len(set(y_binary.tolist())) == 2:
                positive_scores = y_proba[:, positive_index]
                metrics["roc_auc"] = _finite_or_none(
                    roc_auc_score(y_binary, positive_scores)
                )
                metrics["average_precision"] = _finite_or_none(
                    average_precision_score(y_binary, positive_scores)
                )
        except Exception:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None

    return metrics


def select_best_model(
    results: list[ModelTrainingResult],
    task_type: TaskType,
) -> ModelTrainingResult:
    """Select the best successful model overall using CV scores only.

    Ties are resolved by the original result order so final holdout metrics are
    never needed for model selection.
    """

    return select_model_results(results, task_type).selected_result


def select_best_candidate(
    results: list[ModelTrainingResult],
    task_type: TaskType,
) -> ModelTrainingResult | None:
    """Select the strongest non-baseline candidate using CV scores only."""

    successful_results = _successful_selection_results(results)
    candidate_results = [result for result in successful_results if result.role == "candidate"]
    if not candidate_results:
        return None

    lower_is_better = SELECTION_DIRECTION_BY_TASK[task_type] == "lower"
    return _select_by_ordered_score(candidate_results, lower_is_better=lower_is_better)


def select_model_results(
    results: list[ModelTrainingResult],
    task_type: TaskType,
) -> ModelSelectionResult:
    """Select the overall model and retain the strongest trainable candidate."""

    successful_results = _successful_selection_results(results)
    if not successful_results:
        raise RuntimeError("No models completed successfully.")

    lower_is_better = SELECTION_DIRECTION_BY_TASK[task_type] == "lower"
    baseline_result = next((result for result in results if result.role == "baseline"), None)
    valid_baseline_result = (
        baseline_result if baseline_result is not None and _is_selection_ready(baseline_result) else None
    )
    best_candidate_result = select_best_candidate(results, task_type)
    selected_result = _select_by_ordered_score(
        successful_results,
        lower_is_better=lower_is_better,
    )

    candidate_beats_baseline = _candidate_beats_baseline(
        best_candidate_result,
        valid_baseline_result,
        task_type,
    )
    return ModelSelectionResult(
        selected_result=selected_result,
        best_candidate_result=best_candidate_result,
        baseline_result=baseline_result,
        selection_metric=PRIMARY_METRIC_BY_TASK[task_type],
        selection_direction=SELECTION_DIRECTION_BY_TASK[task_type],
        selection_tiebreaker=SELECTION_TIEBREAKER,
        candidate_beats_baseline=candidate_beats_baseline,
        selection_outcome=_selection_outcome(
            selected_result=selected_result,
            best_candidate_result=best_candidate_result,
            baseline_result=baseline_result,
            valid_baseline_result=valid_baseline_result,
            candidate_beats_baseline=candidate_beats_baseline,
            task_type=task_type,
        ),
    )


def _successful_selection_results(
    results: list[ModelTrainingResult],
) -> list[ModelTrainingResult]:
    return [result for result in results if _is_selection_ready(result)]


def _is_selection_ready(result: ModelTrainingResult) -> bool:
    return result.status == "succeeded" and _metric_value(result) is not None


def _metric_value(result: ModelTrainingResult | None) -> float | None:
    if result is None or result.primary_metric_value is None:
        return None
    try:
        value = float(result.primary_metric_value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _select_by_ordered_score(
    results: list[ModelTrainingResult],
    lower_is_better: bool,
) -> ModelTrainingResult:
    best_result = results[0]
    best_value = _metric_value(best_result)
    if best_value is None:
        raise RuntimeError("No models completed successfully.")
    for result in results[1:]:
        value = _metric_value(result)
        if value is None:
            continue
        is_better = value < best_value if lower_is_better else value > best_value
        if is_better:
            best_result = result
            best_value = value
    return best_result


def _candidate_beats_baseline(
    best_candidate_result: ModelTrainingResult | None,
    baseline_result: ModelTrainingResult | None,
    task_type: TaskType,
) -> bool | None:
    candidate_value = _metric_value(best_candidate_result)
    baseline_value = _metric_value(baseline_result)
    if candidate_value is None or baseline_value is None:
        return None
    if task_type == "regression":
        return candidate_value < baseline_value
    return candidate_value > baseline_value


def _metric_values_tie(
    left: ModelTrainingResult | None,
    right: ModelTrainingResult | None,
) -> bool:
    left_value = _metric_value(left)
    right_value = _metric_value(right)
    if left_value is None or right_value is None:
        return False
    return abs(left_value - right_value) < 1e-12


def _selection_outcome(
    *,
    selected_result: ModelTrainingResult,
    best_candidate_result: ModelTrainingResult | None,
    baseline_result: ModelTrainingResult | None,
    valid_baseline_result: ModelTrainingResult | None,
    candidate_beats_baseline: bool | None,
    task_type: TaskType,
) -> str:
    if selected_result.role == "baseline":
        if best_candidate_result is None:
            return "baseline_selected_no_successful_candidates"
        if _metric_values_tie(best_candidate_result, valid_baseline_result):
            return "baseline_selected_tie"
        return "baseline_selected_baseline_outperformed_candidates"

    if baseline_result is None:
        return "candidate_selected_no_baseline_attempted"
    if valid_baseline_result is None:
        return "candidate_selected_baseline_unavailable"
    if candidate_beats_baseline:
        return "candidate_selected_candidate_outperformed_baseline"
    if _metric_values_tie(best_candidate_result, valid_baseline_result):
        return "candidate_selected_tie"
    if task_type == "regression":
        return "candidate_selected_lower_cv_score"
    return "candidate_selected_higher_cv_score"


def baseline_comparison(
    baseline_result: ModelTrainingResult | None,
    task_type: TaskType,
    best_candidate_result: ModelTrainingResult | None = None,
    selected_result: ModelTrainingResult | None = None,
) -> dict[str, Any]:
    """Compare the strongest candidate against the baseline using CV metrics."""

    primary_metric = PRIMARY_METRIC_BY_TASK[task_type]
    selection_direction = SELECTION_DIRECTION_BY_TASK[task_type]
    baseline_value = _metric_value(baseline_result)
    candidate_value = _metric_value(best_candidate_result)
    selected_value = _metric_value(selected_result or best_candidate_result)

    if baseline_value is None:
        return {
            "baseline_metric_value": None,
            "best_candidate_metric_value": candidate_value,
            "selected_metric_value": selected_value,
            "candidate_beats_baseline": None,
            "candidate_tied_baseline": None,
            "absolute_improvement": None,
            "percent_improvement": None,
            "interpretation": "Baseline comparison is unavailable because baseline metrics were not produced.",
        }

    if candidate_value is None:
        return {
            "baseline_metric_value": _finite_or_none(baseline_value),
            "best_candidate_metric_value": None,
            "selected_metric_value": selected_value,
            "candidate_beats_baseline": None,
            "candidate_tied_baseline": None,
            "absolute_improvement": None,
            "percent_improvement": None,
            "interpretation": "No successful candidate produced a valid CV metric; the baseline was selected.",
        }

    if task_type == "regression":
        absolute_improvement = baseline_value - candidate_value
    else:
        absolute_improvement = candidate_value - baseline_value

    percent_improvement = None
    if baseline_value != 0:
        percent_improvement = (absolute_improvement / abs(baseline_value)) * 100

    candidate_tied_baseline = abs(absolute_improvement) < 1e-12
    candidate_beats_baseline = absolute_improvement > 0 and not candidate_tied_baseline
    selected_name = (selected_result or best_candidate_result).model_name
    candidate_name = best_candidate_result.model_name

    if candidate_tied_baseline:
        interpretation = (
            f"The best candidate `{candidate_name}` tied the baseline on mean CV "
            f"{primary_metric}; {SELECTION_TIEBREAKER} selected `{selected_name}`."
        )
    elif candidate_beats_baseline:
        interpretation = (
            f"The best candidate `{candidate_name}` improved on the baseline with a "
            f"{selection_direction} mean CV {primary_metric} and was selected."
        )
    else:
        interpretation = (
            "The baseline outperformed all candidate models during cross-validation "
            "and was selected."
        )

    return {
        "baseline_metric_value": _finite_or_none(baseline_value),
        "best_candidate_metric_value": _finite_or_none(candidate_value),
        "selected_metric_value": _finite_or_none(selected_value),
        "candidate_beats_baseline": candidate_beats_baseline,
        "candidate_tied_baseline": candidate_tied_baseline,
        "absolute_improvement": _finite_or_none(absolute_improvement),
        "percent_improvement": _finite_or_none(percent_improvement),
        "interpretation": interpretation,
    }


def create_evaluation_plots(
    results: list[ModelTrainingResult],
    prepared: PreprocessingResult,
    selected_result: ModelTrainingResult,
    run_root: Path,
    plots_dir: Path,
) -> list[dict[str, str]]:
    """Create task-appropriate evaluation plots and return plot metadata."""

    output_dir = ensure_directory(plots_dir / "evaluation")
    generated: list[dict[str, str]] = []

    comparison_path = create_model_comparison_plot(
        results=results,
        task_type=prepared.task_type,
        output_dir=output_dir,
        selected_model_name=selected_result.model_name,
    )
    _append_plot(
        generated,
        comparison_path,
        run_root,
        "CV Model Comparison",
        "evaluation_model_comparison",
    )

    if selected_result.estimator is None:
        return generated

    predictions = (
        selected_result.holdout_predictions
        if selected_result.holdout_predictions is not None
        else selected_result.estimator.predict(prepared.X_test)
    )

    if prepared.task_type == "regression":
        _append_plot(
            generated,
            create_predicted_vs_actual_plot(prepared.y_test, predictions, output_dir),
            run_root,
            "Predicted vs Actual",
            "evaluation_predicted_vs_actual",
        )
        _append_plot(
            generated,
            create_residuals_plot(prepared.y_test, predictions, output_dir),
            run_root,
            "Residuals",
            "evaluation_residuals",
        )
    else:
        _append_plot(
            generated,
            create_confusion_matrix_plot(prepared.y_test, predictions, output_dir),
            run_root,
            "Confusion Matrix",
            "evaluation_confusion_matrix",
        )

    _append_plot(
        generated,
        create_feature_importance_plot(selected_result, output_dir),
        run_root,
        "Feature Signal",
        "evaluation_feature_importance",
    )

    return generated


def create_model_comparison_plot(
    results: list[ModelTrainingResult],
    task_type: TaskType,
    output_dir: str | Path,
    best_model_name: str | None = None,
    selected_model_name: str | None = None,
) -> Path | None:
    """Create a bar chart comparing the primary metric across models."""

    primary_metric = PRIMARY_METRIC_BY_TASK[task_type]
    successful_results = _successful_selection_results(results)
    if not successful_results:
        return None

    highlighted_model_name = selected_model_name or best_model_name
    output_path = ensure_directory(output_dir) / "model_comparison.png"
    labels = [result.model_name.replace("_", " ").title() for result in successful_results]
    values = [float(result.primary_metric_value) for result in successful_results]
    colors = [
        "#59A14F" if result.model_name == highlighted_model_name else "#9C9C9C"
        if result.role == "baseline"
        else "#4C78A8"
        for result in successful_results
    ]
    direction = "Lower is better" if task_type == "regression" else "Higher is better"

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.35), 5))
    ax.bar(labels, values, color=colors)
    ax.set_title(f"CV Model Comparison by {primary_metric.upper()} ({direction})")
    ax.set_xlabel("Model")
    ax.set_ylabel(f"Mean CV {primary_metric.upper()}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_predicted_vs_actual_plot(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    output_dir: str | Path,
) -> Path | None:
    """Create a predicted-vs-actual regression plot."""

    if len(y_true) == 0:
        return None

    output_path = ensure_directory(output_dir) / "predicted_vs_actual.png"
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    minimum = float(min(np.min(y_true_array), np.min(y_pred_array)))
    maximum = float(max(np.max(y_true_array), np.max(y_pred_array)))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(y_true_array, y_pred_array, alpha=0.75, color="#4C78A8")
    ax.plot([minimum, maximum], [minimum, maximum], color="#E15759", linewidth=1.5)
    ax.set_title("Predicted vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_residuals_plot(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    output_dir: str | Path,
) -> Path | None:
    """Create a residual plot for regression predictions."""

    if len(y_true) == 0:
        return None

    output_path = ensure_directory(output_dir) / "residuals.png"
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    residuals = y_true_array - y_pred_array

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(y_pred_array, residuals, alpha=0.75, color="#4C78A8")
    ax.axhline(0, color="#E15759", linewidth=1.5)
    ax.set_title("Residuals")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual - Predicted")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_confusion_matrix_plot(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    output_dir: str | Path,
) -> Path | None:
    """Create a confusion matrix for classification predictions."""

    labels = sorted({str(value) for value in list(y_true) + list(y_pred)}, key=str)
    if not labels:
        return None

    output_path = ensure_directory(output_dir) / "confusion_matrix.png"
    matrix = confusion_matrix(
        [str(value) for value in y_true],
        [str(value) for value in y_pred],
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.7)))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_feature_importance_plot(
    result: ModelTrainingResult,
    output_dir: str | Path,
    top_n: int = 20,
) -> Path | None:
    """Create a model-specific feature signal chart when supported."""

    if result.estimator is None:
        return None

    named_steps = getattr(result.estimator, "named_steps", {})
    model = named_steps.get("model")
    preprocessor = named_steps.get("preprocessor")
    if model is None or preprocessor is None:
        return None

    names = get_transformed_feature_names(preprocessor)
    values = _extract_feature_values(model)
    if values is None or not names or len(values) != len(names):
        return None

    importance = pd.DataFrame({"feature": names, "value": np.abs(values)})
    importance = importance.sort_values("value", ascending=False).head(top_n)
    if importance.empty or float(importance["value"].max()) <= 0:
        return None

    output_path = ensure_directory(output_dir) / "feature_importance.png"
    fig, ax = plt.subplots(figsize=(8, max(5, len(importance) * 0.32)))
    ax.barh(importance["feature"][::-1], importance["value"][::-1], color="#F28E2B")
    ax.set_title("Feature Signal for Selected Model (Not Causal)")
    ax.set_xlabel("Absolute importance or coefficient")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _predict_probabilities(
    estimator: Any,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not hasattr(estimator, "predict_proba"):
        return None, None
    try:
        probabilities = estimator.predict_proba(X_test)
        classes = None
        model = getattr(estimator, "named_steps", {}).get("model")
        if model is not None and hasattr(model, "classes_"):
            classes = np.asarray(model.classes_)
        return probabilities, classes
    except Exception:
        return None, None


def _extract_feature_values(model: object) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)

    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        if coefficients.ndim == 1:
            return coefficients
        return np.mean(np.abs(coefficients), axis=0)

    return None


def _append_plot(
    generated: list[dict[str, str]],
    path: Path | None,
    run_root: Path,
    label: str,
    category: str,
) -> None:
    if path is None:
        return
    generated.append(
        {
            "path": path.relative_to(run_root).as_posix(),
            "label": label,
            "category": category,
        }
    )


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if math.isfinite(numeric_value) else None
