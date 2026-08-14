import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from app.backend.services.evaluation_service import EvaluationService
from app.backend.services.run_manager import RunManager
from app.tools import modeling as modeling_module
from app.tools.modeling import ModelTrainingResult
from app.tools.preprocessing import prepare_modeling_data


class ConstantRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, value=0.0):
        self.value = value

    def fit(self, X, y):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class CountingEstimator:
    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.fit_calls = 0
        self.predict_calls = 0
        self.named_steps = {}

    def fit(self, X, y):
        self.fit_calls += 1
        return self

    def predict(self, X):
        self.predict_calls += 1
        return self.predictions[: len(X)]


def test_cv_metrics_select_model_before_holdout_is_evaluated(tmp_path):
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6, 7, 8],
            "target": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        }
    )
    prepared = prepare_modeling_data(
        dataframe,
        target_column="target",
        task_type="regression",
        test_size=0.25,
        random_state=42,
    )
    holdout_winner = CountingEstimator(prepared.y_test.tolist())
    cv_winner = CountingEstimator([999.0] * len(prepared.y_test))

    results = [
        ModelTrainingResult(
            model_name="baseline_median",
            role="baseline",
            status="succeeded",
            estimator=CountingEstimator([0.0] * len(prepared.y_test)),
            cv_metrics={"cv_rmse_mean": 10.0},
            metrics={"cv_rmse_mean": 10.0},
            primary_metric_value=10.0,
        ),
        ModelTrainingResult(
            model_name="cv_winner",
            role="candidate",
            status="succeeded",
            estimator=cv_winner,
            cv_metrics={"cv_rmse_mean": 1.0},
            metrics={"cv_rmse_mean": 1.0},
            primary_metric_value=1.0,
        ),
        ModelTrainingResult(
            model_name="holdout_winner",
            role="candidate",
            status="succeeded",
            estimator=holdout_winner,
            cv_metrics={"cv_rmse_mean": 5.0},
            metrics={"cv_rmse_mean": 5.0},
            primary_metric_value=5.0,
        ),
    ]

    manager = RunManager(runs_dir=tmp_path)
    run_id = "cv-selection-test"
    manager.create_run(run_id)
    evaluation = EvaluationService(manager).evaluate_and_save(run_id, prepared, results)

    assert evaluation.best_result.model_name == "cv_winner"
    assert cv_winner.fit_calls == 1
    assert holdout_winner.fit_calls == 0
    assert cv_winner.predict_calls == 1
    assert holdout_winner.predict_calls == 0
    assert evaluation.summary.test_evaluated_model_names == ["cv_winner"]
    assert evaluation.summary.best_model_metrics["rmse"] == pytest.approx(
        prepared.y_test.sub(999.0).pow(2).mean() ** 0.5
    )
    assert evaluation.summary.final_test_metrics == evaluation.summary.best_model_metrics
    assert set(evaluation.summary.candidate_cv_results) == {
        "cv_winner",
        "holdout_winner",
    }
    serialized = {
        result["model_name"]: result
        for result in evaluation.model_results_payload["results"]
    }
    assert serialized["holdout_winner"]["holdout_metrics"] == {}


def test_training_only_cv_beats_candidate_that_would_win_holdout(
    tmp_path,
    monkeypatch,
):
    prepared = _prepare_holdout_sensitive_regression(holdout_target_value=100.0)
    monkeypatch.setattr(modeling_module, "_model_specs", _constant_regression_specs)

    results = modeling_module.train_models(prepared, random_state=42)
    manager = RunManager(runs_dir=tmp_path)
    run_id = "holdout-winner-rejected"
    manager.create_run(run_id)
    evaluation = EvaluationService(manager).evaluate_and_save(run_id, prepared, results)

    assert evaluation.best_result.model_name == "cv_winner"
    assert evaluation.summary.test_evaluated_model_names == ["cv_winner"]
    assert evaluation.summary.final_test_metrics["rmse"] == pytest.approx(100.0)
    assert evaluation.model_results_payload["results"][1]["holdout_metrics"][
        "rmse"
    ] == pytest.approx(100.0)
    assert evaluation.model_results_payload["results"][2]["holdout_metrics"] == {}


def test_changing_only_holdout_targets_does_not_change_selected_model(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(modeling_module, "_model_specs", _constant_regression_specs)
    original = _select_for_holdout_target(tmp_path, "original-holdout", 100.0)
    mutated = _select_for_holdout_target(tmp_path, "mutated-holdout", -250.0)

    assert original.summary.best_model_name == "cv_winner"
    assert mutated.summary.best_model_name == "cv_winner"
    assert (
        original.summary.final_test_metrics["rmse"]
        != mutated.summary.final_test_metrics["rmse"]
    )


def test_cross_validation_receives_training_partition_only(monkeypatch):
    prepared = _prepare_holdout_sensitive_regression(holdout_target_value=100.0)
    calls = []

    def fake_cross_validate(estimator, X, y, cv, scoring, error_score, n_jobs):
        calls.append((X, y))
        return {"test_rmse": np.array([-1.0, -1.0])}

    monkeypatch.setattr(modeling_module, "_model_specs", _constant_regression_specs)
    monkeypatch.setattr(modeling_module, "cross_validate", fake_cross_validate)

    modeling_module.train_models(prepared, random_state=42)

    assert calls
    assert all(X is prepared.X_train for X, _ in calls)
    assert all(y is prepared.y_train for _, y in calls)


def test_selection_ties_keep_original_order_without_holdout_tie_break(tmp_path):
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6, 7, 8],
            "target": [0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    prepared = prepare_modeling_data(
        dataframe,
        target_column="target",
        task_type="regression",
        test_size=0.25,
        random_state=42,
    )
    first = CountingEstimator([999.0] * len(prepared.y_test))
    second = CountingEstimator(prepared.y_test.tolist())
    results = [
        ModelTrainingResult(
            model_name="first_candidate",
            role="candidate",
            status="succeeded",
            estimator=first,
            cv_metrics={"cv_rmse_mean": 1.0},
            metrics={"cv_rmse_mean": 1.0},
            primary_metric_value=1.0,
        ),
        ModelTrainingResult(
            model_name="second_candidate",
            role="candidate",
            status="succeeded",
            estimator=second,
            cv_metrics={"cv_rmse_mean": 1.0},
            metrics={"cv_rmse_mean": 1.0},
            primary_metric_value=1.0,
        ),
    ]
    manager = RunManager(runs_dir=tmp_path)
    run_id = "tie-break-selection"
    manager.create_run(run_id)

    evaluation = EvaluationService(manager).evaluate_and_save(run_id, prepared, results)

    assert evaluation.best_result.model_name == "first_candidate"
    assert first.predict_calls == 1
    assert second.fit_calls == 0
    assert second.predict_calls == 0
    assert evaluation.summary.selection_tiebreaker == "original_result_order"


def test_numeric_imputer_statistic_is_learned_from_training_partition():
    dataframe = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                1000,
                1001,
                1002,
                1003,
                1004,
                1005,
                1006,
                1007,
                1008,
                1009,
            ],
            "target": [float(index) * 2.0 for index in range(20)],
        }
    )
    dataframe.loc[[0, 5, 10], "feature"] = None

    prepared = prepare_modeling_data(
        dataframe,
        target_column="target",
        task_type="regression",
        test_size=0.4,
        random_state=42,
    )
    fitted = prepared.preprocessor.fit(prepared.X_train, prepared.y_train)

    train_median = pd.to_numeric(prepared.X_train["feature"], errors="coerce").median()
    full_median = pd.to_numeric(dataframe["feature"], errors="coerce").median()
    learned_median = fitted.named_transformers_["numeric"].named_steps[
        "imputer"
    ].statistics_[0]

    assert train_median != full_median
    assert learned_median == pytest.approx(train_median)


def _select_for_holdout_target(tmp_path, run_id, holdout_target_value):
    prepared = _prepare_holdout_sensitive_regression(holdout_target_value)
    results = modeling_module.train_models(prepared, random_state=42)
    manager = RunManager(runs_dir=tmp_path)
    manager.create_run(run_id)
    return EvaluationService(manager).evaluate_and_save(run_id, prepared, results)


def _prepare_holdout_sensitive_regression(holdout_target_value):
    row_count = 40
    test_indices = train_test_split(
        list(range(row_count)),
        test_size=0.25,
        random_state=42,
    )[1]
    dataframe = pd.DataFrame(
        {
            "feature": [index % 7 for index in range(row_count)],
            "target": [0.0] * row_count,
        }
    )
    dataframe.loc[test_indices, "target"] = holdout_target_value
    return prepare_modeling_data(
        dataframe,
        target_column="target",
        task_type="regression",
        test_size=0.25,
        random_state=42,
    )


def _constant_regression_specs(task_type, random_state):
    assert task_type == "regression"
    return [
        ("baseline_median", "baseline", ConstantRegressor(0.0)),
        ("cv_winner", "candidate", ConstantRegressor(0.0)),
        ("holdout_winner", "candidate", ConstantRegressor(100.0)),
    ]
