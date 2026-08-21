from __future__ import annotations

import numpy as np
import pandas as pd

from app.deterministic import deterministic_recommendation
from app.deterministic_policy import MAX_ONE_HOT_FEATURES
from app.modeling import _estimator


def _clean_numeric(rows: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, 3))
    frame = pd.DataFrame(values, columns=["x1", "x2", "x3"])
    frame["target"] = np.where(frame["x1"] + 0.2 * frame["x2"] > 0, "yes", "no")
    return frame


def _correlated_numeric(rows: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=rows)
    frame = pd.DataFrame(
        {f"x{index}": base + rng.normal(0, 0.02, rows) for index in range(20)}
    )
    frame["target"] = np.where(base > 0, "yes", "no")
    return frame


def _mixed_nonlinear(rows: int = 250, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, 6))
    frame = pd.DataFrame(values, columns=[f"x{index}" for index in range(6)])
    frame["segment"] = np.where(values[:, 0] > 0, "a", "b")
    frame["region"] = np.where(values[:, 1] > 0, "east", "west")
    frame["channel"] = np.where(values[:, 2] > 0, "online", "store")
    frame["target"] = np.where(
        values[:, 0] ** 2 + values[:, 1] * values[:, 2] > 0.8,
        "yes",
        "no",
    )
    return frame


def _large_nonlinear(rows: int = 1000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, 4))
    frame = pd.DataFrame(values, columns=[f"x{index}" for index in range(4)])
    frame["target"] = np.where(np.abs(values[:, 0]) > 0.7, "yes", "no")
    return frame


def test_clean_numeric_profile_favors_linear_family():
    recommendation = deterministic_recommendation(
        _clean_numeric(), "classify target", target_hint="target", task_type="classification"
    )

    assert recommendation.recommended_method == "linear"
    assert recommendation.ranked_methods[0] == "linear"
    assert recommendation.diagnostics is not None
    assert recommendation.diagnostics.numeric_feature_count == 3
    assert recommendation.diagnostics.categorical_feature_count == 0


def test_multicollinearity_favors_regularized_linear_over_plain_linear():
    recommendation = deterministic_recommendation(
        _correlated_numeric(), "classify target", target_hint="target", task_type="classification"
    )

    assert recommendation.recommended_method == "regularized_linear"
    assert recommendation.method_scores["regularized_linear"] > recommendation.method_scores["linear"]
    assert recommendation.diagnostics is not None
    assert recommendation.diagnostics.max_abs_numeric_correlation > 0.9


def test_mixed_nonlinear_profile_selects_tree_ensemble():
    recommendation = deterministic_recommendation(
        _mixed_nonlinear(), "classify target", target_hint="target", task_type="classification"
    )

    assert recommendation.recommended_method == "tree_ensemble"
    assert recommendation.diagnostics is not None
    assert recommendation.diagnostics.nonlinearity_signal in {"moderate", "high"}
    assert recommendation.diagnostics.interaction_signal == "high"


def test_large_nonlinear_profile_can_select_boosted_tree():
    recommendation = deterministic_recommendation(
        _large_nonlinear(), "classify target", target_hint="target", task_type="classification"
    )

    assert recommendation.recommended_method == "boosted_tree"
    assert recommendation.method_scores["boosted_tree"] > recommendation.method_scores["tree_ensemble"]
    assert recommendation.diagnostics is not None
    assert recommendation.diagnostics.rows >= 600
    assert recommendation.diagnostics.nonlinearity_signal == "high"


def test_oversized_one_hot_families_are_ineligible_but_boosted_tree_uses_ordinal_path():
    rng = np.random.default_rng(3)
    rows = 500
    frame = pd.DataFrame(
        {
            f"category_{index}": [f"v{value}" for value in rng.integers(0, 79, rows)]
            for index in range(60)
        }
    )
    frame["target"] = np.where(np.arange(rows) % 2, "yes", "no")
    recommendation = deterministic_recommendation(
        frame, "classify target", target_hint="target", task_type="classification"
    )

    assert recommendation.diagnostics is not None
    assert recommendation.diagnostics.estimated_one_hot_dimensionality > MAX_ONE_HOT_FEATURES
    assert recommendation.method_assessments["linear"].eligible is False
    assert recommendation.method_assessments["regularized_linear"].eligible is False
    assert recommendation.method_assessments["tree_ensemble"].eligible is False
    assert recommendation.method_assessments["boosted_tree"].eligible is True
    assert recommendation.recommended_method == "boosted_tree"


def test_missingness_is_a_factor_but_does_not_automatically_force_tree():
    frame = _clean_numeric()
    frame.loc[frame.index[::5], "x1"] = np.nan
    recommendation = deterministic_recommendation(
        frame, "classify target", target_hint="target", task_type="classification"
    )

    assert recommendation.recommended_method == "linear"
    assert recommendation.diagnostics is not None
    assert recommendation.diagnostics.overall_missing_fraction > 0
    assert recommendation.preprocessing.numeric_imputation == "median"


def test_target_diagnostics_are_task_specific():
    frame = _clean_numeric()
    frame["target"] = np.where(np.arange(len(frame)) % 20 == 0, "minority", "majority")
    classification = deterministic_recommendation(
        frame, "classify target", target_hint="target", task_type="classification"
    )
    assert classification.diagnostics is not None
    target = classification.diagnostics.target.classification
    assert target is not None
    assert target.minimum_class_size == 30
    assert target.minority_class_fraction < 0.1

    regression_frame = _clean_numeric().drop(columns=["target"])
    regression_frame["target"] = np.exp(regression_frame["x1"])
    regression = deterministic_recommendation(
        regression_frame, "estimate target", target_hint="target", task_type="regression"
    )
    assert regression.diagnostics is not None
    regression_target = regression.diagnostics.target.regression
    assert regression_target is not None
    assert regression_target.variance > 0
    assert regression_target.skewness > 0


def test_scores_are_deterministic_and_row_and_column_order_invariant():
    frame = _mixed_nonlinear()
    first = deterministic_recommendation(
        frame, "classify target", target_hint="target", task_type="classification"
    )
    second = deterministic_recommendation(
        frame, "classify target", target_hint="target", task_type="classification"
    )
    reordered_rows = deterministic_recommendation(
        frame.sample(frac=1, random_state=31).reset_index(drop=True),
        "classify target",
        target_hint="target",
        task_type="classification",
    )
    reordered_columns = deterministic_recommendation(
        frame[["region", "target", "x4", "segment", "x0", "channel", "x1", "x2", "x3", "x5"]],
        "classify target",
        target_hint="target",
        task_type="classification",
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.recommended_method == reordered_rows.recommended_method == reordered_columns.recommended_method
    assert first.ranked_methods == reordered_rows.ranked_methods == reordered_columns.ranked_methods
    assert first.method_scores == reordered_rows.method_scores == reordered_columns.method_scores


def test_classification_linear_estimator_is_actually_unregularized():
    assert _estimator("classification", "linear", 42).get_params()["penalty"] is None
    regularized = _estimator("classification", "regularized_linear", 42)
    assert regularized.get_params()["penalty"] in {"l2", "deprecated"}
    assert regularized.get_params()["C"] == 0.5
    assert regularized.get_params()["l1_ratio"] == 0.0
