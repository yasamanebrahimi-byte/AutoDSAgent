from __future__ import annotations

import numpy as np
import pandas as pd

from app.deterministic import deterministic_recommendation
from app.deterministic_diagnostics import compute_deterministic_diagnostics


def _multiclass_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    class_membership = np.repeat(["A", "B", "C"], 80)
    return pd.DataFrame(
        {
            # The middle class lies between the other two.  An arbitrary class
            # code therefore produces materially different correlations when
            # the code order changes, even though membership is unchanged.
            "separation": np.repeat([-5.0, 5.0, 0.0], 80),
            "noise": rng.normal(size=len(class_membership)),
            "segment": np.repeat(["left", "right", "middle"], 80),
            "target": class_membership,
        }
    )


def _renamed_multiclass(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed["target"] = renamed["target"].map(
        {"A": "zebra", "B": "monkey", "C": "aardvark"}
    )
    return renamed


def _recommendation_relevant(recommendation):
    diagnostics = recommendation.diagnostics
    assert diagnostics is not None
    return {
        "association_measure": diagnostics.association_measure,
        "marginal_association_strength": diagnostics.marginal_association_strength,
        "class_separation_strength": diagnostics.class_separation_strength,
        "mean_univariate_signal": diagnostics.mean_univariate_signal,
        "nonlinearity_score": diagnostics.nonlinearity_score,
        "nonlinearity_signal": diagnostics.nonlinearity_signal,
        "nonlinearity_applicable": diagnostics.nonlinearity_applicable,
        "nonlinear_feature_count": diagnostics.nonlinear_feature_count,
        "nonlinear_feature_fraction": diagnostics.nonlinear_feature_fraction,
        "nonlinearity_heterogeneity": diagnostics.nonlinearity_heterogeneity,
        "structural_complexity_score": diagnostics.structural_complexity_score,
        "structural_complexity_signal": diagnostics.structural_complexity_signal,
        "method_scores": recommendation.method_scores,
        "ranked_methods": recommendation.ranked_methods,
        "recommended_method": recommendation.recommended_method,
        "score_margin": recommendation.score_margin,
        "confidence": recommendation.confidence,
    }


def test_adversarial_multiclass_label_order_does_not_change_diagnostics_or_recommendation():
    frame = _multiclass_frame()
    renamed = _renamed_multiclass(frame)

    # This is the old failure mode: sorted numeric target codes produce a
    # different apparent relationship for the same class membership.
    first_codes = frame["target"].map({"A": 0.0, "B": 1.0, "C": 2.0})
    second_codes = frame["target"].map({"A": 0.0, "C": 1.0, "B": 2.0})
    first_correlation = abs(float(frame["separation"].corr(first_codes)))
    second_correlation = abs(float(frame["separation"].corr(second_codes)))
    assert abs(first_correlation - second_correlation) > 0.10

    original = deterministic_recommendation(
        frame, "classify target", target_hint="target", task_type="classification"
    )
    relabeled = deterministic_recommendation(
        renamed, "classify target", target_hint="target", task_type="classification"
    )

    assert _recommendation_relevant(original) == _recommendation_relevant(relabeled)
    assert original.diagnostics is not None
    assert original.diagnostics.association_measure == "classification_eta_squared_and_cramers_v"
    assert original.diagnostics.class_separation_strength > 0.95
    assert original.diagnostics.nonlinearity_score == 0.0
    assert original.diagnostics.pearson_spearman_gap == 0.0


def test_binary_label_swap_preserves_recommendation_relevant_evidence():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"x": rng.normal(size=240), "noise": rng.normal(size=240)})
    frame["target"] = np.where(frame["x"] > 0, "yes", "no")
    swapped = frame.assign(target=frame["target"].map({"yes": "no", "no": "yes"}))

    original = deterministic_recommendation(
        frame, "classify target", target_hint="target", task_type="classification"
    )
    relabeled = deterministic_recommendation(
        swapped, "classify target", target_hint="target", task_type="classification"
    )

    assert _recommendation_relevant(original) == _recommendation_relevant(relabeled)
    assert original.diagnostics is not None
    assert original.diagnostics.association_measure == "classification_eta_squared"


def test_numeric_multiclass_labels_remain_nominal():
    frame = _multiclass_frame()
    frame["target"] = frame["target"].map({"A": 10, "B": 20, "C": 30})
    diagnostics = compute_deterministic_diagnostics(frame, "target", "classification")

    assert diagnostics.association_measure == "classification_eta_squared_and_cramers_v"
    assert diagnostics.nonlinearity_applicable is False
    assert diagnostics.pearson_spearman_gap == 0.0
    assert diagnostics.nonlinearity_score == 0.0
    assert 0.0 <= diagnostics.marginal_association_strength <= 1.0
    assert 0.0 <= diagnostics.class_separation_strength <= 1.0


def test_categorical_association_is_label_order_invariant_and_bounded():
    frame = _multiclass_frame()
    renamed = _renamed_multiclass(frame)
    first = compute_deterministic_diagnostics(frame, "target", "classification")
    second = compute_deterministic_diagnostics(renamed, "target", "classification")

    assert first.marginal_association_strength == second.marginal_association_strength
    assert first.class_separation_strength == second.class_separation_strength
    assert 0.0 <= first.marginal_association_strength <= 1.0
    assert 0.0 <= first.class_separation_strength <= 1.0


def test_constant_numeric_predictor_and_rare_class_are_finite_and_safe():
    constant = pd.DataFrame(
        {
            "constant": np.ones(90),
            "target": np.repeat(["a", "b", "c"], 30),
        }
    )
    constant_diagnostics = compute_deterministic_diagnostics(
        constant, "target", "classification"
    )
    assert constant_diagnostics.marginal_association_strength == 0.0
    assert constant_diagnostics.class_separation_strength == 0.0

    rare = pd.DataFrame(
        {
            "x": np.arange(100, dtype=float),
            "target": ["majority"] * 99 + ["rare"],
        }
    )
    rare_diagnostics = compute_deterministic_diagnostics(rare, "target", "classification")
    assert rare_diagnostics.target.classification is not None
    assert rare_diagnostics.target.classification.minimum_class_size == 1
    assert np.isfinite(rare_diagnostics.marginal_association_strength)
    assert np.isfinite(rare_diagnostics.class_separation_strength)
    assert 0.0 <= rare_diagnostics.marginal_association_strength <= 1.0
    assert 0.0 <= rare_diagnostics.class_separation_strength <= 1.0


def test_regression_relationship_path_remains_numeric_and_task_specific():
    x = np.linspace(-2.0, 2.0, 160)
    frame = pd.DataFrame({"x": x, "target": x**2})
    diagnostics = compute_deterministic_diagnostics(frame, "target", "regression")

    assert diagnostics.association_measure == "regression_pearson_spearman_binned"
    assert diagnostics.nonlinearity_applicable is True
    assert diagnostics.marginal_association_strength > 0.0
    assert diagnostics.nonlinearity_score > 0.0

