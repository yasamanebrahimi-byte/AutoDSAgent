from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_classification, make_moons

from app.deterministic import deterministic_recommendation
from app.deterministic_diagnostics import compute_deterministic_diagnostics


def _diagnostics(features: np.ndarray, target: np.ndarray):
    frame = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(features.shape[1])])
    frame["target"] = target
    return compute_deterministic_diagnostics(frame, "target", "classification")


def test_linearly_separable_boundary_is_low_and_linear_remains_supported():
    rng = np.random.default_rng(1001)
    features = rng.normal(size=(300, 3))
    target = (2.0 * features[:, 0] - 1.5 * features[:, 1] + rng.normal(0, 0.15, 300) > 0).astype(int)

    diagnostics = _diagnostics(features, target)
    boundary = diagnostics.classification_boundary_signals

    assert boundary.boundary_complexity == "low"
    assert boundary.boundary_complexity_score < 0.18
    assert boundary.linear_separability_score > 0.75
    assert boundary.local_class_consistency > 0.75
    recommendation = deterministic_recommendation(
        pd.DataFrame(features, columns=["x0", "x1", "x2"]).assign(target=target),
        "classify target",
        target_hint="target",
        task_type="classification",
    )
    assert recommendation.method_scores["linear"] >= recommendation.method_scores["tree_ensemble"]


def test_two_moons_and_circles_show_local_non_linear_boundary_evidence():
    moon_features, moon_target = make_moons(n_samples=300, noise=0.20, random_state=41)
    circle_features, circle_target = make_circles(
        n_samples=300,
        noise=0.08,
        factor=0.45,
        random_state=41,
    )

    moon = _diagnostics(moon_features, moon_target).classification_boundary_signals
    circles = _diagnostics(circle_features, circle_target).classification_boundary_signals

    assert moon.boundary_complexity in {"moderate", "high"}
    assert moon.local_class_consistency > moon.linear_boundary_probe_score
    assert moon.nonlinear_advantage_score > 0.0
    assert circles.boundary_complexity == "high"
    assert circles.local_class_consistency > 0.90
    assert circles.linear_separability_score < 0.20
    recommendation = deterministic_recommendation(
        pd.DataFrame(moon_features, columns=["x0", "x1"]).assign(target=moon_target),
        "classify target",
        target_hint="target",
        task_type="classification",
    )
    assert recommendation.recommended_method == "boosted_tree"
    assert any(
        contribution.factor == "boundary_complexity" and contribution.points == -8
        for contribution in recommendation.method_assessments["linear"].contributions
    )
    assert any(
        contribution.factor == "boundary_complexity" and contribution.points == 10
        for contribution in recommendation.method_assessments["boosted_tree"].contributions
    )


def test_random_labels_do_not_turn_weak_linear_signal_into_nonlinearity():
    rng = np.random.default_rng(1002)
    features = rng.normal(size=(300, 5))
    target = rng.integers(0, 2, size=300)

    boundary = _diagnostics(features, target).classification_boundary_signals

    assert boundary.boundary_complexity == "low"
    assert boundary.boundary_complexity_score < 0.18
    assert boundary.nonlinear_advantage_score < 0.10
    assert boundary.local_structure_score < 0.20


def test_imbalanced_linear_and_multiclass_linear_are_not_called_nonlinear():
    features, target = make_classification(
        n_samples=360,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.90, 0.10],
        class_sep=1.5,
        random_state=43,
    )
    imbalanced = _diagnostics(features, target).classification_boundary_signals
    assert imbalanced.boundary_complexity == "low"
    assert imbalanced.linear_boundary_probe_score > 0.65

    features, target = make_classification(
        n_samples=360,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=47,
    )
    multiclass = _diagnostics(features, target).classification_boundary_signals
    assert multiclass.boundary_complexity == "low"
    assert multiclass.linear_separability_score > 0.60


def test_multiclass_concentric_regions_are_supported():
    rng = np.random.default_rng(1003)
    features = rng.normal(size=(360, 2))
    radius = np.sqrt(np.sum(features**2, axis=1))
    target = np.digitize(radius, bins=[0.75, 1.35])

    boundary = _diagnostics(features, target).classification_boundary_signals

    assert boundary.boundary_complexity_applicable is True
    assert boundary.boundary_complexity in {"moderate", "high"}
    assert boundary.local_class_consistency > boundary.linear_boundary_probe_score


def test_boundary_diagnostic_is_explicitly_unavailable_without_numeric_geometry():
    frame = pd.DataFrame(
        {
            "segment": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    diagnostics = compute_deterministic_diagnostics(frame, "target", "classification")
    boundary = diagnostics.classification_boundary_signals

    assert boundary.boundary_complexity_applicable is False
    assert boundary.boundary_diagnostic_confidence == "low"
    assert boundary.boundary_diagnostic_reason == "no_usable_numeric_features_for_geometry"
