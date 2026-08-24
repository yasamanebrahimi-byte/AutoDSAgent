from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from app.deterministic import deterministic_recommendation
from app.deterministic_diagnostics import compute_deterministic_diagnostics
from app.deterministic_policy import DeterministicPolicy
from evaluation.benchmarks import _final_interaction_regression


def _regression_frame(kind: str, seed: int, rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, 5))
    if kind == "product":
        target = values[:, 0] * values[:, 1] + rng.normal(0.0, 0.15, rows)
    elif kind == "sin":
        target = np.sin(values[:, 0] * values[:, 1]) + rng.normal(0.0, 0.10, rows)
    elif kind == "mixed":
        target = 2.0 * values[:, 0] + 3.0 * values[:, 1] * values[:, 2] + rng.normal(0.0, 0.15, rows)
    elif kind == "linear":
        target = 2.0 * values[:, 0] - 3.0 * values[:, 1] + 0.5 * values[:, 2] + rng.normal(0.0, 0.15, rows)
    elif kind == "noise":
        target = rng.normal(size=rows)
    else:
        raise ValueError(kind)
    return pd.DataFrame(values, columns=[f"x{index}" for index in range(values.shape[1])]).assign(target=target)


def _interaction_disabled_policy() -> DeterministicPolicy:
    return replace(
        DeterministicPolicy(),
        interaction_moderate_threshold=2.0,
        interaction_high_threshold=3.0,
        structural_complexity_interaction_weight=0.0,
    )


def test_pure_product_has_weak_marginals_and_strong_interaction_evidence():
    frame = _regression_frame("product", seed=7)
    diagnostics = compute_deterministic_diagnostics(frame, "target", "regression")
    interaction = diagnostics.interaction_signals

    assert diagnostics.marginal_association_strength < 0.30
    assert interaction.interaction_strength == "high"
    assert interaction.interaction_score > 0.35
    assert interaction.interaction_pairs_evaluated > 0
    assert interaction.strong_interaction_pair_count >= 1
    assert interaction.top_interaction_pairs[0].features == ["x0", "x1"]
    assert interaction.top_interaction_pairs[0].transform == "product"
    assert interaction.top_interaction_pairs[0].incremental_strength > 0.50


def test_sinusoidal_product_uses_non_monotonic_capable_signal():
    frame = _regression_frame("sin", seed=11)
    diagnostics = compute_deterministic_diagnostics(frame, "target", "regression")
    interaction = diagnostics.interaction_signals

    assert interaction.interaction_score > 0.25
    assert interaction.interaction_strength in {"moderate", "high"}
    assert interaction.top_interaction_pairs[0].features == ["x0", "x1"]
    assert interaction.top_interaction_pairs[0].joint_strength > interaction.top_interaction_pairs[0].marginal_strength


def test_additive_linear_data_does_not_create_interaction_preference():
    frame = _regression_frame("linear", seed=13)
    aware = deterministic_recommendation(
        frame, "estimate target", target_hint="target", task_type="regression"
    )
    baseline = deterministic_recommendation(
        frame,
        "estimate target",
        target_hint="target",
        task_type="regression",
        policy=_interaction_disabled_policy(),
    )

    assert aware.diagnostics is not None
    assert aware.diagnostics.interaction_signals.interaction_strength == "low"
    assert aware.diagnostics.interaction_signals.interaction_score < 0.10
    assert aware.method_scores["tree_ensemble"] <= baseline.method_scores["tree_ensemble"]
    assert aware.method_scores["boosted_tree"] <= baseline.method_scores["boosted_tree"]


def test_mixed_additive_and_interaction_data_reports_both_signals():
    diagnostics = compute_deterministic_diagnostics(
        _regression_frame("mixed", seed=17), "target", "regression"
    )

    assert diagnostics.marginal_association_strength > 0.20
    assert diagnostics.interaction_signals.interaction_score > 0.20
    assert diagnostics.interaction_signals.interaction_strength in {"moderate", "high"}


def test_noise_only_interaction_score_is_low_across_seeds():
    scores = []
    for seed in range(5):
        diagnostics = compute_deterministic_diagnostics(
            _regression_frame("noise", seed=seed), "target", "regression"
        )
        scores.append(diagnostics.interaction_signals.interaction_score)

    assert max(scores) < 0.15
    assert np.mean(scores) < 0.08


def test_interaction_detection_is_stable_across_seeds_and_scale_changes():
    scores = [
        compute_deterministic_diagnostics(
            _regression_frame("product", seed=seed), "target", "regression"
        ).interaction_signals.interaction_score
        for seed in range(5)
    ]
    assert np.mean(scores) > 0.35
    assert np.var(scores) < 0.02

    frame = _regression_frame("product", seed=23)
    scaled = frame.copy()
    scaled["x0"] *= 1000.0
    scaled["x1"] *= 0.001
    first = compute_deterministic_diagnostics(frame, "target", "regression")
    second = compute_deterministic_diagnostics(scaled, "target", "regression")
    np.testing.assert_allclose(
        second.interaction_signals.interaction_score,
        first.interaction_signals.interaction_score,
        rtol=0.0,
        atol=0.01,
    )


def test_existing_interaction_benchmark_exposes_auditable_evidence():
    recommendation = deterministic_recommendation(
        _final_interaction_regression(),
        "estimate target",
        target_hint="target",
        task_type="regression",
    )
    assert recommendation.diagnostics is not None
    interaction = recommendation.diagnostics.interaction_signals
    assert interaction.interaction_strength in {"moderate", "high"}
    assert interaction.strong_interaction_pair_count >= 1
    assert interaction.top_interaction_pairs
    assert recommendation.method_assessments["tree_ensemble"].contributions
    assert any(
        contribution.factor == "interaction"
        for contribution in recommendation.method_assessments["tree_ensemble"].contributions
    )


def test_missing_infinite_and_constant_features_are_skipped_safely():
    frame = _regression_frame("product", seed=29)
    frame["constant"] = 1.0
    frame.loc[::7, "x0"] = np.inf
    frame.loc[::11, "x1"] = -np.inf
    frame.loc[::13, "target"] = np.nan
    diagnostics = compute_deterministic_diagnostics(frame, "target", "regression")

    assert diagnostics.interaction_signals.interaction_applicable is True
    assert np.isfinite(diagnostics.interaction_signals.interaction_score)
    assert diagnostics.interaction_signals.skipped_pair_count >= 0
