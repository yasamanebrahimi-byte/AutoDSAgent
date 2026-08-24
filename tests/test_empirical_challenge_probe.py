from __future__ import annotations

import numpy as np
import pandas as pd

import app.empirical_challenge_probe as probe
from app.empirical_challenge_probe import EmpiricalProbePolicy, run_pairwise_model_probe
from app.schemas import ModelingPlan, PreprocessingContract


def _plan(method: str) -> ModelingPlan:
    preprocessing = PreprocessingContract(
        numeric_scaling="standard" if method in {"linear", "regularized_linear"} else "none"
    )
    return ModelingPlan(
        recommended_method=method,
        preprocessing=preprocessing,
        reasoning="This deterministic test proposal is a complete, supported plan for the training data.",
        confidence=0.7,
    )


def _regression_frame(rows: int = 72) -> pd.DataFrame:
    x = np.linspace(-2.0, 2.0, rows)
    return pd.DataFrame({"x": x, "noise": np.sin(x * 3), "target": x**2})


def test_probe_compares_only_the_two_proposed_families(monkeypatch):
    seen: list[str] = []
    original = probe._estimator

    def recording_estimator(task_type, method, random_state):
        seen.append(method)
        return original(task_type, method, random_state)

    monkeypatch.setattr(probe, "_estimator", recording_estimator)
    result = run_pairwise_model_probe(
        _regression_frame(),
        "target",
        "regression",
        _plan("linear"),
        _plan("boosted_tree"),
        policy=EmpiricalProbePolicy(random_state=11),
        random_state=11,
    )

    assert result["status"] == "completed"
    assert set(seen) == {"linear", "boosted_tree"}
    assert "regularized_linear" not in seen
    assert "tree_ensemble" not in seen
    assert result["fit_count"] == 6


def test_probe_is_reproducible_and_uses_regression_kfold():
    frame = _regression_frame()
    kwargs = {
        "training_frame": frame,
        "target_column": "target",
        "task_type": "regression",
        "proposal_a": _plan("linear"),
        "proposal_b": _plan("boosted_tree"),
        "policy": EmpiricalProbePolicy(random_state=17),
        "random_state": 17,
    }
    first = run_pairwise_model_probe(**kwargs)
    second = run_pairwise_model_probe(**kwargs)
    assert first == second
    assert first["cv_strategy"] == "kfold"
    assert first["cv_folds"] == 3
    assert first["holdout_used"] is False
    assert first["data_used"] == "frozen_training_partition_only"


def test_probe_classification_uses_stratified_folds_and_only_training_rows():
    frame = pd.DataFrame(
        {
            "signal": np.arange(48, dtype=float),
            "target": ["yes", "no"] * 24,
        }
    )
    result = run_pairwise_model_probe(
        frame.iloc[:36].copy(),
        "target",
        "classification",
        _plan("linear"),
        _plan("tree_ensemble"),
        policy=EmpiricalProbePolicy(random_state=5),
        random_state=5,
    )
    assert result["status"] == "completed"
    assert result["cv_strategy"] == "stratified_kfold"
    assert result["training_rows"] == 36
    assert result["holdout_used"] is False


def test_tiny_difference_is_a_tie(monkeypatch):
    scores = iter(
        (
            [0.821, 0.820, 0.822],
            [0.819, 0.820, 0.821],
        )
    )
    monkeypatch.setattr(probe, "_metric_scores", lambda *args, **kwargs: next(scores))
    result = run_pairwise_model_probe(
        _regression_frame(),
        "target",
        "regression",
        _plan("linear"),
        _plan("boosted_tree"),
        policy=EmpiricalProbePolicy(random_state=2),
        random_state=2,
    )
    assert result["winner"] == "tie"
    assert result["evidence_strength"] == "tie"


def test_clear_consistent_difference_is_strong(monkeypatch):
    scores = iter(( [0.95, 0.96, 0.94], [0.60, 0.61, 0.59] ))
    monkeypatch.setattr(probe, "_metric_scores", lambda *args, **kwargs: next(scores))
    result = run_pairwise_model_probe(
        _regression_frame(),
        "target",
        "regression",
        _plan("linear"),
        _plan("boosted_tree"),
        policy=EmpiricalProbePolicy(random_state=2),
        random_state=2,
    )
    assert result["winner"] == "B"
    assert result["evidence_strength"] == "strong"
    assert result["proposal_a"]["fold_wins"] == 0
    assert result["proposal_b"]["fold_wins"] == 3


def test_probe_failure_is_recorded_without_raising(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("controlled estimator failure")

    monkeypatch.setattr(probe, "_estimator", fail)
    result = run_pairwise_model_probe(
        _regression_frame(),
        "target",
        "regression",
        _plan("linear"),
        _plan("boosted_tree"),
        policy=EmpiricalProbePolicy(random_state=3),
        random_state=3,
    )
    assert result["status"] == "failed"
    assert result["winner"] == "tie"
    assert "controlled estimator failure" in result["error"]
