from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import app.pipeline as pipeline
from app.empirical_challenge_probe import EmpiricalProbePolicy
from app.schemas import DeterministicRecommendation, ModelingPlan, ModelingResolution, PreprocessingContract
from app.validation import freeze_supervised_split
from app.deterministic import profile_dataframe


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": np.linspace(-1.0, 1.0, 60),
            "noise": np.cos(np.linspace(-1.0, 1.0, 60)),
            "target": ["yes", "no"] * 30,
        }
    )


def _plans() -> tuple[ModelingPlan, DeterministicRecommendation]:
    return (
        ModelingPlan(
            recommended_method="linear",
            preprocessing=PreprocessingContract(numeric_scaling="standard"),
            reasoning="The initial plan provides a complete transparent baseline for the task.",
            confidence=0.7,
        ),
        DeterministicRecommendation(
            target_column="target",
            task_type="classification",
            recommended_method="tree_ensemble",
            preprocessing=PreprocessingContract(),
            reasoning="The independent structural recommendation favors a flexible supported family.",
            evidence=["training-only structural evidence"],
            confidence="low",
            score_margin=2.0,
        ),
    )


class _Agents:
    def __init__(self, selected_method: str = "linear") -> None:
        self.calls = 0
        self.payload: dict[str, object] | None = None
        self.selected_method = selected_method

    def reconcile_modeling(self, question, profile, modeling_plan, deterministic):
        del question, profile, modeling_plan
        self.calls += 1
        self.payload = deterministic.get("_blinded_reconciliation_payload")
        return ModelingResolution(
            selected_method=self.selected_method,
            selected_preprocessing=PreprocessingContract(numeric_scaling="standard")
            if self.selected_method == "linear"
            else PreprocessingContract(),
            checks=["both_hard_validated"],
            justification="The blinded reconciler selected one of the two validated proposals.",
            confidence=0.8,
        )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    strength: str,
    agents: _Agents | None = None,
    events: list[str] | None = None,
    mode: str = "selective",
):
    frame = _frame()
    split = freeze_supervised_split(frame, "target", "classification")
    modeling_plan, deterministic = _plans()
    captured: dict[str, pd.DataFrame] = {}

    def probe(training_frame, *args, **kwargs):
        del args, kwargs
        if events is not None:
            events.append("probe")
        captured["frame"] = training_frame.copy()
        return {
            "status": "completed",
            "policy_version": "test",
            "task_type": "classification",
            "metric": "macro_f1",
            "higher_is_better": True,
            "cv_folds": 3,
            "training_rows": len(training_frame),
            "data_used": "frozen_training_partition_only",
            "holdout_used": False,
            "fit_count": 6,
            "winner": "B",
            "difference": 0.2,
            "relative_advantage": 0.2,
            "normalized_advantage": 0.2,
            "evidence_strength": strength,
            "proposal_a": {"model_family": "linear", "mean_score": 0.7},
            "proposal_b": {"model_family": "tree_ensemble", "mean_score": 0.9},
        }

    monkeypatch.setattr(pipeline, "run_pairwise_model_probe", probe)
    result = pipeline._validate_modeling_gate(
        agents or _Agents(),
        profile_dataframe(frame.iloc[list(split.train_row_positions)]),
        "classify target",
        modeling_plan,
        deterministic,
        [],
        {},
        offline=False,
        dataframe=frame,
        test_size=0.2,
        random_state=42,
        reconciliation_profile=profile_dataframe(frame.iloc[list(split.train_row_positions)]),
        split=split,
        row_positions=list(range(len(frame))),
        approved_target="target",
        approved_task="classification",
        empirical_probe_policy=EmpiricalProbePolicy(random_state=42),
        soft_challenge_mode=mode,
    )
    return result, captured, split


def test_probe_precedes_soft_challenge_metadata(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    original = pipeline.decide_soft_challenge

    def record_soft(*args, **kwargs):
        events.append("soft")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "decide_soft_challenge", record_soft)
    _run(monkeypatch, "weak", events=events)
    assert events == ["probe", "soft"]


def test_agreement_skips_probe_and_reconciliation(monkeypatch: pytest.MonkeyPatch):
    frame = _frame()
    split = freeze_supervised_split(frame, "target", "classification")
    plan, deterministic = _plans()
    deterministic = deterministic.model_copy(update={"recommended_method": "linear", "preprocessing": plan.preprocessing})
    calls = {"probe": 0}
    monkeypatch.setattr(pipeline, "run_pairwise_model_probe", lambda *args, **kwargs: calls.__setitem__("probe", 1))
    agents = _Agents()
    result = pipeline._validate_modeling_gate(
        agents,
        profile_dataframe(frame.iloc[list(split.train_row_positions)]),
        "classify target",
        plan,
        deterministic,
        [],
        {},
        offline=False,
        dataframe=frame,
        test_size=0.2,
        random_state=42,
        reconciliation_profile=profile_dataframe(frame.iloc[list(split.train_row_positions)]),
        split=split,
        row_positions=list(range(len(frame))),
        approved_target="target",
        approved_task="classification",
    )
    assert calls["probe"] == 0
    assert agents.calls == 0
    assert result["selected_method"] == "linear"
    assert result["decision_path"] == "agreement"
    assert result["gate_decision"]["reason_code"] == "no_material_disagreement"


@pytest.mark.parametrize("strength", ["tie", "weak"])
def test_weak_or_tied_probe_abstains(monkeypatch: pytest.MonkeyPatch, strength: str):
    agents = _Agents()
    result, captured, split = _run(monkeypatch, strength, agents)
    assert len(captured["frame"]) == len(split.train_row_positions)
    assert result["empirical_probe_invoked"] is True
    assert result["probe_evidence_strength"] == strength
    assert result["soft_challenge_decision"] == "abstain"
    assert result["reconciliation"] is None
    assert agents.calls == 0
    assert result["selected_method"] == "linear"
    assert result["gate_decision"]["reason_code"] in {"probe_inconclusive", "probe_weak"}
    assert "heuristic confidence" not in result["gate_decision"]["reason_text"]


@pytest.mark.parametrize("strength", ["moderate", "strong"])
def test_moderate_or_strong_probe_invokes_blinded_reconciliation(monkeypatch: pytest.MonkeyPatch, strength: str):
    agents = _Agents(selected_method="linear")
    result, captured, split = _run(monkeypatch, strength, agents)
    assert len(captured["frame"]) == len(split.train_row_positions)
    assert result["probe_evidence_strength"] == strength
    assert result["soft_challenge_decision"] == "challenge"
    assert result["reconciliation"] is not None
    assert agents.calls == 1
    assert result["selected_method"] == "linear"
    assert result["gate_decision"]["reason_code"] in {"reconciliation_selected_original", "reconciliation_selected_alternative"}
    assert result["gate_decision"]["trigger_reason_code"] == "probe_supports_intervention"
    assert result["selected_proposal_source"] in {"agent", "deterministic"}
    prompt_payload = {
        "proposal_a": (agents.payload or {}).get("proposal_a"),
        "proposal_b": (agents.payload or {}).get("proposal_b"),
        "empirical_probe": (agents.payload or {}).get("empirical_probe"),
    }
    assert '"deterministic"' not in json.dumps(prompt_payload).lower()
    assert '"agent"' not in json.dumps(prompt_payload).lower()


@pytest.mark.parametrize("strength", ["moderate", "strong"])
def test_probe_direct_selects_probe_winner_without_reconciler(monkeypatch: pytest.MonkeyPatch, strength: str):
    agents = _Agents(selected_method="tree_ensemble")
    result, _, _ = _run(monkeypatch, strength, agents, mode="probe_direct")
    assert result["selected_method"] in {"linear", "tree_ensemble"}
    assert result["selected_proposal_source"] in {"agent", "deterministic"}
    assert result["reconciliation"] is None
    assert agents.calls == 0


def test_always_reconcile_does_not_use_probe_gate(monkeypatch: pytest.MonkeyPatch):
    agents = _Agents(selected_method="linear")
    result, _, _ = _run(monkeypatch, "weak", agents, mode="always_reconcile")
    assert result["reconciliation"] is not None
    assert agents.calls == 1


def test_hard_validation_only_retains_valid_llm_on_disagreement(monkeypatch: pytest.MonkeyPatch):
    agents = _Agents(selected_method="tree_ensemble")
    result, _, _ = _run(monkeypatch, "strong", agents, mode="hard_validation_only")
    assert result["selected_method"] == "linear"
    assert result["reconciliation"] is None
    assert result["empirical_probe_invoked"] is False


def test_probe_disabled_is_safe_abstention(monkeypatch: pytest.MonkeyPatch):
    agents = _Agents()
    frame = _frame()
    split = freeze_supervised_split(frame, "target", "classification")
    plan, deterministic = _plans()
    monkeypatch.setattr(pipeline, "run_pairwise_model_probe", lambda *args, **kwargs: pytest.fail("probe must not run"))
    result = pipeline._validate_modeling_gate(
        agents,
        profile_dataframe(frame.iloc[list(split.train_row_positions)]),
        "classify target",
        plan,
        deterministic,
        [],
        {},
        offline=False,
        dataframe=frame,
        test_size=0.2,
        random_state=42,
        reconciliation_profile=profile_dataframe(frame.iloc[list(split.train_row_positions)]),
        split=split,
        row_positions=list(range(len(frame))),
        approved_target="target",
        approved_task="classification",
        empirical_probe_policy=EmpiricalProbePolicy(enabled=False),
    )
    assert result["empirical_probe_invoked"] is False
    assert result["probe_status"] == "unavailable"
    assert result["reconciliation"] is None
    assert result["selected_method"] == "linear"


def test_hard_invalid_proposal_skips_probe_and_keeps_hard_correction(monkeypatch: pytest.MonkeyPatch):
    frame = _frame()
    frame.loc[0, "signal"] = np.nan
    split = freeze_supervised_split(frame, "target", "classification")
    plan, deterministic = _plans()
    invalid = plan.model_copy(update={"preprocessing": PreprocessingContract(numeric_imputation="none")})
    agents = _Agents(selected_method="tree_ensemble")
    monkeypatch.setattr(pipeline, "run_pairwise_model_probe", lambda *args, **kwargs: pytest.fail("hard-invalid plans must not be probed"))
    result = pipeline._validate_modeling_gate(
        agents,
        profile_dataframe(frame.iloc[list(split.train_row_positions)]),
        "classify target",
        invalid,
        deterministic,
        [],
        {},
        offline=False,
        dataframe=frame,
        test_size=0.2,
        random_state=42,
        reconciliation_profile=profile_dataframe(frame.iloc[list(split.train_row_positions)]),
        split=split,
        row_positions=list(range(len(frame))),
        approved_target="target",
        approved_task="classification",
    )
    assert result["empirical_probe_invoked"] is False
    assert result["decision_path"] == "hard_validation_correction"
    assert result["gate_decision"]["reason_code"] == "hard_validation_failure"
    assert result["reconciliation"] is not None
