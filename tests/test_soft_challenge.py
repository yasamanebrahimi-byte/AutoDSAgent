import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.pipeline import _validate_before_training
from app.schemas import AgentPlan, ConflictResolution, DeterministicRecommendation, PreprocessingContract
from app.soft_challenge import SoftChallengePolicy, decide_soft_challenge
from app.validation import freeze_supervised_split


def _diagnostics() -> dict[str, object]:
    return {
        "sample_to_feature_ratio": 20.0,
        "effective_features_estimate": 5,
        "nonlinearity_signal": "low",
        "nonlinearity_applicable": True,
        "structural_complexity_signal": "low",
    }


def _calibration(rate: float, *, support: int = 20) -> dict[str, object]:
    return {
        "calibration_artifact_version": "test-v1",
        "regimes": {
            "classification/low/low/high": {
                "support": support,
                "challenge_win_rate": rate,
                "challenge_loss_rate": 1.0 - rate,
            }
        },
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"signal": np.linspace(-1.0, 1.0, 80), "target": ["yes", "no"] * 40})


def _plans(confidence: str = "low") -> tuple[AgentPlan, DeterministicRecommendation]:
    agent = AgentPlan(
        target_column="target",
        task_type="classification",
        recommended_method="linear",
        preprocessing=PreprocessingContract(numeric_scaling="standard"),
        reasoning="The initial proposal favors a transparent linear baseline for the observed signal.",
        confidence=0.7,
    )
    deterministic = DeterministicRecommendation(
        target_column="target",
        task_type="classification",
        recommended_method="tree_ensemble",
        preprocessing=PreprocessingContract(),
        reasoning="The deterministic challenger favors a supported tree family for the observed evidence.",
        evidence=["training-only structural evidence"],
        confidence=confidence,
        score_margin=20.0 if confidence == "high" else 4.0,
    )
    return agent, deterministic


def test_low_confidence_disagreement_abstains_and_preserves_agent():
    decision = decide_soft_challenge(
        agent_method="linear",
        deterministic_method="tree_ensemble",
        deterministic_confidence="low",
        score_margin=4.0,
        diagnostics=_diagnostics(),
        task_type="classification",
        calibration_artifact=_calibration(0.99),
    )
    assert decision.decision == "abstain"
    assert decision.decision_reason == "low_deterministic_confidence"


def test_high_confidence_reliable_disagreement_challenges():
    decision = decide_soft_challenge(
        agent_method="linear",
        deterministic_method="tree_ensemble",
        deterministic_confidence="high",
        score_margin=20.0,
        diagnostics=_diagnostics(),
        task_type="classification",
        calibration_artifact=_calibration(0.80),
    )
    assert decision.decision == "challenge"
    assert decision.calibration_support == 20


def test_high_confidence_unreliable_disagreement_abstains():
    decision = decide_soft_challenge(
        agent_method="linear",
        deterministic_method="tree_ensemble",
        deterministic_confidence="high",
        score_margin=20.0,
        diagnostics=_diagnostics(),
        task_type="classification",
        calibration_artifact=_calibration(0.40),
    )
    assert decision.decision == "abstain"


def test_insufficient_support_abstains_even_with_high_confidence():
    decision = decide_soft_challenge(
        agent_method="linear",
        deterministic_method="tree_ensemble",
        deterministic_confidence="high",
        score_margin=20.0,
        diagnostics=_diagnostics(),
        task_type="classification",
        calibration_artifact=_calibration(1.0, support=2),
    )
    assert decision.decision == "abstain"
    assert decision.decision_reason == "insufficient_calibration_support"


def test_agreement_bypasses_calibration():
    decision = decide_soft_challenge(
        agent_method="linear",
        deterministic_method="linear",
        deterministic_confidence="low",
        score_margin=None,
        diagnostics=None,
        task_type="classification",
        calibration_artifact={},
    )
    assert decision.decision == "agree"


def test_runtime_gate_abstention_does_not_reconcile():
    frame = _frame()
    split = freeze_supervised_split(frame, "target", "classification")
    agent, deterministic = _plans("low")

    class NoReconciliation:
        def reconcile(self, *args, **kwargs):
            raise AssertionError("abstention must not invoke reconciliation")

    result = _validate_before_training(
        NoReconciliation(),
        {"rows": len(frame)},
        "classify target",
        agent,
        deterministic,
        [],
        {},
        offline=False,
        dataframe=frame,
        split=split,
        row_positions=list(range(len(frame))),
        established_target="target",
        established_task="classification",
    )
    assert result["soft_challenge_decision"] == "abstain"
    assert result["soft_challenge"]["reconciliation_invoked"] is False
    assert result["final"]["recommended_method"] == "linear"


def test_challenge_can_still_lose_reconciliation(tmp_path: Path):
    frame = _frame()
    split = freeze_supervised_split(frame, "target", "classification")
    agent, deterministic = _plans("high")
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(_calibration(0.90)), encoding="utf-8")

    class AgentWins:
        def reconcile(self, question, profile, agent_plan, deterministic_payload):
            del question, profile, deterministic_payload
            return ConflictResolution(
                selected_target_column=agent_plan.target_column,
                selected_task_type=agent_plan.task_type,
                selected_method=agent_plan.recommended_method,
                selected_preprocessing=agent_plan.preprocessing,
                checks=["both_hard_validated", "agent_preserved"],
                justification="The initial agent plan remains a defensible choice after reviewing both training-only proposals.",
                confidence=0.8,
            )

    result = _validate_before_training(
        AgentWins(),
        {"rows": len(frame)},
        "classify target",
        agent,
        deterministic,
        [],
        {},
        offline=False,
        dataframe=frame,
        split=split,
        row_positions=list(range(len(frame))),
        established_target="target",
        established_task="classification",
        soft_challenge_policy=SoftChallengePolicy(calibration_artifact_path=str(calibration_path)),
    )
    assert result["soft_challenge_decision"] == "challenge"
    assert result["soft_challenge"]["reconciliation_invoked"] is True
    assert result["empirical_probe_invoked"] is True
    assert result["empirical_probe"]["status"] in {"completed", "unavailable", "failed"}
    assert result["final"]["selected_source"] == "agent"


def test_decision_is_reproducible():
    kwargs = {
        "agent_method": "linear",
        "deterministic_method": "tree_ensemble",
        "deterministic_confidence": "high",
        "score_margin": 20.0,
        "diagnostics": _diagnostics(),
        "task_type": "classification",
        "calibration_artifact": _calibration(0.80),
    }
    assert decide_soft_challenge(**kwargs).as_dict() == decide_soft_challenge(**kwargs).as_dict()
