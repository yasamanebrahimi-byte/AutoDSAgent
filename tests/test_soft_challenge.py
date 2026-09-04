from app.soft_challenge import decide_soft_challenge


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
