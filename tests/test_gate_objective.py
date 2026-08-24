import pytest

from evaluation.metrics import (
    GateUtilityWeights,
    catastrophic_transition,
    classify_intervention_outcome,
    normalized_performance_delta,
    regret_reduction,
    summarize_gate_health,
)
from evaluation.policy_calibration import policy_candidates, select_policy_candidate


def _challenge(initial: float, final: float) -> dict:
    return {
        "trial_status": "completed",
        "benchmark_case": "fixture",
        "agent_initial_valid": True,
        "agreement_status": "disagreement",
        "method_disagreement": True,
        "soft_challenge": {"status": "disagreement", "decision": "challenge"},
        "agent_normalized_regret": initial,
        "gated_normalized_regret": final,
        "deterministic_normalized_regret": final,
    }


def test_intervention_outcome_direction_and_tolerance_for_both_tasks():
    assert normalized_performance_delta("classification", 0.70, 0.75) == pytest.approx(0.05)
    assert normalized_performance_delta("regression", 2.5, 2.0) == pytest.approx(0.5)
    assert classify_intervention_outcome(0.01, 0.02) == "neutral"
    assert classify_intervention_outcome(0.03, 0.02) == "improved"
    assert classify_intervention_outcome(-0.03, 0.02) == "worsened"


def test_regret_reduction_and_catastrophic_transitions():
    assert regret_reduction(0.30, 0.10) == pytest.approx(0.20)
    assert regret_reduction(0.10, 0.30) == pytest.approx(-0.20)
    assert catastrophic_transition(0.20, 0.05, 0.10) == {
        "initial_catastrophic": True,
        "final_catastrophic": False,
        "catastrophic_prevented": True,
        "catastrophic_introduced": False,
    }
    assert catastrophic_transition(0.05, 0.20, 0.10)["catastrophic_introduced"] is True


def test_gate_health_reports_precision_yield_harm_recall_and_utility():
    records = [
        _challenge(0.20, 0.00),  # improved and catastrophic prevented
        _challenge(0.00, 0.20),  # worsened and catastrophic introduced
        _challenge(0.05, 0.045),  # neutral
        {
            "trial_status": "completed",
            "benchmark_case": "fixture",
            "agent_initial_valid": True,
            "agreement_status": "disagreement",
            "method_disagreement": True,
            "soft_challenge": {"status": "disagreement", "decision": "abstain"},
            "agent_normalized_regret": 0.20,
            "gated_normalized_regret": 0.20,
            "deterministic_normalized_regret": 0.00,
        },
    ]
    health = summarize_gate_health(
        records,
        neutral_tolerance=0.02,
        catastrophic_threshold=0.10,
        weights=GateUtilityWeights(),
    )
    assert health["intervention_precision"] == pytest.approx(0.5)
    assert health["challenge_yield"] == pytest.approx(1 / 3)
    assert health["harmful_intervention_rate"] == pytest.approx(1 / 3)
    assert health["unnecessary_intervention_rate"] == pytest.approx(1 / 3)
    assert health["missed_rescue_count"] == 1
    assert health["catastrophic_prevented_count"] == 1
    assert health["catastrophic_introduced_count"] == 1
    assert health["net_catastrophic_prevention"] == 0
    assert health["utility"]["total_utility"] == pytest.approx(
        1.0 - 2.0 - 0.25 + 3.0 - 5.0 - 1.0
    )


def test_new_calibration_objective_prefers_safer_intervention_over_exact_match():
    candidates = policy_candidates()[:2]
    aggregates = {
        "current": {
            "policy_candidate": "current",
            "policy_complexity": 0,
            "utility": 1.0,
            "harmful_intervention_rate": 0.30,
            "catastrophic_introduced_count": 2,
            "catastrophic_prevented_count": 3,
            "intervention_precision": 0.55,
            "challenge_recall": 0.80,
            "median_regret_reduction": 0.01,
            "unnecessary_intervention_rate": 0.20,
            "exact_reference_match_rate": 0.90,
        },
        "nonlinear_sensitive": {
            "policy_candidate": "nonlinear_sensitive",
            "policy_complexity": 3,
            "utility": 4.0,
            "harmful_intervention_rate": 0.05,
            "catastrophic_introduced_count": 0,
            "catastrophic_prevented_count": 4,
            "intervention_precision": 0.85,
            "challenge_recall": 0.60,
            "median_regret_reduction": 0.03,
            "unnecessary_intervention_rate": 0.10,
            "exact_reference_match_rate": 0.70,
        },
    }
    selection = select_policy_candidate(aggregates, candidates)
    assert selection["selected_candidate"] == "nonlinear_sensitive"
    assert selection["recommendation"] == "promote"
    assert selection["gate_objective_version"] == "intervention-quality-v1"
    assert selection["selection_rationale"]
