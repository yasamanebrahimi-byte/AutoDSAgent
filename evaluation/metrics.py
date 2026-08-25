"""Transparent, source-aware metrics for evaluation records."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import random
from statistics import median, pstdev
from typing import Any


GATE_OBJECTIVE_VERSION = "intervention-quality-v1"


DEFAULT_THRESHOLDS = {
    "classification_regret": 0.01,
    "regression_normalized_regret": 0.02,
    "paired_normalized_regret": 0.02,
    # This is the single tolerance used to classify an evaluation-only
    # intervention outcome.  It is expressed in normalized-regret units so
    # it is comparable across the supported task metrics.
    "neutral_tolerance": 0.02,
    "catastrophic_regret_threshold": 0.10,
}


@dataclass(frozen=True)
class GateUtilityWeights:
    """Small, explicit, asymmetric weights used by development calibration."""

    improvement: float = 1.0
    worsening: float = 2.0
    neutral_intervention: float = 0.25
    catastrophic_prevention: float = 3.0
    catastrophic_introduction: float = 5.0
    missed_rescue: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


DEFAULT_GATE_UTILITY_WEIGHTS = GateUtilityWeights()


def normalized_performance_delta(
    task_type: str,
    initial_score: float | None,
    final_score: float | None,
) -> float | None:
    """Return a direction-normalized score delta where positive means better."""

    if initial_score is None or final_score is None:
        return None
    if task_type == "classification":
        return float(final_score - initial_score)
    if task_type == "regression":
        return float(initial_score - final_score)
    raise ValueError(f"Unsupported task type: {task_type!r}")


def classify_intervention_outcome(
    normalized_delta: float | None,
    neutral_tolerance: float,
) -> str:
    """Classify a direction-normalized delta without treating microscopic noise as harm."""

    if normalized_delta is None:
        return "not_comparable"
    if abs(float(normalized_delta)) <= float(neutral_tolerance):
        return "neutral"
    return "improved" if normalized_delta > 0 else "worsened"


def regret_reduction(
    initial_regret: float | None,
    final_regret: float | None,
) -> float | None:
    """Return positive regret reduction when the gated choice is better."""

    if initial_regret is None or final_regret is None:
        return None
    return float(initial_regret - final_regret)


def catastrophic_transition(
    initial_regret: float | None,
    final_regret: float | None,
    threshold: float,
) -> dict[str, bool]:
    """Classify catastrophic-regret status and its transition direction."""

    initial = bool(initial_regret is not None and initial_regret >= threshold)
    final = bool(final_regret is not None and final_regret >= threshold)
    return {
        "initial_catastrophic": initial,
        "final_catastrophic": final,
        "catastrophic_prevented": initial and not final,
        "catastrophic_introduced": not initial and final,
    }


def gate_utility(
    *,
    improved_count: int,
    worsened_count: int,
    neutral_count: int,
    catastrophic_prevented_count: int,
    catastrophic_introduced_count: int,
    missed_rescue_count: int = 0,
    weights: GateUtilityWeights = DEFAULT_GATE_UTILITY_WEIGHTS,
) -> dict[str, float | dict[str, float]]:
    """Compute a decomposable utility from counts; no hidden normalization is used."""

    contributions = {
        "improvement_reward": float(improved_count * weights.improvement),
        "worsening_penalty": float(-worsened_count * weights.worsening),
        "unnecessary_intervention_penalty": float(-neutral_count * weights.neutral_intervention),
        "catastrophic_prevention_reward": float(
            catastrophic_prevented_count * weights.catastrophic_prevention
        ),
        "catastrophic_introduction_penalty": float(
            -catastrophic_introduced_count * weights.catastrophic_introduction
        ),
        "missed_rescue_penalty": float(-missed_rescue_count * weights.missed_rescue),
    }
    total = float(sum(contributions.values()))
    return {
        **contributions,
        "total_utility": total,
        "weights": weights.as_dict(),
    }


def regret(task_type: str, best_score: float | None, selected_score: float | None) -> float | None:
    """Return non-negative regret relative to the empirical-reference score."""

    if best_score is None or selected_score is None:
        return None
    if task_type == "classification":
        return float(max(0.0, best_score - selected_score))
    if task_type == "regression":
        return float(max(0.0, selected_score - best_score))
    raise ValueError(f"Unsupported task type: {task_type!r}")


def normalized_regret(
    task_type: str, best_score: float | None, selected_score: float | None
) -> float | None:
    """Normalize regret without division by zero and with larger=worse direction."""

    raw = regret(task_type, best_score, selected_score)
    if raw is None:
        return None
    if task_type == "classification":
        # Macro F1 is bounded [0, 1], so an absolute gap is interpretable.
        return raw
    if task_type == "regression":
        return float(raw / max(abs(float(best_score or 0.0)), 1e-12))
    raise ValueError(f"Unsupported task type: {task_type!r}")


def approximate_match(task_type: str, normalized_value: float | None, thresholds: dict[str, float]) -> bool:
    if normalized_value is None:
        return False
    key = "classification_regret" if task_type == "classification" else "regression_normalized_regret"
    return float(normalized_value) <= float(thresholds[key])


def _mean(values: list[float] | Any) -> float | None:
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def _median(values: list[float] | Any) -> float | None:
    values = list(values)
    return float(median(values)) if values else None


def _std(values: list[float] | Any) -> float | None:
    values = list(values)
    return float(pstdev(values)) if values else None


def _rate(count: int, denominator: int) -> float | None:
    return float(count / denominator) if denominator else None


def _method_distribution(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    methods = [str(record[field]) for record in records if record.get(field) is not None]
    counts = Counter(methods)
    return {
        method: {"count": count, "rate": float(count / len(methods)) if methods else 0.0}
        for method, count in sorted(counts.items())
    }


def _stability(records: list[dict[str, Any]]) -> dict[str, Any]:
    methods = [str(record["agent_initial_method"]) for record in records if record.get("agent_initial_method")]
    counts = Counter(methods)
    pair_count = len(methods) * (len(methods) - 1) // 2
    same_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    modal_method, modal_count = (counts.most_common(1)[0] if counts else (None, 0))
    return {
        "trial_count": len(methods),
        "method_distribution": _method_distribution(records, "agent_initial_method"),
        "unique_initial_methods_selected": len(counts),
        "modal_method": modal_method,
        "modal_method_frequency": modal_count,
        "modal_method_rate": float(modal_count / len(methods)) if methods else None,
        "pairwise_consistency": float(same_pairs / pair_count) if pair_count else None,
    }


def _outcome(record: dict[str, Any], tolerance: float) -> str:
    details = record.get("gate_outcome_details") or {}
    stored = details.get("outcome") or record.get("gate_outcome")
    if stored in {"improved", "worsened", "neutral", "tie"}:
        return "neutral" if stored == "tie" else str(stored)
    initial = record.get("agent_normalized_regret")
    gated = record.get("gated_normalized_regret")
    if initial is None or gated is None:
        return "not_comparable"
    return classify_intervention_outcome(
        regret_reduction(float(initial), float(gated)), tolerance
    )


def _validation_failure_codes(record: dict[str, Any]) -> set[str]:
    codes = {
        str(failure.get("code"))
        for failure in record.get("agent_initial_validation_failures", [])
        if failure.get("code")
    }
    final_validation = record.get("final_validation") or {}
    codes.update(
        str(failure.get("code"))
        for failure in final_validation.get("failures", [])
        if failure.get("code")
    )
    codes.update(
        str(check.get("code"))
        for check in final_validation.get("checks", [])
        if check.get("status") == "failed" and check.get("code")
    )
    return codes


def _paired_improvements(records: list[dict[str, Any]]) -> list[float]:
    return [
        float(record["paired_cv_improvement"])
        for record in records
        if record.get("paired_cv_improvement") is not None
    ]


def _holdout_improvements(records: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        initial = record.get("agent_initial_holdout_metrics") or {}
        gated = record.get("gated_final_holdout_metrics") or {}
        task_type = record.get("task_type")
        if task_type == "classification" and initial.get("macro_f1") is not None and gated.get("macro_f1") is not None:
            values.append(float(gated["macro_f1"]) - float(initial["macro_f1"]))
        elif task_type == "regression" and initial.get("rmse") is not None and gated.get("rmse") is not None:
            values.append(float(initial["rmse"]) - float(gated["rmse"]))
    return values


def _reconciliation_rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("reconciliation_status") == "succeeded"]
    sourced = [
        record for record in successful
        if record.get("reconciliation_method_source") in {"agent", "deterministic"}
    ]
    agent_count = sum(record.get("reconciliation_method_source") == "agent" for record in sourced)
    deterministic_count = sum(record.get("reconciliation_method_source") == "deterministic" for record in sourced)
    a_count = sum(record.get("selected_proposal") == "A" for record in successful)
    b_count = sum(record.get("selected_proposal") == "B" for record in successful)
    proposal_labeled = [
        record for record in successful if record.get("selected_proposal") in {"A", "B"}
    ]
    a_rate = _rate(a_count, len(proposal_labeled))
    b_rate = _rate(b_count, len(proposal_labeled))
    pairs: dict[str, list[dict[str, Any]]] = {}
    for record in proposal_labeled:
        pair_id = record.get("order_swap_pair_id")
        if pair_id is not None:
            pairs.setdefault(str(pair_id), []).append(record)
    pair_results: list[bool] = []
    for pair in pairs.values():
        sources = {
            record.get("selected_proposal_source") or record.get("reconciliation_method_source")
            for record in pair
        }
        if len(pair) >= 2 and len(sources) == 1 and None not in sources:
            pair_results.append(True)
        elif len(pair) >= 2:
            pair_results.append(False)
    return {
        "reconciliation_sided_with_agent_count": agent_count,
        "reconciliation_sided_with_deterministic_count": deterministic_count,
        "reconciliation_sided_with_agent_rate": _rate(agent_count, len(sourced)),
        "reconciliation_sided_with_deterministic_rate": _rate(deterministic_count, len(sourced)),
        "reconciliation_a_selected_count": a_count,
        "reconciliation_b_selected_count": b_count,
        "reconciliation_a_selected_rate": a_rate,
        "reconciliation_b_selected_rate": b_rate,
        "reconciliation_a_b_selection_imbalance": (
            abs(a_rate - b_rate) if a_rate is not None and b_rate is not None else None
        ),
        "order_swap_pair_count": len(pair_results),
        "order_swap_consistency_rate": _rate(sum(pair_results), len(pair_results)),
        "order_swap_flip_rate": (
            1.0 - (sum(pair_results) / len(pair_results)) if pair_results else None
        ),
    }


def _initial_hard_invalid(record: dict[str, Any]) -> bool:
    artifact = record.get("hard_validation") or {}
    if "initial_hard_invalid" in artifact:
        return bool(artifact["initial_hard_invalid"])
    if record.get("initial_hard_invalid") is not None:
        return bool(record["initial_hard_invalid"])
    return record.get("agent_initial_valid") is False


def _hard_intervention(record: dict[str, Any]) -> bool:
    artifact = record.get("hard_validation") or {}
    if "intervention_required" in artifact:
        return bool(artifact["intervention_required"])
    if record.get("hard_validation_intervened") is not None:
        return bool(record["hard_validation_intervened"])
    return record.get("unsafe_plan_intercepted") is True


def _final_hard_invalid(record: dict[str, Any]) -> bool:
    if record.get("final_hard_invalid") is not None:
        return bool(record["final_hard_invalid"])
    return record.get("final_valid") is False


def _soft_status(record: dict[str, Any]) -> str | None:
    artifact = record.get("soft_challenge") or {}
    if artifact.get("status") in {"agreement", "disagreement", "invalid", "unavailable"}:
        return str(artifact["status"])
    if record.get("soft_challenge_status") in {"agreement", "disagreement", "invalid", "unavailable"}:
        return str(record["soft_challenge_status"])
    if record.get("agreement_status") in {"agreement", "disagreement"}:
        return str(record["agreement_status"])
    if record.get("agreement_status") == "llm_only":
        return "disagreement" if record.get("method_disagreement") is True else "agreement"
    return None


def _soft_decision(record: dict[str, Any]) -> str | None:
    artifact = record.get("soft_challenge") or {}
    decision = artifact.get("decision")
    if decision in {"agree", "challenge", "abstain"}:
        return str(decision)
    status_detail = artifact.get("status_detail")
    if status_detail == "challenged":
        return "challenge"
    if status_detail == "abstained":
        return "abstain"
    if status_detail == "agreement":
        return "agree"
    if record.get("reconciliation_invoked") is True and _soft_status(record) == "disagreement":
        return "challenge"
    if _soft_status(record) == "disagreement":
        return "abstain"
    if _soft_status(record) == "agreement":
        return "agree"
    return None


def _soft_challenges(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if _soft_status(record) == "disagreement"
        and (
            record.get("hard_validation_status") == "passed"
            or record.get("agent_initial_valid") is True
            or not record.get("hard_validation")
        )
    ]


def _soft_decision_records(records: list[dict[str, Any]], decision: str) -> list[dict[str, Any]]:
    return [record for record in _soft_challenges(records) if _soft_decision(record) == decision]


def _soft_outcome_counts(records: list[dict[str, Any]], tolerance: float) -> dict[str, int]:
    challenges = _soft_decision_records(records, "challenge")
    outcomes = Counter(_outcome(record, tolerance) for record in challenges)
    return {
        "improved": outcomes.get("improved", 0),
        "worsened": outcomes.get("worsened", 0),
        "neutral": outcomes.get("neutral", 0),
        "not_comparable": outcomes.get("not_comparable", 0),
    }


def _bootstrap_interval(values: list[float], seed: int = 20260824, samples: int = 1000) -> dict[str, Any]:
    """Return a deterministic percentile bootstrap interval, or an explicit small-sample result."""

    if not values:
        return {"lower": None, "upper": None, "support": 0, "stable": False}
    if len(values) < 2:
        value = float(values[0])
        return {"lower": value, "upper": value, "support": len(values), "stable": False}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(draw) / len(draw))
    estimates.sort()
    low_index = int(0.025 * (len(estimates) - 1))
    high_index = int(0.975 * (len(estimates) - 1))
    return {
        "lower": float(estimates[low_index]),
        "upper": float(estimates[high_index]),
        "support": len(values),
        "stable": len(values) >= 20,
    }


def _decision_path(record: dict[str, Any]) -> str:
    decision = _soft_decision(record)
    if decision in {None, "agree"} or _soft_status(record) == "agreement":
        return "agreement"
    if decision == "abstain":
        return "abstention"
    if decision != "challenge":
        return "unavailable"
    probe = record.get("empirical_probe") or {}
    if record.get("empirical_probe_invoked") is not True or probe.get("status") != "completed":
        return "challenge_without_successful_probe"
    strength = str(probe.get("evidence_strength") or "unknown")
    return f"challenge_plus_{strength}_probe"


def _confidence_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str) and value.lower() in {"low", "medium", "high"}:
        return value.lower()
    value = float(value)
    if value < 0.4:
        return "low"
    if value < 0.7:
        return "medium"
    return "high"


def _record_regret_reduction(record: dict[str, Any]) -> float | None:
    details = record.get("gate_outcome_details") or {}
    if details.get("regret_reduction") is not None:
        return float(details["regret_reduction"])
    return regret_reduction(
        record.get("agent_normalized_regret"), record.get("gated_normalized_regret")
    )


def _alternative_delta(record: dict[str, Any]) -> float | None:
    initial = record.get("agent_normalized_regret")
    deterministic = record.get("deterministic_normalized_regret")
    return regret_reduction(initial, deterministic)


def _gate_health(
    records: list[dict[str, Any]],
    *,
    tolerance: float,
    catastrophic_threshold: float,
    weights: GateUtilityWeights,
) -> dict[str, Any]:
    """Summarize intervention quality without relying on family-name correctness."""

    valid = [record for record in records if record.get("trial_status") != "failed"]
    paired = [record for record in valid if _record_regret_reduction(record) is not None]
    challenges = _soft_decision_records(valid, "challenge")
    abstentions = _soft_decision_records(valid, "abstain")
    challenge_outcomes = [
        _outcome(record, tolerance)
        for record in challenges
        if _record_regret_reduction(record) is not None
    ]
    improved = challenge_outcomes.count("improved")
    worsened = challenge_outcomes.count("worsened")
    neutral = challenge_outcomes.count("neutral")
    comparable_challenges = improved + worsened + neutral
    deltas = [float(_record_regret_reduction(record)) for record in paired]
    challenge_deltas = [
        float(_record_regret_reduction(record))
        for record in challenges
        if _record_regret_reduction(record) is not None
    ]
    positive = [delta for delta in challenge_deltas if delta > tolerance]
    negative = [delta for delta in challenge_deltas if delta < -tolerance]
    initial_catastrophic = sum(
        bool(
            record.get("agent_normalized_regret") is not None
            and float(record["agent_normalized_regret"]) >= catastrophic_threshold
        )
        for record in paired
    )
    final_catastrophic = sum(
        bool(
            record.get("gated_normalized_regret") is not None
            and float(record["gated_normalized_regret"]) >= catastrophic_threshold
        )
        for record in paired
    )
    prevented = sum(
        bool(
            record.get("agent_normalized_regret") is not None
            and record.get("gated_normalized_regret") is not None
            and float(record["agent_normalized_regret"]) >= catastrophic_threshold
            and float(record["gated_normalized_regret"]) < catastrophic_threshold
        )
        for record in paired
    )
    introduced = sum(
        bool(
            record.get("agent_normalized_regret") is not None
            and record.get("gated_normalized_regret") is not None
            and float(record["agent_normalized_regret"]) < catastrophic_threshold
            and float(record["gated_normalized_regret"]) >= catastrophic_threshold
        )
        for record in paired
    )
    missed_rescue = 0
    good_abstention = 0
    avoided_harm = 0
    neutral_abstention = 0
    for record in abstentions:
        alternative_delta = _alternative_delta(record)
        if alternative_delta is None:
            continue
        outcome = classify_intervention_outcome(alternative_delta, tolerance)
        if outcome == "improved":
            missed_rescue += 1
        elif outcome == "worsened":
            good_abstention += 1
            avoided_harm += 1
        else:
            neutral_abstention += 1
    beneficial_opportunities = sum(
        _alternative_delta(record) is not None
        and _alternative_delta(record) > tolerance
        for record in valid
        if _soft_status(record) == "disagreement"
    )
    beneficial_challenges = sum(outcome == "improved" for outcome in challenge_outcomes)
    utility = gate_utility(
        improved_count=improved,
        worsened_count=worsened,
        neutral_count=neutral,
        catastrophic_prevented_count=prevented,
        catastrophic_introduced_count=introduced,
        missed_rescue_count=missed_rescue,
        weights=weights,
    )
    outcome_counts = {
        "improved": improved,
        "worsened": worsened,
        "neutral": neutral,
        # Legacy aliases remain in the serialized summary for historical data.
        "tie": neutral,
        "not_comparable": len(challenges) - comparable_challenges,
    }
    path_metrics: dict[str, dict[str, Any]] = {}
    for path in sorted({_decision_path(record) for record in valid}):
        path_records = [record for record in valid if _decision_path(record) == path]
        path_deltas = [
            float(_record_regret_reduction(record))
            for record in path_records
            if _record_regret_reduction(record) is not None
        ]
        path_outcomes = Counter(
            _outcome(record, tolerance)
            for record in path_records
            if _record_regret_reduction(record) is not None
        )
        path_metrics[path] = {
            "count": len(path_records),
            "mean_gate_delta": _mean(path_deltas),
            "median_gate_delta": _median(path_deltas),
            "improved": path_outcomes.get("improved", 0),
            "worsened": path_outcomes.get("worsened", 0),
            "neutral": path_outcomes.get("neutral", 0),
            "mean_regret_reduction": _mean(path_deltas),
        }
    confidence_metrics: dict[str, dict[str, Any]] = {}
    for bucket in sorted({_confidence_bucket(record.get("deterministic_confidence")) for record in valid}):
        bucket_records = [
            record
            for record in challenges
            if _confidence_bucket(record.get("deterministic_confidence")) == bucket
        ]
        bucket_outcomes = Counter(
            _outcome(record, tolerance)
            for record in bucket_records
            if _record_regret_reduction(record) is not None
        )
        bucket_deltas = [
            float(_record_regret_reduction(record))
            for record in bucket_records
            if _record_regret_reduction(record) is not None
        ]
        confidence_metrics[bucket] = {
            "challenge_count": len(bucket_records),
            "improvement_rate": _rate(bucket_outcomes.get("improved", 0), len(bucket_records)),
            "harm_rate": _rate(bucket_outcomes.get("worsened", 0), len(bucket_records)),
            "mean_regret_reduction": _mean(bucket_deltas),
            "catastrophic_prevention_count": sum(
                bool((record.get("gate_outcome_details") or {}).get("catastrophic_prevented"))
                for record in bucket_records
            ),
        }
    probe_metrics: dict[str, dict[str, Any]] = {}
    for strength in sorted({
        str((record.get("empirical_probe") or {}).get("evidence_strength"))
        for record in challenges
        if (record.get("empirical_probe") or {}).get("evidence_strength")
    }):
        probe_records = [
            record for record in challenges
            if (record.get("empirical_probe") or {}).get("evidence_strength") == strength
        ]
        probe_outcomes = Counter(
            _outcome(record, tolerance)
            for record in probe_records
            if _record_regret_reduction(record) is not None
        )
        probe_metrics[strength] = {
            "challenge_count": len(probe_records),
            "improved": probe_outcomes.get("improved", 0),
            "worsened": probe_outcomes.get("worsened", 0),
            "neutral": probe_outcomes.get("neutral", 0),
            "harm_rate": _rate(probe_outcomes.get("worsened", 0), len(probe_records)),
            "mean_regret_reduction": _mean([
                float(_record_regret_reduction(record))
                for record in probe_records
                if _record_regret_reduction(record) is not None
            ]),
        }
    total_positive_gain = float(sum(positive)) if positive else 0.0
    sorted_positive = sorted(positive, reverse=True)
    gain_concentration = {
        "total_positive_gain": total_positive_gain,
        "top_1_fraction": (
            float(sum(sorted_positive[:1]) / total_positive_gain) if total_positive_gain else None
        ),
        "top_5_fraction": (
            float(sum(sorted_positive[:5]) / total_positive_gain) if total_positive_gain else None
        ),
        "positive_intervention_count": len(positive),
    }
    regime_groups: dict[str, list[dict[str, Any]]] = {}
    for record in valid:
        deterministic_artifact = record.get("deterministic_recommendation") or {}
        diagnostics = deterministic_artifact.get("diagnostics", {})
        if not diagnostics:
            diagnostics = record.get("diagnostics") or {}
        interaction = diagnostics.get("interaction_signals") or {}
        boundary = diagnostics.get("classification_boundary_signals") or {}
        labels = {
            "task": str(record.get("task_type", "unknown")),
            "interaction": str(interaction.get("interaction_strength", "unknown")),
            "boundary": str(boundary.get("boundary_complexity", "unknown")),
        }
        for dimension, value in labels.items():
            if value != "unknown":
                regime_groups.setdefault(f"{dimension}:{value}", []).append(record)
    regime_metrics: dict[str, dict[str, Any]] = {}
    suppressed_regimes: list[str] = []
    for key, group in sorted(regime_groups.items()):
        if len(group) < 2:
            suppressed_regimes.append(key)
            continue
        group_deltas = [
            float(_record_regret_reduction(record))
            for record in group
            if _record_regret_reduction(record) is not None
        ]
        group_outcomes = Counter(
            _outcome(record, tolerance)
            for record in group
            if _record_regret_reduction(record) is not None
        )
        regime_metrics[key] = {
            "support": len(group),
            "improved": group_outcomes.get("improved", 0),
            "worsened": group_outcomes.get("worsened", 0),
            "neutral": group_outcomes.get("neutral", 0),
            "mean_regret_reduction": _mean(group_deltas),
        }
    return {
        "total_disagreements": sum(_soft_status(record) == "disagreement" for record in valid),
        "total_challenges": len(challenges),
        "total_abstentions": len(abstentions),
        "improved_interventions": improved,
        "worsened_interventions": worsened,
        "neutral_interventions": neutral,
        "intervention_precision": _rate(improved, improved + worsened),
        "intervention_precision_including_neutral": _rate(improved, len(challenges)),
        "challenge_yield": _rate(improved, len(challenges)),
        "harmful_intervention_rate": _rate(worsened, len(challenges)),
        "unnecessary_intervention_count": neutral,
        "unnecessary_intervention_rate": _rate(neutral, len(challenges)),
        "challenge_recall": _rate(beneficial_challenges, beneficial_opportunities),
        "beneficial_challenge_count": beneficial_challenges,
        "beneficial_opportunity_count": beneficial_opportunities,
        "missed_rescue_count": missed_rescue,
        "good_abstention_count": good_abstention,
        "avoided_harm_count": avoided_harm,
        "neutral_abstention_count": neutral_abstention,
        "intervention_f1": (
            _rate(2 * improved, 2 * improved + worsened + (beneficial_opportunities - beneficial_challenges))
            if 2 * improved + worsened + (beneficial_opportunities - beneficial_challenges)
            else None
        ),
        "mean_regret_reduction": _mean(deltas),
        "median_regret_reduction": _median(deltas),
        "regret_reduction_ci": _bootstrap_interval(deltas),
        "mean_intervention_regret_reduction": _mean(challenge_deltas),
        "median_intervention_regret_reduction": _median(challenge_deltas),
        "positive_delta_mean": _mean(positive),
        "positive_delta_median": _median(positive),
        "negative_delta_mean": _mean(negative),
        "negative_delta_median": _median(negative),
        "worst_regression": min(negative) if negative else None,
        "largest_improvement": max(positive) if positive else None,
        "initial_catastrophic_count": initial_catastrophic,
        "final_catastrophic_count": final_catastrophic,
        "catastrophic_prevented_count": prevented,
        "catastrophic_introduced_count": introduced,
        "initial_agent_catastrophic_count": initial_catastrophic,
        "final_gate_catastrophic_count": final_catastrophic,
        "catastrophic_regret_prevented_count": prevented,
        "catastrophic_regret_introduced_count": introduced,
        "catastrophic_prevented_rate": _rate(prevented, initial_catastrophic),
        "catastrophic_introduced_rate": _rate(introduced, len(paired) - initial_catastrophic),
        "net_catastrophic_prevention": prevented - introduced,
        "utility": utility,
        "outcome_counts": outcome_counts,
        "path_metrics": path_metrics,
        "confidence_metrics": confidence_metrics,
        "probe_strength_metrics": probe_metrics,
        "regime_metrics": regime_metrics,
        "suppressed_regime_slices": suppressed_regimes,
        "gain_concentration": gain_concentration,
        "comparable_trial_count": len(paired),
    }


def _dataset_weighted_health(
    records: list[dict[str, Any]],
    *,
    tolerance: float,
    catastrophic_threshold: float,
    weights: GateUtilityWeights,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("benchmark_case", record.get("dataset_id", "unknown"))), []).append(record)
    per_dataset = {
        dataset: _gate_health(group, tolerance=tolerance, catastrophic_threshold=catastrophic_threshold, weights=weights)
        for dataset, group in sorted(grouped.items())
    }
    numeric = (
        "intervention_precision", "challenge_yield", "harmful_intervention_rate",
        "unnecessary_intervention_rate", "challenge_recall", "mean_regret_reduction",
        "median_regret_reduction", "catastrophic_prevented_rate", "catastrophic_introduced_rate",
    )
    return {
        "dataset_count": len(per_dataset),
        "per_dataset": per_dataset,
        **{name: _mean(
            float(item[name]) for item in per_dataset.values() if item.get(name) is not None
        ) for name in numeric},
    }


def summarize_gate_health(
    records: list[dict[str, Any]],
    *,
    neutral_tolerance: float = DEFAULT_THRESHOLDS["neutral_tolerance"],
    catastrophic_threshold: float = DEFAULT_THRESHOLDS["catastrophic_regret_threshold"],
    weights: GateUtilityWeights = DEFAULT_GATE_UTILITY_WEIGHTS,
) -> dict[str, Any]:
    """Public shared utility for trial- and policy-calibration intervention summaries."""

    return _gate_health(
        records,
        tolerance=neutral_tolerance,
        catastrophic_threshold=catastrophic_threshold,
        weights=weights,
    )


def _empirical_probe_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize runtime probe use without exposing it to runtime decisions."""

    invoked = [record for record in records if record.get("empirical_probe_invoked") is True]
    completed = [
        record for record in invoked
        if (record.get("empirical_probe") or {}).get("status") == "completed"
    ]
    unavailable = [
        record for record in invoked
        if (record.get("empirical_probe") or {}).get("status") == "unavailable"
    ]
    failed = [
        record for record in invoked
        if (record.get("empirical_probe") or {}).get("status") == "failed"
    ]
    winners = Counter((record.get("empirical_probe") or {}).get("winner") for record in completed)
    strengths = Counter((record.get("empirical_probe") or {}).get("evidence_strength") for record in completed)
    fits = [
        float((record.get("empirical_probe") or {}).get("fit_count"))
        for record in completed
        if (record.get("empirical_probe") or {}).get("fit_count") is not None
    ]
    winner_final_matches = []
    winner_reference_matches = []
    useful = []
    for record in completed:
        probe = record.get("empirical_probe") or {}
        winner = probe.get("winner")
        if winner not in {"A", "B"}:
            continue
        proposal = probe.get(f"proposal_{winner.lower()}") or {}
        method = proposal.get("model_family")
        if method is not None and record.get("final_method") is not None:
            winner_final_matches.append(record.get("final_method") == method)
        if method is not None and record.get("empirical_best_method") is not None:
            winner_reference_matches.append(method == record.get("empirical_best_method"))
        if probe.get("evidence_strength") in {"moderate", "strong"}:
            initial = record.get("agent_normalized_regret")
            gated = record.get("gated_normalized_regret")
            if initial is not None and gated is not None:
                useful.append(float(gated) <= float(initial))
    return {
        "probe_invocation_count": len(invoked),
        "probe_invocation_rate": _rate(
            len(invoked),
            sum(record.get("method_disagreement") is True for record in records),
        ),
        "probe_completion_count": len(completed),
        "probe_unavailable_count": len(unavailable),
        "probe_failed_count": len(failed),
        "probe_completion_rate": _rate(len(completed), len(invoked)),
        "probe_a_win_count": winners.get("A", 0),
        "probe_b_win_count": winners.get("B", 0),
        "probe_tie_count": winners.get("tie", 0),
        "probe_evidence_strength_counts": dict(sorted(strengths.items())),
        "probe_evidence_strength_distribution": dict(sorted(Counter(
            (record.get("empirical_probe") or {}).get("evidence_strength") or "unknown"
            for record in invoked
        ).items())),
        "probe_winner_matched_final_selection_count": sum(winner_final_matches),
        "probe_winner_matched_final_selection_rate": _rate(
            sum(winner_final_matches), len(winner_final_matches)
        ),
        "probe_winner_matched_empirical_reference_count": sum(winner_reference_matches),
        "probe_winner_matched_empirical_reference_rate": _rate(
            sum(winner_reference_matches), len(winner_reference_matches)
        ),
        "probe_moderate_or_strong_following_improved_or_tied_count": sum(useful),
        "probe_moderate_or_strong_following_improved_or_tied_rate": _rate(sum(useful), len(useful)),
        "probe_conditional_usefulness_rate": _rate(sum(useful), len(useful)),
        "probe_mean_model_fits": _mean(fits),
        "probe_median_model_fits": _median(fits),
    }


def summarize_trials(
    trials: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
    utility_weights: GateUtilityWeights | dict[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate rows while keeping operational and OpenAI-only views separate."""

    configured = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    neutral_tolerance = float(
        configured.get("neutral_tolerance", configured["paired_normalized_regret"])
    )
    if utility_weights is None:
        weights = DEFAULT_GATE_UTILITY_WEIGHTS
    elif isinstance(utility_weights, GateUtilityWeights):
        weights = utility_weights
    else:
        weights = GateUtilityWeights(**utility_weights)
    total = len(trials)
    failed = [record for record in trials if record.get("trial_status") == "failed"]
    completed = [record for record in trials if record.get("trial_status") != "failed"]
    openai = [record for record in completed if record.get("agent_source") == "openai"]
    clean = [record for record in trials if record.get("perturbation_id", "clean") == "clean"]
    deterministic_available = [
        record for record in completed if record.get("deterministic_recommendation") is not None
    ]
    initial_invalid = [record for record in completed if record.get("agent_initial_valid") is False]
    initial_valid = [record for record in completed if record.get("agent_initial_valid") is True]
    final_valid = [record for record in completed if record.get("final_valid") is True]
    agreement = [record for record in deterministic_available if record.get("agreement_status") == "agreement"]
    disagreements = [record for record in deterministic_available if record.get("agreement_status") == "disagreement"]
    recon = [record for record in completed if record.get("reconciliation_invoked")]
    recon_success = [record for record in recon if record.get("reconciliation_status") == "succeeded"]
    soft_challenge_records = _soft_challenges(completed)
    intercepted = [record for record in initial_invalid if record.get("unsafe_plan_intercepted") is True]
    intentionally_unsafe = [
        record for record in completed
        if record.get("perturbation_id") != "clean"
        and record.get("perturbation", {}).get("kind") == "deterministic_invariant_violation"
    ]
    intentionally_unsafe_intercepted = [
        record for record in intentionally_unsafe if record.get("unsafe_plan_intercepted") is True
    ]
    failure_counts = Counter(code for record in trials for code in _validation_failure_codes(record))

    def method_match(records: list[dict[str, Any]], field: str) -> tuple[int, int]:
        eligible = [
            record for record in records
            if record.get(field) is not None and record.get("empirical_best_method") is not None
        ]
        return (
            sum(record.get(field) == record.get("empirical_best_method") for record in eligible),
            len(eligible),
        )

    def aggregate_for(records: list[dict[str, Any]]) -> dict[str, Any]:
        valid_records = [record for record in records if record.get("trial_status") != "failed"]
        initial_valid_records = [record for record in valid_records if record.get("agent_initial_valid") is True]
        final_valid_records = [record for record in valid_records if record.get("final_valid") is True]
        soft_challenge_records = _soft_challenges(valid_records)
        challenge_records = _soft_decision_records(valid_records, "challenge")
        abstained_records = _soft_decision_records(valid_records, "abstain")
        agent_matches, agent_denominator = method_match(valid_records, "agent_initial_method")
        gated_matches, gated_denominator = method_match(valid_records, "final_method")
        paired = [
            record for record in valid_records
            if record.get("agent_normalized_regret") is not None
            and record.get("gated_normalized_regret") is not None
        ]
        health = _gate_health(
            valid_records,
            tolerance=neutral_tolerance,
            catastrophic_threshold=float(configured["catastrophic_regret_threshold"]),
            weights=weights,
        )
        improvements = _paired_improvements(paired)
        holdout_improvements = _holdout_improvements(paired)
        outcomes = Counter(_outcome(record, float(configured["paired_normalized_regret"])) for record in paired)
        task_unsafe = [
            record for record in valid_records
            if record.get("perturbation_id") != "clean"
            and record.get("perturbation", {}).get("kind") == "deterministic_invariant_violation"
        ]
        unnecessary = [
            record for record in initial_valid_records
            if record.get("method_disagreement") is True
            and record.get("final_method") is not None
            and record.get("final_method") != record.get("agent_initial_method")
            and record.get("agent_normalized_regret") is not None
            and approximate_match(record.get("task_type", "classification"), record["agent_normalized_regret"], configured)
        ]
        challenge_outcomes = [
            _outcome(record, float(configured["paired_normalized_regret"]))
            for record in challenge_records
            if record.get("agent_normalized_regret") is not None
            and record.get("gated_normalized_regret") is not None
        ]
        challenge_improved = challenge_outcomes.count("improved")
        challenge_worsened = challenge_outcomes.count("worsened")
        challenge_neutral = challenge_outcomes.count("neutral")
        abstained_comparable = [
            record for record in abstained_records
            if record.get("agent_normalized_regret") is not None
            and record.get("deterministic_normalized_regret") is not None
        ]
        abstained_agent_better = sum(
            float(record["agent_normalized_regret"]) < float(record["deterministic_normalized_regret"]) - float(configured["paired_normalized_regret"])
            for record in abstained_comparable
        )
        abstained_deterministic_better = sum(
            float(record["deterministic_normalized_regret"]) < float(record["agent_normalized_regret"]) - float(configured["paired_normalized_regret"])
            for record in abstained_comparable
        )
        challenge_deltas = [
            float(record["challenge_regret_delta"])
            for record in challenge_records
            if record.get("challenge_regret_delta") is not None
        ]
        catastrophic_agent = [
            record for record in valid_records
            if record.get("agent_normalized_regret") is not None
            and float(record["agent_normalized_regret"]) >= float(configured["catastrophic_regret_threshold"])
        ]
        catastrophic_challenges = [record for record in challenge_records if record in catastrophic_agent]
        catastrophic_prevented = sum(
            record.get("gated_normalized_regret") is not None
            and float(record["gated_normalized_regret"]) < float(configured["catastrophic_regret_threshold"])
            for record in catastrophic_challenges
        )
        return {
            **_empirical_probe_summary(valid_records),
            "trial_count": len(records),
            "completed_trial_count": len(valid_records),
            "valid_trial_count": len(final_valid_records),
            "invalid_trial_count": sum(record.get("final_valid") is False for record in valid_records),
            "agent_initial_valid_count": len(initial_valid_records),
            "agent_initial_invalid_count": sum(record.get("agent_initial_valid") is False for record in valid_records),
            "agent_initial_validity_rate": _rate(len(initial_valid_records), len(valid_records)),
            "initial_hard_invalid_proposal_count": sum(
                _initial_hard_invalid(record) for record in valid_records
            ),
            "hard_validation_intervention_count": sum(
                _hard_intervention(record) for record in valid_records
            ),
            "hard_validation_interception_rate": _rate(
                sum(_hard_intervention(record) for record in valid_records),
                sum(_initial_hard_invalid(record) for record in valid_records),
            ),
            "final_hard_invalid_count": sum(_final_hard_invalid(record) for record in valid_records),
            "agreement_rate": _rate(
                sum(record.get("agreement_status") == "agreement" for record in valid_records),
                sum(record.get("agreement_status") in {"agreement", "disagreement"} for record in valid_records),
            ),
            "disagreement_rate": _rate(
                sum(record.get("agreement_status") == "disagreement" for record in valid_records),
                sum(record.get("agreement_status") in {"agreement", "disagreement"} for record in valid_records),
            ),
            "reconciliation_success_rate": _rate(
                sum(record.get("reconciliation_status") == "succeeded" for record in valid_records),
                sum(bool(record.get("reconciliation_invoked")) for record in valid_records),
            ),
            "reconciliation_invocation_rate": _rate(
                sum(bool(record.get("reconciliation_invoked")) for record in valid_records), len(valid_records)
            ),
            **_reconciliation_rates(valid_records),
            "model_family_disagreement_rate": _rate(
                sum(record.get("method_disagreement") is True for record in valid_records),
                sum(record.get("deterministic_recommendation") is not None for record in valid_records),
            ),
            "preprocessing_disagreement_rate": _rate(
                sum(record.get("preprocessing_disagreement") is True for record in valid_records),
                sum(record.get("deterministic_recommendation") is not None for record in valid_records),
            ),
            "soft_challenge_count": len(soft_challenge_records),
            "total_disagreements": len(soft_challenge_records),
            "challenges": len(challenge_records),
            "abstentions": len(abstained_records),
            "challenge_rate": _rate(len(challenge_records), len(soft_challenge_records)),
            "abstention_rate": _rate(len(abstained_records), len(soft_challenge_records)),
            "soft_challenge_reconciliation_invocation_count": sum(
                bool(record.get("reconciliation_invoked")) for record in soft_challenge_records
            ),
            "soft_challenge_reconciliation_invocation_rate": _rate(
                sum(bool(record.get("reconciliation_invoked")) for record in soft_challenge_records),
                len(soft_challenge_records),
            ),
            **_reconciliation_rates(valid_records),
            "soft_challenge_reconciliation_sided_with_agent_count": _reconciliation_rates(
                soft_challenge_records
            )["reconciliation_sided_with_agent_count"],
            "soft_challenge_reconciliation_sided_with_deterministic_count": _reconciliation_rates(
                soft_challenge_records
            )["reconciliation_sided_with_deterministic_count"],
            "soft_challenge_reconciliation_sided_with_agent_rate": _reconciliation_rates(
                soft_challenge_records
            )["reconciliation_sided_with_agent_rate"],
            "soft_challenge_reconciliation_sided_with_deterministic_rate": _reconciliation_rates(
                soft_challenge_records
            )["reconciliation_sided_with_deterministic_rate"],
            "soft_challenge_outcome_counts": _soft_outcome_counts(
                valid_records, float(configured["paired_normalized_regret"])
            ),
            "soft_challenge_improved_count": _soft_outcome_counts(
                valid_records, float(configured["paired_normalized_regret"])
            )["improved"],
            "soft_challenge_worsened_count": _soft_outcome_counts(
                valid_records, float(configured["paired_normalized_regret"])
            )["worsened"],
            "soft_challenge_neutral_count": _soft_outcome_counts(
                valid_records, float(configured["paired_normalized_regret"])
            )["neutral"],
            "challenge_improved_count": challenge_improved,
            "challenge_worsened_count": challenge_worsened,
            "challenge_neutral_count": challenge_neutral,
            "intervention_precision": health["intervention_precision"],
            "challenge_yield": health["challenge_yield"],
            "harmful_intervention_rate": health["harmful_intervention_rate"],
            "challenge_recall": health["challenge_recall"],
            "missed_rescue_count": health["missed_rescue_count"],
            "good_abstention_count": health["good_abstention_count"],
            "neutral_abstention_count": health["neutral_abstention_count"],
            "mean_regret_reduction": health["mean_regret_reduction"],
            "median_regret_reduction": health["median_regret_reduction"],
            "regret_reduction_ci": health["regret_reduction_ci"],
            "initial_catastrophic_count": health["initial_catastrophic_count"],
            "final_catastrophic_count": health["final_catastrophic_count"],
            "catastrophic_prevented_count": health["catastrophic_prevented_count"],
            "catastrophic_introduced_count": health["catastrophic_introduced_count"],
            "net_catastrophic_prevention": health["net_catastrophic_prevention"],
            "gate_utility": health["utility"],
            "mean_performance_delta_conditional_on_challenge": _mean(challenge_deltas),
            "abstained_agent_better_count": abstained_agent_better,
            "abstained_deterministic_better_count": abstained_deterministic_better,
            "abstained_comparable_count": len(abstained_comparable),
            "catastrophic_regret_rate": _rate(len(catastrophic_agent), len(valid_records)),
            "catastrophic_regret_prevented_by_challenge_count": catastrophic_prevented,
            "catastrophic_regret_prevented_by_challenge_rate": _rate(
                catastrophic_prevented, len(catastrophic_challenges)
            ),
            "unnecessary_intervention_count": health["unnecessary_intervention_count"],
            "unnecessary_intervention_rate": health["unnecessary_intervention_rate"],
            "unsafe_plan_interception_count": sum(record.get("unsafe_plan_intercepted") is True for record in valid_records),
            "unsafe_plan_interception_rate": _rate(
                sum(record.get("unsafe_plan_intercepted") is True for record in valid_records),
                sum(record.get("agent_initial_valid") is False for record in valid_records),
            ),
            "final_invalid_count": sum(record.get("final_valid") is False for record in valid_records),
            "final_invalid_rate": _rate(sum(record.get("final_valid") is False for record in valid_records), len(valid_records)),
            "deterministic_validation_intervention_count": sum(
                record.get("deterministic_validation_intervened") is True for record in valid_records
            ),
            "deterministic_validation_intervention_rate": _rate(
                sum(record.get("deterministic_validation_intervened") is True for record in valid_records),
                len(valid_records),
            ),
            "initial_method_distribution": _method_distribution(valid_records, "agent_initial_method"),
            "final_method_distribution": _method_distribution(valid_records, "final_method"),
            "agent_empirical_reference_match_rate": _rate(agent_matches, agent_denominator),
            "gated_empirical_reference_match_rate": _rate(gated_matches, gated_denominator),
            "agent_normalized_regret_mean": _mean([
                float(record["agent_normalized_regret"])
                for record in valid_records if record.get("agent_normalized_regret") is not None
            ]),
            "agent_normalized_regret_median": _median([
                float(record["agent_normalized_regret"])
                for record in valid_records if record.get("agent_normalized_regret") is not None
            ]),
            "gated_normalized_regret_mean": _mean([
                float(record["gated_normalized_regret"])
                for record in valid_records if record.get("gated_normalized_regret") is not None
            ]),
            "gated_normalized_regret_median": _median([
                float(record["gated_normalized_regret"])
                for record in valid_records if record.get("gated_normalized_regret") is not None
            ]),
            "paired_cv_improvement_mean": _mean(improvements),
            "paired_cv_improvement_median": _median(improvements),
            "paired_cv_improvement_std": _std(improvements),
            "paired_holdout_improvement_mean": _mean(holdout_improvements),
            "paired_holdout_improvement_median": _median(holdout_improvements),
            "paired_holdout_improvement_std": _std(holdout_improvements),
            "performance_comparable_count": len(paired),
            "gating_outcome_counts": {
                "improved": outcomes.get("improved", 0),
                "worsened": outcomes.get("worsened", 0),
                "neutral": outcomes.get("neutral", 0),
                "tie": outcomes.get("neutral", 0),
                "not_comparable": outcomes.get("not_comparable", 0),
                "gated_better_count": outcomes.get("improved", 0),
                "agent_better_count": outcomes.get("worsened", 0),
                "tie_count": outcomes.get("neutral", 0),
                "eligible_count": len(paired),
            },
            "potentially_unnecessary_intervention_count": len(unnecessary),
            "potentially_unnecessary_intervention_rate": _rate(
                len(unnecessary),
                sum(record.get("method_disagreement") is True for record in initial_valid_records),
            ),
            "validation_interception_count": sum(record.get("unsafe_plan_intercepted") is True for record in task_unsafe),
            "validation_interception_rate": _rate(
                sum(record.get("unsafe_plan_intercepted") is True for record in task_unsafe), len(task_unsafe)
            ),
            "gate_health": health,
        }

    all_matches_initial, all_initial_denominator = method_match(completed, "agent_initial_method")
    all_matches_gated, all_gated_denominator = method_match(completed, "final_method")
    paired = [
        record for record in completed
        if record.get("agent_normalized_regret") is not None
        and record.get("gated_normalized_regret") is not None
    ]
    outcomes = Counter(_outcome(record, float(configured["paired_normalized_regret"])) for record in paired)
    unnecessary = [
        record for record in completed
        if record.get("agent_initial_valid") is True
        and record.get("method_disagreement") is True
        and record.get("final_method") is not None
        and record.get("final_method") != record.get("agent_initial_method")
        and record.get("agent_normalized_regret") is not None
        and approximate_match(record.get("task_type", "classification"), record["agent_normalized_regret"], configured)
    ]
    openai_initial, openai_initial_denominator = method_match(openai, "agent_initial_method")
    openai_gated, openai_gated_denominator = method_match(openai, "final_method")
    openai_paired = [
        record for record in openai
        if record.get("agent_normalized_regret") is not None
        and record.get("gated_normalized_regret") is not None
    ]
    openai_holdout_improvements = _holdout_improvements(openai_paired)
    openai_agreement_records = [
        record for record in openai if record.get("agreement_status") in {"agreement", "disagreement"}
    ]
    openai_preprocessing_records = [
        record for record in openai if record.get("preprocessing_disagreement") is not None
    ]
    by_dataset = {
        case: aggregate_for([record for record in completed if record.get("benchmark_case") == case])
        for case in sorted({str(record.get("benchmark_case")) for record in completed})
    }
    for case, item in by_dataset.items():
        dataset_openai = [record for record in openai if record.get("benchmark_case") == case]
        item["openai_trial_count"] = len(dataset_openai)
        item["openai_only"] = aggregate_for(dataset_openai)
    by_task = {
        task: aggregate_for([record for record in completed if record.get("task_type") == task])
        for task in ("classification", "regression")
    }
    overall_selective = aggregate_for(completed)
    overall_gate_health = overall_selective["gate_health"]
    dataset_gate_health = _dataset_weighted_health(
        completed,
        tolerance=neutral_tolerance,
        catastrophic_threshold=float(configured["catastrophic_regret_threshold"]),
        weights=weights,
    )
    leave_one_dataset_out: dict[str, Any] = {}
    dataset_names = sorted({str(record.get("benchmark_case")) for record in completed})
    if len(dataset_names) >= 2:
        for dataset_name in dataset_names:
            remaining = [
                record for record in completed
                if str(record.get("benchmark_case")) != dataset_name
            ]
            leave_one_dataset_out[dataset_name] = _gate_health(
                remaining,
                tolerance=neutral_tolerance,
                catastrophic_threshold=float(configured["catastrophic_regret_threshold"]),
                weights=weights,
            )
    source_counts = {
        source: sum(record.get("agent_source") == source for record in trials)
        for source in sorted({str(record.get("agent_source")) for record in trials})
    }
    return {
        **_empirical_probe_summary(completed),
        "formulas": {
            "classification_regret": "max(0, best_macro_f1 - selected_macro_f1)",
            "regression_regret": "max(0, selected_rmse - best_rmse)",
            "classification_normalized_regret": "classification_regret",
            "regression_normalized_regret": "regression_regret / max(abs(best_rmse), 1e-12)",
            "paired_cv_improvement_classification": "gated_macro_f1 - initial_macro_f1",
            "paired_cv_improvement_regression": "initial_rmse - gated_rmse",
            "normalized_performance_delta": "classification=final_macro_f1-initial_macro_f1; regression=initial_rmse-final_rmse",
            "regret_reduction": "initial_normalized_regret-final_normalized_regret",
            "gate_outcome": "improved/worsened when regret reduction differs from zero by more than neutral_tolerance; otherwise neutral",
            "intervention_precision": "improved_interventions/(improved_interventions+worsened_interventions); neutral interventions excluded",
            "challenge_yield": "improved_interventions/total_challenges",
            "harmful_intervention_rate": "worsened_interventions/total_challenges",
            "unnecessary_intervention_rate": "neutral_interventions/total_challenges",
            "challenge_recall": "beneficial_challenges_made/all_disagreements_where_deterministic_alternative_would_help",
            "potentially_unnecessary_intervention": "valid initial plan AND method disagreement AND final method changed AND initial regret within the task tolerance",
        },
        "gate_objective_version": GATE_OBJECTIVE_VERSION,
        "gate_evaluation_objective": {
            "version": GATE_OBJECTIVE_VERSION,
            "neutral_tolerance": neutral_tolerance,
            "catastrophic_regret_threshold": configured["catastrophic_regret_threshold"],
            "utility_weights": weights.as_dict(),
            "calibration_data_role": "policy development only; final evaluation is never used to tune these values",
        },
        "thresholds": configured,
        "trial_count": total,
        "completed_trial_count": len(completed),
        "failed_trial_count": len(failed),
        "clean_trial_count": len(clean),
        "perturbation_trial_count": total - len(clean),
        "requested_live_trials": sum(bool(record.get("requested_live_trial")) for record in trials),
        "successful_openai_trials": sum(record.get("agent_source") == "openai" for record in trials),
        "offline_fallback_trials": sum(record.get("agent_source") == "offline_fallback" for record in trials),
        "failed_trials": len(failed),
        "live_request_failed_trials": sum(record.get("live_request_failed") is True for record in trials),
        "mock_trials": sum(record.get("agent_source") == "mock" for record in trials),
        "valid_trial_count": len(final_valid),
        "invalid_trial_count": sum(record.get("final_valid") is False for record in completed),
        "agent_initial_valid_count": len(initial_valid),
        "agent_initial_invalid_count": len(initial_invalid),
        "agent_initial_validity_rate": _rate(len(initial_valid), len(completed)),
        "initial_hard_invalid_proposal_count": sum(_initial_hard_invalid(record) for record in completed),
        "hard_validation_intervention_count": sum(_hard_intervention(record) for record in completed),
        "hard_validation_interception_rate": _rate(
            sum(_hard_intervention(record) for record in completed),
            sum(_initial_hard_invalid(record) for record in completed),
        ),
        "final_hard_invalid_count": sum(_final_hard_invalid(record) for record in completed),
        "agreement_rate": _rate(len(agreement), len(deterministic_available)),
        "disagreement_rate": _rate(len(disagreements), len(deterministic_available)),
        "reconciliation_success_rate": _rate(len(recon_success), len(recon)),
        "reconciliation_invocation_rate": _rate(len(recon), len(completed)),
        "model_family_disagreement_rate": _rate(
            sum(record.get("method_disagreement") is True for record in deterministic_available),
            len(deterministic_available),
        ),
        "preprocessing_disagreement_rate": _rate(
            sum(record.get("preprocessing_disagreement") is True for record in deterministic_available),
            len(deterministic_available),
        ),
        "soft_challenge_count": len(soft_challenge_records),
        "total_disagreements": overall_selective["total_disagreements"],
        "challenges": overall_selective["challenges"],
        "abstentions": overall_selective["abstentions"],
        "challenge_rate": overall_selective["challenge_rate"],
        "abstention_rate": overall_selective["abstention_rate"],
        "soft_challenge_reconciliation_invocation_count": sum(
            bool(record.get("reconciliation_invoked")) for record in soft_challenge_records
        ),
        "soft_challenge_reconciliation_invocation_rate": _rate(
            sum(bool(record.get("reconciliation_invoked")) for record in soft_challenge_records),
            len(soft_challenge_records),
        ),
        **_reconciliation_rates(completed),
        "soft_challenge_reconciliation_sided_with_agent_count": _reconciliation_rates(
            soft_challenge_records
        )["reconciliation_sided_with_agent_count"],
        "soft_challenge_reconciliation_sided_with_deterministic_count": _reconciliation_rates(
            soft_challenge_records
        )["reconciliation_sided_with_deterministic_count"],
        "soft_challenge_reconciliation_sided_with_agent_rate": _reconciliation_rates(
            soft_challenge_records
        )["reconciliation_sided_with_agent_rate"],
        "soft_challenge_reconciliation_sided_with_deterministic_rate": _reconciliation_rates(
            soft_challenge_records
        )["reconciliation_sided_with_deterministic_rate"],
        "soft_challenge_outcome_counts": _soft_outcome_counts(
            completed, float(configured["paired_normalized_regret"])
        ),
        "soft_challenge_improved_count": _soft_outcome_counts(
            completed, float(configured["paired_normalized_regret"])
        )["improved"],
        "soft_challenge_worsened_count": _soft_outcome_counts(
            completed, float(configured["paired_normalized_regret"])
        )["worsened"],
        "soft_challenge_neutral_count": _soft_outcome_counts(
            completed, float(configured["paired_normalized_regret"])
        )["neutral"],
        "challenge_improved_count": overall_selective["challenge_improved_count"],
        "challenge_worsened_count": overall_selective["challenge_worsened_count"],
        "challenge_neutral_count": overall_selective["challenge_neutral_count"],
        "intervention_precision": overall_selective["intervention_precision"],
        "challenge_yield": overall_selective["challenge_yield"],
        "harmful_intervention_rate": overall_selective["harmful_intervention_rate"],
        "unnecessary_intervention_rate": overall_selective["unnecessary_intervention_rate"],
        "challenge_recall": overall_selective["challenge_recall"],
        "missed_rescue_count": overall_selective["missed_rescue_count"],
        "good_abstention_count": overall_selective["good_abstention_count"],
        "neutral_abstention_count": overall_selective["neutral_abstention_count"],
        "gate_health": overall_gate_health,
        "trial_weighted_gate_health": overall_gate_health,
        "dataset_weighted_gate_health": dataset_gate_health,
        "total_challenges": overall_gate_health["total_challenges"],
        "total_abstentions": overall_gate_health["total_abstentions"],
        "improved_interventions": overall_gate_health["improved_interventions"],
        "worsened_interventions": overall_gate_health["worsened_interventions"],
        "neutral_interventions": overall_gate_health["neutral_interventions"],
        "path_metrics": overall_gate_health["path_metrics"],
        "confidence_metrics": overall_gate_health["confidence_metrics"],
        "probe_strength_metrics": overall_gate_health["probe_strength_metrics"],
        "regime_metrics": overall_gate_health["regime_metrics"],
        "suppressed_regime_slices": overall_gate_health["suppressed_regime_slices"],
        "gain_concentration": overall_gate_health["gain_concentration"],
        "mean_regret_reduction": overall_gate_health["mean_regret_reduction"],
        "median_regret_reduction": overall_gate_health["median_regret_reduction"],
        "regret_reduction_ci": overall_gate_health["regret_reduction_ci"],
        "initial_catastrophic_count": overall_gate_health["initial_catastrophic_count"],
        "final_catastrophic_count": overall_gate_health["final_catastrophic_count"],
        "catastrophic_prevented_count": overall_gate_health["catastrophic_prevented_count"],
        "catastrophic_introduced_count": overall_gate_health["catastrophic_introduced_count"],
        "initial_agent_catastrophic_count": overall_gate_health["initial_catastrophic_count"],
        "final_gate_catastrophic_count": overall_gate_health["final_catastrophic_count"],
        "catastrophic_regret_prevented_count": overall_gate_health["catastrophic_prevented_count"],
        "catastrophic_regret_introduced_count": overall_gate_health["catastrophic_introduced_count"],
        "net_catastrophic_prevention": overall_gate_health["net_catastrophic_prevention"],
        "gate_utility": overall_gate_health["utility"],
        "leave_one_dataset_out": leave_one_dataset_out,
        "mean_performance_delta_conditional_on_challenge": overall_selective[
            "mean_performance_delta_conditional_on_challenge"
        ],
        "abstained_agent_better_count": overall_selective["abstained_agent_better_count"],
        "abstained_deterministic_better_count": overall_selective["abstained_deterministic_better_count"],
        "abstained_comparable_count": overall_selective["abstained_comparable_count"],
        "catastrophic_regret_rate": overall_selective["catastrophic_regret_rate"],
        "catastrophic_regret_prevented_by_challenge_count": overall_selective[
            "catastrophic_regret_prevented_by_challenge_count"
        ],
        "catastrophic_regret_prevented_by_challenge_rate": overall_selective[
            "catastrophic_regret_prevented_by_challenge_rate"
        ],
        "unnecessary_intervention_count": overall_selective["unnecessary_intervention_count"],
        "unsafe_plan_interception_count": len(intercepted),
        "unsafe_plan_interception_rate": _rate(len(intercepted), len(initial_invalid)),
        "validation_interception_count": len(intentionally_unsafe_intercepted),
        "validation_interception_rate": _rate(len(intentionally_unsafe_intercepted), len(intentionally_unsafe)),
        "final_invalid_rate": _rate(sum(record.get("final_valid") is False for record in completed), len(completed)),
        "final_invalid_count": sum(record.get("final_valid") is False for record in completed),
        "deterministic_validation_intervention_count": sum(
            record.get("deterministic_validation_intervened") is True for record in completed
        ),
        "deterministic_validation_intervention_rate": _rate(
            sum(record.get("deterministic_validation_intervened") is True for record in completed),
            len(completed),
        ),
        "initial_method_distribution": _method_distribution(completed, "agent_initial_method"),
        "final_method_distribution": _method_distribution(completed, "final_method"),
        "agent_empirical_reference_match_rate": _rate(all_matches_initial, all_initial_denominator),
        "gated_empirical_reference_match_rate": _rate(all_matches_gated, all_gated_denominator),
        "agent_normalized_regret_mean": _mean([
            float(record["agent_normalized_regret"])
            for record in completed if record.get("agent_normalized_regret") is not None
        ]),
        "agent_normalized_regret_median": _median([
            float(record["agent_normalized_regret"])
            for record in completed if record.get("agent_normalized_regret") is not None
        ]),
        "gated_normalized_regret_mean": _mean([
            float(record["gated_normalized_regret"])
            for record in completed if record.get("gated_normalized_regret") is not None
        ]),
        "gated_normalized_regret_median": _median([
            float(record["gated_normalized_regret"])
            for record in completed if record.get("gated_normalized_regret") is not None
        ]),
        "paired_cv_improvement_mean": _mean(_paired_improvements(paired)),
        "paired_cv_improvement_median": _median(_paired_improvements(paired)),
        "paired_cv_improvement_std": _std(_paired_improvements(paired)),
        "paired_holdout_improvement_mean": _mean(_holdout_improvements(paired)),
        "paired_holdout_improvement_median": _median(_holdout_improvements(paired)),
        "paired_holdout_improvement_std": _std(_holdout_improvements(paired)),
        "gating_outcome_counts": {
            "improved": outcomes.get("improved", 0),
            "worsened": outcomes.get("worsened", 0),
            "neutral": outcomes.get("neutral", 0),
            "tie": outcomes.get("neutral", 0),
            "not_comparable": outcomes.get("not_comparable", 0),
            "gated_better_count": outcomes.get("improved", 0),
            "agent_better_count": outcomes.get("worsened", 0),
            "tie_count": outcomes.get("neutral", 0),
            "eligible_count": len(paired),
        },
        "potentially_unnecessary_intervention_count": len(unnecessary),
        "potentially_unnecessary_intervention_rate": _rate(
            len(unnecessary),
            sum(record.get("agent_initial_valid") is True and record.get("method_disagreement") is True for record in completed),
        ),
        "failure_counts_by_validation_code": dict(sorted(failure_counts.items())),
        "by_task": by_task,
        "by_dataset": by_dataset,
        "openai_only": {
            **({} if not openai else {
                **aggregate_for(openai),
                "stability_by_dataset": {
                    case: _stability([record for record in openai if record.get("benchmark_case") == case])
                    for case in sorted({str(record.get("benchmark_case")) for record in openai})
                },
            }),
            "requested_live_trials": sum(bool(record.get("requested_live_trial")) for record in openai),
            "successful_openai_trials": len(openai),
        },
        "openai_only_match_rates": {
            "initial_reference_match_rate": _rate(openai_initial, openai_initial_denominator),
            "gated_reference_match_rate": _rate(openai_gated, openai_gated_denominator),
        },
        "initial_reference_match_rate": _rate(openai_initial, openai_initial_denominator),
        "gated_reference_match_rate": _rate(openai_gated, openai_gated_denominator),
        "openai_only_method_agreement_rate": _rate(
            sum(not bool(record.get("method_disagreement")) for record in openai_agreement_records),
            len(openai_agreement_records),
        ),
        "openai_only_preprocessing_agreement_rate": _rate(
            sum(not bool(record.get("preprocessing_disagreement")) for record in openai_preprocessing_records),
            len(openai_preprocessing_records),
        ),
        "openai_only_reconciliation_invocation_rate": _rate(
            sum(bool(record.get("reconciliation_invoked")) for record in openai), len(openai)
        ),
        "openai_only_paired_stats": {
            "trial_count": len(openai_paired),
            "mean_paired_improvement": _mean(_paired_improvements(openai_paired)),
            "median_paired_improvement": _median(_paired_improvements(openai_paired)),
            "std_paired_improvement": _std(_paired_improvements(openai_paired)),
            "mean_paired_holdout_improvement": _mean(openai_holdout_improvements),
            "median_paired_holdout_improvement": _median(openai_holdout_improvements),
            "std_paired_holdout_improvement": _std(openai_holdout_improvements),
            "improved_count": sum(_outcome(record, float(configured["paired_normalized_regret"])) == "improved" for record in openai_paired),
            "worsened_count": sum(_outcome(record, float(configured["paired_normalized_regret"])) == "worsened" for record in openai_paired),
            "neutral_count": sum(_outcome(record, neutral_tolerance) == "neutral" for record in openai_paired),
            "tie_count": sum(_outcome(record, neutral_tolerance) == "neutral" for record in openai_paired),
        },
        "stability_by_dataset": {
            case: _stability([record for record in openai if record.get("benchmark_case") == case])
            for case in sorted({str(record.get("benchmark_case")) for record in openai})
        },
        "source_counts": source_counts,
    }
