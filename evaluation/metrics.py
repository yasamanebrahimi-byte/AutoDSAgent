"""Transparent, source-aware metrics for evaluation records."""

from __future__ import annotations

from collections import Counter
from statistics import median, pstdev
from typing import Any


DEFAULT_THRESHOLDS = {
    "classification_regret": 0.01,
    "regression_normalized_regret": 0.02,
    "paired_normalized_regret": 0.02,
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


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _std(values: list[float]) -> float | None:
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
    if record.get("gate_outcome") in {"improved", "worsened", "tie"}:
        return str(record["gate_outcome"])
    initial = record.get("agent_normalized_regret")
    gated = record.get("gated_normalized_regret")
    if initial is None or gated is None:
        return "not_comparable"
    if gated < initial - tolerance:
        return "improved"
    if gated > initial + tolerance:
        return "worsened"
    return "tie"


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
    return {
        "reconciliation_sided_with_agent_count": agent_count,
        "reconciliation_sided_with_deterministic_count": deterministic_count,
        "reconciliation_sided_with_agent_rate": _rate(agent_count, len(sourced)),
        "reconciliation_sided_with_deterministic_rate": _rate(deterministic_count, len(sourced)),
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


def _soft_outcome_counts(records: list[dict[str, Any]], tolerance: float) -> dict[str, int]:
    challenges = _soft_challenges(records)
    outcomes = Counter(_outcome(record, tolerance) for record in challenges)
    return {
        "improved": outcomes.get("improved", 0),
        "worsened": outcomes.get("worsened", 0),
        "neutral": outcomes.get("tie", 0),
        "not_comparable": outcomes.get("not_comparable", 0),
    }


def summarize_trials(
    trials: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate rows while keeping operational and OpenAI-only views separate."""

    configured = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
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
        agent_matches, agent_denominator = method_match(valid_records, "agent_initial_method")
        gated_matches, gated_denominator = method_match(valid_records, "final_method")
        paired = [
            record for record in valid_records
            if record.get("agent_normalized_regret") is not None
            and record.get("gated_normalized_regret") is not None
        ]
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
        return {
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
            "model_family_disagreement_rate": _rate(
                sum(record.get("method_disagreement") is True for record in valid_records),
                sum(record.get("deterministic_recommendation") is not None for record in valid_records),
            ),
            "preprocessing_disagreement_rate": _rate(
                sum(record.get("preprocessing_disagreement") is True for record in valid_records),
                sum(record.get("deterministic_recommendation") is not None for record in valid_records),
            ),
            "soft_challenge_count": len(soft_challenge_records),
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
                "tie": outcomes.get("tie", 0),
                "not_comparable": outcomes.get("not_comparable", 0),
                "gated_better_count": outcomes.get("improved", 0),
                "agent_better_count": outcomes.get("worsened", 0),
                "tie_count": outcomes.get("tie", 0),
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
    source_counts = {
        source: sum(record.get("agent_source") == source for record in trials)
        for source in sorted({str(record.get("agent_source")) for record in trials})
    }
    return {
        "formulas": {
            "classification_regret": "max(0, best_macro_f1 - selected_macro_f1)",
            "regression_regret": "max(0, selected_rmse - best_rmse)",
            "classification_normalized_regret": "classification_regret",
            "regression_normalized_regret": "regression_regret / max(abs(best_rmse), 1e-12)",
            "paired_cv_improvement_classification": "gated_macro_f1 - initial_macro_f1",
            "paired_cv_improvement_regression": "initial_rmse - gated_rmse",
            "gate_outcome": "improved/worsened when normalized regret differs by more than paired_normalized_regret; otherwise tie",
            "potentially_unnecessary_intervention": "valid initial plan AND method disagreement AND final method changed AND initial regret within the task tolerance",
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
            "tie": outcomes.get("tie", 0),
            "not_comparable": outcomes.get("not_comparable", 0),
            "gated_better_count": outcomes.get("improved", 0),
            "agent_better_count": outcomes.get("worsened", 0),
            "tie_count": outcomes.get("tie", 0),
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
            "tie_count": sum(_outcome(record, float(configured["paired_normalized_regret"])) == "tie" for record in openai_paired),
        },
        "stability_by_dataset": {
            case: _stability([record for record in openai if record.get("benchmark_case") == case])
            for case in sorted({str(record.get("benchmark_case")) for record in openai})
        },
        "source_counts": source_counts,
    }
