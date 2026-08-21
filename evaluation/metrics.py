"""Transparent regret and aggregate metric formulas for evaluation records."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any


DEFAULT_THRESHOLDS = {
    "classification_regret": 0.01,
    "regression_normalized_regret": 0.02,
    "paired_normalized_regret": 0.02,
}


def regret(task_type: str, best_score: float | None, selected_score: float | None) -> float | None:
    """Return larger-is-worse family regret for the configured primary metric."""

    if best_score is None or selected_score is None:
        return None
    if task_type == "classification":
        return float(best_score - selected_score)
    if task_type == "regression":
        return float(selected_score - best_score)
    raise ValueError(f"Unsupported task type: {task_type!r}")


def normalized_regret(
    task_type: str, best_score: float | None, selected_score: float | None
) -> float | None:
    """Normalize regret for cross-dataset aggregation without averaging RMSE."""

    raw = regret(task_type, best_score, selected_score)
    if raw is None:
        return None
    if task_type == "classification":
        return float(max(0.0, raw))
    scale = max(abs(float(best_score or 0.0)), 1e-12)
    return float(max(0.0, raw) / scale)


def approximate_match(task_type: str, normalized_value: float | None, thresholds: dict[str, float]) -> bool:
    if normalized_value is None:
        return False
    threshold_key = (
        "classification_regret" if task_type == "classification" else "regression_normalized_regret"
    )
    return float(normalized_value) <= float(thresholds[threshold_key])


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _records_with(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get(field) is not None]


def _rate(count: int, denominator: int) -> float | None:
    return float(count / denominator) if denominator else None


def _validation_failure_codes(record: dict[str, Any]) -> set[str]:
    """Return one set of failed validation codes observed in a trial.

    A code can appear in both the initial and gated validation reports.  The
    summary counts a code once per trial so the aggregate describes how many
    trials encountered the invariant, rather than how many internal reports
    happened to repeat it.
    """

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


def summarize_trials(
    trials: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate auditable trial rows without mixing incompatible raw metrics."""

    configured = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    total = len(trials)
    clean = sum(record.get("perturbation_id") == "clean" for record in trials)
    perturbation = total - clean
    deterministic_available = [
        record for record in trials if record.get("deterministic_recommendation") is not None
    ]
    initial_invalid = [record for record in trials if record.get("agent_initial_valid") is False]
    initial_valid = [record for record in trials if record.get("agent_initial_valid") is True]
    final_valid = [record for record in trials if record.get("final_valid") is True]
    final_invalid = [record for record in trials if record.get("final_valid") is not True]
    agreement = [record for record in deterministic_available if record.get("agreement_status") == "agreement"]
    disagreements = [
        record for record in deterministic_available if record.get("agreement_status") == "disagreement"
    ]
    recon = [record for record in trials if record.get("reconciliation_invoked")]
    recon_success = [record for record in recon if record.get("reconciliation_status") == "succeeded"]
    intercepted = [
        record
        for record in initial_invalid
        if record.get("unsafe_plan_intercepted") is True
    ]
    intentionally_unsafe = [
        record
        for record in trials
        if record.get("perturbation_id") != "clean"
        and record.get("perturbation", {}).get("kind") == "deterministic_invariant_violation"
    ]
    intentionally_unsafe_intercepted = [
        record for record in intentionally_unsafe if record.get("unsafe_plan_intercepted") is True
    ]
    failure_counts = Counter(
        code
        for record in trials
        for code in _validation_failure_codes(record)
    )

    def method_match(condition: str) -> tuple[int, int]:
        eligible = [
            record
            for record in trials
            if record.get(condition) is not None and record.get("empirical_best_method") is not None
        ]
        matched = sum(record.get(condition) == record.get("empirical_best_method") for record in eligible)
        return matched, len(eligible)

    agent_matches, agent_match_denominator = method_match("agent_initial_method")
    gated_matches, gated_match_denominator = method_match("final_method")
    performance_records = [
        record
        for record in trials
        if record.get("agent_normalized_regret") is not None
        and record.get("gated_normalized_regret") is not None
    ]
    paired_threshold = float(configured["paired_normalized_regret"])
    gated_better = sum(
        record["gated_normalized_regret"] + paired_threshold < record["agent_normalized_regret"]
        for record in performance_records
    )
    agent_better = sum(
        record["agent_normalized_regret"] + paired_threshold < record["gated_normalized_regret"]
        for record in performance_records
    )
    tie = len(performance_records) - gated_better - agent_better

    def performance_counts(records: list[dict[str, Any]]) -> dict[str, int]:
        improved = sum(
            record.get("gated_normalized_regret", 0.0)
            < record.get("agent_normalized_regret", 0.0) - paired_threshold
            for record in records
        )
        worsened = sum(
            record.get("gated_normalized_regret", 0.0)
            > record.get("agent_normalized_regret", 0.0) + paired_threshold
            for record in records
        )
        return {"improved": improved, "worsened": worsened, "unchanged": len(records) - improved - worsened}

    unnecessary = [
        record
        for record in trials
        if record.get("agent_initial_valid") is True
        and record.get("method_disagreement") is True
        and record.get("agent_normalized_regret") is not None
        and approximate_match(record["task_type"], record["agent_normalized_regret"], configured)
    ]

    def aggregate_for(records: list[dict[str, Any]]) -> dict[str, Any]:
        initial_regrets = [record["agent_normalized_regret"] for record in records if record.get("agent_normalized_regret") is not None]
        gated_regrets = [record["gated_normalized_regret"] for record in records if record.get("gated_normalized_regret") is not None]
        paired = [
            record for record in records
            if record.get("agent_normalized_regret") is not None
            and record.get("gated_normalized_regret") is not None
        ]
        task_unsafe = [
            record
            for record in records
            if record.get("perturbation_id") != "clean"
            and record.get("perturbation", {}).get("kind") == "deterministic_invariant_violation"
        ]
        return {
            "trial_count": len(records),
            "valid_trial_count": sum(record.get("final_valid") is True for record in records),
            "invalid_trial_count": sum(record.get("final_valid") is not True for record in records),
            "agent_initial_valid_count": sum(record.get("agent_initial_valid") is True for record in records),
            "agent_initial_invalid_count": sum(record.get("agent_initial_valid") is False for record in records),
            "agent_initial_validity_rate": _rate(
                sum(record.get("agent_initial_valid") is True for record in records), len(records)
            ),
            "agreement_rate": _rate(
                sum(record.get("agreement_status") == "agreement" for record in records),
                sum(record.get("agreement_status") in {"agreement", "disagreement"} for record in records),
            ),
            "disagreement_rate": _rate(
                sum(record.get("agreement_status") == "disagreement" for record in records),
                sum(record.get("agreement_status") in {"agreement", "disagreement"} for record in records),
            ),
            "reconciliation_success_rate": _rate(
                sum(record.get("reconciliation_status") == "succeeded" for record in records),
                sum(bool(record.get("reconciliation_invoked")) for record in records),
            ),
            "reconciliation_invocation_rate": _rate(
                sum(bool(record.get("reconciliation_invoked")) for record in records),
                len(records),
            ),
            "unsafe_plan_interception_count": sum(record.get("unsafe_plan_intercepted") is True for record in records),
            "unsafe_plan_interception_rate": _rate(
                sum(record.get("unsafe_plan_intercepted") is True for record in records),
                sum(record.get("agent_initial_valid") is False for record in records),
            ),
            "final_invalid_count": sum(record.get("final_valid") is not True for record in records),
            "final_invalid_rate": _rate(sum(record.get("final_valid") is not True for record in records), len(records)),
            "validation_interception_count": sum(
                record.get("unsafe_plan_intercepted") is True for record in task_unsafe
            ),
            "validation_interception_rate": _rate(
                sum(record.get("unsafe_plan_intercepted") is True for record in task_unsafe),
                len(task_unsafe),
            ),
            "agent_empirical_reference_match_rate": _rate(
                sum(
                    record.get("agent_initial_method") == record.get("empirical_best_method")
                    for record in records
                    if record.get("agent_initial_method") is not None and record.get("empirical_best_method") is not None
                ),
                sum(
                    record.get("agent_initial_method") is not None and record.get("empirical_best_method") is not None
                    for record in records
                ),
            ),
            "gated_empirical_reference_match_rate": _rate(
                sum(
                    record.get("final_method") == record.get("empirical_best_method")
                    for record in records
                    if record.get("final_method") is not None and record.get("empirical_best_method") is not None
                ),
                sum(
                    record.get("final_method") is not None and record.get("empirical_best_method") is not None
                    for record in records
                ),
            ),
            "agent_normalized_regret_mean": _mean(initial_regrets),
            "agent_normalized_regret_median": _median(initial_regrets),
            "gated_normalized_regret_mean": _mean(gated_regrets),
            "gated_normalized_regret_median": _median(gated_regrets),
            "performance_comparable_count": len(paired),
            "gating_outcome_counts": performance_counts(paired),
            "potentially_unnecessary_intervention_count": sum(
                record in unnecessary for record in records
            ),
        }

    by_task = {
        task: aggregate_for([record for record in trials if record.get("task_type") == task])
        for task in ("classification", "regression")
    }
    return {
        "formulas": {
            "classification_regret": "best_macro_f1 - selected_macro_f1",
            "regression_regret": "selected_rmse - best_rmse",
            "regression_normalized_regret": "max(0, regression_regret) / max(abs(best_rmse), 1e-12)",
            "unsafe_plan_interception_rate": "invalid initial proposals not proceeding unchanged / all invalid initial proposals",
            "potentially_unnecessary_intervention": "valid initial plan AND model-family disagreement AND normalized regret within configured threshold",
        },
        "thresholds": configured,
        "trial_count": total,
        "clean_trial_count": clean,
        "perturbation_trial_count": perturbation,
        "valid_trial_count": len(final_valid),
        "invalid_trial_count": len(final_invalid),
        "agent_initial_valid_count": len(initial_valid),
        "agent_initial_invalid_count": len(initial_invalid),
        "agent_initial_validity_rate": _rate(total - len(initial_invalid), total),
        "agreement_rate": _rate(len(agreement), len(deterministic_available)),
        "disagreement_rate": _rate(len(disagreements), len(deterministic_available)),
        "reconciliation_success_rate": _rate(len(recon_success), len(recon)),
        "reconciliation_invocation_rate": _rate(len(recon), total),
        "unsafe_plan_interception_count": len(intercepted),
        "unsafe_plan_interception_rate": _rate(len(intercepted), len(initial_invalid)),
        "validation_interception_count": len(intentionally_unsafe_intercepted),
        "validation_interception_rate": _rate(
            len(intentionally_unsafe_intercepted), len(intentionally_unsafe)
        ),
        "final_invalid_rate": _rate(len(final_invalid), total),
        "final_invalid_count": len(final_invalid),
        "agent_empirical_reference_match_rate": _rate(agent_matches, agent_match_denominator),
        "gated_empirical_reference_match_rate": _rate(gated_matches, gated_match_denominator),
        "agent_normalized_regret_mean": _mean(
            [record["agent_normalized_regret"] for record in trials if record.get("agent_normalized_regret") is not None]
        ),
        "agent_normalized_regret_median": _median(
            [record["agent_normalized_regret"] for record in trials if record.get("agent_normalized_regret") is not None]
        ),
        "gated_normalized_regret_mean": _mean(
            [record["gated_normalized_regret"] for record in trials if record.get("gated_normalized_regret") is not None]
        ),
        "gated_normalized_regret_median": _median(
            [record["gated_normalized_regret"] for record in trials if record.get("gated_normalized_regret") is not None]
        ),
        "gating_outcome_counts": {
            "gated_better_count": gated_better,
            "agent_better_count": agent_better,
            "tie_count": tie,
            "eligible_count": len(performance_records),
        },
        "potentially_unnecessary_intervention_count": len(unnecessary),
        "potentially_unnecessary_intervention_rate": _rate(
            len(unnecessary),
            sum(record.get("agent_initial_valid") is True and record.get("method_disagreement") is True for record in trials),
        ),
        "failure_counts_by_validation_code": dict(sorted(failure_counts.items())),
        "by_task": by_task,
        "source_counts": {
            source: sum(record.get("agent_source") == source for record in trials)
            for source in sorted({str(record.get("agent_source")) for record in trials})
        },
    }
