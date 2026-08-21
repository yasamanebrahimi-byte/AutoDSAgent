"""Offline calibration and audit utilities for the deterministic policy.

This module is deliberately outside :mod:`app`.  It may fit candidate models
through the evaluation-only empirical reference, but the runtime recommender
never imports it and never consumes its output.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from app.deterministic import deterministic_recommendation, profile_dataframe
from app.deterministic_policy import DeterministicPolicy
from app.validation import freeze_supervised_split, training_profile_frame
from evaluation.benchmarks import (
    BENCHMARK_SUITE_VERSION,
    BenchmarkCase,
    BenchmarkRole,
    default_benchmark_cases,
)
from evaluation.empirical_reference import evaluate_empirical_reference, evaluate_holdout_plan
from evaluation.metrics import normalized_regret


CALIBRATION_SCHEMA_VERSION = "1"
DEFAULT_SPLIT_SEEDS: tuple[int, ...] = (42, 123, 2027)
CATASTROPHIC_REGRET_THRESHOLD = 0.10
PROMOTION_MIN_REGRET_IMPROVEMENT = 0.005
PROMOTION_MIN_CATASTROPHIC_IMPROVEMENT = 0.02
POLICY_SELECTION_RULE = (
    "rank by lowest dataset-level mean normalized regret, then lowest "
    "dataset-level catastrophic-regret rate, then highest dataset-level top-2 "
    "reference inclusion, then lowest policy complexity; retain the current "
    "policy unless the selected candidate clears the predefined promotion margin"
)


@dataclass(frozen=True)
class PolicyCandidate:
    """A small, named, interpretable policy variant."""

    name: str
    policy: DeterministicPolicy
    complexity: int
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy_version": self.policy.version,
            "complexity": self.complexity,
            "rationale": self.rationale,
            "parameters": asdict(self.policy),
        }


def policy_candidates(baseline: DeterministicPolicy | None = None) -> list[PolicyCandidate]:
    """Return four intentionally small candidates around the frozen baseline."""

    current = baseline or DeterministicPolicy()
    return [
        PolicyCandidate(
            name="current",
            policy=current,
            complexity=0,
            rationale="Existing frozen runtime policy; mandatory baseline.",
        ),
        PolicyCandidate(
            name="nonlinear_sensitive",
            policy=replace(
                current,
                version=f"{current.version}-candidate-nonlinear",
                nonlinear_moderate_threshold=0.12,
                nonlinear_high_threshold=0.30,
                structural_complexity_high_threshold=0.55,
            ),
            complexity=3,
            rationale="Tests a modestly earlier response to nonlinear evidence.",
        ),
        PolicyCandidate(
            name="high_dimensional_sensitive",
            policy=replace(
                current,
                version=f"{current.version}-candidate-dimensionality",
                low_sample_feature_ratio=5.0,
                moderate_sample_feature_ratio=10.0,
                high_effective_features=80,
            ),
            complexity=3,
            rationale="Tests nearby sample-to-feature and effective-dimension bands.",
        ),
        PolicyCandidate(
            name="missingness_sensitive",
            policy=replace(
                current,
                version=f"{current.version}-candidate-missingness",
                moderate_missing_fraction=0.08,
                high_missing_fraction=0.20,
                widespread_missing_feature_fraction=0.30,
            ),
            complexity=3,
            rationale="Tests a modestly earlier response to missingness evidence.",
        ),
    ]


def _validate_registry(cases: Sequence[BenchmarkCase]) -> list[BenchmarkCase]:
    if not cases:
        raise ValueError("At least one benchmark case is required.")
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Benchmark case names must be unique within an evaluation run.")
    for case in cases:
        if not isinstance(case.role, BenchmarkRole):
            raise ValueError(f"Benchmark case {case.name!r} has no explicit BenchmarkRole.")
    return list(cases)


def calibration_cases(cases: Sequence[BenchmarkCase] | None = None) -> list[BenchmarkCase]:
    """Return only development cases and reject final cases explicitly."""

    supplied_cases = cases is not None
    registry = _validate_registry(list(cases) if supplied_cases else default_benchmark_cases())
    final_cases = [case.name for case in registry if case.role is BenchmarkRole.FINAL_EVALUATION]
    if supplied_cases and final_cases:
        raise ValueError(
            "Policy calibration cannot receive final-evaluation cases: "
            + ", ".join(final_cases)
        )
    selected = [case for case in registry if case.role is BenchmarkRole.POLICY_DEVELOPMENT]
    if not selected:
        raise ValueError("Policy calibration requires at least one policy-development case.")
    return selected


def final_evaluation_cases(cases: Sequence[BenchmarkCase] | None = None) -> list[BenchmarkCase]:
    """Return only frozen final cases and reject development cases explicitly."""

    supplied_cases = cases is not None
    registry = _validate_registry(list(cases) if supplied_cases else default_benchmark_cases())
    development_cases = [case.name for case in registry if case.role is BenchmarkRole.POLICY_DEVELOPMENT]
    if supplied_cases and development_cases:
        raise ValueError(
            "Final policy evaluation cannot receive policy-development cases: "
            + ", ".join(development_cases)
        )
    selected = [case for case in registry if case.role is BenchmarkRole.FINAL_EVALUATION]
    if not selected:
        raise ValueError("Final policy evaluation requires at least one final-evaluation case.")
    return selected


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(median(values)) if values else None


def _repository_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _metric_value(record: dict[str, Any], field: str) -> float | None:
    value = record.get(field)
    return float(value) if value is not None else None


def _evaluate_case_seed(
    case: BenchmarkCase,
    candidate: PolicyCandidate,
    seed: int,
    *,
    include_holdout: bool = False,
    reference_cache: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    frame = case.load()
    split = freeze_supervised_split(
        frame,
        case.target_column,
        case.expected_task_type,
        random_state=seed,
    )
    training_frame = training_profile_frame(
        frame,
        case.target_column,
        case.expected_task_type,
        test_size=split.test_size,
        random_state=seed,
        split=split,
    )
    training_profile = profile_dataframe(training_frame)
    cache_key = (case.name, int(seed))
    if reference_cache is not None and cache_key in reference_cache:
        reference = reference_cache[cache_key]
    else:
        reference = evaluate_empirical_reference(
            training_frame,
            case.target_column,
            case.expected_task_type,
            training_profile,
            random_state=seed,
        )
        if reference_cache is not None:
            reference_cache[cache_key] = reference
    recommendation = deterministic_recommendation(
        training_frame,
        case.question,
        target_hint=case.target_column,
        task_type=case.expected_task_type,
        policy=candidate.policy,
    )
    selected_method = recommendation.recommended_method
    selected_result = reference.get("candidate_metrics", {}).get(selected_method, {})
    best_score = reference.get("best_primary_mean")
    selected_score = (
        selected_result.get("primary_mean")
        if selected_result.get("status") == "evaluated"
        else None
    )
    regret = normalized_regret(case.expected_task_type, best_score, selected_score)
    empirical_ranking = list(reference.get("ranking", []))
    deterministic_ranking = [
        method for method in recommendation.ranked_methods if method in empirical_ranking
    ]
    if not deterministic_ranking:
        deterministic_ranking = list(recommendation.ranked_methods)
    record: dict[str, Any] = {
        "dataset_id": case.name,
        "dataset": case.name,
        "benchmark_role": case.role.value,
        "task_type": case.expected_task_type,
        "seed": int(seed),
        "policy_candidate": candidate.name,
        "policy_version": candidate.policy.version,
        "split_contract": split.as_dict(),
        "training_rows": int(len(training_frame)),
        "diagnostics": recommendation.diagnostics.model_dump(mode="json")
        if recommendation.diagnostics is not None
        else None,
        "deterministic_selected_method": selected_method,
        "deterministic_ranking": deterministic_ranking,
        "deterministic_scores": recommendation.method_scores,
        "policy_score_contributions": {
            method: [item.model_dump(mode="json") for item in assessment.contributions]
            for method, assessment in recommendation.method_assessments.items()
        },
        "empirical_reference_ranking": empirical_ranking,
        "empirical_best_method": reference.get("best_method"),
        "candidate_cv_metrics": reference.get("candidate_metrics", {}),
        "best_primary_mean": best_score,
        "selected_primary_mean": selected_score,
        "normalized_regret": regret,
        "exact_reference_match": bool(selected_method == reference.get("best_method"))
        if reference.get("best_method") is not None
        else False,
        "catastrophic_regret": bool(
            regret is not None and regret >= CATASTROPHIC_REGRET_THRESHOLD
        ),
        "top2_compatibility_success": bool(
            reference.get("best_method") is not None
            and reference.get("best_method") in deterministic_ranking[:2]
        ),
        "holdout_used": False,
        "holdout_evaluation": None,
    }
    if include_holdout:
        holdout = evaluate_holdout_plan(
            frame,
            split,
            case.target_column,
            case.expected_task_type,
            selected_method,
            recommendation.preprocessing,
            random_state=seed,
        )
        record["holdout_used"] = True
        record["holdout_evaluation"] = holdout
    return record


def evaluate_policy_cases(
    cases: Sequence[BenchmarkCase],
    *,
    policy_candidate: PolicyCandidate,
    seeds: Sequence[int],
    expected_role: BenchmarkRole,
    include_holdout: bool = False,
    reference_cache: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a frozen policy on one explicit benchmark role."""

    registry = _validate_registry(cases)
    wrong_role = [case.name for case in registry if case.role is not expected_role]
    if wrong_role:
        raise ValueError(
            f"Expected only {expected_role.value} cases; received cases with another role: "
            + ", ".join(wrong_role)
        )
    if not seeds:
        raise ValueError("At least one deterministic split seed is required.")
    return [
        _evaluate_case_seed(
            case,
            policy_candidate,
            int(seed),
            include_holdout=include_holdout,
            reference_cache=reference_cache,
        )
        for case in registry
        for seed in seeds
    ]


def aggregate_candidate_records(
    records: Sequence[dict[str, Any]],
    candidate: PolicyCandidate,
) -> dict[str, Any]:
    """Aggregate repeated seeds within dataset before summarizing across datasets."""

    candidate_records = [record for record in records if record["policy_candidate"] == candidate.name]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        by_dataset[str(record["dataset_id"])].append(record)

    per_dataset: list[dict[str, Any]] = []
    all_methods: list[str] = []
    for dataset_id in sorted(by_dataset):
        dataset_records = by_dataset[dataset_id]
        regrets = [
            float(record["normalized_regret"])
            for record in dataset_records
            if record.get("normalized_regret") is not None
        ]
        exact = [bool(record["exact_reference_match"]) for record in dataset_records]
        catastrophic = [bool(record["catastrophic_regret"]) for record in dataset_records]
        top2 = [bool(record["top2_compatibility_success"]) for record in dataset_records]
        selections = [str(record["deterministic_selected_method"]) for record in dataset_records]
        all_methods.extend(selections)
        counts = Counter(selections)
        modal_count = counts.most_common(1)[0][1] if counts else 0
        per_dataset.append(
            {
                "dataset_id": dataset_id,
                "seed_count": len(dataset_records),
                "mean_normalized_regret": _mean(regrets),
                "median_normalized_regret": _median(regrets),
                "exact_reference_match_rate": _mean(float(value) for value in exact),
                "catastrophic_regret_rate": _mean(float(value) for value in catastrophic),
                "top2_compatibility_rate": _mean(float(value) for value in top2),
                "family_selection_distribution": {
                    method: {"count": count, "rate": count / max(len(selections), 1)}
                    for method, count in sorted(counts.items())
                },
                "selected_family_modal_rate": modal_count / max(len(selections), 1),
                "selected_families": sorted(counts),
            }
        )

    def dataset_metric(name: str) -> list[float]:
        return [float(item[name]) for item in per_dataset if item.get(name) is not None]

    selection_counts = Counter(all_methods)
    dataset_count = len(per_dataset)
    family_distribution = {
        method: {"count": count, "rate": count / max(len(all_methods), 1)}
        for method, count in sorted(selection_counts.items())
    }
    return {
        "policy_candidate": candidate.name,
        "policy_version": candidate.policy.version,
        "policy_complexity": candidate.complexity,
        "record_count": len(candidate_records),
        "dataset_count": dataset_count,
        "aggregation_unit": "unique_dataset; repeated seeds averaged within dataset first",
        "mean_normalized_regret": _mean(dataset_metric("mean_normalized_regret")),
        "median_normalized_regret": _median(dataset_metric("mean_normalized_regret")),
        "exact_reference_match_rate": _mean(dataset_metric("exact_reference_match_rate")),
        "catastrophic_regret_rate": _mean(dataset_metric("catastrophic_regret_rate")),
        "top2_compatibility_rate": _mean(dataset_metric("top2_compatibility_rate")),
        "policy_stability": {
            "mean_dataset_modal_family_rate": _mean(
                item["selected_family_modal_rate"] for item in per_dataset
            ),
            "datasets_with_seed_selection_variation": sum(
                len(item["selected_families"]) > 1 for item in per_dataset
            ),
            "dataset_count": dataset_count,
        },
        "family_selection_distribution": family_distribution,
        "family_collapse_warning": bool(
            max((item["rate"] for item in family_distribution.values()), default=0.0) > 0.80
        ),
        "per_dataset": per_dataset,
    }


def policy_rank_key(aggregate: dict[str, Any]) -> tuple[float, float, float, int]:
    """The predeclared lexicographic objective used for candidate selection."""

    return (
        float(aggregate["mean_normalized_regret"])
        if aggregate.get("mean_normalized_regret") is not None
        else math.inf,
        float(aggregate["catastrophic_regret_rate"])
        if aggregate.get("catastrophic_regret_rate") is not None
        else math.inf,
        -float(aggregate["top2_compatibility_rate"])
        if aggregate.get("top2_compatibility_rate") is not None
        else math.inf,
        int(aggregate["policy_complexity"]),
    )


def select_policy_candidate(
    aggregates: dict[str, dict[str, Any]],
    candidates: Sequence[PolicyCandidate],
) -> dict[str, Any]:
    """Select by the fixed objective and decide whether promotion is justified."""

    ordered = sorted(aggregates.values(), key=policy_rank_key)
    if not ordered:
        raise ValueError("No candidate policy aggregates are available for selection.")
    best = ordered[0]
    current = aggregates.get("current")
    if current is None:
        raise ValueError("The current policy baseline must be included in calibration.")
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    regret_improvement = None
    catastrophic_improvement = None
    if current.get("mean_normalized_regret") is not None and best.get("mean_normalized_regret") is not None:
        regret_improvement = float(current["mean_normalized_regret"] - best["mean_normalized_regret"])
    if current.get("catastrophic_regret_rate") is not None and best.get("catastrophic_regret_rate") is not None:
        catastrophic_improvement = float(current["catastrophic_regret_rate"] - best["catastrophic_regret_rate"])
    materially_better = bool(
        best["policy_candidate"] != "current"
        and (
            (regret_improvement is not None and regret_improvement >= PROMOTION_MIN_REGRET_IMPROVEMENT)
            or (
                regret_improvement is not None
                and abs(regret_improvement) < PROMOTION_MIN_REGRET_IMPROVEMENT
                and catastrophic_improvement is not None
                and catastrophic_improvement >= PROMOTION_MIN_CATASTROPHIC_IMPROVEMENT
            )
        )
    )
    return {
        "selection_rule": POLICY_SELECTION_RULE,
        "ranked_candidates": [item["policy_candidate"] for item in ordered],
        "selected_candidate": best["policy_candidate"],
        "baseline_candidate": "current",
        "recommendation": "promote" if materially_better else "retain_current",
        "promotion_margins": {
            "minimum_mean_regret_improvement": PROMOTION_MIN_REGRET_IMPROVEMENT,
            "minimum_catastrophic_rate_improvement": PROMOTION_MIN_CATASTROPHIC_IMPROVEMENT,
            "observed_mean_regret_improvement_vs_current": regret_improvement,
            "observed_catastrophic_rate_improvement_vs_current": catastrophic_improvement,
        },
        "selected_policy_version": candidate_by_name[best["policy_candidate"]].policy.version,
    }


def _failure_cases(records: Sequence[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    failures = [record for record in records if record.get("normalized_regret") is not None]
    failures = sorted(failures, key=lambda record: float(record["normalized_regret"]), reverse=True)
    return [
        {
            "dataset_id": record["dataset_id"],
            "seed": record["seed"],
            "task_type": record["task_type"],
            "diagnostics": record["diagnostics"],
            "deterministic_selected_method": record["deterministic_selected_method"],
            "empirical_best_method": record["empirical_best_method"],
            "deterministic_ranking": record["deterministic_ranking"],
            "empirical_reference_ranking": record["empirical_reference_ranking"],
            "normalized_regret": record["normalized_regret"],
            "policy_score_contributions": record["policy_score_contributions"],
        }
        for record in failures[:limit]
    ]


def _sensitivity_rows(aggregates: dict[str, dict[str, Any]], candidates: Sequence[PolicyCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "candidate": candidate.name,
            "policy_version": candidate.policy.version,
            "thresholds": {
                "nonlinear_moderate_threshold": candidate.policy.nonlinear_moderate_threshold,
                "nonlinear_high_threshold": candidate.policy.nonlinear_high_threshold,
                "sample_feature_ratio": [
                    candidate.policy.low_sample_feature_ratio,
                    candidate.policy.moderate_sample_feature_ratio,
                    candidate.policy.healthy_sample_feature_ratio,
                ],
                "missing_fraction": [
                    candidate.policy.moderate_missing_fraction,
                    candidate.policy.high_missing_fraction,
                ],
                "high_effective_features": candidate.policy.high_effective_features,
            },
            "mean_normalized_regret": aggregates[candidate.name].get("mean_normalized_regret"),
            "median_normalized_regret": aggregates[candidate.name].get("median_normalized_regret"),
            "catastrophic_regret_rate": aggregates[candidate.name].get("catastrophic_regret_rate"),
            "exact_reference_match_rate": aggregates[candidate.name].get("exact_reference_match_rate"),
            "top2_compatibility_rate": aggregates[candidate.name].get("top2_compatibility_rate"),
            "family_selection_distribution": aggregates[candidate.name].get("family_selection_distribution", {}),
        }
        for candidate in candidates
    ]


def build_calibration_artifact(
    cases: Sequence[BenchmarkCase],
    candidates: Sequence[PolicyCandidate],
    records: Sequence[dict[str, Any]],
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    aggregates = {
        candidate.name: aggregate_candidate_records(records, candidate) for candidate in candidates
    }
    selection = select_policy_candidate(aggregates, candidates)
    selected_name = str(selection["selected_candidate"])
    current_records = [record for record in records if record["policy_candidate"] == "current"]
    selected_records = [record for record in records if record["policy_candidate"] == selected_name]
    return {
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "evaluation_role": BenchmarkRole.POLICY_DEVELOPMENT.value,
        "benchmark_role": BenchmarkRole.POLICY_DEVELOPMENT.value,
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        "policy_version_under_test": DeterministicPolicy().version,
        "git_commit": _repository_commit(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(cases),
        "case_count": len(cases),
        "record_count": len(records),
        "dataset_ids": [case.name for case in cases],
        "random_seeds": [int(seed) for seed in seeds],
        "metric_definitions": {
            "classification_primary_metric": "macro_f1; larger is better",
            "regression_primary_metric": "rmse; smaller is better",
            "normalized_regret": "classification=max(0,best-selected); regression=max(0,selected-best)/max(abs(best),1e-12)",
            "catastrophic_regret_threshold": CATASTROPHIC_REGRET_THRESHOLD,
            "aggregation": "average repeated seeds within each unique dataset, then summarize across datasets",
        },
        "candidate_policies": [candidate.as_dict() for candidate in candidates],
        "selection_rule": POLICY_SELECTION_RULE,
        "selection": selection,
        "selected_candidate": selected_name,
        "recommendation": selection["recommendation"],
        "aggregate_metrics": aggregates,
        "per_dataset_metrics": {
            name: aggregate["per_dataset"] for name, aggregate in aggregates.items()
        },
        "sensitivity_analysis": _sensitivity_rows(aggregates, candidates),
        "failure_cases": _failure_cases(selected_records),
        "current_policy_failure_cases": _failure_cases(current_records),
        "raw_records": list(records),
    }


def render_calibration_report(artifact: dict[str, Any]) -> str:
    """Render a concise human-readable report including negative cases."""

    aggregates = artifact["aggregate_metrics"]
    lines = [
        "# Deterministic Policy Calibration Report",
        "",
        f"- Policy version under test: `{artifact['policy_version_under_test']}`",
        f"- Benchmark suite: `{artifact['benchmark_suite_version']}`",
        f"- Role: `{artifact['evaluation_role']}`",
        f"- Unique datasets: **{artifact['dataset_count']}**; repeated seeds are not treated as independent datasets",
        f"- Split seeds: `{artifact['random_seeds']}`",
        f"- Git commit: `{artifact.get('git_commit') or 'unavailable'}`",
        "",
        "## Development benchmark composition",
        "",
        "The registry assigns these cases permanently to policy development. Final-evaluation cases are rejected by the calibration runner.",
        "",
        *[f"- `{case}`" for case in artifact["dataset_ids"]],
        "",
        "## Candidate configurations and selection criterion",
        "",
        f"{artifact['selection_rule']}.",
        "",
        "| Candidate | Mean regret | Catastrophic rate | Exact match | Top-2 rate | Collapse warning |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, aggregate in aggregates.items():
        lines.append(
            f"| `{name}` | {aggregate['mean_normalized_regret']!s} | "
            f"{aggregate['catastrophic_regret_rate']!s} | {aggregate['exact_reference_match_rate']!s} | "
            f"{aggregate['top2_compatibility_rate']!s} | "
            f"{'yes' if aggregate['family_collapse_warning'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity analysis",
            "",
            "The four candidates are a deliberately small neighborhood around the current interpretable thresholds; no continuous optimizer or LLM is used.",
            "",
        ]
    )
    for row in artifact["sensitivity_analysis"]:
        lines.append(
            f"- `{row['candidate']}`: nonlinear thresholds "
            f"{row['thresholds']['nonlinear_moderate_threshold']}/{row['thresholds']['nonlinear_high_threshold']}; "
            f"sample-feature bands `{row['thresholds']['sample_feature_ratio']}`; "
            f"missingness bands `{row['thresholds']['missing_fraction']}`; "
            f"mean regret `{row['mean_normalized_regret']}`."
        )
    lines.extend(
        [
            "",
            "## Per-dataset results and failure cases",
            "",
            "Per-dataset means below preserve dataset identity before the across-dataset summary.",
            "",
        ]
    )
    current = aggregates["current"]
    lines.append("| Dataset | Seeds | Mean regret | Catastrophic rate | Top-2 rate | Selected families |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in current["per_dataset"]:
        lines.append(
            f"| `{row['dataset_id']}` | {row['seed_count']} | {row['mean_normalized_regret']!s} | "
            f"{row['catastrophic_regret_rate']!s} | {row['top2_compatibility_rate']!s} | "
            f"{', '.join(row['selected_families'])} |"
        )
    lines.extend(["", "Largest current-policy regret cases:", ""])
    for failure in artifact["current_policy_failure_cases"]:
        lines.append(
            f"- `{failure['dataset_id']}` seed `{failure['seed']}`: "
            f"deterministic `{failure['deterministic_selected_method']}`, "
            f"empirical best `{failure['empirical_best_method']}`, "
            f"regret `{failure['normalized_regret']}`."
        )
    lines.extend(
        [
            "",
            "## Family-selection distribution",
            "",
            f"{json.dumps(current['family_selection_distribution'], sort_keys=True)}",
            "",
            "## Recommendation",
            "",
            f"- Objective-selected candidate: `{artifact['selected_candidate']}`",
            f"- Recommendation: **{artifact['recommendation']}**",
            "- Final benchmark results are not used to modify the policy version under test.",
            "- Compatibility scores remain interpretable compatibility points, not probabilities of empirical optimality.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_artifacts(output_dir: Path, artifact: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "policy_calibration.json"
    markdown_path = output_dir / "policy_calibration_report.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown_path.write_text(render_calibration_report(artifact), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def run_policy_calibration(
    output_dir: str | Path = "evaluation_results/policy_calibration",
    *,
    cases: Sequence[BenchmarkCase] | None = None,
    seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    candidates: Sequence[PolicyCandidate] | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """Run offline policy development using training-only empirical CV."""

    selected_cases = calibration_cases(cases)
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive when provided.")
        selected_cases = selected_cases[:max_cases]
    selected_candidates = list(candidates or policy_candidates())
    if not any(candidate.name == "current" for candidate in selected_candidates):
        raise ValueError("Calibration candidates must include the current policy baseline.")
    records: list[dict[str, Any]] = []
    reference_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in selected_candidates:
        records.extend(
            evaluate_policy_cases(
                selected_cases,
                policy_candidate=candidate,
                seeds=seeds,
                expected_role=BenchmarkRole.POLICY_DEVELOPMENT,
                reference_cache=reference_cache,
            )
        )
    artifact = build_calibration_artifact(
        selected_cases,
        selected_candidates,
        records,
        seeds=seeds,
    )
    artifact["artifacts"] = _write_artifacts(Path(output_dir), artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the interpretable deterministic policy offline.")
    parser.add_argument("--output", default="evaluation_results/policy_calibration")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run two development cases with one seed.")
    args = parser.parse_args()
    seeds = tuple(args.seeds or ((42,) if args.smoke else DEFAULT_SPLIT_SEEDS))
    result = run_policy_calibration(
        args.output,
        seeds=seeds,
        max_cases=2 if args.smoke else args.max_cases,
    )
    print(
        json.dumps(
            {
                "recommendation": result["recommendation"],
                "selected_candidate": result["selected_candidate"],
                "dataset_count": result["dataset_count"],
                "record_count": result["record_count"],
                "artifacts": result["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
