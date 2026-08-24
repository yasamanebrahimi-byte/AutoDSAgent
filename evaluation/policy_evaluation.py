"""Frozen-policy evaluation on the untouched final benchmark role."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from evaluation.benchmarks import BENCHMARK_SUITE_VERSION, BenchmarkRole
from evaluation.policy_calibration import (
    CALIBRATION_SCHEMA_VERSION,
    DEFAULT_SPLIT_SEEDS,
    _failure_cases,
    _repository_commit,
    aggregate_candidate_records,
    evaluate_policy_cases,
    final_evaluation_cases,
    policy_candidates,
)
from evaluation.metrics import GATE_OBJECTIVE_VERSION


def _final_holdout_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        holdout = record.get("holdout_evaluation") or {}
        if holdout.get("status") == "evaluated":
            by_dataset[str(record["dataset_id"])].append(record)
    per_dataset: list[dict[str, Any]] = []
    for dataset_id, dataset_records in sorted(by_dataset.items()):
        metrics: dict[str, list[float]] = defaultdict(list)
        for record in dataset_records:
            for name, value in (record["holdout_evaluation"].get("holdout_metrics") or {}).items():
                if value is not None:
                    metrics[name].append(float(value))
        per_dataset.append(
            {
                "dataset_id": dataset_id,
                "seed_count": len(dataset_records),
                "mean_holdout_metrics": {
                    name: sum(values) / len(values) for name, values in sorted(metrics.items()) if values
                },
            }
        )
    return {
        "aggregation_unit": "unique_dataset; repeated seeds averaged within dataset first",
        "evaluated_dataset_count": len(per_dataset),
        "per_dataset": per_dataset,
    }


def render_final_report(artifact: dict[str, Any]) -> str:
    aggregate = artifact["aggregate_metrics"]
    lines = [
        "# Frozen Deterministic Policy Final Evaluation",
        "",
        f"- Evaluation role: `{artifact['evaluation_role']}`",
        f"- Frozen policy version: `{artifact['policy_version']}`",
        f"- Benchmark suite: `{artifact['benchmark_suite_version']}`",
        f"- Unique datasets: **{artifact['dataset_count']}**",
        f"- Split seeds: `{artifact['random_seeds']}`",
        f"- Git commit: `{artifact.get('git_commit') or 'unavailable'}`",
        "",
        "## Frozen evaluation protocol",
        "",
        "Only final-evaluation cases are accepted. Diagnostics and empirical-reference CV use each case's training partition; the holdout is scored after the policy decision and cannot modify policy parameters.",
        "",
        "## Policy quality metrics",
        "",
        f"- Gate objective: `{artifact.get('gate_objective_version', GATE_OBJECTIVE_VERSION)}`",
        f"- Intervention precision: `{aggregate.get('intervention_precision')}`; challenge yield: `{aggregate.get('challenge_yield')}`; harmful-intervention rate: `{aggregate.get('harmful_intervention_rate')}`",
        f"- Challenge recall: `{aggregate.get('challenge_recall')}`; unnecessary-intervention rate: `{aggregate.get('unnecessary_intervention_rate')}`",
        f"- Mean regret reduction: `{aggregate.get('mean_regret_reduction')}`; median: `{aggregate.get('median_regret_reduction')}`",
        f"- Catastrophic prevented: `{aggregate.get('catastrophic_prevented_count', 0)}`; introduced: `{aggregate.get('catastrophic_introduced_count', 0)}`; net: `{aggregate.get('net_catastrophic_prevention')}`",
        f"- Mean dataset-level normalized regret: `{aggregate['mean_normalized_regret']}`",
        f"- Median dataset-level normalized regret: `{aggregate['median_normalized_regret']}`",
        f"- Empirical-reference match rate: `{aggregate['exact_reference_match_rate']}`",
        f"- Catastrophic-regret rate: `{aggregate['catastrophic_regret_rate']}`",
        f"- Top-2 compatibility success: `{aggregate['top2_compatibility_rate']}`",
        f"- Family-selection distribution: `{json.dumps(aggregate['family_selection_distribution'], sort_keys=True)}`",
        "",
        "## Per-dataset final results",
        "",
        "| Dataset | Seeds | Interaction | Score | Mean regret | Catastrophic rate | Top-2 rate | Selected families |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in aggregate["per_dataset"]:
        interaction = row["interaction_diagnostics"]
        lines.append(
            f"| `{row['dataset_id']}` | {row['seed_count']} | {interaction['modal_strength']} | "
            f"{interaction['mean_score']!s} | {row['mean_normalized_regret']!s} | "
            f"{row['catastrophic_regret_rate']!s} | {row['top2_compatibility_rate']!s} | "
            f"{', '.join(row['selected_families'])} |"
        )
    interaction = aggregate["interaction_diagnostics"]
    boundary = aggregate["boundary_diagnostics"]
    lines.extend(
        [
            "",
            "## Interaction diagnostics",
            "",
            f"- Mean interaction score: `{interaction['mean_score']}`; median: `{interaction['median_score']}`",
            f"- Strength distribution: `{json.dumps(interaction['strength_distribution'], sort_keys=True)}`",
            f"- Mean evaluated pairs: `{interaction['mean_pairs_evaluated']}`; mean strong-pair fraction: `{interaction['mean_strong_pair_fraction']}`",
            f"- Regime metrics: `{json.dumps(aggregate['interaction_regime_metrics'], sort_keys=True)}`",
            f"- Top interaction evidence: `{json.dumps(interaction['top_interaction_evidence'], sort_keys=True)}`",
            "",
            "## Classification boundary diagnostics",
            "",
            f"- Mean boundary complexity score: `{boundary['mean_score']}`; median: `{boundary['median_score']}`",
            f"- Mean linear probe score: `{boundary['mean_linear_boundary_probe_score']}`; mean normalized linear separability: `{boundary['mean_linear_separability_score']}`",
            f"- Mean local class consistency: `{boundary['mean_local_class_consistency']}`; mean nonlinear advantage: `{boundary['mean_nonlinear_advantage_score']}`",
            f"- Category distribution: `{json.dumps(boundary['category_distribution'], sort_keys=True)}`",
            f"- Confidence distribution: `{json.dumps(boundary['confidence_distribution'], sort_keys=True)}`",
            f"- Regime metrics: `{json.dumps(aggregate['boundary_regime_metrics'], sort_keys=True)}`",
            "",
            "## Final holdout metrics",
            "",
        ]
    )
    for row in artifact["holdout_metrics"]["per_dataset"]:
        lines.append(f"- `{row['dataset_id']}`: `{row['mean_holdout_metrics']}`")
    lines.extend(["", "## Largest policy failure cases", ""])
    for failure in artifact["failure_cases"]:
        lines.append(
            f"- `{failure['dataset_id']}` seed `{failure['seed']}`: deterministic "
            f"`{failure['deterministic_selected_method']}`, empirical best "
            f"`{failure['empirical_best_method']}`, regret `{failure['normalized_regret']}`."
        )
    lines.extend(
        [
            "",
            "Final benchmark results are descriptive evidence for the frozen policy. They are not used to tune or rewrite the policy version evaluated here.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_policy_evaluation(
    output_dir: str | Path = "evaluation_results/policy_evaluation",
    *,
    cases: Sequence[Any] | None = None,
    seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """Evaluate the current frozen runtime policy without candidate selection."""

    selected_cases = final_evaluation_cases(cases)
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive when provided.")
        selected_cases = selected_cases[:max_cases]
    current = next(candidate for candidate in policy_candidates() if candidate.name == "current")
    records = evaluate_policy_cases(
        selected_cases,
        policy_candidate=current,
        seeds=seeds,
        expected_role=BenchmarkRole.FINAL_EVALUATION,
        include_holdout=True,
    )
    aggregate = aggregate_candidate_records(records, current)
    artifact: dict[str, Any] = {
        "evaluation_schema_version": CALIBRATION_SCHEMA_VERSION,
        "gate_objective_version": GATE_OBJECTIVE_VERSION,
        "evaluation_role": BenchmarkRole.FINAL_EVALUATION.value,
        "benchmark_role": BenchmarkRole.FINAL_EVALUATION.value,
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        "policy_version": current.policy.version,
        "git_commit": _repository_commit(),
        "dataset_count": len(selected_cases),
        "case_count": len(selected_cases),
        "record_count": len(records),
        "dataset_ids": [case.name for case in selected_cases],
        "random_seeds": [int(seed) for seed in seeds],
        "metric_definitions": {
            "classification_primary_metric": "macro_f1; larger is better",
            "regression_primary_metric": "rmse; smaller is better",
            "normalized_regret": "training-only CV reference regret; holdout is never used for policy choice",
            "intervention_quality": "the frozen policy is judged by intervention outcomes; family matches are secondary diagnostics",
        },
        "aggregate_metrics": aggregate,
        "holdout_metrics": _final_holdout_summary(records),
        "failure_cases": _failure_cases(records),
        "raw_records": records,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "policy_evaluation.json"
    markdown_path = output_path / "policy_evaluation_report.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown_path.write_text(render_final_report(artifact), encoding="utf-8")
    artifact["artifacts"] = {"json": str(json_path), "markdown": str(markdown_path)}
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen deterministic policy on final benchmarks.")
    parser.add_argument("--output", default="evaluation_results/policy_evaluation")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run two final cases with one seed.")
    args = parser.parse_args()
    seeds = tuple(args.seeds or ((42,) if args.smoke else DEFAULT_SPLIT_SEEDS))
    result = run_policy_evaluation(
        args.output,
        seeds=seeds,
        max_cases=2 if args.smoke else args.max_cases,
    )
    print(
        json.dumps(
            {
                "evaluation_role": result["evaluation_role"],
                "policy_version": result["policy_version"],
                "dataset_count": result["dataset_count"],
                "record_count": result["record_count"],
                "artifacts": result["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
