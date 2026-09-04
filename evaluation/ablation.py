"""Controlled, paired modeling-gate ablation studies.

This module owns experiment orchestration.  The production pipeline remains a
single implementation; each preset only supplies explicit runtime settings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

from evaluation.benchmarks import BenchmarkCase, default_benchmark_cases
from evaluation.runner import run_evaluation
from evaluation.statistics import cluster_bootstrap_ci


ABLATION_SCHEMA_VERSION = "modeling-gate-ablation-v1"
EXPERIMENT_FREEZE_METADATA_VERSION = "experiment-freeze-v1"
PRIMARY_ABLATION_NAMES = (
    "llm_only",
    "hard_validation_only",
    "deterministic_only",
    "always_reconcile",
    "probe_direct",
    "full",
)


@dataclass(frozen=True)
class AblationSpec:
    """Complete, serializable definition of one architecture stage."""

    name: str
    decision_mode: str
    soft_challenge_strategy: str
    interaction_diagnostics: bool = True
    classification_boundary_diagnostics: bool = True
    empirical_probe: bool = False
    challenger_enabled: bool = True
    hard_validation_enabled: bool = True
    reconciliation_enabled: bool = False
    reconcile_on_any_disagreement: bool = False
    direct_probe_selection_enabled: bool = False
    abstention_enabled: bool = True
    legacy: bool = False
    schema_version: str = ABLATION_SCHEMA_VERSION
    deterministic_policy_version: str = "4"
    soft_challenge_policy_version: str = "v1"
    empirical_probe_policy_version: str = "v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "decision_mode": self.decision_mode,
            "soft_challenge_strategy": self.soft_challenge_strategy,
            "interaction_diagnostics": self.interaction_diagnostics,
            "classification_boundary_diagnostics": self.classification_boundary_diagnostics,
            "empirical_probe": self.empirical_probe,
            "challenger_enabled": self.challenger_enabled,
            "hard_validation_enabled": self.hard_validation_enabled,
            "reconciliation_enabled": self.reconciliation_enabled,
            "reconcile_on_any_disagreement": self.reconcile_on_any_disagreement,
            "direct_probe_selection_enabled": self.direct_probe_selection_enabled,
            "abstention_enabled": self.abstention_enabled,
            "legacy": self.legacy,
            "schema_version": self.schema_version,
            "deterministic_policy_version": self.deterministic_policy_version,
            "soft_challenge_policy_version": self.soft_challenge_policy_version,
            "empirical_probe_policy_version": self.empirical_probe_policy_version,
        }


def ablation_presets() -> dict[str, AblationSpec]:
    """Return versioned presets, with no benchmark-specific behavior."""

    common = {
        "interaction_diagnostics": True,
        "classification_boundary_diagnostics": True,
    }
    return {
        "llm_only": AblationSpec(
            "llm_only", "llm_only", "calibrated", empirical_probe=False,
            challenger_enabled=False, hard_validation_enabled=True, abstention_enabled=False, **common
        ),
        "hard_validation_only": AblationSpec(
            "hard_validation_only", "hard_validation_only", "calibrated", empirical_probe=False,
            reconciliation_enabled=False, abstention_enabled=False, **common
        ),
        "deterministic_only": AblationSpec(
            "deterministic_only", "deterministic_only", "calibrated", empirical_probe=False,
            reconciliation_enabled=False, abstention_enabled=False, **common
        ),
        "always_reconcile": AblationSpec(
            "always_reconcile", "always_reconcile", "calibrated", empirical_probe=False,
            reconciliation_enabled=True, reconcile_on_any_disagreement=True, abstention_enabled=False, **common
        ),
        # Backward-compatible name for the former primary baseline.
        "blinded_always_reconcile": AblationSpec(
            "blinded_always_reconcile", "always_reconcile", "calibrated", empirical_probe=False,
            reconciliation_enabled=True, reconcile_on_any_disagreement=True, legacy=True, **common
        ),
        "probe_direct": AblationSpec(
            "probe_direct", "probe_direct", "calibrated", empirical_probe=True,
            reconciliation_enabled=False, direct_probe_selection_enabled=True, **common
        ),
        "high_confidence_only": AblationSpec(
            "high_confidence_only", "selective", "high_confidence_only", empirical_probe=False, **common
        ),
        # The calibrated baseline disables the newer evidence sources so the
        # interaction/boundary stage has a clean paired comparison.
        "selective_calibrated": AblationSpec(
            "selective_calibrated", "selective", "calibrated",
            interaction_diagnostics=False,
            classification_boundary_diagnostics=False,
            empirical_probe=False,
            legacy=True,
        ),
        "interaction_boundary_aware": AblationSpec(
            "interaction_boundary_aware", "selective", "calibrated", empirical_probe=False, **common
        ),
        "empirical_probe": AblationSpec(
            "empirical_probe", "selective", "calibrated", empirical_probe=True, **common
        ),
        "probe_first": AblationSpec(
            "probe_first", "probe_direct", "calibrated", empirical_probe=True,
            reconciliation_enabled=False, direct_probe_selection_enabled=True, legacy=True, **common
        ),
        "full": AblationSpec(
            "full", "full", "calibrated", empirical_probe=True,
            reconciliation_enabled=True, **common
        ),
    }


def _read_trials(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _health_row(name: str, result: dict[str, Any], spec: AblationSpec) -> dict[str, Any]:
    summary = result["summary"]
    health = summary.get("gate_health", {})
    live = {
        "requested_live_trials": sum(bool(row.get("requested_live_trial")) for row in result.get("trials", [])),
        "successful_initial_openai_calls": sum(
            bool(row.get("initial_modeling_call_made")) and row.get("agent_source") == "openai"
            for row in result.get("trials", [])
        ),
        "failed_initial_openai_calls": sum(
            bool(row.get("requested_live_trial")) and row.get("agent_request_status") == "failed"
            for row in result.get("trials", [])
        ),
        "successful_reconciliation_calls": sum(
            bool(row.get("reconciliation_api_call_made")) for row in result.get("trials", [])
        ),
        "failed_reconciliation_calls": sum(
            bool(row.get("reconciliation_request_failed")) for row in result.get("trials", [])
        ),
        "fallback_rows": sum(bool(row.get("fallback_row")) for row in result.get("trials", [])),
    }
    return {
        "ablation": name,
        "spec": spec.as_dict(),
        "trial_count": summary.get("trial_count", 0),
        "improved": health.get("improved_interventions", 0),
        "worsened": health.get("worsened_interventions", 0),
        "neutral": health.get("neutral_interventions", 0),
        "intervention_precision": health.get("intervention_precision"),
        "harmful_intervention_rate": health.get("harmful_intervention_rate"),
        "unnecessary_intervention_rate": health.get("unnecessary_intervention_rate"),
        "challenge_recall": health.get("challenge_recall"),
        "mean_regret_reduction": health.get("mean_regret_reduction"),
        "median_regret_reduction": health.get("median_regret_reduction"),
        "catastrophic_regret_prevented": health.get("catastrophic_prevented_count", 0),
        "catastrophic_regret_introduced": health.get("catastrophic_introduced_count", 0),
        "catastrophic_net": health.get("net_catastrophic_prevention"),
        "api_usage": live,
        "integrity": {
            "strict_live": bool(result.get("config", {}).get("require_live", False)),
            "fallback_rows_zero": live["fallback_rows"] == 0,
        },
    }


def _unit_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("benchmark_case"),
        row.get("perturbation_id", "clean"),
        row.get("split_seed"),
        row.get("trial"),
        row.get("evaluation_variant", "standard"),
    )


def _paired_comparison(
    rows_by_name: dict[str, list[dict[str, Any]]],
    first: str,
    second: str,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    left = {_unit_key(row): row for row in rows_by_name.get(first, []) if row.get("trial_status") != "failed"}
    right = {_unit_key(row): row for row in rows_by_name.get(second, []) if row.get("trial_status") != "failed"}
    shared = sorted(set(left) & set(right), key=str)
    differences: list[float] = []
    paired_rows: list[dict[str, Any]] = []
    first_better = second_better = tied = 0
    for key in shared:
        a = left[key].get("gated_normalized_regret")
        b = right[key].get("gated_normalized_regret")
        if a is None or b is None:
            continue
        # Positive means the first configuration has lower regret.
        difference = float(b) - float(a)
        differences.append(difference)
        paired_rows.append({"benchmark_case": key[0], "difference": difference})
        if difference > tolerance:
            first_better += 1
        elif difference < -tolerance:
            second_better += 1
        else:
            tied += 1
    difference_ci = cluster_bootstrap_ci(
        paired_rows,
        lambda sample: mean(row["difference"] for row in sample) if sample else None,
        "benchmark_case",
    )
    return {
        "first": first,
        "second": second,
        "paired_units": len(differences),
        "n_paired_datasets": len({row["benchmark_case"] for row in paired_rows}),
        "first_better": first_better,
        "second_better": second_better,
        "tied": tied,
        "mean_paired_regret_difference_first_advantage": mean(differences) if differences else None,
        "median_paired_regret_difference_first_advantage": median(differences) if differences else None,
        "paired_regret_difference_ci": difference_ci,
    }


def _render_combined_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Paired Modeling-Gate Ablation Study",
        "",
        f"- Ablation schema: `{payload['ablation_schema_version']}`",
        f"- Benchmark suite: `{payload.get('suite', 'local')}`; tier: `{payload.get('tier')}`",
        f"- Split seeds: `{payload['split_seeds']}`",
        f"- LLM repetitions per split: `{payload['llm_repetitions']}`",
        f"- Planner model: `{payload.get('planner_model', payload.get('model'))}`",
        f"- Reconciler model: `{payload.get('reconciler_model', payload.get('model'))}`",
        f"- Strict live: `{payload['require_live']}`",
        "",
        "## Central Comparison",
        "",
        "| Ablation | Improved | Worsened | Neutral | Precision | Harm rate | Median regret reduction | Catastrophic net | Initial calls | Reconciliation calls | Probe invocations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["central_table"]:
        api = row["api_usage"]
        probe = payload["summaries"][row["ablation"]].get("probe_invocation_count", 0)
        lines.append(
            f"| {row['ablation']} | {row['improved']} | {row['worsened']} | {row['neutral']} | {row['intervention_precision']} | {row['harmful_intervention_rate']} | {row['median_regret_reduction']} | {row['catastrophic_net']} | {api['successful_initial_openai_calls']} | {api['successful_reconciliation_calls']} | {probe} |"
        )
    lines.extend(["", "## Paired Comparisons", ""])
    for item in payload["paired_comparisons"]:
        lines.append(
            f"- `{item['first']}` vs `{item['second']}`: first better `{item['first_better']}`, second better `{item['second_better']}`, tied `{item['tied']}`, mean first advantage `{item['mean_paired_regret_difference_first_advantage']}`."
        )
    lines.extend(["", "## Live-Trial Integrity", ""])
    for name, row in payload["central_by_ablation"].items():
        api = row["api_usage"]
        lines.append(
            f"- `{name}`: requested `{api['requested_live_trials']}`, initial failures `{api['failed_initial_openai_calls']}`, reconciliation failures `{api['failed_reconciliation_calls']}`, fallback rows `{api['fallback_rows']}`."
        )
    lines.extend([
        "",
        "Initial proposals are keyed by case, perturbation, split seed, LLM repetition, model, prompt schema, training-profile digest, target, and task. They are generated once and reused across compatible ablations; reconciliation outputs are never shared across prompt variants.",
        "",
        "Split-seed variation is represented by `split_seed`; stochastic LLM variation is represented independently by `trial`/LLM repetition. Every paired comparison uses the same unit key.",
    ])
    return "\n".join(lines) + "\n"


def run_ablation_study(
    output_dir: str | Path,
    *,
    cases: Sequence[BenchmarkCase] | None = None,
    split_seeds: Sequence[int] = (42,),
    repetitions: int = 1,
    model: str = "gpt-4.1-mini",
    planner_model: str | None = None,
    reconciler_model: str | None = None,
    offline: bool = False,
    require_live: bool = False,
    include_perturbations: bool = False,
    ablations: Sequence[str] | None = None,
    thresholds: dict[str, float] | None = None,
    case_names: Sequence[str] | None = None,
    modeling_plan_factory: Any | None = None,
    reconciliation_factory: Any | None = None,
    resume: bool = False,
    suite: str = "local",
    tier: str | None = None,
) -> dict[str, Any]:
    if require_live and offline:
        raise ValueError("require_live cannot be combined with offline mode.")
    if suite not in {"local", "external"}:
        raise ValueError("suite must be 'local' or 'external'.")
    all_specs = ablation_presets()
    selected_names = list(ablations) if ablations is not None else list(PRIMARY_ABLATION_NAMES)
    unknown = sorted(set(selected_names) - set(all_specs))
    if unknown:
        raise ValueError(f"Unknown ablation preset(s): {', '.join(unknown)}")
    selected_specs = [all_specs[name] for name in selected_names]
    if cases is not None:
        selected_cases = list(cases)
    elif suite == "external":
        from evaluation.external_benchmarks import external_benchmark_cases

        selected_cases = external_benchmark_cases()
    else:
        selected_cases = default_benchmark_cases()
    if case_names:
        wanted = set(case_names)
        selected_cases = [case for case in selected_cases if case.name in wanted]
    if tier is not None:
        selected_cases = [case for case in selected_cases if case.tier == tier]
    if not selected_cases:
        raise ValueError("No benchmark cases selected.")
    resolved_planner_model = planner_model or model
    resolved_reconciler_model = reconciler_model or model
    split_seeds = tuple(int(seed) for seed in split_seeds)
    if not split_seeds:
        raise ValueError("At least one split seed is required.")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    def repository_commit() -> str | None:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                check=True, timeout=2,
            ).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    root_config = {
        "experiment_freeze_metadata_version": EXPERIMENT_FREEZE_METADATA_VERSION,
        "repository_commit": repository_commit(),
        "ablation_schema_version": ABLATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_seeds": list(split_seeds),
        "llm_repetitions": repetitions,
        "model": model,
        "planner_model": resolved_planner_model,
        "reconciler_model": resolved_reconciler_model,
        "planner_model_requested": resolved_planner_model,
        "reconciler_model_requested": resolved_reconciler_model,
        "suite": suite,
        "tier": tier,
        "offline": offline,
        "require_live": require_live,
        "include_perturbations": include_perturbations,
        "selected_ablations": selected_names,
        "ablation_definitions": {name: spec.as_dict() for name, spec in all_specs.items()},
        "benchmark_cases": [case.as_dict() for case in selected_cases],
        "benchmark_task_ids": [
            case.openml_task_id for case in selected_cases
            if case.openml_task_id is not None
        ],
        "prompt_schema_version": "modeling-gate-v1",
        "deterministic_policy_version": "captured_in_each_child_evaluation_config",
        "empirical_probe_policy_version": "captured_in_each_child_evaluation_config",
        "thresholds": thresholds or "captured_in_each_child_evaluation_config",
        "evaluation_objective": "intervention-quality-v1",
    }
    if suite == "external":
        root_config["benchmark_suite_version"] = (
            selected_cases[0].benchmark_suite_version or "unknown"
        )
    if resume and config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "ablation_schema_version",
            "split_seeds",
            "llm_repetitions",
            "selected_ablations",
            "require_live",
            "suite",
            "tier",
            "planner_model",
            "reconciler_model",
        ):
            if existing.get(key) != root_config.get(key):
                raise ValueError(f"Existing ablation configuration is incompatible for {key!r}.")
        root_config = {
            **existing,
            "suite": existing.get("suite", "local"),
            "tier": existing.get("tier"),
            "planner_model": existing["planner_model"],
            "reconciler_model": existing["reconciler_model"],
            "planner_model_requested": existing["planner_model_requested"],
            "reconciler_model_requested": existing["reconciler_model_requested"],
        }
    elif config_path.exists():
        raise ValueError("Output directory already contains an ablation study; use --resume or choose a new directory.")
    else:
        config_path.write_text(json.dumps(root_config, indent=2, sort_keys=True), encoding="utf-8")

    proposal_cache_path = root / "proposal_cache.jsonl"
    results: dict[str, dict[str, Any]] = {}
    trial_rows: dict[str, list[dict[str, Any]]] = {}
    for spec in selected_specs:
        spec_dir = root / spec.name
        result = run_evaluation(
            spec_dir,
            cases=selected_cases,
            repetitions=repetitions,
            seed=split_seeds[0],
            split_seeds=split_seeds,
            model=model,
            planner_model=resolved_planner_model,
            reconciler_model=resolved_reconciler_model,
            offline=offline,
            require_live=require_live,
            include_perturbations=include_perturbations,
            thresholds=thresholds,
            case_names=case_names,
            modeling_plan_factory=modeling_plan_factory,
            reconciliation_factory=reconciliation_factory,
            ablation_spec=spec,
            proposal_cache_path=proposal_cache_path,
            empirical_reference_cache_path=root / "empirical_reference_cache.json",
            resume=resume and (spec_dir / "config.json").is_file(),
            suite=suite,
            tier=tier,
        )
        results[spec.name] = result
        trial_rows[spec.name] = result["trials"]

    central = [_health_row(name, results[name], all_specs[name]) for name in selected_names]
    summaries = {name: results[name]["summary"] for name in selected_names}
    rows_by_name = {name: trial_rows[name] for name in selected_names}
    pairs = [
        ("full", "llm_only"),
        ("hard_validation_only", "llm_only"),
        ("deterministic_only", "hard_validation_only"),
        ("always_reconcile", "full"),
        ("probe_direct", "full"),
    ]
    paired = [
        _paired_comparison(rows_by_name, first, second)
        for first, second in pairs
        if first in rows_by_name and second in rows_by_name
    ]
    combined = {
        **root_config,
        "central_table": central,
        "central_by_ablation": {row["ablation"]: row for row in central},
        "summaries": summaries,
        "paired_comparisons": paired,
        "live_integrity": {
            row["ablation"]: row["api_usage"] for row in central
        },
    }
    (root / "ablation_summary.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "ablation_summary.md").write_text(
        _render_combined_markdown(combined), encoding="utf-8"
    )
    return {
        "output_dir": str(root),
        "config": str(config_path),
        "summary": combined,
        "paths": {
            "proposal_cache": str(proposal_cache_path),
            "empirical_reference_cache": str(root / "empirical_reference_cache.json"),
            "summary_json": str(root / "ablation_summary.json"),
            "summary_markdown": str(root / "ablation_summary.md"),
        },
    }


def _parse_split_seeds(values: list[str] | None) -> list[int]:
    if not values:
        return [42]
    result: list[int] = []
    for value in values:
        result.extend(int(item.strip()) for item in value.split(",") if item.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paired AutoDSAgent modeling-gate ablation study.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", action="append", dest="split_seeds")
    parser.add_argument("--split-seeds", action="append", dest="split_seeds_alias")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--planner-model")
    parser.add_argument("--reconciler-model")
    parser.add_argument("--suite", choices=("local", "external"), default="local")
    parser.add_argument("--tier", choices=("core", "stress"))
    parser.add_argument("--ablation", action="append", dest="ablations")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--include-perturbations", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_ablation_study(
        args.output,
        split_seeds=_parse_split_seeds((args.split_seeds or []) + (args.split_seeds_alias or [])),
        repetitions=args.repetitions,
        model=args.model,
        planner_model=args.planner_model,
        reconciler_model=args.reconciler_model,
        ablations=args.ablations,
        case_names=args.cases,
        offline=args.offline,
        require_live=args.require_live,
        include_perturbations=args.include_perturbations,
        resume=args.resume,
        suite=args.suite,
        tier=args.tier,
    )
    print(json.dumps(result["summary"]["central_table"], indent=2, default=str))


if __name__ == "__main__":
    main()
