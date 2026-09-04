"""Controlled, paired modeling-gate ablation studies.

This module owns experiment orchestration.  The production pipeline remains a
single implementation; each preset only supplies explicit runtime settings.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

from evaluation.benchmarks import BenchmarkCase, default_benchmark_cases
from evaluation.runner import EXPERIMENT_CONFIG_VERSION, run_evaluation
from evaluation.statistics import (
    DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    cluster_bootstrap_ci,
)
from evaluation.metrics import DEFAULT_THRESHOLDS, holdout_neutral_tolerance, paper_holdout_delta
from evaluation.confirmatory import (
    CONFIRMATORY_EXPERIMENT_NAME,
    load_confirmatory_manifest,
    runtime_manifest_values,
    validate_confirmatory_manifest,
    deterministic_policy_config,
    empirical_probe_config,
    config_sha256,
    repository_commit as current_repository_commit,
    experiment_code_sha256,
)
from evaluation.external_benchmarks import external_benchmark_manifest_sha256, external_benchmark_specs
from app.deterministic_policy import DeterministicPolicy
from app.empirical_challenge_probe import EmpiricalProbePolicy
from app.llm import PROMPT_SCHEMA_VERSION
from app.reconciliation import BLINDED_RECONCILIATION_PROMPT_VERSION


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
        "planner_live_success": sum(
            bool(row.get("requested_live_trial")) and row.get("agent_source") == "openai"
            for row in result.get("trials", [])
        ),
        "reconciler_live_success": sum(
            bool(row.get("reconciliation_api_call_made"))
            for row in result.get("trials", [])
        ),
    }
    return {
        "ablation": name,
        "spec": spec.as_dict(),
        "trial_count": summary.get("trial_count", 0),
        "valid_trial_count": summary.get("valid_trial_count", 0),
        "invalid_trial_count": summary.get("failed_trial_count", 0) + summary.get("invalid_trial_count", 0),
        "n_datasets": summary.get("dataset_macro_gate_health", {}).get("dataset_count", 0),
        "improved": health.get("improved_interventions", 0),
        "worsened": health.get("worsened_interventions", 0),
        "neutral": health.get("neutral_interventions", 0),
        "intervention_precision": health.get("intervention_precision"),
        "intervention_rate": health.get("intervention_rate"),
        "abstention_rate": health.get("abstention_rate"),
        "abstention_preservation_rate": health.get("abstention_preservation_rate"),
        "beneficial_intervention_rate": health.get("beneficial_intervention_rate"),
        "harmful_intervention_rate": health.get("harmful_intervention_rate"),
        "harm_rate": health.get("harm_rate", health.get("harmful_intervention_rate")),
        "neutral_intervention_rate": health.get("neutral_intervention_rate"),
        "unnecessary_intervention_rate": health.get("unnecessary_intervention_rate"),
        "paper_holdout_delta_mean": summary.get("dataset_macro_paper_holdout_delta_mean"),
        "paper_holdout_delta_median": summary.get("dataset_macro_paper_holdout_delta_median"),
        "paper_holdout_delta_ci": summary.get("dataset_macro_paper_holdout_delta_ci"),
        "paper_holdout_outcome_cis": {
            name: summary.get("dataset_macro_gate_health", {})
            .get("confidence_intervals", {}).get(name)
            for name in (
                "beneficial_intervention_rate", "harmful_intervention_rate",
                "neutral_intervention_rate", "intervention_precision",
            )
        },
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
            "strict_live_valid": summary.get("strict_live_valid", True),
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
    tolerance: float | dict[str, float] = 1e-12,
) -> dict[str, Any]:
    left = {_unit_key(row): row for row in rows_by_name.get(first, []) if row.get("trial_status") != "failed"}
    right = {_unit_key(row): row for row in rows_by_name.get(second, []) if row.get("trial_status") != "failed"}
    shared = sorted(set(left) & set(right), key=str)
    holdout_differences: list[float] = []
    holdout_task_types: list[str] = []
    holdout_by_dataset: dict[str, list[float]] = {}
    dataset_task_types: dict[str, set[str]] = {}
    diagnostic_differences: list[float] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for key in shared:
        first_row = left[key]
        second_row = right[key]
        first_delta = first_row.get("paper_holdout_delta")
        second_delta = second_row.get("paper_holdout_delta")
        if first_delta is None:
            first_delta = paper_holdout_delta(
                str(first_row.get("task_type", "classification")),
                first_row.get("initial_holdout_metric"),
                first_row.get("final_holdout_metric"),
            )
        if second_delta is None:
            second_delta = paper_holdout_delta(
                str(second_row.get("task_type", "classification")),
                second_row.get("initial_holdout_metric"),
                second_row.get("final_holdout_metric"),
            )
        if first_delta is not None and second_delta is not None:
            # Positive means the first configuration produced the better
            # untouched-holdout intervention outcome.
            difference = float(first_delta) - float(second_delta)
            holdout_differences.append(difference)
            holdout_task_types.append(str(first_row.get("task_type", "classification")))
            dataset = str(key[0])
            holdout_by_dataset.setdefault(dataset, []).append(difference)
            dataset_task_types.setdefault(dataset, set()).add(
                str(first_row.get("task_type", "classification"))
            )
        a = first_row.get("gated_normalized_regret")
        b = second_row.get("gated_normalized_regret")
        if a is not None and b is not None:
            # Training/reference regret remains available as a diagnostic,
            # never as the primary paired comparison.
            diagnostic_difference = float(b) - float(a)
            diagnostic_differences.append(diagnostic_difference)
            diagnostic_rows.append({"benchmark_case": key[0], "difference": diagnostic_difference})
        if first_delta is None or second_delta is None:
            continue
    dataset_effects = [
        {
            "benchmark_case": dataset,
            "difference": mean(differences),
            "paired_trial_count": len(differences),
            "task_type": sorted(dataset_task_types.get(dataset, {"classification"}))[0],
        }
        for dataset, differences in sorted(holdout_by_dataset.items())
    ]
    dataset_means = [row["difference"] for row in dataset_effects]
    dataset_better = {"first": 0, "second": 0, "tied": 0}
    for row in dataset_effects:
        task_type = row["task_type"]
        pair_tolerance = (
            holdout_neutral_tolerance(task_type, tolerance)
            if isinstance(tolerance, dict)
            else float(tolerance)
        )
        if row["difference"] > pair_tolerance:
            dataset_better["first"] += 1
        elif row["difference"] < -pair_tolerance:
            dataset_better["second"] += 1
        else:
            dataset_better["tied"] += 1
    # One row per dataset is intentional: the clustered bootstrap below is
    # therefore a bootstrap of the dataset-level paired effects, not of
    # individual split/repetition rows.
    holdout_difference_ci = cluster_bootstrap_ci(
        dataset_effects,
        lambda sample: mean(row["difference"] for row in sample) if sample else None,
        "benchmark_case",
    )
    trial_weighted_mean = mean(holdout_differences) if holdout_differences else None
    trial_weighted_median = median(holdout_differences) if holdout_differences else None
    diagnostic_difference_ci = cluster_bootstrap_ci(
        diagnostic_rows,
        lambda sample: mean(row["difference"] for row in sample) if sample else None,
        "benchmark_case",
    )
    return {
        "first": first,
        "second": second,
        "paired_units": len(holdout_differences),
        "paired_holdout_units": len(holdout_differences),
        "paired_training_diagnostic_units": len(diagnostic_differences),
        "n_paired_datasets": len(dataset_effects),
        "paired_holdout_dataset_effects": dataset_effects,
        # Paper-primary pairwise values: one equal-weighted effect per
        # benchmark dataset/task, with win/tie/loss also classified per task.
        "first_better": dataset_better["first"],
        "second_better": dataset_better["second"],
        "tied": dataset_better["tied"],
        "dataset_macro_first_better": dataset_better["first"],
        "dataset_macro_second_better": dataset_better["second"],
        "dataset_macro_tied": dataset_better["tied"],
        "mean_paired_holdout_delta_difference_first_advantage": mean(dataset_means) if dataset_means else None,
        "median_paired_holdout_delta_difference_first_advantage": median(dataset_means) if dataset_means else None,
        "dataset_macro_mean_paired_holdout_delta_difference_first_advantage": mean(dataset_means) if dataset_means else None,
        "dataset_macro_median_paired_holdout_delta_difference_first_advantage": median(dataset_means) if dataset_means else None,
        "paired_holdout_delta_ci": holdout_difference_ci,
        # Explicitly secondary, trial-weighted diagnostics retained for
        # reproducibility with earlier reports.
        "trial_weighted_mean_paired_holdout_delta_difference_first_advantage": trial_weighted_mean,
        "trial_weighted_median_paired_holdout_delta_difference_first_advantage": trial_weighted_median,
        "trial_weighted_first_better": sum(
            difference > (
                holdout_neutral_tolerance(task_type, tolerance)
                if isinstance(tolerance, dict) else float(tolerance)
            ) for difference, task_type in zip(holdout_differences, holdout_task_types)
        ),
        "trial_weighted_second_better": sum(
            difference < -(
                holdout_neutral_tolerance(task_type, tolerance)
                if isinstance(tolerance, dict) else float(tolerance)
            ) for difference, task_type in zip(holdout_differences, holdout_task_types)
        ),
        "trial_weighted_tied": sum(
            abs(difference) <= (
                holdout_neutral_tolerance(task_type, tolerance)
                if isinstance(tolerance, dict) else float(tolerance)
            ) for difference, task_type in zip(holdout_differences, holdout_task_types)
        ),
        "mean_paired_regret_difference_first_advantage": mean(diagnostic_differences) if diagnostic_differences else None,
        "median_paired_regret_difference_first_advantage": median(diagnostic_differences) if diagnostic_differences else None,
        "trial_weighted_mean_paired_regret_difference_first_advantage": mean(diagnostic_differences) if diagnostic_differences else None,
        "trial_weighted_median_paired_regret_difference_first_advantage": median(diagnostic_differences) if diagnostic_differences else None,
        "paired_regret_difference_ci": diagnostic_difference_ci,
        "trial_weighted_paired_regret_difference_ci": diagnostic_difference_ci,
        "training_reference_comparison_role": "secondary diagnostic; primary comparison uses untouched holdout",
        "paired_holdout_difference_sign": "first ablation paper_holdout_delta minus second ablation paper_holdout_delta; positive favors first",
        "win_loss_tie_unit": "dataset/task mean paired difference",
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
        "| Ablation | Datasets | Valid | Failed/invalid | Challenge rate | Intervention rate | Abstention rate | Beneficial | Harmful | Neutral | Holdout delta (dataset macro, descriptive) | Holdout CI | Planner calls | Reconciler calls | Probe invocations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["central_table"]:
        api = row["api_usage"]
        probe = payload["summaries"][row["ablation"]].get("probe_invocation_count", 0)
        lines.append(
            f"| {row['ablation']} | {row['n_datasets']} | {row['valid_trial_count']} | {row['invalid_trial_count']} | {row.get('challenge_rate')} | {row.get('intervention_rate')} | {row.get('abstention_rate')} | {row.get('beneficial_intervention_rate')} | {row.get('harmful_intervention_rate')} | {row.get('neutral_intervention_rate')} | {row.get('paper_holdout_delta_mean')} | {row.get('paper_holdout_delta_ci')} | {api['successful_initial_openai_calls']} | {api['successful_reconciliation_calls']} | {probe} |"
        )
    lines.extend(["", "## Paired Comparisons", ""])
    for item in payload["paired_comparisons"]:
        lines.append(
            f"- `{item['first']}` vs `{item['second']}`: dataset-macro untouched-holdout first better `{item['first_better']}`, second better `{item['second_better']}`, tied `{item['tied']}`, mean first holdout advantage `{item['mean_paired_holdout_delta_difference_first_advantage']}` (CI `{item['paired_holdout_delta_ci']}`). Trial-weighted diagnostic mean: `{item['trial_weighted_mean_paired_holdout_delta_difference_first_advantage']}`. Training-reference regret difference is secondary diagnostic: `{item['mean_paired_regret_difference_first_advantage']}`."
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
    confirmatory_config_path: str | Path | None = None,
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
    configured_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    confirmatory_metadata: dict[str, Any] | None = None
    if confirmatory_config_path is not None:
        manifest = load_confirmatory_manifest(confirmatory_config_path)
        runtime_values = runtime_manifest_values(
            experiment_name=CONFIRMATORY_EXPERIMENT_NAME,
            planner_model=resolved_planner_model,
            reconciler_model=resolved_reconciler_model,
            split_seeds=split_seeds,
            llm_repetitions=repetitions,
            holdout_fraction=0.2,
            selected_ablations=selected_names,
            deterministic_policy_version=DeterministicPolicy().version,
            deterministic_policy_sha256=config_sha256(deterministic_policy_config()),
            empirical_probe_policy_version=EmpiricalProbePolicy().policy_version,
            empirical_probe_policy_sha256=config_sha256(empirical_probe_config()),
            planner_prompt_schema_version=PROMPT_SCHEMA_VERSION,
            reconciler_prompt_schema_version=BLINDED_RECONCILIATION_PROMPT_VERSION,
            candidate_model_families=[
                "linear", "regularized_linear", "tree_ensemble", "boosted_tree"
            ],
            preprocessing_option_space=[
                "one_hot/categorical_unknown_handling=ignore",
                "ordinal/categorical_unknown_handling=use_encoded_value",
                "none/categorical_unknown_handling=ignore",
            ],
            classification_neutral_tolerance=holdout_neutral_tolerance(
                "classification", configured_thresholds
            ),
            regression_neutral_tolerance=holdout_neutral_tolerance(
                "regression", configured_thresholds
            ),
            benchmark_manifest_version=(
                selected_cases[0].benchmark_suite_version
                if selected_cases[0].benchmark_suite_version
                else "local-2"
            ),
            benchmark_manifest_sha256=external_benchmark_manifest_sha256(),
            benchmark_task_ids=[case.openml_task_id for case in selected_cases if case.openml_task_id is not None],
            benchmark_tranches={
                "core": [spec.task_id for spec in external_benchmark_specs() if spec.tier == "core"],
                "stress": [spec.task_id for spec in external_benchmark_specs() if spec.tier == "stress"],
            },
            benchmark_tier=tier,
            strict_live_required=require_live,
            bootstrap_settings={
                "method": "dataset_cluster_bootstrap_percentile",
                "replicates": DEFAULT_BOOTSTRAP_REPLICATES,
                "confidence_level": DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL,
                "seed": DEFAULT_BOOTSTRAP_SEED,
            },
            experiment_config_version=EXPERIMENT_CONFIG_VERSION,
            expected_experiment_code_sha256=experiment_code_sha256(),
            source_git_commit=current_repository_commit(),
        )
        if suite != "external":
            raise ValueError("Confirmatory manifest enforcement requires suite='external'.")
        confirmatory_metadata = validate_confirmatory_manifest(manifest, runtime_values)
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
        "experiment_config_version": EXPERIMENT_CONFIG_VERSION,
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
        "benchmark_manifest_version": (
            selected_cases[0].benchmark_suite_version
            if selected_cases[0].benchmark_suite_version
            else "local-2"
        ),
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
        "planner_prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "reconciler_prompt_schema_version": BLINDED_RECONCILIATION_PROMPT_VERSION,
        "deterministic_policy_version": DeterministicPolicy().version,
        "deterministic_policy": {
            "version": DeterministicPolicy().version,
            "parameters": "captured in evaluation/configs/paper_confirmatory_v1.json",
        },
        "empirical_probe_policy_version": EmpiricalProbePolicy().policy_version,
        "empirical_probe_policy": EmpiricalProbePolicy().as_dict(),
        "reconciliation_prompt_version": BLINDED_RECONCILIATION_PROMPT_VERSION,
        "thresholds": configured_thresholds,
        "holdout_fraction": 0.2,
        "candidate_model_families": ["linear", "regularized_linear", "tree_ensemble", "boosted_tree"],
        "preprocessing_option_space": [
            "one_hot/categorical_unknown_handling=ignore",
            "ordinal/categorical_unknown_handling=use_encoded_value",
            "none/categorical_unknown_handling=ignore",
        ],
        "statistical_settings": {
            "independent_unit": "dataset/task",
            "bootstrap_method": "dataset_cluster_bootstrap_percentile",
            "bootstrap_replicates": 10000,
            "bootstrap_confidence_level": 0.95,
            "bootstrap_seed": 20260824,
        },
        "confirmatory_config_snapshot": "evaluation/configs/paper_confirmatory_v1.json",
        "confirmatory_mode": confirmatory_metadata is not None,
        "confirmatory_config_status": (
            confirmatory_metadata["status"] if confirmatory_metadata else "not_selected"
        ),
        "experiment_config_path": str(Path(confirmatory_config_path).resolve()) if confirmatory_config_path else None,
        "experiment_config_sha256": confirmatory_metadata.get("experiment_config_sha256") if confirmatory_metadata else None,
        "expected_experiment_code_sha256": confirmatory_metadata.get("expected_experiment_code_sha256") if confirmatory_metadata else None,
        "source_git_commit": confirmatory_metadata.get("source_git_commit") if confirmatory_metadata else repository_commit(),
        "frozen_manifest_path": (
            str(root / "frozen_confirmatory_manifest.json")
            if confirmatory_metadata else None
        ),
        "strict_live_required": require_live,
        "fallback_rows": None,
        "config_mismatch_detected": False,
        "confirmatory_valid": None,
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
            "benchmark_manifest_version",
            "thresholds",
            "confirmatory_config_snapshot",
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

    if confirmatory_metadata is not None:
        shutil.copyfile(
            Path(confirmatory_config_path),
            root / "frozen_confirmatory_manifest.json",
        )

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
            confirmatory_config_path=confirmatory_config_path,
            confirmatory_selected_ablations=selected_names if confirmatory_metadata else None,
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
        _paired_comparison(
            rows_by_name,
            first,
            second,
            tolerance={
                "classification": holdout_neutral_tolerance(
                    "classification", {**DEFAULT_THRESHOLDS, **(thresholds or {})}
                ),
                "regression": holdout_neutral_tolerance(
                    "regression", {**DEFAULT_THRESHOLDS, **(thresholds or {})}
                ),
            },
        )
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
    if confirmatory_metadata is not None:
        fallback_rows = sum(
            int(result["summary"].get("fallback_rows", 0)) for result in results.values()
        )
        root_config.update({
            "fallback_rows": fallback_rows,
            "external_benchmark_manifest_matches": bool(
                all(result["summary"].get("external_benchmark_manifest_matches") is True for result in results.values())
            ),
            "confirmatory_valid": bool(
                all(result["summary"].get("confirmatory_valid") is True for result in results.values())
            ),
        })
        combined.update(root_config)
        combined["confirmatory_valid"] = root_config["confirmatory_valid"]
        config_path.write_text(json.dumps(root_config, indent=2, sort_keys=True), encoding="utf-8")
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
    parser.add_argument(
        "--confirmatory-config",
        help="Opt into a frozen external confirmatory manifest; development runs omit this flag.",
    )
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
        confirmatory_config_path=args.confirmatory_config,
    )
    print(json.dumps(result["summary"]["central_table"], indent=2, default=str))


if __name__ == "__main__":
    main()
