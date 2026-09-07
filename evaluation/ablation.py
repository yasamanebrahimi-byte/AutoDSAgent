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
from evaluation.metrics import (
    DEFAULT_THRESHOLDS,
    REGRESSION_HOLDOUT_RMSE_EPSILON,
    holdout_neutral_tolerance,
    paper_holdout_delta,
    relative_rmse_improvement,
    summarize_trials,
)
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
    validate_resume_manifest_identity,
    model_conditions,
    condition_repetition_ids,
    expand_confirmatory_evaluation_units,
    validate_confirmatory_completeness,
)
from evaluation.external_benchmarks import external_benchmark_manifest_sha256, external_benchmark_specs
from evaluation.provenance import environment_provenance
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
SECONDARY_PAIRED_COMPARISON_PAIRS = (
    ("llm_with_diagnostics", "llm_only"),
)
SECONDARY_ABLATION_NAMES = ("llm_with_diagnostics",)


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
    analysis_role: str = "secondary"
    planner_evidence_mode: str = "training_profile_only"

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
            "analysis_role": self.analysis_role,
            "planner_evidence_mode": self.planner_evidence_mode,
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
            challenger_enabled=False, hard_validation_enabled=True, abstention_enabled=False,
            analysis_role="primary", **common
        ),
        "hard_validation_only": AblationSpec(
            "hard_validation_only", "hard_validation_only", "calibrated", empirical_probe=False,
            reconciliation_enabled=False, abstention_enabled=False, analysis_role="primary", **common
        ),
        "deterministic_only": AblationSpec(
            "deterministic_only", "deterministic_only", "calibrated", empirical_probe=False,
            reconciliation_enabled=False, abstention_enabled=False, analysis_role="primary", **common
        ),
        "always_reconcile": AblationSpec(
            "always_reconcile", "always_reconcile", "calibrated", empirical_probe=False,
            reconciliation_enabled=True, reconcile_on_any_disagreement=True, abstention_enabled=False,
            analysis_role="primary", **common
        ),
        # Backward-compatible name for the former primary baseline.
        "blinded_always_reconcile": AblationSpec(
            "blinded_always_reconcile", "always_reconcile", "calibrated", empirical_probe=False,
            reconciliation_enabled=True, reconcile_on_any_disagreement=True, legacy=True, **common
        ),
        "probe_direct": AblationSpec(
            "probe_direct", "probe_direct", "calibrated", empirical_probe=True,
            reconciliation_enabled=False, direct_probe_selection_enabled=True, analysis_role="primary", **common
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
            reconciliation_enabled=True, analysis_role="primary", **common
        ),
        "llm_with_diagnostics": AblationSpec(
            "llm_with_diagnostics", "llm_only", "calibrated", empirical_probe=False,
            challenger_enabled=False, hard_validation_enabled=True,
            reconciliation_enabled=False, abstention_enabled=False,
            analysis_role="secondary",
            planner_evidence_mode="training_only_structural_diagnostics",
            **common,
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
        row.get("model_condition_id", "default"),
        row.get("llm_repetition_id"),
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


def _initial_holdout_metric(row: dict[str, Any]) -> float | None:
    """Return the persisted initial untouched-holdout metric for a trial."""

    direct = row.get("initial_holdout_metric")
    if direct is not None:
        return float(direct)
    metric_name = "macro_f1" if str(row.get("task_type", "classification")) == "classification" else "rmse"
    nested = (row.get("agent_initial_holdout_metrics") or {}).get(metric_name)
    return float(nested) if nested is not None else None


def _initial_planner_quality_difference(
    task_type: str,
    diagnostics_metric: float,
    ordinary_metric: float,
    *,
    epsilon: float,
) -> float:
    """Return a direction-normalized diagnostics-minus-ordinary effect."""

    if task_type == "classification":
        return float(diagnostics_metric - ordinary_metric)
    if task_type == "regression":
        return float(relative_rmse_improvement(ordinary_metric, diagnostics_metric, epsilon=epsilon))
    raise ValueError(f"Unsupported task type: {task_type!r}")


def _paired_initial_planner_comparison(
    rows_by_name: dict[str, list[dict[str, Any]]],
    first: str,
    second: str,
    tolerance: float | dict[str, float] = 1e-12,
    *,
    rmse_epsilon: float = REGRESSION_HOLDOUT_RMSE_EPSILON,
) -> dict[str, Any]:
    """Compare initial planner quality for a secondary information control.

    ``first`` and ``second`` are paired at the same trial unit.  Only jointly
    valid initial plans with persisted initial holdout metrics contribute to
    the quality effect; validity itself is reported separately by
    :func:`_paired_initial_plan_validity`.
    """

    left = {
        _unit_key(row): row
        for row in rows_by_name.get(first, [])
        if row.get("trial_status") != "failed"
    }
    right = {
        _unit_key(row): row
        for row in rows_by_name.get(second, [])
        if row.get("trial_status") != "failed"
    }
    shared = sorted(set(left) & set(right), key=str)
    differences: list[float] = []
    task_types: list[str] = []
    by_dataset: dict[str, list[float]] = {}
    dataset_task_types: dict[str, set[str]] = {}
    for key in shared:
        first_row = left[key]
        second_row = right[key]
        if first_row.get("agent_initial_valid") is not True or second_row.get("agent_initial_valid") is not True:
            continue
        first_metric = _initial_holdout_metric(first_row)
        second_metric = _initial_holdout_metric(second_row)
        if first_metric is None or second_metric is None:
            continue
        task_type = str(first_row.get("task_type", "classification"))
        difference = _initial_planner_quality_difference(
            task_type,
            first_metric,
            second_metric,
            epsilon=rmse_epsilon,
        )
        differences.append(difference)
        task_types.append(task_type)
        dataset = str(key[0])
        by_dataset.setdefault(dataset, []).append(difference)
        dataset_task_types.setdefault(dataset, set()).add(task_type)

    dataset_effects = [
        {
            "benchmark_case": dataset,
            "difference": mean(values),
            "paired_trial_count": len(values),
            "task_type": sorted(dataset_task_types.get(dataset, {"classification"}))[0],
        }
        for dataset, values in sorted(by_dataset.items())
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
    def task_type_summary(task_type: str) -> dict[str, Any]:
        task_effects = [row for row in dataset_effects if row["task_type"] == task_type]
        task_means = [row["difference"] for row in task_effects]
        task_ci = cluster_bootstrap_ci(
            task_effects,
            lambda sample: mean(row["difference"] for row in sample) if sample else None,
            "benchmark_case",
        )
        formula = (
            "diagnostics_initial_macro_f1 - ordinary_initial_macro_f1"
            if task_type == "classification"
            else "(ordinary_initial_rmse - diagnostics_initial_rmse) / "
            "max(abs(ordinary_initial_rmse), rmse_epsilon)"
        )
        return {
            "task_type": task_type,
            "dataset_count": len(task_effects),
            "eligible_dataset_count": len(task_effects),
            "dataset_macro_effect": mean(task_means) if task_means else None,
            "confidence_interval": task_ci,
            "effect_formula": formula,
            "dataset_effects": task_effects,
        }

    task_summaries = {
        task_type: task_type_summary(task_type)
        for task_type in ("classification", "regression")
    }
    descriptive_only = {
        "role": "descriptive_only",
        "estimand": "mixed_task_initial_planner_quality_magnitude",
        "dataset_count": len(dataset_effects),
        "dataset_macro_effect": mean(dataset_means) if dataset_means else None,
        "confidence_interval": cluster_bootstrap_ci(
            dataset_effects,
            lambda sample: mean(row["difference"] for row in sample) if sample else None,
            "benchmark_case",
        ),
        "warning": (
            "Classification macro-F1-point effects and regression relative-RMSE "
            "effects are on different measurement scales; this mixed-task magnitude "
            "is audit-only and is not the headline secondary estimand."
        ),
    }
    return {
        "first": first,
        "second": second,
        "analysis_role": "secondary_information_asymmetry_control",
        "comparison_scope": "within_model_condition",
        "estimand": "initial_planner_holdout_performance",
        "paired_units": len(shared),
        "jointly_evaluable_initial_plan_units": len(differences),
        "n_paired_datasets": len(dataset_effects),
        "paired_initial_planner_dataset_effects": dataset_effects,
        "first_better": dataset_better["first"],
        "second_better": dataset_better["second"],
        "tied": dataset_better["tied"],
        "dataset_macro_first_better": dataset_better["first"],
        "dataset_macro_second_better": dataset_better["second"],
        "dataset_macro_tied": dataset_better["tied"],
        "quality_estimand_condition": (
            "conditional on jointly valid and evaluable initial plans"
        ),
        "classification": task_summaries["classification"],
        "regression": task_summaries["regression"],
        "directional_dataset_outcomes": {
            "diagnostics_better": dataset_better["first"],
            "ordinary_better": dataset_better["second"],
            "tied": dataset_better["tied"],
        },
        "descriptive_only": descriptive_only,
        "initial_planner_quality_difference_sign": (
            "diagnostics_initial_minus_ordinary_initial; positive favors diagnostics"
        ),
        "classification_effect_formula": "diagnostics_initial_macro_f1 - ordinary_initial_macro_f1",
        "regression_effect_formula": (
            "(ordinary_initial_rmse - diagnostics_initial_rmse) "
            "/ max(abs(ordinary_initial_rmse), rmse_epsilon)"
        ),
        "pairing_unit": "dataset/task, perturbation, split seed, trial, model condition, LLM repetition, evaluation variant",
        "aggregation": "mean repetitions within dataset/task, then equal-weighted dataset macro",
        "uncertainty": "dataset_cluster_bootstrap_percentile",
        "win_loss_tie_unit": "dataset/task mean paired initial-planner-quality difference",
        "quality_magnitude_scope": (
            "classification and regression magnitudes are summarized separately; "
            "directional dataset outcomes may remain cross-task-type"
        ),
        "paired_initial_planner_quality_differences": differences,
        "paired_initial_planner_quality_task_types": task_types,
    }


def _paired_initial_plan_validity(
    rows_by_name: dict[str, list[dict[str, Any]]],
    first: str,
    second: str,
) -> dict[str, Any]:
    """Report paired validity outcomes without conditioning on holdout metrics."""

    left = {
        _unit_key(row): row
        for row in rows_by_name.get(first, [])
        if row.get("trial_status") != "failed"
    }
    right = {
        _unit_key(row): row
        for row in rows_by_name.get(second, [])
        if row.get("trial_status") != "failed"
    }
    shared = sorted(set(left) & set(right), key=str)
    counts = {
        "both_initial_valid": 0,
        "diagnostics_valid_ordinary_invalid": 0,
        "diagnostics_invalid_ordinary_valid": 0,
        "both_initial_invalid": 0,
    }
    unclassified = 0
    for key in shared:
        diagnostics_valid = left[key].get("agent_initial_valid")
        ordinary_valid = right[key].get("agent_initial_valid")
        if not isinstance(diagnostics_valid, bool) or not isinstance(ordinary_valid, bool):
            unclassified += 1
        elif diagnostics_valid and ordinary_valid:
            counts["both_initial_valid"] += 1
        elif diagnostics_valid and not ordinary_valid:
            counts["diagnostics_valid_ordinary_invalid"] += 1
        elif not diagnostics_valid and ordinary_valid:
            counts["diagnostics_invalid_ordinary_valid"] += 1
        else:
            counts["both_initial_invalid"] += 1
    classified = sum(counts.values())
    ordinary_valid_count = counts["both_initial_valid"] + counts["diagnostics_invalid_ordinary_valid"]
    diagnostics_valid_count = counts["both_initial_valid"] + counts["diagnostics_valid_ordinary_invalid"]
    ordinary_rate = ordinary_valid_count / classified if classified else None
    diagnostics_rate = diagnostics_valid_count / classified if classified else None
    return {
        "first": first,
        "second": second,
        "analysis_role": "secondary_information_asymmetry_control",
        "comparison_scope": "within_model_condition",
        "estimand": "paired_initial_plan_validity",
        "paired_units": len(shared),
        "classified_paired_units": classified,
        "unclassified_paired_units": unclassified,
        **{f"{name}_count": value for name, value in counts.items()},
        "ordinary_initial_plan_valid_count": ordinary_valid_count,
        "diagnostics_initial_plan_valid_count": diagnostics_valid_count,
        "ordinary_initial_plan_valid_rate": ordinary_rate,
        "diagnostics_initial_plan_valid_rate": diagnostics_rate,
        "validity_rate_difference_diagnostics_minus_ordinary": (
            diagnostics_rate - ordinary_rate
            if diagnostics_rate is not None and ordinary_rate is not None
            else None
        ),
        "pairing_unit": "dataset/task, perturbation, split seed, trial, model condition, LLM repetition, evaluation variant",
        "invalid_plan_handling": (
            "invalid initial plans remain in validity outcomes and are excluded from "
            "quality magnitudes; quality is conditional on jointly valid and evaluable "
            "initial plans"
        ),
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
        f"- Primary ablations: `{payload.get('selected_primary_ablations', [])}`",
        f"- Secondary ablations: `{payload.get('selected_secondary_ablations', [])}`",
        "- Primary and secondary ablations are separate analysis strata; secondary diagnostics are not pooled into the primary claim.",
        "- Paper-primary estimates are reported separately for each declared model condition. Repetitions remain nested within dataset/task.",
        "- Repetitions are aligned by declared repetition slot for balanced analysis; `rep_001` across model conditions or ablations is not a shared-seed stochastic match, because those are separate planner calls.",
        "- Any across-model aggregate below is explicitly descriptive/audit-only and is not a paper-primary estimand.",
        "",
        "## Paper-Primary Results by Model Condition",
        "",
        "| Model condition | Ablation | Datasets | Challenge rate | Intervention rate | Abstention rate | Beneficial | Harmful | Neutral | Holdout delta (dataset macro) | Holdout CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    by_condition = payload.get("analysis_summaries_by_model_condition", {})
    for condition_id, condition_payload in by_condition.items():
        for ablation_name, summary in condition_payload.get("primary", {}).items():
            lines.append(
                f"| {condition_id} | {ablation_name} | {summary.get('dataset_macro_gate_health', {}).get('dataset_count', 0)} | {summary.get('challenge_rate')} | {summary.get('intervention_rate')} | {summary.get('abstention_rate')} | {summary.get('beneficial_intervention_rate')} | {summary.get('harmful_intervention_rate')} | {summary.get('neutral_intervention_rate')} | {summary.get('dataset_macro_paper_holdout_delta_mean')} | {summary.get('dataset_macro_paper_holdout_delta_ci')} |"
            )
    lines.extend(["", "## Paper-Primary Paired Comparisons by Model Condition", ""])
    for condition_id, items in payload.get("paired_comparisons_by_model_condition", {}).items():
        lines.append(f"### `{condition_id}`")
        for item in items:
            lines.append(
                f"- `{item['first']}` vs `{item['second']}`: dataset-macro first better `{item['first_better']}`, second better `{item['second_better']}`, tied `{item['tied']}`, mean first holdout advantage `{item['mean_paired_holdout_delta_difference_first_advantage']}` (CI `{item['paired_holdout_delta_ci']}`)."
            )
    lines.extend([
        "",
        "## Secondary information-asymmetry control",
        "",
        "Within each model condition, `llm_with_diagnostics` is compared directly with `llm_only` "
        "using initial untouched-holdout planner quality, not intervention delta. This tests whether "
        "giving the initial LLM planner the richer pre-specified training-only structural diagnostics "
        "available to the deterministic challenger improves its initial plan. Initial-plan validity "
        "is reported separately, and planner-quality magnitude is conditional on jointly valid and "
        "evaluable initial plans. Classification and regression magnitudes are reported separately; "
        "directional dataset outcomes may remain cross-task-type. This secondary information-asymmetry "
        "analysis does not enter the paper-primary confirmatory claim.",
        "",
    ])
    for condition_id, items in payload.get("secondary_paired_comparisons_by_model_condition", {}).items():
        lines.append(f"### `{condition_id}` — initial planner quality")
        if not items:
            lines.append("- No secondary paired comparison was available.")
            continue
        for item in items:
            classification = item.get("classification", {})
            regression = item.get("regression", {})
            outcomes = item.get("directional_dataset_outcomes", {})
            lines.append(
                f"- `{item['first']}` vs `{item['second']}`: "
                f"classification diagnostics effect `{classification.get('dataset_macro_effect')}` "
                f"(n=`{classification.get('dataset_count')}`, CI `{classification.get('confidence_interval')}`); "
                f"regression diagnostics effect `{regression.get('dataset_macro_effect')}` "
                f"(n=`{regression.get('dataset_count')}`, CI `{regression.get('confidence_interval')}`); "
                f"directional outcomes diagnostics better `{outcomes.get('diagnostics_better', 0)}`, "
                f"ordinary better `{outcomes.get('ordinary_better', 0)}`, tied `{outcomes.get('tied', 0)}`."
            )
    lines.extend(["", "### Secondary initial-plan validity", ""])
    for condition_id, items in payload.get("secondary_initial_planner_validity_by_model_condition", {}).items():
        lines.append(f"- `{condition_id}`:")
        if not items:
            lines.append("  - No paired validity comparison was available.")
            continue
        for item in items:
            lines.append(
                f"  - ordinary valid rate `{item['ordinary_initial_plan_valid_rate']}`, "
                f"diagnostics valid rate `{item['diagnostics_initial_plan_valid_rate']}`, "
                f"both valid `{item['both_initial_valid_count']}`, "
                f"diagnostics valid/ordinary invalid `{item['diagnostics_valid_ordinary_invalid_count']}`, "
                f"diagnostics invalid/ordinary valid `{item['diagnostics_invalid_ordinary_valid_count']}`, "
                f"both invalid `{item['both_initial_invalid_count']}`."
            )
    lines.extend(["", "## Combined Cross-Model Descriptive Audit", "", "The following totals pool model conditions only for audit/descriptive purposes; they are not paper-primary estimates.", "", "| Analysis role | Ablation | Datasets | Valid | Failed/invalid | Challenge rate | Intervention rate | Abstention rate | Beneficial | Harmful | Neutral | Holdout delta (descriptive) | Holdout CI | Planner calls | Reconciler calls | Probe invocations |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"])
    for row in payload["central_table"]:
        api = row["api_usage"]
        probe = payload["summaries"][row["ablation"]].get("probe_invocation_count", 0)
        role = (payload.get("ablation_definitions", {}).get(row["ablation"], {}) or {}).get(
            "analysis_role", "secondary"
        )
        lines.append(
            f"| {role} / descriptive-only | {row['ablation']} | {row['n_datasets']} | {row['valid_trial_count']} | {row['invalid_trial_count']} | {row.get('challenge_rate')} | {row.get('intervention_rate')} | {row.get('abstention_rate')} | {row.get('beneficial_intervention_rate')} | {row.get('harmful_intervention_rate')} | {row.get('neutral_intervention_rate')} | {row.get('paper_holdout_delta_mean')} | {row.get('paper_holdout_delta_ci')} | {api['successful_initial_openai_calls']} | {api['successful_reconciliation_calls']} | {probe} |"
        )
    lines.extend(["", "### Combined Cross-Model Descriptive Paired Comparisons", ""])
    for item in payload.get("descriptive_combined_paired_comparisons", {}).get("comparisons", []):
        lines.append(
            f"- `{item['first']}` vs `{item['second']}` (descriptive-only): first better `{item['first_better']}`, second better `{item['second_better']}`, tied `{item['tied']}`, mean first holdout advantage `{item['mean_paired_holdout_delta_difference_first_advantage']}` (CI `{item['paired_holdout_delta_ci']}`)."
        )
    lines.extend(["", "## Live-Trial Integrity", ""])
    for name, row in payload["central_by_ablation"].items():
        api = row["api_usage"]
        lines.append(
            f"- `{name}`: requested `{api['requested_live_trials']}`, initial failures `{api['failed_initial_openai_calls']}`, reconciliation failures `{api['failed_reconciliation_calls']}`, fallback rows `{api['fallback_rows']}`."
        )
    lines.extend([
        "",
        "Initial proposals are keyed by case, perturbation, split seed, LLM repetition, provider, model condition, model, prompt schema, training-profile digest, target, task, evidence mode, and diagnostics digest. Ordinary paired ablations reuse the same proposal; the diagnostics-enabled planner has a distinct cache namespace.",
        "",
        "Split-seed variation is represented by `split_seed`; stochastic LLM variation is represented independently by `trial`/LLM repetition. Repetitions are aligned by declared repetition slot for balanced analysis, not shared-seed stochastic matches across separate planner calls. Every paired comparison uses the same unit key.",
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
    selected_primary_names = [
        name for name in selected_names if all_specs[name].analysis_role == "primary"
    ]
    selected_secondary_names = [
        name for name in selected_names if all_specs[name].analysis_role == "secondary"
    ]
    resolved_planner_model = planner_model or model
    resolved_reconciler_model = reconciler_model or model
    split_seeds = tuple(int(seed) for seed in split_seeds)
    if not split_seeds:
        raise ValueError("At least one split seed is required.")
    configured_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    root = Path(output_dir).resolve()
    config_path = root / "config.json"
    if resume and confirmatory_config_path is not None:
        if not config_path.is_file():
            raise ValueError(
                "--resume requires an existing confirmatory study config before any trial execution."
            )
        existing_for_identity = json.loads(config_path.read_text(encoding="utf-8"))
        validate_resume_manifest_identity(
            existing_for_identity,
            confirmatory_config_path,
            frozen_manifest_path=root / "frozen_confirmatory_manifest.json",
        )
    confirmatory_metadata: dict[str, Any] | None = None
    frozen_conditions: list[dict[str, Any]] | None = None
    if confirmatory_config_path is not None:
        manifest = load_confirmatory_manifest(confirmatory_config_path)
        manifest_prompt_schema_version = str(
            (manifest.get("prompts") or {}).get("planner_schema_version")
        )
        manifest_experiment_config_version = str(
            manifest.get("experiment_config_version")
        )
        manifest_snapshot = Path(confirmatory_config_path).as_posix()
        frozen_conditions = model_conditions(manifest)
        # In strict mode the frozen manifest is the sole experiment design
        # source.  Runtime flags cannot narrow, add, or replace its matrix.
        manifest_primary_names = list((manifest.get("ablations") or {}).get("primary", []))
        manifest_secondary_names = list((manifest.get("ablations") or {}).get("secondary", []))
        selected_names = manifest_primary_names + manifest_secondary_names
        if not selected_names:
            raise ValueError("Confirmatory manifest declares no ablations.")
        if include_perturbations:
            raise ValueError(
                "Strict confirmatory perturbations must be declared by the frozen manifest; "
                "the current manifest declares none."
            )
        split_seeds = tuple(
            int(value) for value in (manifest.get("splits_and_repetitions", {}) or {}).get("split_seeds", [])
        )
        if not split_seeds:
            raise ValueError("Frozen confirmatory manifest declares no split seeds.")
        # The initial validation uses one declared condition only to validate
        # the manifest identity.  Each condition is executed below with its
        # own models, repetitions, IDs, and settings.
        first_condition = frozen_conditions[0]
        runtime_values = runtime_manifest_values(
            experiment_name=CONFIRMATORY_EXPERIMENT_NAME,
            planner_model=first_condition["planner_model"],
            reconciler_model=first_condition["reconciler_model"],
            split_seeds=split_seeds,
            llm_repetitions=int((manifest.get("splits_and_repetitions", {}) or {}).get("llm_repetitions", first_condition["llm_repetitions"])),
            holdout_fraction=0.2,
            # Manifest validation compares the primary paper stratum. The
            # complete execution matrix below includes secondary controls too.
            selected_ablations=manifest_primary_names,
            deterministic_policy_version=DeterministicPolicy().version,
            deterministic_policy_sha256=config_sha256(deterministic_policy_config()),
            empirical_probe_policy_version=EmpiricalProbePolicy().policy_version,
            empirical_probe_policy_sha256=config_sha256(empirical_probe_config()),
            planner_prompt_schema_version=manifest_prompt_schema_version,
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
            experiment_config_version=manifest_experiment_config_version,
            expected_experiment_code_sha256=experiment_code_sha256(),
            source_git_commit=current_repository_commit(),
            model_conditions=frozen_conditions,
            generation_settings={
                key: value for key, value in (manifest.get("generation_settings", {}) or {}).items()
                if value is not None
            },
            llm_repetition_ids=condition_repetition_ids(manifest, first_condition),
            selected_model_condition_id=first_condition["condition_id"],
        )
        if suite != "external":
            raise ValueError("Confirmatory manifest enforcement requires suite='external'.")
        confirmatory_metadata = validate_confirmatory_manifest(manifest, runtime_values)
        unknown = sorted(set(selected_names) - set(all_specs))
        if unknown:
            raise ValueError(f"Frozen manifest contains unknown ablation preset(s): {', '.join(unknown)}")
        selected_specs = [all_specs[name] for name in selected_names]
    else:
        selected_specs = [all_specs[name] for name in selected_names]
    selected_primary_names = [
        name for name in selected_names if all_specs[name].analysis_role == "primary"
    ]
    selected_secondary_names = [
        name for name in selected_names if all_specs[name].analysis_role == "secondary"
    ]
    root.mkdir(parents=True, exist_ok=True)
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
        "experiment_config_version": (
            manifest_experiment_config_version
            if confirmatory_metadata is not None
            else EXPERIMENT_CONFIG_VERSION
        ),
        "repository_commit": repository_commit(),
        "ablation_schema_version": ABLATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_seeds": list(split_seeds),
        "llm_repetitions": (
            int((manifest.get("splits_and_repetitions", {}) or {}).get("llm_repetitions", repetitions))
            if frozen_conditions is not None else repetitions
        ),
        "llm_repetitions_by_model_condition": {
            str(condition["condition_id"]): int(condition["llm_repetitions"])
            for condition in (frozen_conditions or [{"condition_id": "default", "llm_repetitions": repetitions}])
        },
        "model": model,
        "planner_model": resolved_planner_model,
        "reconciler_model": resolved_reconciler_model,
        "model_conditions": frozen_conditions or [{
            "condition_id": "default",
            "planner_model": resolved_planner_model,
            "reconciler_model": resolved_reconciler_model,
            "llm_repetitions": repetitions,
            "llm_repetition_ids": [f"rep_{index + 1:03d}" for index in range(repetitions)],
            "generation_settings": dict({}),
        }],
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
        "selected_primary_ablations": selected_primary_names,
        "selected_secondary_ablations": selected_secondary_names,
        "analysis_separation": {
            "primary": selected_primary_names,
            "secondary": selected_secondary_names,
            "rule": "primary and secondary ablations are reported in separate analysis strata and are never pooled for the confirmatory claim",
        },
        "model_condition_reporting": {
            "condition_ids": [str(condition["condition_id"]) for condition in (frozen_conditions or [{"condition_id": "default"}])],
            "primary_model_condition_reporting": "separate",
            "cross_model_condition_aggregation": "descriptive_only",
            "per_condition_results": "by_model_condition.<condition_id>.by_ablation",
            "combined_summary_role": "descriptive audit total only; never a pseudo-model estimate or confirmatory model comparison",
        },
        "ablation_definitions": {name: spec.as_dict() for name, spec in all_specs.items()},
        "benchmark_cases": [case.as_dict() for case in selected_cases],
        "benchmark_task_ids": [
            case.openml_task_id for case in selected_cases
            if case.openml_task_id is not None
        ],
        "planner_prompt_schema_version": (
            manifest_prompt_schema_version
            if confirmatory_metadata is not None
            else PROMPT_SCHEMA_VERSION
        ),
        "reconciler_prompt_schema_version": BLINDED_RECONCILIATION_PROMPT_VERSION,
        "deterministic_policy_version": DeterministicPolicy().version,
        "deterministic_policy": {
            "version": DeterministicPolicy().version,
            "parameters": f"captured in {manifest_snapshot}" if confirmatory_metadata is not None else "captured in the selected development configuration",
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
        "confirmatory_config_snapshot": (
            manifest_snapshot
            if confirmatory_metadata is not None
            else "evaluation/configs/paper_confirmatory_v1.json"
        ),
        "confirmatory_mode": confirmatory_metadata is not None,
        "confirmatory_config_status": (
            confirmatory_metadata["status"] if confirmatory_metadata else "not_selected"
        ),
        "experiment_config_path": str(Path(confirmatory_config_path).resolve()) if confirmatory_config_path else None,
        "experiment_config_sha256": confirmatory_metadata.get("experiment_config_sha256") if confirmatory_metadata else None,
        "confirmatory_manifest_sha256": confirmatory_metadata.get("experiment_config_sha256") if confirmatory_metadata else None,
        "benchmark_manifest_sha256": external_benchmark_manifest_sha256() if suite == "external" else None,
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
    if frozen_conditions is not None:
        root_config.update({
            "model": "multiple_frozen_conditions",
            "planner_model": "multiple_frozen_conditions",
            "reconciler_model": "multiple_frozen_conditions",
            "generation_settings_by_model_condition": {
                str(condition["condition_id"]): dict(condition.get("generation_settings", {}) or {})
                for condition in frozen_conditions
            },
            "environment_provenance": environment_provenance(manifest=confirmatory_config_path),
        })
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

    if confirmatory_metadata is not None and not resume:
        shutil.copyfile(
            Path(confirmatory_config_path),
            root / "frozen_confirmatory_manifest.json",
        )

    proposal_cache_path = root / "proposal_cache.jsonl"
    results: dict[str, dict[str, Any]] = {}
    trial_rows: dict[str, list[dict[str, Any]]] = {}
    execution_conditions = frozen_conditions or [{
        "condition_id": "default",
        "planner_model": resolved_planner_model,
        "reconciler_model": resolved_reconciler_model,
        "llm_repetitions": repetitions,
        "llm_repetition_ids": [f"rep_{index + 1:03d}" for index in range(repetitions)],
        "generation_settings": {},
    }]
    for spec in selected_specs:
        combined_rows: list[dict[str, Any]] = []
        condition_results: dict[str, dict[str, Any]] = {}
        for condition in execution_conditions:
            condition_id = str(condition["condition_id"])
            spec_dir = root / spec.name
            if len(execution_conditions) > 1:
                spec_dir = spec_dir / condition_id
            condition_reps = int(condition["llm_repetitions"])
            condition_ids = condition_repetition_ids(
                manifest, condition
            ) if frozen_conditions is not None else list(condition["llm_repetition_ids"])
            result = run_evaluation(
                spec_dir,
                cases=selected_cases,
                repetitions=condition_reps,
                seed=split_seeds[0],
                split_seeds=split_seeds,
                model=condition["planner_model"],
                planner_model=condition["planner_model"],
                reconciler_model=condition["reconciler_model"],
                provider=str(condition.get("provider", "openai")),
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
                confirmatory_selected_ablations=(
                    manifest_primary_names if confirmatory_metadata else None
                ),
                model_condition_id=condition_id,
                llm_repetition_ids=condition_ids,
                generation_settings=dict(condition.get("generation_settings", {}) or {}),
            )
            condition_results[condition_id] = result
            combined_rows.extend(result["trials"])
        results[spec.name] = {
            "summary": summarize_trials(combined_rows, thresholds=thresholds),
            "trials": combined_rows,
            "condition_results": condition_results,
        }
        trial_rows[spec.name] = combined_rows

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
    completeness: dict[str, Any] | None = None
    if confirmatory_metadata is not None:
        expected_units = expand_confirmatory_evaluation_units(
            manifest,
            dataset_ids=[case.name for case in selected_cases],
            split_seeds=split_seeds,
            ablations=selected_names,
        )
        completeness = validate_confirmatory_completeness(
            expected_units,
            [row for rows in trial_rows.values() for row in rows],
        )
    primary_summaries_by_condition = {
        str(condition["condition_id"]): {
            name: results[name]["condition_results"][str(condition["condition_id"])]["summary"]
            for name in selected_primary_names
        }
        for condition in execution_conditions
    }
    secondary_summaries_by_condition = {
        str(condition["condition_id"]): {
            name: results[name]["condition_results"][str(condition["condition_id"])]["summary"]
            for name in selected_secondary_names
        }
        for condition in execution_conditions
    }
    paired_comparisons_by_condition = {}
    secondary_paired_comparisons_by_condition = {}
    secondary_initial_planner_validity_by_condition = {}
    pair_tolerance = {
        "classification": holdout_neutral_tolerance(
            "classification", {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        ),
        "regression": holdout_neutral_tolerance(
            "regression", {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        ),
    }
    for condition in execution_conditions:
        condition_id = str(condition["condition_id"])
        primary_rows_by_name = {
            name: results[name]["condition_results"][condition_id]["trials"]
            for name in selected_primary_names
        }
        paired_comparisons_by_condition[condition_id] = [
            _paired_comparison(primary_rows_by_name, first, second, tolerance=pair_tolerance)
            for first, second in pairs
            if first in primary_rows_by_name and second in primary_rows_by_name
        ]
        secondary_rows_by_name = {
            name: results[name]["condition_results"][condition_id]["trials"]
            for first, second in SECONDARY_PAIRED_COMPARISON_PAIRS
            for name in (first, second)
            if name in results
        }
        secondary_paired_comparisons_by_condition[condition_id] = []
        for first, second in SECONDARY_PAIRED_COMPARISON_PAIRS:
            if first not in secondary_rows_by_name or second not in secondary_rows_by_name:
                continue
            comparison = _paired_initial_planner_comparison(
                secondary_rows_by_name,
                first,
                second,
                tolerance=pair_tolerance,
                rmse_epsilon=configured_thresholds["holdout_rmse_epsilon"],
            )
            secondary_paired_comparisons_by_condition[condition_id].append(comparison)
        secondary_initial_planner_validity_by_condition[condition_id] = []
        for first, second in SECONDARY_PAIRED_COMPARISON_PAIRS:
            if first not in secondary_rows_by_name or second not in secondary_rows_by_name:
                continue
            secondary_initial_planner_validity_by_condition[condition_id].append(
                _paired_initial_plan_validity(secondary_rows_by_name, first, second)
            )
    combined = {
        **root_config,
        "central_table": central,
        "central_by_ablation": {row["ablation"]: row for row in central},
        "summaries": summaries,
        "analysis_summaries_by_model_condition": {
            condition_id: {
                "primary": primary_summaries_by_condition[condition_id],
                "secondary": secondary_summaries_by_condition[condition_id],
                "independent_unit": "dataset/task",
                "repetition_nesting": "repetitions nested within dataset/task × model condition",
                "primary_estimand": "dataset-macro within this model condition",
                "secondary_estimand": "initial planner holdout performance; dataset-macro within this model condition",
                "secondary_validity_estimand": "paired initial-plan validity",
                "independent_dataset_unit_count_by_ablation": {
                    name: summary.get("dataset_macro_gate_health", {}).get("dataset_count", 0)
                    for name, summary in primary_summaries_by_condition[condition_id].items()
                },
            }
            for condition_id in primary_summaries_by_condition
        },
        "paired_comparisons_by_model_condition": paired_comparisons_by_condition,
        "secondary_paired_comparisons_by_model_condition": secondary_paired_comparisons_by_condition,
        "secondary_initial_planner_quality_by_model_condition": secondary_paired_comparisons_by_condition,
        "secondary_initial_planner_validity_by_model_condition": secondary_initial_planner_validity_by_condition,
        # Compatibility aliases for older consumers.  Their role is explicit
        # so they cannot be mistaken for the paper-primary estimand.
        "analysis_summaries": {
            "primary": {name: summaries[name] for name in selected_primary_names},
            "secondary": {name: summaries[name] for name in selected_secondary_names},
            "role": "descriptive_only_compatibility_alias",
            "warning": "Use analysis_summaries_by_model_condition for paper-primary reporting.",
        },
        "paired_comparisons": {
            "comparisons": paired,
            "role": "descriptive_only_compatibility_alias",
            "warning": "Use paired_comparisons_by_model_condition for paper-primary reporting.",
        },
        "descriptive_combined_summary": {
            "primary": {name: summaries[name] for name in selected_primary_names},
            "secondary": {name: summaries[name] for name in selected_secondary_names},
            "role": "descriptive_only",
            "warning": "Across-model totals are audit/descriptive aggregates and are not paper-primary estimands.",
        },
        "descriptive_combined_paired_comparisons": {
            "comparisons": paired,
            "role": "descriptive_only",
            "warning": "Across-model paired comparisons are audit/descriptive aggregates and are not paper-primary estimands.",
        },
        "live_integrity": {
            row["ablation"]: row["api_usage"] for row in central
        },
        "by_model_condition": {
            str(condition["condition_id"]): {
                "by_ablation": {
                    name: results[name]["condition_results"][str(condition["condition_id"])]
                    ["summary"]
                    for name in selected_names
                },
                "primary_ablations": selected_primary_names,
                "secondary_ablations": selected_secondary_names,
                "aggregation_rule": "reported separately by model condition; no cross-condition pooling",
            }
            for condition in execution_conditions
        },
        "confirmatory_matrix": completeness,
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
                completeness is not None and completeness.get("complete", False)
                and
                all(
                    condition_result["summary"].get("confirmatory_valid") is True
                    for result in results.values()
                    for condition_result in result.get("condition_results", {}).values()
                )
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
