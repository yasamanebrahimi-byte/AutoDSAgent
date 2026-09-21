"""Retrospective conventional model-selection baselines.

The functions in this module are deliberately outside ``app.pipeline`` and
``evaluation.runner``.  They consume frozen evaluation artifacts after the
runtime gate has made its decision.  In particular, neither baseline can see a
holdout value while selecting a model family and neither baseline makes an
LLM call.

The default path reuses the persisted training-only CV artifacts.  Optional
frame loading is provided only for the cases where a historical row did not
persist the selected baseline's final holdout metric; every such recomputation
is recorded in the derived row and in the summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.deterministic import profile_dataframe
from app.schemas import PreprocessingContract
from app.validation import (
    FrozenSplit,
    freeze_supervised_split,
    training_profile_frame,
)
from evaluation.empirical_reference import (
    evaluate_empirical_reference,
    evaluate_holdout_plan,
    select_empirical_reference_from_candidates,
    training_only_contract,
)
from evaluation.metrics import (
    DEFAULT_THRESHOLDS,
    classify_holdout_intervention_outcome,
    holdout_neutral_tolerance,
    normalized_performance_delta,
    normalized_regret,
    paper_holdout_delta,
    relative_rmse_improvement,
)
from evaluation.statistics import (
    DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    cluster_bootstrap_ci,
)


BASELINE_ANALYSIS_ROLE = "retrospective_posthoc_baseline"
PROSPECTIVE_ANALYSIS_ROLE = "prospective_generalization"
BASELINE_ANALYSIS_SCHEMA_VERSION = "conventional-baselines-v2"
SUPPORTED_METHODS = ("linear", "regularized_linear", "tree_ensemble", "boosted_tree")
BASELINE_NAMES = ("llm_only", "pairwise_cv_always", "probe_direct", "all_four_cv")
STRICT_NUMERICAL_RTOL = 1e-12
STRICT_NUMERICAL_ATOL = 1e-12


class BaselineAnalysisError(ValueError):
    """Raised when a frozen artifact is insufficient for a scientific derivation."""


class MissingHistoricalFields(BaselineAnalysisError):
    """Raised by strict callers when required persisted provenance is absent."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _family(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    return value if value in SUPPORTED_METHODS else None


def _strict_tie(left: float, right: float) -> bool:
    """Match the empirical probe's strict numerical comparison convention."""

    return bool(np.isclose(left, right, rtol=STRICT_NUMERICAL_RTOL, atol=STRICT_NUMERICAL_ATOL))


def _direction(task_type: str) -> bool:
    if task_type == "classification":
        return True
    if task_type == "regression":
        return False
    raise ValueError(f"Unsupported task type: {task_type!r}")


def _preprocessing_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, PreprocessingContract):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        try:
            return PreprocessingContract.model_validate(value).model_dump(mode="json")
        except Exception:
            return dict(value)
    return None


def _source_for_label(
    label: str,
    proposal: Mapping[str, Any],
    sources: Mapping[str, str | None],
    initial_family: str,
    challenger_family: str,
) -> str:
    source = sources.get(label)
    if source in {"agent", "initial_llm", "llm"}:
        return "initial_llm"
    if source in {"deterministic", "challenger"}:
        return "deterministic_challenger"
    proposal_family = _family(proposal.get("model_family"))
    if proposal_family == initial_family and proposal_family != challenger_family:
        return "initial_llm"
    if proposal_family == challenger_family and proposal_family != initial_family:
        return "deterministic_challenger"
    raise MissingHistoricalFields(
        "Cannot map empirical_probe proposal "
        f"{label!r} to the initial LLM/challenger source; persist proposal_a_source "
        "and proposal_b_source (or distinct proposal model_family values)."
    )


def _validity_value(value: Any) -> bool | None:
    """Normalize persisted validation/actionability values without guessing."""

    if isinstance(value, (bool, np.bool_)):
        return value
    if isinstance(value, Mapping):
        if value.get("valid") is not None:
            return _validity_value(value.get("valid"))
        for key in ("status", "overall_status", "actionability_status"):
            if value.get(key) is not None:
                return _validity_value(value.get(key))
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"passed", "pass", "valid", "actionable", "ready", "ok", "true", "yes"}:
            return True
        if normalized in {
            "failed", "fail", "invalid", "not_actionable", "unavailable",
            "false", "no", "rejected", "not_valid",
        }:
            return False
    return None


def select_pairwise_cv_always(
    task_type: str,
    initial_llm_family: str | None,
    deterministic_challenger_family: str | None,
    probe: Mapping[str, Any] | None,
    *,
    initial_valid: bool = True,
    challenger_valid: bool | None = None,
    initial_preprocessing: Mapping[str, Any] | PreprocessingContract | None = None,
    challenger_preprocessing: Mapping[str, Any] | PreprocessingContract | None = None,
    proposal_a_source: str | None = None,
    proposal_b_source: str | None = None,
) -> dict[str, Any]:
    """Select the raw mean-CV winner of the two compared candidate plans.

    The selective probe's ``winner`` is intentionally ignored.  It can be
    ``"tie"`` after applying evidence-strength thresholds even when the raw
    means are numerically different.  This baseline only uses the persisted
    proposal means and the metric direction; on a strict numerical tie it
    retains the initial LLM incumbent.
    """

    initial_family = _family(initial_llm_family)
    challenger_family = _family(deterministic_challenger_family)
    if initial_llm_family is not None and initial_family is None:
        raise BaselineAnalysisError(f"Unsupported initial LLM family: {initial_llm_family!r}")
    if deterministic_challenger_family is not None and challenger_family is None:
        raise BaselineAnalysisError(
            f"Unsupported deterministic challenger family: {deterministic_challenger_family!r}"
        )
    if initial_family is None:
        raise MissingHistoricalFields("initial LLM model family is missing")

    initial_contract = _preprocessing_dict(initial_preprocessing)
    challenger_contract = _preprocessing_dict(challenger_preprocessing)
    result: dict[str, Any] = {
        "status": "evaluated",
        "task_type": task_type,
        "metric": "macro_f1" if task_type == "classification" else "rmse",
        "higher_is_better": _direction(task_type),
        "initial_llm_family": initial_family,
        "deterministic_challenger_family": challenger_family,
        "selected_baseline_family": initial_family,
        "selected_preprocessing": initial_contract,
        "selection_source": "initial_llm_incumbent",
        "selection_rule": "agreement_or_hard_validation_semantics",
        "proposal_a_mean_score": None,
        "proposal_b_mean_score": None,
        "raw_mean_cv_winner": None,
        "raw_mean_cv_difference": None,
        "evidence_strength_ignored": True,
        "counterfactual_selection_fit_count": 0,
        "probe_fit_count": 0,
        "reconciler_call_count": 0,
    }

    if challenger_family is None:
        raise MissingHistoricalFields("deterministic challenger model family is missing")
    if challenger_valid is None and challenger_family != initial_family:
        raise MissingHistoricalFields("deterministic challenger validity/actionability status is missing")

    # Hard validation remains authoritative.  This is not an ordinary soft
    # comparison and therefore does not require a probe artifact.
    if not initial_valid:
        if challenger_family is not None and challenger_valid:
            result.update(
                selected_baseline_family=challenger_family,
                selected_preprocessing=challenger_contract,
                selection_source="hard_validation_repair",
                selection_rule="existing_universal_hard_validation_repair",
            )
        else:
            result.update(
                status="not_evaluable",
                selection_source="hard_validation_failed",
                selection_rule="existing_universal_hard_validation_semantics",
            )
        return result

    if challenger_valid is None and challenger_family == initial_family:
        result.update(
            selection_source="initial_llm_incumbent",
            selection_rule="model_family_agreement_preserve_initial",
        )
        return result

    if not challenger_valid:
        result.update(
            selection_source="initial_llm_incumbent",
            selection_rule="challenger_hard_invalid_preserve_initial",
        )
        return result
    if initial_family == challenger_family:
        result.update(
            selection_source="initial_llm_incumbent",
            selection_rule="model_family_agreement_preserve_initial",
        )
        return result

    probe_map = _mapping(probe)
    proposal_a = _mapping(probe_map.get("proposal_a"))
    proposal_b = _mapping(probe_map.get("proposal_b"))
    mean_a = _float(proposal_a.get("mean_score"))
    mean_b = _float(proposal_b.get("mean_score"))
    if mean_a is None or mean_b is None:
        raise MissingHistoricalFields(
            "pairwise_cv_always requires raw mean-CV values in "
            "empirical_probe.proposal_a.mean_score and "
            "empirical_probe.proposal_b.mean_score for every actionable family disagreement"
        )
    sources = {
        "A": proposal_a_source,
        "B": proposal_b_source,
    }
    source_a = _source_for_label("A", proposal_a, sources, initial_family, challenger_family)
    source_b = _source_for_label("B", proposal_b, sources, initial_family, challenger_family)
    if source_a == source_b:
        raise MissingHistoricalFields("pairwise probe proposals do not identify one LLM and one challenger plan")

    source_mean = {source_a: mean_a, source_b: mean_b}
    raw_winner = "tie"
    if not _strict_tie(mean_a, mean_b):
        a_wins = (mean_a > mean_b) == _direction(task_type)
        raw_winner = "A" if a_wins else "B"
    winner_source = "initial_llm" if raw_winner == "tie" else (source_a if raw_winner == "A" else source_b)
    selected_family = initial_family if winner_source == "initial_llm" else challenger_family
    proposal_contracts = {
        "initial_llm": initial_contract,
        "deterministic_challenger": challenger_contract,
    }
    if raw_winner == "A":
        selected_contract = _preprocessing_dict(proposal_a.get("preprocessing")) or proposal_contracts[source_a]
    elif raw_winner == "B":
        selected_contract = _preprocessing_dict(proposal_b.get("preprocessing")) or proposal_contracts[source_b]
    else:
        selected_contract = initial_contract
    cv_folds = probe_map.get("cv_folds")
    fit_count = probe_map.get("fit_count")
    if fit_count is None and cv_folds is not None:
        fit_count = 2 * int(cv_folds)
    result.update(
        selected_baseline_family=selected_family,
        selected_preprocessing=selected_contract,
        selection_source=("initial_llm_tie" if raw_winner == "tie" else "raw_pairwise_mean_cv"),
        selection_rule=(
            "strict_raw_mean_cv_tie_preserves_initial_llm"
            if raw_winner == "tie"
            else "higher_mean_cv_macro_f1" if task_type == "classification" else "lower_mean_cv_rmse"
        ),
        proposal_a_mean_score=mean_a,
        proposal_b_mean_score=mean_b,
        raw_mean_cv_winner=raw_winner,
        raw_mean_cv_difference=float(mean_a - mean_b),
        probe_fit_count=int(fit_count or 0),
        counterfactual_selection_fit_count=int(fit_count or 0),
    )
    result["source_mean_scores"] = source_mean
    return result


# Descriptive aliases for downstream scripts and notebooks.
derive_pairwise_cv_always = select_pairwise_cv_always
choose_pairwise_cv_always = select_pairwise_cv_always


def select_all_four_cv(
    task_type: str,
    candidate_cv_metrics: Mapping[str, Mapping[str, Any]],
    *,
    training_profile: Mapping[str, Any] | None = None,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Select the persisted all-family empirical reference.

    Selection delegates to the same order and tie break as
    ``evaluate_empirical_reference``.  The selected preprocessing contract is
    read from that family's persisted validation artifact; only if it was not
    persisted do we derive the canonical training-only contract from the
    training profile.
    """

    if not isinstance(candidate_cv_metrics, Mapping):
        raise MissingHistoricalFields("candidate_cv_metrics is missing")
    selected = select_empirical_reference_from_candidates(candidate_cv_metrics, task_type)
    best = selected.get("best_method")
    candidate = _mapping(candidate_cv_metrics.get(best)) if best else {}
    validation = _mapping(candidate.get("validation"))
    preprocessing = _preprocessing_dict(
        _first(
            validation,
            "approved_preprocessing",
            "preprocessing_contract",
        )
        or candidate.get("preprocessing")
    )
    if preprocessing is None and training_profile is not None and target_column is not None and best:
        preprocessing = training_only_contract(
            dict(training_profile), target_column, task_type, best
        ).model_dump(mode="json")
    if best is not None and preprocessing is None:
        raise MissingHistoricalFields(
            f"all-four winner {best!r} has no persisted canonical preprocessing contract"
        )
    cv_fit_count = sum(
        int(_mapping(candidate_cv_metrics.get(method)).get("cv_folds", 0) or 0)
        for method in SUPPORTED_METHODS
        if _mapping(candidate_cv_metrics.get(method)).get("status") == "evaluated"
    )
    return {
        **selected,
        "selected_baseline_family": best,
        "selected_preprocessing": preprocessing,
        "counterfactual_selection_fit_count": cv_fit_count,
        "distinct_candidate_families_cv_evaluated": sum(
            _mapping(candidate_cv_metrics.get(method)).get("status") == "evaluated"
            for method in SUPPORTED_METHODS
        ),
        "selection_source": "persisted_empirical_reference" if candidate_cv_metrics else "recomputed_empirical_reference",
        "selection_rule": selected.get("selection_rule"),
    }


derive_all_four_cv = select_all_four_cv


def _initial_family(row: Mapping[str, Any]) -> str | None:
    return _family(_first(row, "agent_initial_method") or _mapping(row.get("agent_initial")).get("method"))


def _challenger_family(row: Mapping[str, Any]) -> str | None:
    return _family(
        _first(row, "deterministic_method")
        or _mapping(row.get("deterministic_recommendation")).get("recommended_method")
    )


def _initial_preprocessing(row: Mapping[str, Any]) -> dict[str, Any] | None:
    return _preprocessing_dict(
        _first(row, "agent_initial_preprocessing")
        or _mapping(row.get("agent_initial")).get("preprocessing")
    )


def _challenger_preprocessing(row: Mapping[str, Any]) -> dict[str, Any] | None:
    return _preprocessing_dict(
        _first(row, "deterministic_preprocessing")
        or _mapping(row.get("deterministic_recommendation")).get("preprocessing")
    )


def _initial_valid(row: Mapping[str, Any]) -> bool | None:
    value = _first(row, "agent_initial_valid")
    if value is not None:
        return _validity_value(value)
    nested = _mapping(row.get("agent_initial")).get("valid")
    if nested is not None:
        return _validity_value(nested)
    validation = _mapping(row.get("hard_validation")).get("initial_proposal")
    if validation:
        parsed = _validity_value(validation)
        if parsed is not None:
            return parsed
    validation = _mapping(row.get("agent_initial_validation"))
    parsed = _validity_value(validation)
    if parsed is not None:
        return parsed
    return None


def _challenger_valid(row: Mapping[str, Any]) -> bool | None:
    validation = _mapping(row.get("hard_validation")).get("deterministic_challenger")
    parsed = _validity_value(validation)
    if parsed is not None:
        return parsed
    for key in ("deterministic_valid", "deterministic_challenger_valid"):
        if row.get(key) is not None:
            parsed = _validity_value(row.get(key))
            if parsed is not None:
                return parsed
    parsed = _validity_value(_mapping(row.get("deterministic_recommendation")).get("valid"))
    if parsed is not None:
        return parsed
    # A family name is not evidence that the persisted challenger passed hard
    # validation.  Returning None makes the missing-data path explicit.
    return None if _challenger_family(row) else None


def _candidate_metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = row.get("candidate_cv_metrics")
    if isinstance(candidate, Mapping) and candidate:
        return candidate
    return _mapping(_mapping(row.get("empirical_reference")).get("candidate_metrics"))


def _reference_key(row: Mapping[str, Any], candidate_metrics: Mapping[str, Any]) -> str:
    supplied = row.get("empirical_reference_cache_key")
    if supplied:
        return str(supplied)
    payload = {
        "benchmark_case": row.get("benchmark_case", row.get("dataset_id")),
        "task_type": row.get("task_type"),
        "split_seed": row.get("split_seed"),
        "candidate_metrics": candidate_metrics,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _all_four_compute_cost_key(
    row: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
) -> str:
    """Key one shared conventional four-family CV search.

    The key includes the dataset/split and CV configuration, while explicitly
    excluding the ablation, model condition, and LLM repetition.  Those
    dimensions can change the paired observation without changing this fixed
    training-side search.
    """

    candidate_configuration = {
        method: {
            "status": _mapping(candidate_metrics.get(method)).get("status"),
            "cv_folds": _mapping(candidate_metrics.get(method)).get("cv_folds"),
            "cv_strategy": _mapping(candidate_metrics.get(method)).get("cv_strategy"),
        }
        for method in SUPPORTED_METHODS
    }
    payload = {
        "benchmark_case": row.get("benchmark_case", row.get("dataset_id", row.get("task_id"))),
        "dataset_id": row.get("dataset_id"),
        "dataset_version": row.get("dataset_version"),
        "benchmark_suite_version": row.get("benchmark_suite_version"),
        "task_type": row.get("task_type"),
        "target_column": row.get("agent_initial_target") or row.get("final_target") or row.get("target_column"),
        "split_seed": row.get("split_seed"),
        "split_random_state": row.get("split_random_state", row.get("split_seed")),
        "split_contract": _mapping(row.get("split_contract")),
        "test_size": row.get("test_size"),
        "candidate_methods": list(SUPPORTED_METHODS),
        "candidate_cv_configuration": candidate_configuration,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _holdout_cache_key(
    row: Mapping[str, Any],
    selected_family: str,
    selected_preprocessing: Mapping[str, Any],
) -> str:
    payload = {
        "benchmark_case": row.get("benchmark_case", row.get("dataset_id")),
        "task_type": row.get("task_type"),
        "split_seed": row.get("split_seed"),
        "selected_family": selected_family,
        "selected_preprocessing": selected_preprocessing,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _logical_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identify one experimental realization across ablation directories.

    The ablation arm is intentionally absent.  ``evaluation_variant`` and
    ``order_swap_pair_id`` remain because swapped-order controls are separate
    experimental realizations even when their dataset, split, and repetition
    match the standard row.
    """

    return (
        row.get("model_condition_id", "default"),
        row.get("provider"),
        row.get("planner_model_effective") or row.get("planner_model") or row.get("agent_model"),
        row.get("benchmark_case", row.get("dataset_id", row.get("task_id"))),
        row.get("dataset_id"),
        row.get("task_id"),
        row.get("task_type"),
        row.get("perturbation_id", "clean"),
        row.get("split_seed"),
        row.get("split_random_state", row.get("split_seed")),
        row.get("llm_repetition_id", row.get("trial")),
        row.get("trial"),
        row.get("evaluation_variant", "standard"),
        row.get("order_swap_pair_id"),
    )


def _source_ablation(row: Mapping[str, Any]) -> str | None:
    value = row.get("source_ablation") or row.get("ablation_name")
    if value is None:
        value = row.get("gate_mode")
    return str(value) if value else None


def _has_initial_plan(row: Mapping[str, Any]) -> bool:
    return _initial_family(row) is not None and _initial_valid(row) is not None


def _has_direct_result(row: Mapping[str, Any]) -> bool:
    source = _source_ablation(row)
    return (
        source == "probe_direct"
        or row.get("gate_mode") == "probe_direct"
    ) and _family(row.get("final_method")) is not None


def _row_quality(row: Mapping[str, Any]) -> tuple[int, int, int]:
    probe = _mapping(row.get("empirical_probe"))
    has_probe = int(probe.get("status") == "completed")
    has_reference = int(bool(_candidate_metrics(row)))
    is_probe_arm = int(_source_ablation(row) == "probe_direct" or row.get("gate_mode") == "probe_direct")
    is_full_arm = int(_source_ablation(row) == "full")
    return has_probe, has_reference, is_probe_arm + is_full_arm


def _group_source_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("trial_status") == "failed":
            continue
        groups[_logical_key(row)].append(row)
    selected: list[dict[str, Any]] = []
    for logical_key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        ordered = sorted(
            group_rows,
            key=lambda row: (
                -_row_quality(row)[0],
                -_row_quality(row)[1],
                -_row_quality(row)[2],
                str(row.get("source_file", "")),
                str(row.get("trial_id", "")),
            ),
        )
        probe_rows = [
            row for row in ordered
            if _mapping(row.get("empirical_probe")).get("status") == "completed"
        ]
        llm_rows = [row for row in ordered if _source_ablation(row) == "llm_only" and _has_initial_plan(row)]
        all4_rows = [
            row for row in ordered
            if _source_ablation(row) == "full" and _candidate_metrics(row)
        ] or [row for row in ordered if _candidate_metrics(row)]
        direct_rows = [
            row for row in ordered
            if _has_direct_result(row)
        ]
        probe_direct_rows = [row for row in direct_rows if _source_ablation(row) == "probe_direct"]
        probe_row = probe_direct_rows[0] if probe_direct_rows else (probe_rows[0] if probe_rows else None)
        representative = ordered[0]
        # Flat historical bundles can contain all artifacts in one row.  Use
        # that row when it really contains the required artifact, while still
        # reporting missing dedicated source arms in the provenance map.
        llm_row = llm_rows[0] if llm_rows else (representative if _has_initial_plan(representative) else None)
        all4_row = all4_rows[0] if all4_rows else None
        direct_row = direct_rows[0] if direct_rows else (probe_row if _has_direct_result(probe_row or {}) else None)
        available = sorted({str(_source_ablation(row) or "unknown") for row in ordered})
        role_rows = {
            "llm_only_holdout": llm_row,
            "empirical_probe": probe_row,
            "probe_direct_selective": direct_row,
            "full_all_four_empirical_reference": all4_row,
        }
        missing_roles = [role for role, row in role_rows.items() if row is None]
        selected.append({
            "logical_key": logical_key,
            "representative": representative,
            "llm": llm_row,
            "probe": probe_row,
            "direct": direct_row,
            "all4": all4_row,
            "role_rows": role_rows,
            "available_source_arms": available,
            "missing_source_roles": missing_roles,
            "source_rows": ordered,
        })
    return selected


def _cached_holdout_metric(
    row: Mapping[str, Any],
    baseline: str,
    selected_family: str | None,
    selected_preprocessing: Mapping[str, Any] | None,
) -> tuple[float | None, str | None]:
    metric_name = "macro_f1" if row.get("task_type") == "classification" else "rmse"
    for field in ("baseline_holdout_metrics", "candidate_holdout_metrics", "all_four_holdout_metrics"):
        values = _mapping(row.get(field))
        if selected_family in values:
            value = values[selected_family]
            if isinstance(value, Mapping):
                value = value.get(metric_name)
            parsed = _float(value)
            if parsed is not None:
                return parsed, "reused_existing_candidate_holdout_artifact"
    direct_fields = {
        "all_four_cv": ("all_four_holdout_metric", "all_four_holdout_metrics"),
        "llm_only": ("initial_holdout_metric",),
        "probe_direct": ("final_holdout_metric",),
    }
    for field in direct_fields.get(baseline, ()):
        value = row.get(field)
        if isinstance(value, Mapping):
            value = value.get(metric_name)
        parsed = _float(value)
        if parsed is not None:
            if baseline == "all_four_cv" and selected_family is not None:
                final_family = _family(row.get("final_method"))
                final_preprocessing = _preprocessing_dict(row.get("final_preprocessing"))
                initial_family = _initial_family(row)
                initial_preprocessing = _initial_preprocessing(row)
                if (
                    final_family == selected_family
                    and final_preprocessing == selected_preprocessing
                ) or (
                    initial_family == selected_family
                    and initial_preprocessing == selected_preprocessing
                ):
                    return parsed, "reused_existing_exact_plan_holdout_artifact"
                continue
            return parsed, "reused_existing_holdout_artifact"
    # A pairwise retrospective observation is valid only when the persisted
    # holdout artifact is for the exact selected proposal.  In particular, a
    # probe-direct final value must not be treated as the pairwise value when
    # the two policies selected different families.
    if baseline == "pairwise_cv_always" and selected_family is not None:
        initial_family = _initial_family(row)
        initial_preprocessing = _initial_preprocessing(row)
        final_family = _family(row.get("final_method"))
        final_preprocessing = _preprocessing_dict(row.get("final_preprocessing"))
        if initial_family == selected_family and initial_preprocessing == selected_preprocessing:
            value = row.get("initial_holdout_metric")
            parsed = _float(value)
            if parsed is not None:
                return parsed, "reused_existing_exact_initial_plan_holdout_artifact"
        if final_family == selected_family and final_preprocessing == selected_preprocessing:
            value = row.get("final_holdout_metric")
            parsed = _float(value)
            if parsed is not None:
                return parsed, "reused_existing_exact_final_plan_holdout_artifact"
    if baseline == "llm_only" and _initial_valid(row) is False:
        parsed = _float(row.get("final_holdout_metric"))
        if parsed is not None:
            return parsed, "reused_existing_repair_holdout_artifact"
    if baseline == "all_four_cv" and selected_family is not None:
        final_family = _family(row.get("final_method"))
        final_preprocessing = _preprocessing_dict(row.get("final_preprocessing"))
        initial_family = _initial_family(row)
        initial_preprocessing = _initial_preprocessing(row)
        for value, family, preprocessing in (
            (row.get("final_holdout_metric"), final_family, final_preprocessing),
            (row.get("initial_holdout_metric"), initial_family, initial_preprocessing),
        ):
            parsed = _float(value)
            if parsed is not None and family == selected_family and preprocessing == selected_preprocessing:
                return parsed, "reused_existing_exact_plan_holdout_artifact"
    return None, None


def _split_from_row(frame: pd.DataFrame, row: Mapping[str, Any]) -> FrozenSplit:
    target = str(row.get("agent_initial_target") or row.get("final_target") or row.get("target_column") or "")
    task_type = str(row.get("task_type") or "")
    contract = _mapping(row.get("split_contract"))
    if not target or task_type not in {"classification", "regression"} or not contract:
        raise MissingHistoricalFields(
            "holdout recomputation requires split_contract, task_type, and target identity"
        )
    split_seed = int(row.get("split_seed", contract.get("random_state", 42)))
    test_size = float(row.get("test_size", contract.get("test_size", 0.2)))
    split = freeze_supervised_split(
        frame, target, task_type, test_size=test_size, random_state=split_seed
    )
    observed = split.as_dict()
    identity_fields = (
        "target_column", "task_type", "random_state", "test_size", "strategy",
        "dataset_fingerprint", "valid_rows", "train_rows", "holdout_rows",
        "train_positions_digest", "holdout_positions_digest", "valid_positions_digest",
    )
    mismatches = [
        field for field in identity_fields
        if field in contract and contract.get(field) != observed.get(field)
    ]
    if mismatches:
        raise BaselineAnalysisError(
            "raw frame does not reproduce the persisted frozen split contract; mismatched fields: "
            + ", ".join(mismatches)
        )
    return split


def _load_frame(
    row: Mapping[str, Any],
    loader: Callable[[Mapping[str, Any]], pd.DataFrame] | Mapping[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    if loader is None:
        raise MissingHistoricalFields("raw frame loader was not supplied")
    if isinstance(loader, Mapping):
        key = row.get("benchmark_case", row.get("dataset_id"))
        frame = loader.get(key)
    else:
        frame = loader(row)
    if not isinstance(frame, pd.DataFrame):
        raise BaselineAnalysisError("raw frame loader must return a pandas DataFrame")
    return frame.copy()


def _holdout_with_recompute(
    row: Mapping[str, Any],
    selected_family: str,
    selected_preprocessing: Mapping[str, Any],
    frame_loader: Callable[[Mapping[str, Any]], pd.DataFrame] | Mapping[str, pd.DataFrame],
) -> tuple[float, dict[str, Any]]:
    frame = _load_frame(row, frame_loader)
    split = _split_from_row(frame, row)
    result = evaluate_holdout_plan(
        frame,
        split,
        split.target_column,
        split.task_type,
        selected_family,
        PreprocessingContract.model_validate(selected_preprocessing),
        random_state=split.random_state,
        row_positions=list(range(len(frame))),
    )
    metric_name = "macro_f1" if split.task_type == "classification" else "rmse"
    value = _float(_mapping(result.get("holdout_metrics")).get(metric_name))
    if value is None or result.get("status") != "evaluated":
        raise BaselineAnalysisError(
            f"selected baseline plan holdout evaluation failed for trial {row.get('trial_id')!r}: "
            f"{result.get('error') or result.get('status')}"
        )
    return value, {
        "status": "recomputed_after_selection",
        "fit_count": 1,
        "fit_wall_clock_seconds": result.get("fit_wall_clock_seconds"),
        "split_contract_verified": True,
    }


def _base_provenance(row: Mapping[str, Any], source_run: str | None) -> dict[str, Any]:
    return {
        "source_evaluation_run": row.get("source_run") or source_run,
        "source_file": row.get("source_file"),
        "source_ablation": row.get("source_ablation") or row.get("ablation_name"),
        "source_model_condition": row.get("source_model_condition") or row.get("model_condition_id", "default"),
        "source_experiment_result_directory": row.get("source_experiment_result_directory"),
        "source_trial_id": row.get("trial_id"),
        "source_ablation_name": row.get("source_ablation") or row.get("ablation_name"),
        "source_evaluation_id": row.get("evaluation_id"),
        "dataset_task": row.get("benchmark_case", row.get("dataset_id", row.get("task_id"))),
        "benchmark_case": row.get("benchmark_case", row.get("dataset_id", row.get("task_id"))),
        "dataset_source": row.get("dataset_source"),
        "dataset_id": row.get("dataset_id"),
        "task_id": row.get("task_id"),
        "openml_task_id": row.get("openml_task_id"),
        "dataset_version": row.get("dataset_version"),
        "benchmark_suite_version": row.get("benchmark_suite_version"),
        "task_type": row.get("task_type"),
        "model_condition_id": row.get("model_condition_id", "default"),
        "llm_repetition_id": row.get("llm_repetition_id"),
        "split_seed": row.get("split_seed"),
        "trial": row.get("trial"),
        "provider": row.get("provider"),
        "planner_model": row.get("planner_model"),
        "reconciler_model": row.get("reconciler_model"),
    }


def _derived_row(
    source_row: Mapping[str, Any],
    baseline: str,
    source_run: str | None,
    *,
    all4_source_row: Mapping[str, Any] | None,
    source_roles: Mapping[str, Mapping[str, Any] | None] | None,
    missing_source_roles: Sequence[str] = (),
    reference: Mapping[str, Any] | None,
    reference_recomputed: bool,
    pairwise: Mapping[str, Any] | None,
    holdout_cache: dict[str, tuple[float, str]],
    thresholds: Mapping[str, float] | None,
    frame_loader: Callable[[Mapping[str, Any]], pd.DataFrame] | Mapping[str, pd.DataFrame] | None,
    recompute_holdout: bool,
) -> dict[str, Any]:
    task_type = str(source_row.get("task_type") or "")
    if task_type not in {"classification", "regression"}:
        raise MissingHistoricalFields("task_type is missing or unsupported")
    initial_family = _initial_family(source_row)
    challenger_family = _challenger_family(source_row)
    initial_valid = _initial_valid(source_row)
    challenger_valid = _challenger_valid(source_row)
    if initial_valid is None:
        raise MissingHistoricalFields("initial hard-validity status is missing")
    provenance = _base_provenance(source_row, source_run)
    candidate_metrics = _candidate_metrics(source_row) or (
        _candidate_metrics(all4_source_row or {}) if baseline != "all_four_cv" else {}
    )
    reference_selection = dict(reference or {})
    pair_selection = dict(pairwise or {})
    selected_family: str | None = None
    selected_preprocessing: dict[str, Any] | None = None
    selection_source: str | None = None
    selection_rule: str | None = None
    selection_scores: dict[str, Any] = {}
    reused_cached_scores = False
    recomputed_cv = reference_recomputed
    baseline_required_role = {
        "llm_only": "llm_only_holdout",
        "pairwise_cv_always": "empirical_probe",
        "probe_direct": "probe_direct_selective",
        "all_four_cv": "full_all_four_empirical_reference",
    }[baseline]

    if baseline == "llm_only":
        if initial_valid:
            selected_family = initial_family
            selected_preprocessing = _initial_preprocessing(source_row)
            selection_source = "initial_llm_incumbent"
            selection_rule = "historical_initial_llm_plan"
        else:
            selected_family = _family(source_row.get("final_method")) or (
                challenger_family if challenger_valid else None
            )
            selected_preprocessing = _preprocessing_dict(source_row.get("final_preprocessing")) or (
                _challenger_preprocessing(source_row) if challenger_valid else None
            )
            selection_source = "historical_hard_validation_repair"
            selection_rule = "existing_universal_hard_validation_repair"
    elif baseline == "pairwise_cv_always":
        if pair_selection:
            selected_family = pair_selection.get("selected_baseline_family")
            selected_preprocessing = _preprocessing_dict(pair_selection.get("selected_preprocessing"))
            selection_source = pair_selection.get("selection_source")
            selection_rule = pair_selection.get("selection_rule")
            selection_scores = {
                key: pair_selection.get(key)
                for key in ("proposal_a_mean_score", "proposal_b_mean_score", "raw_mean_cv_winner", "raw_mean_cv_difference")
            }
            reused_cached_scores = bool(_mapping(source_row.get("empirical_probe")))
        else:
            selection_source = "missing_pairwise_probe"
            selection_rule = "raw_pairwise_mean_cv"
    elif baseline == "probe_direct":
        direct = source_row
        if direct.get("ablation_name") != "probe_direct" and direct.get("gate_mode") != "probe_direct":
            direct = {}
        selected_family = _family(_mapping(direct).get("final_method")) if direct else None
        selected_preprocessing = _preprocessing_dict(_mapping(direct).get("final_preprocessing")) if direct else None
        if selected_family is not None:
            selection_source = str(direct.get("final_selection_source") or "probe_direct_historical_decision")
            selection_rule = "existing_selective_pairwise_probe_direct_policy"
        else:
            selection_source = "missing_probe_direct_trial"
            selection_rule = "existing_selective_pairwise_probe_direct_policy"
    elif baseline == "all_four_cv":
        if reference_selection:
            selected_family = _family(reference_selection.get("selected_baseline_family"))
            selected_preprocessing = _preprocessing_dict(reference_selection.get("selected_preprocessing"))
            selection_source = str(reference_selection.get("selection_source") or "persisted_empirical_reference")
            selection_rule = str(reference_selection.get("selection_rule"))
            reused_cached_scores = bool(reference_selection) and not recomputed_cv
        else:
            selection_source = "missing_empirical_reference"
            selection_rule = "all_four_training_only_cv"
    else:
        raise ValueError(f"Unsupported baseline: {baseline!r}")

    # A missing companion arm is a missing historical observation.  Do not
    # turn another arm's final decision into a counterfactual baseline value.
    if (source_roles or {}).get(baseline_required_role) is None:
        selected_family = None
        selected_preprocessing = None
        selection_source = f"missing_source_arm:{baseline_required_role}"
        reused_cached_scores = False

    missing: list[str] = [f"missing companion source role: {role}" for role in missing_source_roles]
    if (source_roles or {}).get(baseline_required_role) is None:
        missing.append(f"required source arm for {baseline}: {baseline_required_role}")
    if baseline in {"pairwise_cv_always", "probe_direct"} and challenger_valid is None:
        missing.append("deterministic challenger validity/actionability status")
    if selected_family is None and baseline != "pairwise_cv_always":
        missing.append("selected_baseline_family")
    if baseline == "pairwise_cv_always" and not pair_selection:
        missing.append("empirical_probe.proposal_a/proposal_b raw mean scores")
    if selected_family is not None and selected_preprocessing is None:
        missing.append("selected candidate preprocessing contract")

    holdout_metric = None
    holdout_source = None
    recomputed_holdout = False
    holdout_fit_count = 0
    fit_wall_clock = None
    if selected_family is not None and selected_preprocessing is not None:
        holdout_key = _holdout_cache_key(source_row, selected_family, selected_preprocessing)
        if baseline == "all_four_cv" and recompute_holdout and holdout_key in holdout_cache:
            holdout_metric, holdout_source = holdout_cache[holdout_key]
            holdout_source = "reused_analysis_holdout_cache"
        else:
            holdout_metric, holdout_source = _cached_holdout_metric(
                source_row, baseline, selected_family, selected_preprocessing
            )
        if holdout_metric is None and recompute_holdout:
            if frame_loader is None:
                missing.append("selected baseline holdout metric or raw frame loader")
            else:
                holdout_metric, recomputed = _holdout_with_recompute(
                    source_row, selected_family, selected_preprocessing, frame_loader
                )
                holdout_source = recomputed["status"]
                fit_wall_clock = recomputed.get("fit_wall_clock_seconds")
                recomputed_holdout = True
                holdout_fit_count = int(recomputed.get("fit_count", 1))
                if baseline == "all_four_cv":
                    holdout_cache[holdout_key] = (holdout_metric, holdout_source)
        elif holdout_metric is None:
            missing.append("selected baseline holdout metric")

    all4_family = _family(reference_selection.get("selected_baseline_family")) if reference_selection else None
    all4_score = _float(reference_selection.get("best_primary_mean")) if reference_selection else None
    selected_score = _float(_mapping(candidate_metrics.get(selected_family)).get("primary_mean")) if selected_family else None
    training_regret = normalized_regret(task_type, all4_score, selected_score) if all4_score is not None else None
    initial_holdout = _float(source_row.get("initial_holdout_metric"))
    if initial_holdout is None:
        initial_holdout = _float(
            _mapping(source_row.get("agent_initial_holdout_metrics")).get(
                "macro_f1" if task_type == "classification" else "rmse"
            )
        )
    paper_delta = paper_holdout_delta(task_type, initial_holdout, holdout_metric)
    intervention = bool(selected_family is not None and initial_family is not None and selected_family != initial_family)
    tolerance = holdout_neutral_tolerance(
        task_type,
        _mapping(source_row.get("holdout_policy")).get("thresholds")
        if _mapping(source_row.get("holdout_policy")).get("thresholds")
        else dict(source_row.get("source_thresholds") or thresholds or DEFAULT_THRESHOLDS),
    )
    outcome = classify_holdout_intervention_outcome(
        paper_delta, tolerance, intervention_occurred=intervention
    )
    if baseline == "all_four_cv":
        outcome = "not_intervened" if not intervention else outcome
    all4_holdout = None
    if baseline != "all_four_cv":
        all4_holdout, _ = _cached_holdout_metric(
            all4_source_row or source_row,
            "all_four_cv",
            all4_family,
            _preprocessing_dict(reference_selection.get("selected_preprocessing")),
        ) if all4_family else (None, None)
    holdout_difference = None
    if holdout_metric is not None and all4_holdout is not None:
        holdout_difference = (
            float(holdout_metric - all4_holdout)
            if task_type == "classification"
            else float(relative_rmse_improvement(all4_holdout, holdout_metric))
        )
    probe = _mapping(source_row.get("empirical_probe"))
    cv_fit_count = 0
    distinct_families = 0
    actionable_disagreement = bool(
        initial_family
        and challenger_family
        and initial_family != challenger_family
        and initial_valid is True
        and challenger_valid is True
    )
    if baseline in {"pairwise_cv_always", "probe_direct"}:
        cv_fit_count = int(
            pair_selection.get("counterfactual_selection_fit_count", 0)
            or probe.get("fit_count", 0)
            or (2 * int(probe.get("cv_folds", 0) or 0) if actionable_disagreement else 0)
        )
        distinct_families = 2 if initial_family and challenger_family and initial_family != challenger_family else int(bool(initial_family))
    elif baseline == "all_four_cv":
        cv_fit_count = int(reference_selection.get("counterfactual_selection_fit_count", 0) or 0)
        distinct_families = int(reference_selection.get("distinct_candidate_families_cv_evaluated", 0) or 0)
    selection_missing = {
        "selected_baseline_family",
        "selected candidate preprocessing contract",
        "empirical_probe.proposal_a/proposal_b raw mean scores",
    }
    selection_artifact_missing = bool(
        selection_missing.intersection(missing)
        or (source_roles or {}).get(baseline_required_role) is None
        or (baseline in {"pairwise_cv_always", "probe_direct"} and challenger_valid is None)
    )
    counterfactual_probe_invocations = int(
        baseline in {"pairwise_cv_always", "probe_direct"} and actionable_disagreement
    )
    counterfactual_planner_calls = int(
        baseline in {"llm_only", "pairwise_cv_always", "probe_direct"}
    )
    counterfactual_reconciler_calls = 0
    counterfactual_final_fits = int(
        selected_family is not None and selected_preprocessing is not None
    )
    actual_cv_fits = int(
        reference_selection.get("counterfactual_selection_fit_count", 0) or 0
    ) if baseline == "all_four_cv" and recomputed_cv else 0
    role_metadata = {
        role: {
            "source_file": row.get("source_file"),
            "source_ablation": row.get("source_ablation") or row.get("ablation_name"),
            "source_trial_id": row.get("trial_id"),
        } if row is not None else None
        for role, row in (source_roles or {}).items()
    }
    row_out = {
        **provenance,
        "analysis_schema_version": BASELINE_ANALYSIS_SCHEMA_VERSION,
        "analysis_role": BASELINE_ANALYSIS_ROLE,
        "baseline_name": baseline,
        "baseline_status": "missing_artifact" if selection_artifact_missing else "evaluated",
        "selection_status": "missing_artifact" if selection_artifact_missing else "evaluated",
        "missing_fields": missing,
        "logical_trial_key": list(_logical_key(source_row)),
        "source_arm_for_baseline": source_row.get("source_ablation") or source_row.get("ablation_name") or source_row.get("gate_mode"),
        "source_arm_roles": role_metadata,
        "available_source_arms": sorted({
            str((role or {}).get("source_ablation") or (role or {}).get("ablation_name") or (role or {}).get("gate_mode"))
            for role in (source_roles or {}).values() if role is not None
        }),
        "missing_source_roles": list(missing_source_roles),
        "initial_llm_family": initial_family,
        "deterministic_challenger_family": challenger_family,
        "initial_llm_hard_valid": initial_valid,
        "challenger_hard_valid": challenger_valid,
        "selected_baseline_family": selected_family,
        "selected_baseline_preprocessing": selected_preprocessing,
        "selection_source": selection_source,
        "selection_rule": selection_rule,
        "selection_metric": "macro_f1" if task_type == "classification" else "rmse",
        "selection_higher_is_better": _direction(task_type),
        "selection_cv_values": selection_scores,
        "all_four_cv_selected_family": all4_family,
        "all_four_cv_best_primary_mean": all4_score,
        "candidate_set_contains_all4_winner": bool(
            all4_family is not None and all4_family in {initial_family, challenger_family}
        ) if all4_family is not None else None,
        "llm_hit_all4_winner": bool(initial_family == all4_family) if all4_family else None,
        "challenger_incremental_hit": bool(
            initial_family is not None and all4_family is not None
            and initial_family != all4_family and challenger_family == all4_family
        ) if all4_family else None,
        "two_candidate_coverage": bool(
            all4_family is not None and all4_family in {initial_family, challenger_family}
        ) if all4_family else None,
        "selected_training_primary_mean": selected_score,
        "training_normalized_regret_relative_to_all_four": training_regret,
        "holdout_metric_name": "macro_f1" if task_type == "classification" else "rmse",
        "holdout_metric": holdout_metric,
        "holdout_metric_source": holdout_source,
        "all_four_holdout_metric_for_comparison": all4_holdout,
        "holdout_performance_difference_baseline_minus_all_four": holdout_difference,
        "initial_holdout_metric": initial_holdout,
        "paper_holdout_delta_initial_to_baseline": paper_delta,
        "holdout_intervention_occurred": intervention,
        "holdout_intervention_outcome": outcome,
        "holdout_neutral_tolerance": tolerance,
        "holdout_neutral_tolerance_units": (
            "absolute macro-F1 points" if task_type == "classification" else "relative RMSE improvement"
        ),
        "counterfactual_selection_fit_count": cv_fit_count,
        "counterfactual_final_fit_count": counterfactual_final_fits,
        "counterfactual_planner_llm_call_count": counterfactual_planner_calls,
        "counterfactual_reconciler_llm_call_count": counterfactual_reconciler_calls,
        "counterfactual_probe_invocation_count": counterfactual_probe_invocations,
        "final_training_fit_count": counterfactual_final_fits,
        "empirical_probe_fit_count": cv_fit_count if baseline in {"pairwise_cv_always", "probe_direct"} else 0,
        "distinct_candidate_families_cv_evaluated": distinct_families,
        "planner_llm_call_count": counterfactual_planner_calls,
        "reconciler_llm_call_count": counterfactual_reconciler_calls,
        "probe_invocation_count": counterfactual_probe_invocations,
        "analysis_reused_cached_scores": bool(reused_cached_scores),
        "analysis_recomputed_cv": recomputed_cv,
        "analysis_recomputed_holdout": recomputed_holdout,
        "actual_analysis_fit_count": actual_cv_fits + holdout_fit_count,
        "actual_analysis_llm_call_count": 0,
        "fit_wall_clock_seconds": fit_wall_clock,
        "reference_cache_key": _reference_key(source_row, candidate_metrics) if candidate_metrics else None,
        "all_four_compute_cost_key": _all_four_compute_cost_key(
            all4_source_row or source_row,
            _candidate_metrics(all4_source_row or source_row) or candidate_metrics,
        ) if baseline == "all_four_cv" and (_candidate_metrics(all4_source_row or source_row) or candidate_metrics) else None,
        "reference_reused_across_repetitions": True,
        "comparison_key": json.dumps(_logical_key(source_row), default=str),
        "reconciler_invoked": False,
    }
    return row_out


def _reference_for_row(
    row: Mapping[str, Any],
    *,
    frame_loader: Callable[[Mapping[str, Any]], pd.DataFrame] | Mapping[str, pd.DataFrame] | None,
    recompute_missing: bool,
    reference_cache: dict[str, dict[str, Any]],
    frame_cache: dict[Any, pd.DataFrame],
) -> tuple[dict[str, Any] | None, bool]:
    candidate = _candidate_metrics(row)
    if candidate:
        selected = select_all_four_cv(
            str(row.get("task_type")),
            candidate,
            training_profile=_mapping(_mapping(row.get("agent_initial_input")).get("training_profile")) or None,
            target_column=str(row.get("agent_initial_target") or row.get("final_target") or "") or None,
        )
        selected["candidate_metrics"] = candidate
        return selected, True
    if not recompute_missing or frame_loader is None:
        return None, False
    key = _reference_key(row, {})
    if key in reference_cache:
        return reference_cache[key], True
    frame = frame_cache.get(row.get("benchmark_case"))
    if frame is None:
        frame = _load_frame(row, frame_loader)
        frame_cache[row.get("benchmark_case")] = frame
    split = _split_from_row(frame, row)
    training = training_profile_frame(
        frame,
        split.target_column,
        split.task_type,
        test_size=split.test_size,
        random_state=split.random_state,
        split=split,
    )
    reference = evaluate_empirical_reference(
        training,
        split.target_column,
        split.task_type,
        profile_dataframe(training),
        random_state=split.random_state,
    )
    selected = select_all_four_cv(
        split.task_type,
        reference.get("candidate_metrics", {}),
        training_profile=profile_dataframe(training),
        target_column=split.target_column,
    )
    selected["candidate_metrics"] = reference.get("candidate_metrics", {})
    selected["selection_source"] = "newly_recomputed_empirical_reference"
    reference_cache[key] = selected
    return selected, False


def derive_baseline_trials(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_run: str | None = None,
    frame_loader: Callable[[Mapping[str, Any]], pd.DataFrame] | Mapping[str, pd.DataFrame] | None = None,
    recompute_missing: bool = False,
    baselines: Sequence[str] = BASELINE_NAMES,
    strict: bool = False,
    thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Derive baseline rows from frozen trial artifacts without LLM calls."""

    unknown = sorted(set(baselines) - set(BASELINE_NAMES))
    if unknown:
        raise ValueError(f"Unknown baseline(s): {', '.join(unknown)}")
    reference_cache: dict[str, dict[str, Any]] = {}
    holdout_cache: dict[str, tuple[float, str]] = {}
    frame_cache: dict[Any, pd.DataFrame] = {}
    derived: list[dict[str, Any]] = []
    errors: list[str] = []
    for group in _group_source_rows(rows):
        all4_source = group["all4"]
        reference_error: str | None = None
        reference_recomputed = False
        reference: dict[str, Any] | None = None
        if all4_source is not None:
            try:
                reference, reference_reused = _reference_for_row(
                    all4_source,
                    frame_loader=frame_loader,
                    recompute_missing=recompute_missing,
                    reference_cache=reference_cache,
                    frame_cache=frame_cache,
                )
                reference_recomputed = reference is not None and not reference_reused
            except BaselineAnalysisError as exc:
                reference_error = str(exc)
                errors.append(f"trial {all4_source.get('trial_id')!r}, all_four_cv reference: {exc}")
        else:
            reference_error = "full/all-four empirical-reference source row is missing"
        probe_source = group["probe"]
        pairwise = None
        initial_family = _initial_family(probe_source or {})
        challenger_family = _challenger_family(probe_source or {})
        initial_valid = _initial_valid(probe_source or {})
        challenger_valid = _challenger_valid(probe_source or {})
        if initial_family and challenger_family and initial_valid is not None and challenger_valid is not None:
            try:
                pairwise = select_pairwise_cv_always(
                    str((probe_source or {}).get("task_type")),
                    initial_family,
                    challenger_family,
                    _mapping(probe_source.get("empirical_probe")) or None,
                    initial_valid=initial_valid,
                    challenger_valid=challenger_valid,
                    initial_preprocessing=_initial_preprocessing(probe_source),
                    challenger_preprocessing=_challenger_preprocessing(probe_source),
                    proposal_a_source=probe_source.get("proposal_a_source"),
                    proposal_b_source=probe_source.get("proposal_b_source"),
                )
            except BaselineAnalysisError as exc:
                errors.append(f"trial {(probe_source or {}).get('trial_id')!r}: {exc}")
        for baseline in baselines:
            if baseline == "llm_only":
                source = group["llm"]
            elif baseline == "probe_direct":
                source = group["direct"]
            elif baseline == "pairwise_cv_always":
                source = group["probe"]
            else:
                source = group["all4"]
            if source is None:
                source = group["representative"]
            try:
                derived_row = _derived_row(
                    source,
                    baseline,
                    source_run,
                    all4_source_row=all4_source,
                    source_roles=group["role_rows"],
                    missing_source_roles=group["missing_source_roles"],
                    reference=reference,
                    reference_recomputed=reference_recomputed,
                    pairwise=pairwise,
                    holdout_cache=holdout_cache,
                    thresholds=thresholds,
                    frame_loader=frame_loader,
                    recompute_holdout=recompute_missing,
                )
                if reference_error and baseline == "all_four_cv":
                    derived_row.setdefault("missing_fields", []).append(reference_error)
                derived.append(derived_row)
                if strict and derived_row.get("missing_fields"):
                    errors.append(
                        f"trial {source.get('trial_id')!r}, baseline {baseline!r}: "
                        + "; ".join(map(str, derived_row.get("missing_fields", [])))
                    )
            except BaselineAnalysisError as exc:
                errors.append(f"trial {source.get('trial_id')!r}, baseline {baseline!r}: {exc}")
                derived.append({
                    **_base_provenance(source, source_run),
                    "analysis_schema_version": BASELINE_ANALYSIS_SCHEMA_VERSION,
                    "analysis_role": BASELINE_ANALYSIS_ROLE,
                    "baseline_name": baseline,
                    "baseline_status": "missing_artifact",
                    "missing_fields": [str(exc), *[
                        f"missing companion source role: {role}"
                        for role in group["missing_source_roles"]
                    ]],
                    "initial_llm_family": _initial_family(source),
                    "deterministic_challenger_family": _challenger_family(source),
                    "selected_baseline_family": None,
                    "holdout_metric": None,
                    "logical_trial_key": list(_logical_key(source)),
                    "source_arm_for_baseline": _source_ablation(source),
                    "source_arm_roles": {
                        role: {
                            "source_file": role_row.get("source_file"),
                            "source_ablation": _source_ablation(role_row),
                            "source_trial_id": role_row.get("trial_id"),
                        } if role_row is not None else None
                        for role, role_row in group["role_rows"].items()
                    },
                    "counterfactual_selection_fit_count": 0,
                    "counterfactual_final_fit_count": 0,
                    "counterfactual_planner_llm_call_count": 0,
                    "counterfactual_reconciler_llm_call_count": 0,
                    "counterfactual_probe_invocation_count": 0,
                    "actual_analysis_fit_count": 0,
                    "actual_analysis_llm_call_count": 0,
                    "analysis_reused_cached_scores": False,
                    "reconciler_invoked": False,
                })
    if strict and errors:
        raise MissingHistoricalFields("Historical artifacts are insufficient:\n- " + "\n- ".join(errors))
    return derived


def _dataset_macro_value(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row.get(field))
        dataset = row.get("benchmark_case", row.get("dataset_task"))
        if value is not None and dataset is not None:
            by_dataset[str(dataset)].append(value)
    means = [statistics.mean(values) for values in by_dataset.values() if values]
    return float(statistics.mean(means)) if means else None


def _dataset_macro_ci(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    usable = [
        {**row, "benchmark_case": str(row.get("benchmark_case", row.get("dataset_task")))}
        for row in rows if _float(row.get(field)) is not None and row.get("benchmark_case", row.get("dataset_task")) is not None
    ]
    return cluster_bootstrap_ci(
        usable,
        lambda sample: _dataset_macro_value(sample, field),
        "benchmark_case",
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )


def _method_summary(
    rows: Sequence[Mapping[str, Any]],
    baseline: str,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    subset = [row for row in rows if row.get("baseline_name") == baseline]
    evaluated = [row for row in subset if row.get("baseline_status") == "evaluated"]
    holdout = [row for row in evaluated if _float(row.get("holdout_metric")) is not None]
    interventions = [row for row in evaluated if row.get("holdout_intervention_occurred")]
    comparable_interventions = [row for row in interventions if row.get("holdout_intervention_outcome") in {"beneficial", "harmful", "neutral"}]
    counts = defaultdict(int)
    for row in comparable_interventions:
        counts[str(row.get("holdout_intervention_outcome"))] += 1
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted({str(row.get("model_condition_id", "default")) for row in subset}):
        condition_rows = [row for row in subset if str(row.get("model_condition_id", "default")) == condition]
        by_condition[condition] = {
            "trial_count": len(condition_rows),
            "evaluable_holdout_count": sum(_float(row.get("holdout_metric")) is not None for row in condition_rows),
            "holdout_metric_dataset_macro_mean": _dataset_macro_value(condition_rows, "holdout_metric"),
            "training_normalized_regret_dataset_macro_mean": _dataset_macro_value(condition_rows, "training_normalized_regret_relative_to_all_four"),
        }
    def cost_total(field: str) -> int:
        if baseline != "all_four_cv" or field != "counterfactual_selection_fit_count":
            return sum(int(row.get(field, 0) or 0) for row in evaluated)
        # The fixed four-family search is shared by repetitions and model
        # conditions when dataset, split, and CV configuration are identical.
        unique_costs: dict[str, int] = {}
        for row in evaluated:
            key = str(row.get("all_four_compute_cost_key") or row.get("comparison_key"))
            unique_costs[key] = max(unique_costs.get(key, 0), int(row.get(field, 0) or 0))
        return sum(unique_costs.values())

    return {
        "baseline": baseline,
        "trial_count": len(subset),
        "evaluated_trial_count": len(evaluated),
        "missing_artifact_trial_count": len(subset) - len(evaluated),
        "partial_artifact_trial_count": sum(bool(row.get("missing_fields")) for row in evaluated),
        "evaluable_holdout_count": len(holdout),
        "missing_holdout_count": len(evaluated) - len(holdout),
        "holdout_metric_name": next((row.get("holdout_metric_name") for row in evaluated if row.get("holdout_metric_name")), None),
        "holdout_metric_dataset_macro_mean": _dataset_macro_value(evaluated, "holdout_metric"),
        "holdout_metric_dataset_macro_ci": _dataset_macro_ci(evaluated, "holdout_metric", n_bootstrap=n_bootstrap, random_state=bootstrap_seed),
        "training_normalized_regret_dataset_macro_mean": _dataset_macro_value(evaluated, "training_normalized_regret_relative_to_all_four"),
        "training_normalized_regret_dataset_macro_ci": _dataset_macro_ci(evaluated, "training_normalized_regret_relative_to_all_four", n_bootstrap=n_bootstrap, random_state=bootstrap_seed),
        "intervention_count": len(interventions),
        "intervention_rate": float(len(interventions) / len(evaluated)) if evaluated else None,
        "beneficial_intervention_count": counts["beneficial"],
        "harmful_intervention_count": counts["harmful"],
        "neutral_intervention_count": counts["neutral"],
        "beneficial_intervention_rate": float(counts["beneficial"] / len(comparable_interventions)) if comparable_interventions else None,
        "harmful_intervention_rate": float(counts["harmful"] / len(comparable_interventions)) if comparable_interventions else None,
        "neutral_intervention_rate": float(counts["neutral"] / len(comparable_interventions)) if comparable_interventions else None,
        "counterfactual_selection_fit_count_total": cost_total("counterfactual_selection_fit_count"),
        "counterfactual_selection_fit_count_per_derived_trial": (
            float(statistics.mean([int(row.get("counterfactual_selection_fit_count", 0) or 0) for row in evaluated])) if evaluated else None
        ),
        "final_training_fit_count_total": sum(int(row.get("final_training_fit_count", 0) or 0) for row in evaluated),
        "empirical_probe_fit_count_total": sum(int(row.get("empirical_probe_fit_count", 0) or 0) for row in evaluated),
        "distinct_candidate_families_cv_evaluated_mean": (
            float(statistics.mean([int(row.get("distinct_candidate_families_cv_evaluated", 0) or 0) for row in evaluated])) if evaluated else None
        ),
        "counterfactual_final_fit_count_total": sum(int(row.get("counterfactual_final_fit_count", 0) or 0) for row in evaluated),
        "counterfactual_planner_llm_call_count": sum(int(row.get("counterfactual_planner_llm_call_count", 0) or 0) for row in evaluated),
        "counterfactual_reconciler_llm_call_count": sum(int(row.get("counterfactual_reconciler_llm_call_count", 0) or 0) for row in evaluated),
        "counterfactual_probe_invocation_count": sum(int(row.get("counterfactual_probe_invocation_count", 0) or 0) for row in evaluated),
        "planner_llm_call_count": sum(int(row.get("counterfactual_planner_llm_call_count", 0) or 0) for row in evaluated),
        "reconciler_llm_call_count": sum(int(row.get("counterfactual_reconciler_llm_call_count", 0) or 0) for row in evaluated),
        "probe_invocation_count": sum(int(row.get("counterfactual_probe_invocation_count", 0) or 0) for row in evaluated),
        "analysis_reused_cached_scores_count": sum(bool(row.get("analysis_reused_cached_scores")) for row in evaluated),
        "analysis_recomputed_holdout_count": sum(bool(row.get("analysis_recomputed_holdout")) for row in evaluated),
        "actual_analysis_fit_count": sum(int(row.get("actual_analysis_fit_count", 0) or 0) for row in evaluated),
        "by_model_condition": by_condition,
    }


def _coverage_slice(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row.get("baseline_name") == "all_four_cv"
        and row.get("initial_llm_hard_valid") is True
        and row.get("challenger_hard_valid") is True
        and row.get("all_four_cv_selected_family") is not None
    ]
    datasets = sorted({str(row.get("benchmark_case")) for row in eligible})
    def rate(field: str, population: Sequence[Mapping[str, Any]]) -> float | None:
        values = [row.get(field) for row in population if row.get(field) is not None]
        return float(sum(bool(value) for value in values) / len(values)) if values else None
    dataset_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        subset = [row for row in eligible if str(row.get("benchmark_case")) == dataset]
        dataset_rows.append({
            "benchmark_case": dataset,
            "llm_hit_all4_winner": float(statistics.mean(bool(row.get("llm_hit_all4_winner")) for row in subset)),
            "challenger_incremental_hit": float(statistics.mean(bool(row.get("challenger_incremental_hit")) for row in subset)),
            "two_candidate_coverage": float(statistics.mean(bool(row.get("two_candidate_coverage")) for row in subset)),
        })
    return {
        "eligible_hard_valid_evaluable_trial_count": len(eligible),
        "missing_or_not_applicable_count": len(rows) - len(eligible),
        "dataset_count": len(datasets),
        "trial_weighted_diagnostic_rates": {
            field: rate(field, eligible)
            for field in ("llm_hit_all4_winner", "challenger_incremental_hit", "two_candidate_coverage")
        },
        "dataset_macro_rates": {
            field: (
                float(statistics.mean(row[field] for row in dataset_rows)) if dataset_rows else None
            )
            for field in ("llm_hit_all4_winner", "challenger_incremental_hit", "two_candidate_coverage")
        },
        "by_dataset": dataset_rows,
    }


def _comparison_summary(rows: Sequence[Mapping[str, Any]], first: str, second: str) -> dict[str, Any]:
    left = {row.get("comparison_key"): row for row in rows if row.get("baseline_name") == first}
    right = {row.get("comparison_key"): row for row in rows if row.get("baseline_name") == second}
    paired: list[dict[str, Any]] = []
    for key in sorted(set(left) & set(right), key=str):
        a, b = left[key], right[key]
        task_type = str(a.get("task_type"))
        first_metric = _float(a.get("holdout_metric"))
        second_metric = _float(b.get("holdout_metric"))
        if first_metric is None or second_metric is None:
            continue
        # Positive values always favor the named first baseline.
        advantage = normalized_performance_delta(task_type, second_metric, first_metric)
        paired.append({
            "comparison_key": key,
            "benchmark_case": a.get("benchmark_case"),
            "task_type": task_type,
            "holdout_performance_difference_first_minus_second": advantage,
        })
    return {
        "first": first,
        "second": second,
        "comparison_scope": "one_model_condition_and_task_type",
        "paired_evaluable_count": len(paired),
        "independent_dataset_count": len({str(item.get("benchmark_case")) for item in paired}),
        "repetitions_and_splits_are_nested": True,
        "holdout_performance_difference_first_minus_second_dataset_macro_mean": _dataset_macro_value(
            paired, "holdout_performance_difference_first_minus_second"
        ),
        "paired_rows": paired,
    }


def summarize_baseline_trials(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize derived rows with dataset/task as the statistical unit."""

    rows = list(rows)
    by_task_type: dict[str, dict[str, Any]] = {}
    for task_type in ("classification", "regression"):
        task_rows = [row for row in rows if row.get("task_type") == task_type]
        by_task_type[task_type] = {
            "trial_count": len(task_rows),
            "baselines": {
                baseline: _method_summary(
                    task_rows,
                    baseline,
                    n_bootstrap=bootstrap_replicates,
                    bootstrap_seed=bootstrap_seed,
                )
                for baseline in BASELINE_NAMES
            },
            "candidate_set_coverage": _coverage_slice(task_rows),
        }
    by_condition: dict[str, Any] = {}
    for condition in sorted({str(row.get("model_condition_id", "default")) for row in rows}):
        condition_rows = [row for row in rows if str(row.get("model_condition_id", "default")) == condition]
        by_condition[condition] = {
            "trial_count": len(condition_rows),
            "by_task_type": {
                task_type: {
                    "candidate_set_coverage": _coverage_slice(
                        [row for row in condition_rows if row.get("task_type") == task_type]
                    ),
                    "baselines": {
                        baseline: _method_summary(
                            [row for row in condition_rows if row.get("task_type") == task_type],
                            baseline,
                            n_bootstrap=bootstrap_replicates,
                            bootstrap_seed=bootstrap_seed,
                        )
                        for baseline in BASELINE_NAMES
                    },
                }
                for task_type in ("classification", "regression")
            },
        }
    baseline_summaries = {
        baseline: _method_summary(
            rows,
            baseline,
            n_bootstrap=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        for baseline in BASELINE_NAMES
    }
    comparisons_by_condition: dict[str, dict[str, Any]] = {}
    comparisons_by_condition_and_task: dict[str, dict[str, Any]] = {}
    condition_values = sorted({str(row.get("model_condition_id", "default")) for row in rows})
    comparison_pairs = (
        ("llm_only", "all_four_cv"),
        ("pairwise_cv_always", "all_four_cv"),
        ("probe_direct", "all_four_cv"),
        ("pairwise_cv_always", "probe_direct"),
    )
    for condition in condition_values:
        condition_rows = [
            row for row in rows
            if str(row.get("model_condition_id", "default")) == condition
        ]
        comparisons_by_condition[condition] = {
            f"{first}_vs_{second}": _comparison_summary(condition_rows, first, second)
            for first, second in comparison_pairs
        }
        comparisons_by_condition_and_task[condition] = {
            task_type: {
                f"{first}_vs_{second}": _comparison_summary(
                    [row for row in condition_rows if row.get("task_type") == task_type],
                    first,
                    second,
                )
                for first, second in comparison_pairs
            }
            for task_type in ("classification", "regression")
        }
    descriptive_combined = {
        "scope": "descriptive_audit_only_pooled_model_conditions",
        "warning": "Model conditions are pooled here for audit only; this is not a primary estimate.",
        "comparisons": {
            f"{first}_vs_{second}": _comparison_summary(rows, first, second)
            for first, second in comparison_pairs
        },
    }
    descriptive_combined["comparisons"]["probe_direct_vs_pairwise_cv_always"] = _comparison_summary(
        rows, "probe_direct", "pairwise_cv_always"
    )
    companion_diagnostics: dict[str, dict[str, Any]] = {}
    for row in rows:
        missing_roles = row.get("missing_source_roles") or []
        if not missing_roles:
            continue
        key = str(row.get("comparison_key") or row.get("logical_trial_key"))
        companion_diagnostics.setdefault(
            key,
            {
                "logical_trial_key": row.get("logical_trial_key"),
                "model_condition_id": row.get("model_condition_id"),
                "benchmark_case": row.get("benchmark_case"),
                "split_seed": row.get("split_seed"),
                "llm_repetition_id": row.get("llm_repetition_id"),
                "missing_source_roles": list(missing_roles),
                "available_source_arms": row.get("available_source_arms", []),
            },
        )
    missing_data_diagnostics = [
        {
            "logical_trial_key": row.get("logical_trial_key"),
            "baseline_name": row.get("baseline_name"),
            "source_file": row.get("source_file"),
            "missing_fields": list(row.get("missing_fields") or []),
        }
        for row in rows
        if row.get("missing_fields")
    ]
    return {
        "analysis_schema_version": BASELINE_ANALYSIS_SCHEMA_VERSION,
        "analysis_role": BASELINE_ANALYSIS_ROLE,
        "independent_statistical_unit": "dataset/task",
        "repetition_nesting": "LLM repetitions and split seeds are nested within dataset/task; they are not IID datasets",
        "baseline_definitions": {
            "pairwise_cv_always": "On a valid model-family disagreement, choose the raw mean-CV winner of the two persisted candidate plans; strict numerical ties retain the initial LLM plan.",
            "all_four_cv": "Choose the best eligible family from the fixed portfolio linear, regularized_linear, tree_ensemble, boosted_tree using training-only CV and the existing deterministic tie break.",
            "probe_direct": "Existing selective pairwise probe-direct decision, including its abstention threshold.",
            "llm_only": "Initial LLM plan after universal hard validation, with no soft override.",
        },
        "baseline_names": list(BASELINE_NAMES),
        "derived_trial_count": len(rows),
        "missing_artifact_count": sum(
            row.get("baseline_status") != "evaluated" or bool(row.get("missing_fields"))
            for row in rows
        ),
        "selection_missing_artifact_count": sum(
            row.get("baseline_status") != "evaluated" for row in rows
        ),
        "partial_artifact_count": sum(
            row.get("baseline_status") == "evaluated" and bool(row.get("missing_fields"))
            for row in rows
        ),
        "incomplete_logical_trial_count": len(companion_diagnostics),
        "missing_companion_diagnostics": list(companion_diagnostics.values()),
        "missing_data_diagnostics": missing_data_diagnostics,
        "baseline_summaries": baseline_summaries,
        "candidate_set_coverage": _coverage_slice(rows),
        "by_task_type": by_task_type,
        "by_model_condition": by_condition,
        "descriptive_combined_only": {
            "warning": "Combined model-condition values are descriptive/audit output and are not a pooled primary estimate.",
            "baseline_summaries": baseline_summaries,
            "candidate_set_coverage": _coverage_slice(rows),
        },
        "primary_comparison_scope": "model_condition_and_task_type",
        "comparisons_by_model_condition": comparisons_by_condition,
        "comparisons_by_model_condition_and_task_type": comparisons_by_condition_and_task,
        "descriptive_combined_condition_comparisons": descriptive_combined,
        # Compatibility alias retained for consumers of v1.  Its metadata
        # makes clear that it is descriptive/audit-only.
        "comparisons": {
            "scope": descriptive_combined["scope"],
            "warning": descriptive_combined["warning"],
            **descriptive_combined["comparisons"],
        },
        "bootstrap": {
            "method": "dataset_cluster_bootstrap_percentile",
            "replicates": bootstrap_replicates,
            "confidence_level": DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL,
            "seed": bootstrap_seed,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BaselineAnalysisError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise BaselineAnalysisError(f"Expected an object in {path} line {line_number}")
        rows.append(value)
    return rows


def discover_trial_files(source_dir: str | Path) -> list[Path]:
    """Find result bundles without assuming one historical directory layout."""

    source = Path(source_dir).resolve()
    files = sorted(source.rglob("trials.jsonl"))
    if not files:
        raise MissingHistoricalFields(
            f"No trials.jsonl found under {source}. Historical summary/config files alone do not contain enough trial-level provenance for retrospective baselines."
        )
    return files


def _config_for_trial_file(trials_path: Path) -> dict[str, Any]:
    candidates = (trials_path.parent / "config.json", trials_path.parent.parent / "config.json")
    for path in candidates:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
    return {}


def _source_run_label(trials_path: Path, config: Mapping[str, Any]) -> str:
    return str(config.get("evaluation_id") or trials_path.parent)


def _path_ablation_and_condition(source: Path, trials_path: Path) -> tuple[str | None, str | None]:
    """Infer arm/condition only as a provenance fallback for sparse rows."""

    try:
        parts = trials_path.parent.relative_to(source).parts
    except ValueError:
        parts = trials_path.parent.parts
    if not parts:
        return None, None
    if len(parts) == 1:
        return None, None
    return str(parts[0]), str(parts[1]) if len(parts) > 1 else None


def _annotate_source_row(
    row: Mapping[str, Any],
    *,
    source: Path,
    trials_path: Path,
    config: Mapping[str, Any],
    source_run: str,
) -> dict[str, Any]:
    path_ablation, path_condition = _path_ablation_and_condition(source, trials_path)
    ablation = row.get("ablation_name") or config.get("ablation_name") or path_ablation
    condition = (
        row.get("model_condition_id")
        or config.get("model_condition_id")
        or path_condition
        or "default"
    )
    annotated = dict(row)
    annotated.update({
        "source_file": str(trials_path),
        "source_ablation": str(ablation) if ablation is not None else None,
        "source_model_condition": str(condition),
        "source_experiment_result_directory": str(source),
        "source_run": source_run,
    })
    # These defaults make grouping robust when a source arm persisted only
    # the trial payload and kept run metadata in config.json.
    for key in (
        "ablation_name", "model_condition_id", "provider", "planner_model",
        "planner_model_effective", "reconciler_model", "evaluation_id",
        "experiment_config_version", "benchmark_suite_version", "test_size",
    ):
        if annotated.get(key) is None and config.get(key) is not None:
            annotated[key] = config.get(key)
    if annotated.get("ablation_name") is None and ablation is not None:
        annotated["ablation_name"] = ablation
    if annotated.get("model_condition_id") is None:
        annotated["model_condition_id"] = condition
    if annotated.get("source_thresholds") is None and isinstance(config.get("thresholds"), Mapping):
        annotated["source_thresholds"] = dict(config["thresholds"])
    return annotated


def load_unified_trial_rows(source_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load every historical trial file before deriving any baseline."""

    source = Path(source_dir).resolve()
    all_rows: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    for trial_file in discover_trial_files(source):
        config = _config_for_trial_file(trial_file)
        raw_rows = _read_jsonl(trial_file)
        run_label = _source_run_label(trial_file, config)
        path_ablation, path_condition = _path_ablation_and_condition(source, trial_file)
        source_runs.append({
            "source_run": run_label,
            "trials_path": str(trial_file),
            "source_ablation": config.get("ablation_name") or path_ablation,
            "source_model_condition": config.get("model_condition_id") or path_condition,
            "config_path": str(trial_file.parent / "config.json") if (trial_file.parent / "config.json").is_file() else None,
            "trial_count": len(raw_rows),
            "experiment_config_version": config.get("experiment_config_version"),
            "provider": config.get("provider"),
            "analysis_role_of_source": "historical_frozen_experiment" if config.get("confirmatory_mode") else "source_evaluation_run",
        })
        all_rows.extend(
            _annotate_source_row(
                row,
                source=source,
                trials_path=trial_file,
                config=config,
                source_run=run_label,
            )
            for row in raw_rows
        )
    return all_rows, source_runs


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat_rows.append({
            key: json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else value
            for key, value in row.items()
        })
    fields = sorted({key for row in flat_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)


def _render_markdown(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Retrospective conventional baseline analysis",
        "",
        f"- Analysis role: `{summary.get('analysis_role')}`.",
        f"- Derived rows: **{summary.get('derived_trial_count', 0)}**; rows with missing artifacts: **{summary.get('missing_artifact_count', 0)}**.",
        "- Independent statistical unit: **dataset/task**. LLM repetitions and split seeds are nested observations, not independent datasets.",
        "- Historical source artifacts are read-only; this directory contains only derived analysis outputs.",
        "",
        "## Baseline definitions",
        "",
        "- `all_four_cv`: conventional full-portfolio model-family selection over the four supported families using training-only CV.",
        "- `pairwise_cv_always`: always trust the raw two-candidate mean-CV result, ignoring the selective evidence-strength/abstention threshold.",
        "- `probe_direct`: the existing selective pairwise comparator, including its abstention behavior.",
        "- `llm_only`: the initial LLM plan after universal hard validation.",
        "",
        "## Overall descriptive comparison",
        "",
        "| Baseline | Evaluated rows | Holdout dataset-macro mean | Training normalized regret vs all-four | Counterfactual selection fits | Planner calls | Reconciler calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for baseline, item in summary.get("baseline_summaries", {}).items():
        lines.append(
            f"| `{baseline}` | {item.get('evaluated_trial_count', 0)} | {item.get('holdout_metric_dataset_macro_mean')} | {item.get('training_normalized_regret_dataset_macro_mean')} | {item.get('counterfactual_selection_fit_count_total', 0)} | {item.get('planner_llm_call_count', 0)} | {item.get('reconciler_llm_call_count', 0)} |"
        )
    lines.extend(["", "## Candidate-set coverage", ""])
    coverage = summary.get("candidate_set_coverage", {})
    lines.append(f"- Hard-valid/evaluable denominator: **{coverage.get('eligible_hard_valid_evaluable_trial_count', 0)}** trials across **{coverage.get('dataset_count', 0)}** datasets/tasks.")
    for key, value in (coverage.get("dataset_macro_rates") or {}).items():
        lines.append(f"- `{key}` dataset-macro rate: **{value}**.")
    lines.extend(["", "## Primary comparisons by model condition", ""])
    lines.append("Comparisons below are stratified by model condition and task type; repetitions and split seeds remain nested within dataset/task.")
    lines.extend(["", "| Model condition | Task type | Comparison | Paired observations | Independent datasets | Dataset-macro difference |", "|---|---|---|---:|---:|---:|"])
    for condition, task_map in (summary.get("comparisons_by_model_condition_and_task_type") or {}).items():
        for task_type, comparison_map in task_map.items():
            for name, item in comparison_map.items():
                lines.append(
                    f"| `{condition}` | `{task_type}` | `{name}` | {item.get('paired_evaluable_count', 0)} | {item.get('independent_dataset_count', 0)} | {item.get('holdout_performance_difference_first_minus_second_dataset_macro_mean')} |"
                )
    lines.extend(["", "## Descriptive combined-condition audit", "", "The combined-condition section is descriptive/audit-only and is not a primary pooled estimate.", ""])
    lines.extend(["", "## Missing or not-applicable provenance", ""])
    missing_rows = [
        row for row in rows
        if row.get("baseline_status") != "evaluated" or row.get("missing_fields")
    ]
    if missing_rows:
        lines.extend(
            f"- `{row.get('source_trial_id')}` / `{row.get('baseline_name')}`: {', '.join(map(str, row.get('missing_fields', [])))}"
            for row in missing_rows[:100]
        )
        if len(missing_rows) > 100:
            lines.append(f"- ... {len(missing_rows) - 100} additional rows; see `baseline_trials.jsonl`.")
    else:
        lines.append("- None.")
    lines.extend(["", "## Directionality", "", "Positive holdout differences in paired comparisons favor the named first baseline. Classification uses macro-F1 difference; regression uses relative RMSE improvement.", ""])
    return "\n".join(lines)


def analyze_result_directory(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    frame_loader: Callable[[Mapping[str, Any]], pd.DataFrame] | Mapping[str, pd.DataFrame] | None = None,
    recompute_missing: bool = False,
    strict: bool = False,
    baselines: Sequence[str] = BASELINE_NAMES,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Analyze one result tree and write four new, machine-readable outputs."""

    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if source == output:
        raise ValueError("Output directory must be distinct from the source experiment directory.")
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory must not be inside the source experiment directory.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty; existing derived outputs are not overwritten.")
    all_source_rows, source_runs = load_unified_trial_rows(source)
    # Derivation happens once, after every arm has been loaded.  This is the
    # boundary that prevents one logical trial from being emitted once per
    # ablation directory.
    all_rows = derive_baseline_trials(
        all_source_rows,
        source_run=str(source),
        frame_loader=frame_loader,
        recompute_missing=recompute_missing,
        baselines=baselines,
        strict=strict,
        thresholds=None,
    )
    summary = summarize_baseline_trials(
        all_rows,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    summary.update({
        "source_directory": str(source),
        "source_runs": source_runs,
        "source_artifacts_modified": False,
        "analysis_recompute_policy": (
            "recompute_missing_deterministic_cv_or_holdout" if recompute_missing else "reuse_persisted_values_only"
        ),
        "strict": strict,
    })
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline_trials.jsonl").write_text(
        "".join(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    (output / "baseline_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    csv_rows: list[dict[str, Any]] = []
    for baseline, item in summary.get("baseline_summaries", {}).items():
        csv_rows.append({
            "summary_scope": "descriptive_combined_condition",
            "model_condition_id": "",
            "task_type": "",
            "baseline": baseline,
            **{key: value for key, value in item.items() if not isinstance(value, (dict, list))},
        })
    for condition, task_map in (summary.get("by_model_condition") or {}).items():
        for task_type, task_item in (task_map.get("by_task_type") or {}).items():
            for baseline, item in (task_item.get("baselines") or {}).items():
                csv_rows.append({
                    "summary_scope": "model_condition_task_type",
                    "model_condition_id": condition,
                    "task_type": task_type,
                    "baseline": baseline,
                    **{key: value for key, value in item.items() if not isinstance(value, (dict, list))},
                })
    for condition, task_map in (summary.get("comparisons_by_model_condition_and_task_type") or {}).items():
        for task_type, comparison_map in task_map.items():
            for comparison_name, item in comparison_map.items():
                csv_rows.append({
                    "summary_scope": "primary_comparison_by_model_condition_and_task_type",
                    "model_condition_id": condition,
                    "task_type": task_type,
                    "comparison": comparison_name,
                    **{key: value for key, value in item.items() if not isinstance(value, (dict, list))},
                })
    _write_csv(output / "baseline_summary.csv", csv_rows)
    (output / "baseline_summary.md").write_text(
        _render_markdown(summary, all_rows),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output),
        "paths": {name: str(output / name) for name in (
            "baseline_trials.jsonl", "baseline_summary.json", "baseline_summary.csv", "baseline_summary.md"
        )},
        "summary": summary,
        "trials": all_rows,
    }


def _default_frame_loader(row: Mapping[str, Any]) -> pd.DataFrame:
    suite = str(row.get("benchmark_suite") or row.get("suite") or "")
    if suite == "external" or row.get("openml_task_id") is not None:
        from evaluation.external_benchmarks import external_benchmark_cases
        cases = {case.name: case for case in external_benchmark_cases()}
    else:
        from evaluation.benchmarks import default_benchmark_cases
        cases = {case.name: case for case in default_benchmark_cases()}
    name = str(row.get("benchmark_case", ""))
    if name not in cases:
        raise MissingHistoricalFields(f"no built-in benchmark case matches {name!r}")
    return cases[name].load()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive retrospective conventional baselines from frozen evaluation trials.")
    parser.add_argument("--source", required=True, help="Existing evaluation/result directory; it is never overwritten.")
    parser.add_argument("--output", required=True, help="New directory for baseline outputs.")
    parser.add_argument("--recompute-missing", action="store_true", help="Load raw benchmark frames and recompute only missing deterministic CV/holdout values.")
    parser.add_argument("--strict", action="store_true", help="Fail after listing missing historical fields instead of writing partial rows.")
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    loader = _default_frame_loader if args.recompute_missing else None
    result = analyze_result_directory(
        args.source,
        args.output,
        frame_loader=loader,
        recompute_missing=args.recompute_missing,
        strict=args.strict,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"output_dir": result["output_dir"], "paths": result["paths"], "missing_artifact_count": result["summary"].get("missing_artifact_count")}, indent=2))


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    main()


__all__ = [
    "BASELINE_ANALYSIS_ROLE",
    "BASELINE_ANALYSIS_SCHEMA_VERSION",
    "BASELINE_NAMES",
    "BaselineAnalysisError",
    "MissingHistoricalFields",
    "analyze_result_directory",
    "choose_pairwise_cv_always",
    "derive_all_four_cv",
    "derive_baseline_trials",
    "derive_pairwise_cv_always",
    "discover_trial_files",
    "load_unified_trial_rows",
    "select_all_four_cv",
    "select_pairwise_cv_always",
    "summarize_baseline_trials",
]
