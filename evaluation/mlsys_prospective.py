"""Shared metadata and validation helpers for the prospective MLSys study.

The prospective experiment is deliberately a fresh artifact family.  These
helpers do not read, repair, or reinterpret historical result summaries.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


MLSYS_PROSPECTIVE_CONFIG_SCHEMA_VERSION = "mlsys-prospective-config-v1"
MLSYS_TRIAL_SCHEMA_VERSION = "mlsys-prospective-trial-v1"
SUPPORTED_MLSYS_TRIAL_SCHEMA_VERSIONS = frozenset({MLSYS_TRIAL_SCHEMA_VERSION})
MLSYS_SOURCE_ARMS = ("llm_only", "probe_direct", "full")
MLSYS_FIXED_FAMILIES = ("linear", "regularized_linear", "tree_ensemble", "boosted_tree")


def jsonable(value: Any) -> Any:
    """Convert common project values to deterministic JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        jsonable(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def manifest_hash_without_hash_fields(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest independently of its derived hash fields."""

    payload = dict(manifest)
    for field in (
        "content_sha256",
        "manifest_sha256",
        "canonical_manifest_sha256",
    ):
        payload.pop(field, None)
    return canonical_sha256(payload)


def trial_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the source-arm-independent identity of one logical trial."""

    explicit = row.get("logical_trial_id")
    if explicit:
        return ("logical_trial_id", str(explicit))
    return (
        "identity",
        row.get("model_condition_id", "default"),
        row.get("provider"),
        row.get("planner_model_effective") or row.get("planner_model") or row.get("agent_model"),
        row.get("benchmark_case", row.get("dataset_id", row.get("task_id"))),
        row.get("dataset_id"),
        row.get("dataset_version"),
        row.get("task_id"),
        row.get("task_type"),
        row.get("target_column") or row.get("agent_initial_target") or row.get("final_target"),
        row.get("perturbation_id", "clean"),
        row.get("split_seed"),
        row.get("split_random_state", row.get("split_seed")),
        row.get("llm_repetition_id", row.get("trial")),
        row.get("trial"),
        row.get("evaluation_variant", "standard"),
        row.get("order_swap_pair_id"),
    )


def identity_label(identity: tuple[Any, ...]) -> str:
    return canonical_sha256(list(identity))[:16]


def source_arm(row: Mapping[str, Any]) -> str | None:
    value = row.get("source_arm") or row.get("ablation_name") or row.get("gate_mode")
    return str(value) if value else None


def has_preprocessing(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and (
        value.get("fit_inside_pipeline") is True
        or "numeric_imputation" in value
        or "categorical_encoding" in value
    )


def has_initial_plan(row: Mapping[str, Any]) -> bool:
    nested = row.get("agent_initial")
    method = row.get("agent_initial_method") or (
        nested.get("method") if isinstance(nested, Mapping) else None
    )
    preprocessing = row.get("agent_initial_preprocessing") or (
        nested.get("preprocessing") if isinstance(nested, Mapping) else None
    )
    validity = row.get("agent_initial_valid")
    if validity is None and isinstance(nested, Mapping):
        validity = nested.get("valid")
    return method in MLSYS_FIXED_FAMILIES and has_preprocessing(preprocessing) and validity is not None


def candidate_metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = row.get("candidate_cv_metrics")
    if isinstance(direct, Mapping) and direct:
        return direct
    nested = row.get("empirical_reference")
    if isinstance(nested, Mapping) and isinstance(nested.get("candidate_metrics"), Mapping):
        return nested["candidate_metrics"]
    records = row.get("four_family_cv_records")
    if isinstance(records, Mapping) and records:
        return records
    return {}


def has_four_family_reference(row: Mapping[str, Any]) -> bool:
    metrics = candidate_metrics(row)
    if not all(family in metrics and isinstance(metrics[family], Mapping) for family in MLSYS_FIXED_FAMILIES):
        return False
    for family in MLSYS_FIXED_FAMILIES:
        value = metrics[family]
        if not value.get("status"):
            return False
        if value.get("status") != "evaluated":
            return False
        if value.get("primary_mean", value.get("cv_mean")) is None:
            return False
        preprocessing = value.get("preprocessing") or value.get("preprocessing_contract")
        validation = value.get("validation")
        if not has_preprocessing(preprocessing) and not (
            isinstance(validation, Mapping)
            and has_preprocessing(validation.get("approved_preprocessing"))
        ):
            return False
    return True


def pairwise_evidence(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    explicit = row.get("pairwise_cv_evidence")
    if isinstance(explicit, Mapping):
        return explicit
    probe = row.get("empirical_probe")
    if not isinstance(probe, Mapping):
        return None
    a = probe.get("proposal_a")
    b = probe.get("proposal_b")
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return None
    if a.get("mean_score") is None or b.get("mean_score") is None:
        return None
    return probe


def has_pairwise_evidence(row: Mapping[str, Any]) -> bool:
    evidence = pairwise_evidence(row)
    if not isinstance(evidence, Mapping):
        return False
    a = evidence.get("candidate_a") or evidence.get("proposal_a")
    b = evidence.get("candidate_b") or evidence.get("proposal_b")
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return False
    return (
        a.get("model_family") in MLSYS_FIXED_FAMILIES
        and b.get("model_family") in MLSYS_FIXED_FAMILIES
        and (a.get("mean_score", a.get("raw_mean_cv_score")) is not None)
        and (b.get("mean_score", b.get("raw_mean_cv_score")) is not None)
        and evidence.get("metric") in {"macro_f1", "rmse"}
        and evidence.get("higher_is_better") is not None
        and evidence.get("raw_mean_difference") is not None
        and evidence.get("probe_status", evidence.get("status"))
    )


def has_holdout(row: Mapping[str, Any]) -> bool:
    return (
        row.get("final_holdout_metric") is not None
        or isinstance(row.get("gated_final_holdout_metrics"), Mapping)
        and bool(row.get("gated_final_holdout_metrics"))
    )


__all__ = [
    "MLSYS_FIXED_FAMILIES",
    "MLSYS_PROSPECTIVE_CONFIG_SCHEMA_VERSION",
    "MLSYS_SOURCE_ARMS",
    "MLSYS_TRIAL_SCHEMA_VERSION",
    "SUPPORTED_MLSYS_TRIAL_SCHEMA_VERSIONS",
    "candidate_metrics",
    "canonical_json_bytes",
    "canonical_sha256",
    "has_four_family_reference",
    "has_holdout",
    "has_initial_plan",
    "has_pairwise_evidence",
    "has_preprocessing",
    "identity_label",
    "manifest_hash_without_hash_fields",
    "pairwise_evidence",
    "source_arm",
    "trial_identity",
]
