"""Opt-in validation and hashing for frozen confirmatory experiment manifests.

The manifest is intentionally a plain JSON document.  This module only
provides the small amount of machinery needed to associate a run with that
document and to fail before execution when the declared runtime differs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONFIRMATORY_MANIFEST_SCHEMA_VERSION = "confirmatory-manifest-v1"
CONFIRMATORY_EXPERIMENT_NAME = "selective-intervention-reliability"


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a manifest deterministically for hashing and comparison."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any] | str | Path) -> str:
    """Return the SHA-256 of a canonical JSON manifest representation."""

    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def load_confirmatory_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a confirmatory manifest document."""

    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Confirmatory manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Confirmatory manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Confirmatory manifest root must be a JSON object.")
    return value


def _normalise(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalise(item) for item in value]
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    return value


def _same(left: Any, right: Any) -> bool:
    return _normalise(left) == _normalise(right)


def _manifest_values(manifest: Mapping[str, Any]) -> dict[str, Any]:
    external = manifest.get("external_benchmark", {})
    holdout = manifest.get("holdout", {})
    prompts = manifest.get("prompts", {})
    modeling = manifest.get("modeling", {})
    repetitions = manifest.get("splits_and_repetitions", {})
    statistics = manifest.get("statistics", {})
    return {
        "manifest_schema_version": manifest.get("manifest_schema_version"),
        "experiment_name": manifest.get("experiment_name"),
        "experiment_config_version": manifest.get("experiment_config_version"),
        "planner_model": modeling.get("planner_model"),
        "reconciler_model": modeling.get("reconciler_model"),
        "split_seeds": repetitions.get("split_seeds"),
        "llm_repetitions": repetitions.get("llm_repetitions"),
        "holdout_fraction": holdout.get("fraction"),
        "selected_ablations": (manifest.get("ablations") or {}).get("primary"),
        "deterministic_policy_version": (manifest.get("deterministic_policy") or {}).get("version"),
        "empirical_probe_policy_version": (manifest.get("empirical_probe_policy") or {}).get("policy_version"),
        "planner_prompt_schema_version": prompts.get("planner_schema_version"),
        "reconciler_prompt_schema_version": prompts.get("reconciliation_prompt_version"),
        "candidate_model_families": modeling.get("candidate_model_families"),
        "classification_neutral_tolerance": holdout.get("classification_neutral_tolerance"),
        "regression_neutral_tolerance": holdout.get("regression_neutral_tolerance"),
        "bootstrap_settings": {
            "method": statistics.get("bootstrap_method"),
            "replicates": statistics.get("bootstrap_replicates"),
            "confidence_level": statistics.get("confidence_level"),
            "seed": statistics.get("bootstrap_seed"),
        },
        "benchmark_manifest_version": external.get("manifest_version"),
        "strict_live_required": manifest.get("strict_live_required"),
    }


def runtime_manifest_values(
    *,
    experiment_name: str,
    planner_model: str,
    reconciler_model: str,
    split_seeds: list[int] | tuple[int, ...],
    llm_repetitions: int,
    holdout_fraction: float,
    selected_ablations: list[str] | tuple[str, ...] | None,
    deterministic_policy_version: str,
    empirical_probe_policy_version: str,
    planner_prompt_schema_version: str,
    reconciler_prompt_schema_version: str,
    candidate_model_families: list[str] | tuple[str, ...],
    classification_neutral_tolerance: float,
    regression_neutral_tolerance: float,
    benchmark_manifest_version: str,
    strict_live_required: bool,
    bootstrap_settings: Mapping[str, Any],
    experiment_config_version: str,
) -> dict[str, Any]:
    """Build the runtime projection compared with the frozen manifest."""

    return {
        "manifest_schema_version": CONFIRMATORY_MANIFEST_SCHEMA_VERSION,
        "experiment_name": experiment_name,
        "experiment_config_version": experiment_config_version,
        "planner_model": planner_model,
        "reconciler_model": reconciler_model,
        "split_seeds": list(split_seeds),
        "llm_repetitions": int(llm_repetitions),
        "holdout_fraction": float(holdout_fraction),
        "selected_ablations": list(selected_ablations) if selected_ablations is not None else None,
        "deterministic_policy_version": deterministic_policy_version,
        "empirical_probe_policy_version": empirical_probe_policy_version,
        "planner_prompt_schema_version": planner_prompt_schema_version,
        "reconciler_prompt_schema_version": reconciler_prompt_schema_version,
        "candidate_model_families": list(candidate_model_families),
        "classification_neutral_tolerance": float(classification_neutral_tolerance),
        "regression_neutral_tolerance": float(regression_neutral_tolerance),
        "bootstrap_settings": dict(bootstrap_settings),
        "benchmark_manifest_version": benchmark_manifest_version,
        "strict_live_required": bool(strict_live_required),
    }


def validate_confirmatory_manifest(
    manifest: Mapping[str, Any] | str | Path,
    runtime_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a frozen manifest and return its reproducibility metadata.

    ``selected_ablations`` may be omitted for a single evaluation run.  An
    ablation study supplies it explicitly, which makes the frozen set
    enforceable without blocking ordinary development runs.
    """

    loaded = load_confirmatory_manifest(manifest) if isinstance(manifest, (str, Path)) else dict(manifest)
    if loaded.get("manifest_schema_version") != CONFIRMATORY_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Confirmatory manifest schema mismatch: expected "
            f"{CONFIRMATORY_MANIFEST_SCHEMA_VERSION!r}, got {loaded.get('manifest_schema_version')!r}."
        )
    if loaded.get("status") != "frozen":
        raise ValueError(
            "Confirmatory manifest is not frozen; set status to 'frozen' only after "
            "the experiment definition has been intentionally selected."
        )
    expected = _manifest_values(loaded)
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if expected_value is None:
            mismatches.append(f"{key} missing from frozen manifest")
            continue
        actual_value = runtime_values.get(key)
        if key == "selected_ablations" and actual_value is None:
            continue
        if not _same(expected_value, actual_value):
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    if mismatches:
        raise ValueError("Confirmatory manifest mismatch; refusing to run: " + "; ".join(mismatches))
    return {
        "status": "frozen",
        "experiment_config_version": loaded["experiment_config_version"],
        "experiment_config_sha256": manifest_sha256(loaded),
        "manifest_schema_version": loaded["manifest_schema_version"],
    }


# Descriptive aliases for callers that use "config" rather than "manifest".
validate_confirmatory_config = validate_confirmatory_manifest
config_sha256 = manifest_sha256
