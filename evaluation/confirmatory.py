"""Opt-in validation and hashing for frozen confirmatory experiment manifests.

The manifest is intentionally a plain JSON document.  This module only
provides the small amount of machinery needed to associate a run with that
document and to fail before execution when the declared runtime differs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


CONFIRMATORY_MANIFEST_SCHEMA_VERSION = "confirmatory-manifest-v1"
CONFIRMATORY_EXPERIMENT_NAME = "selective-intervention-reliability"
EXPERIMENT_CODE_PATHS = ("app", "evaluation", "pyproject.toml")
CONFIRMATORY_MANIFEST_RELATIVE_PATH = "evaluation/configs/paper_confirmatory_v1.json"
_EXCLUDED_DIRECTORY_NAMES = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".cache", "cache", "caches", "evaluation_results", "results", "tmp", "temp",
}
_EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo", ".tmp", ".temp"}


def canonical_json_bytes(value: Any) -> bytes:
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


def experiment_code_sha256(repository_root: str | Path | None = None) -> str:
    """Hash the canonical result-affecting source/configuration tree.

    Paths and file bytes are framed separately, and paths are visited in
    lexical POSIX order.  The confirmatory manifest is deliberately excluded:
    its expected digest must not hash itself.
    """

    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files: list[Path] = []
    for relative_root in EXPERIMENT_CODE_PATHS:
        candidate = root / relative_root
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(path for path in candidate.rglob("*") if path.is_file())
    included: list[tuple[str, Path]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        parts = set(Path(relative).parts)
        if parts & _EXCLUDED_DIRECTORY_NAMES:
            continue
        if path.suffix.lower() in _EXCLUDED_FILE_SUFFIXES:
            continue
        if relative == CONFIRMATORY_MANIFEST_RELATIVE_PATH:
            continue
        included.append((relative, path))
    for relative, path in sorted(included, key=lambda item: item[0]):
        path_bytes = relative.encode("utf-8")
        file_bytes = path.read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(file_bytes).to_bytes(8, "big"))
        digest.update(file_bytes)
    return digest.hexdigest()


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


def _comparable_model_conditions(value: Any) -> Any:
    """Compare frozen condition declarations without treating null defaults as drift."""

    if not isinstance(value, list):
        return value
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            result.append(item)
            continue
        normalized = dict(item)
        normalized["generation_settings"] = {
            key: value for key, value in (item.get("generation_settings", {}) or {}).items()
            if value is not None
        }
        if normalized.get("llm_repetition_ids") is None:
            normalized.pop("llm_repetition_ids", None)
        result.append(normalized)
    return result


def deterministic_policy_config(policy: Any | None = None) -> dict[str, Any]:
    """Return every runtime deterministic-policy parameter in canonical form."""

    if policy is None:
        from app.deterministic_policy import DeterministicPolicy

        policy = DeterministicPolicy()
    return _normalise(asdict(policy))


def empirical_probe_config(policy: Any | None = None) -> dict[str, Any]:
    """Return every runtime empirical-probe parameter in canonical form."""

    if policy is None:
        from app.empirical_challenge_probe import EmpiricalProbePolicy

        policy = EmpiricalProbePolicy()
    return _normalise(policy.as_dict())


def config_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def repository_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=2,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


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
        "expected_experiment_code_sha256": manifest.get("expected_experiment_code_sha256"),
        "planner_model": modeling.get("planner_model"),
        "reconciler_model": modeling.get("reconciler_model"),
        "model_conditions": model_conditions(manifest),
        "generation_settings": {key: value for key, value in (manifest.get("generation_settings", {}) or {}).items() if value is not None},
        "llm_repetition_ids": repetition_ids(manifest),
        "split_seeds": repetitions.get("split_seeds"),
        "llm_repetitions": repetitions.get("llm_repetitions"),
        "holdout_fraction": holdout.get("fraction"),
        "selected_ablations": (manifest.get("ablations") or {}).get("primary"),
        "deterministic_policy_version": (manifest.get("deterministic_policy") or {}).get("version"),
        "deterministic_policy_sha256": (manifest.get("deterministic_policy") or {}).get("configuration_sha256"),
        "empirical_probe_policy_version": (manifest.get("empirical_probe_policy") or {}).get("policy_version"),
        "empirical_probe_policy_sha256": (manifest.get("empirical_probe_policy") or {}).get("configuration_sha256"),
        "planner_prompt_schema_version": prompts.get("planner_schema_version"),
        "reconciler_prompt_schema_version": prompts.get("reconciliation_prompt_version"),
        "candidate_model_families": modeling.get("candidate_model_families"),
        "preprocessing_option_space": modeling.get("preprocessing_option_space"),
        "classification_neutral_tolerance": holdout.get("classification_neutral_tolerance"),
        "regression_neutral_tolerance": holdout.get("regression_neutral_tolerance"),
        "bootstrap_settings": {
            "method": statistics.get("bootstrap_method"),
            "replicates": statistics.get("bootstrap_replicates"),
            "confidence_level": statistics.get("confidence_level"),
            "seed": statistics.get("bootstrap_seed"),
        },
        "benchmark_manifest_version": external.get("manifest_version"),
        "benchmark_manifest_sha256": external.get("manifest_sha256"),
        "benchmark_tranches": external.get("tranches"),
        "strict_live_required": manifest.get("strict_live_required"),
    }


def model_conditions(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen model matrix in a canonical, auditable form.

    The legacy single-model fields remain readable for exploratory callers,
    but a confirmatory manifest is normalized to explicit conditions here.
    """
    declared = manifest.get("model_conditions")
    if declared is None:
        modeling = manifest.get("modeling", {})
        declared = [{
            "condition_id": "default",
            "planner_model": modeling.get("planner_model"),
            "reconciler_model": modeling.get("reconciler_model"),
            "llm_repetitions": (manifest.get("splits_and_repetitions", {}) or {}).get("llm_repetitions", 1),
        }]
    if not isinstance(declared, list) or not declared:
        raise ValueError("Confirmatory manifest model_conditions must be a non-empty list.")
    result = []
    seen: set[str] = set()
    for condition in declared:
        if not isinstance(condition, Mapping):
            raise ValueError("Each model condition must be an object.")
        condition_id = str(condition.get("condition_id", "")).strip()
        planner = str(condition.get("planner_model", "")).strip()
        reconciler = str(condition.get("reconciler_model", planner)).strip()
        repetitions = condition.get("llm_repetitions")
        if not condition_id or condition_id in seen:
            raise ValueError("Model condition IDs must be non-empty and unique.")
        if not planner or not reconciler:
            raise ValueError(f"Model condition {condition_id!r} must declare planner and reconciler models.")
        if not isinstance(repetitions, int) or repetitions < 1:
            raise ValueError(f"Model condition {condition_id!r} must declare positive llm_repetitions.")
        seen.add(condition_id)
        condition_repetition_ids = condition.get("llm_repetition_ids")
        if condition_repetition_ids is not None:
            if not isinstance(condition_repetition_ids, list) or len(condition_repetition_ids) != repetitions:
                raise ValueError(
                    f"Model condition {condition_id!r} llm_repetition_ids must contain exactly "
                    "llm_repetitions identifiers."
                )
            condition_repetition_ids = [str(item).strip() for item in condition_repetition_ids]
            if not all(condition_repetition_ids) or len(set(condition_repetition_ids)) != len(condition_repetition_ids):
                raise ValueError(
                    f"Model condition {condition_id!r} llm_repetition_ids must be non-empty and unique."
                )
        result.append({
            "condition_id": condition_id,
            "planner_model": planner,
            "reconciler_model": reconciler,
            "llm_repetitions": repetitions,
            "llm_repetition_ids": condition_repetition_ids,
            # Preserve nulls: null means provider default and is part of the
            # frozen declaration even though it is omitted from the request.
            "generation_settings": dict(condition.get("generation_settings", {}) or {}),
        })
    return result


def repetition_ids(manifest: Mapping[str, Any]) -> list[str]:
    repetitions = manifest.get("splits_and_repetitions", {}) or {}
    declared = repetitions.get("llm_repetition_ids")
    count = repetitions.get("llm_repetitions", 1)
    if declared is None:
        declared = [f"rep_{index:03d}" for index in range(1, int(count) + 1)]
    if not isinstance(declared, list) or len(declared) != int(count):
        raise ValueError("llm_repetition_ids must contain exactly llm_repetitions identifiers.")
    values = [str(item).strip() for item in declared]
    if not all(values) or len(set(values)) != len(values):
        raise ValueError("llm_repetition_ids must be non-empty and unique.")
    return values


def condition_repetition_ids(
    manifest: Mapping[str, Any], condition: Mapping[str, Any]
) -> list[str]:
    """Return the stable repetition IDs for one frozen model condition."""

    explicit = condition.get("llm_repetition_ids")
    if explicit is not None:
        return [str(item) for item in explicit]
    return repetition_ids(manifest)[: int(condition["llm_repetitions"])]


def expand_confirmatory_matrix(
    manifest: Mapping[str, Any],
    *,
    dataset_ids: list[str] | tuple[str, ...] = (),
    ablations: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Expand the declared condition/repetition matrix without making calls."""
    names = list(ablations if ablations is not None else (manifest.get("ablations", {}) or {}).get("primary", []))
    rows = []
    for dataset_id in dataset_ids or [None]:
        for condition in model_conditions(manifest):
            for repetition_id in condition_repetition_ids(manifest, condition):
                for ablation in names or [None]:
                    rows.append({
                        "dataset_id": dataset_id,
                        "model_condition_id": condition["condition_id"],
                        **condition,
                        "llm_repetition_id": repetition_id,
                        "ablation": ablation,
                    })
    return rows


def expand_confirmatory_evaluation_units(
    manifest: Mapping[str, Any],
    *,
    dataset_ids: list[str] | tuple[str, ...],
    split_seeds: list[int] | tuple[int, ...],
    ablations: list[str] | tuple[str, ...] | None = None,
    perturbation_ids: list[str] | tuple[str, ...] = ("clean",),
    evaluation_variants: list[str] | tuple[str, ...] = ("standard",),
) -> list[dict[str, Any]]:
    """Expand planned result units from the frozen manifest dimensions."""

    base = expand_confirmatory_matrix(
        manifest, dataset_ids=dataset_ids, ablations=ablations
    )
    units: list[dict[str, Any]] = []
    for row in base:
        for split_seed in split_seeds:
            for perturbation_id in perturbation_ids:
                for evaluation_variant in evaluation_variants:
                    units.append({
                        **row,
                        "benchmark_case": row["dataset_id"],
                        "split_seed": int(split_seed),
                        "perturbation_id": perturbation_id,
                        "ablation_name": row["ablation"],
                        "evaluation_variant": evaluation_variant,
                    })
    return units


def confirmatory_unit_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Canonical identity for a planned/evaluated confirmatory unit."""

    return (
        str(row.get("model_condition_id", "")),
        str(row.get("llm_repetition_id", "")),
        str(row.get("benchmark_case", row.get("dataset_id", ""))),
        str(row.get("perturbation_id", "clean")),
        str(row.get("split_seed", "")),
        str(row.get("ablation_name", row.get("ablation", ""))),
        str(row.get("evaluation_variant", "standard")),
    )


def validate_confirmatory_completeness(
    expected_units: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    observed_rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Audit the planned evaluation units against persisted result rows.

    Failed rows are observed records but do not satisfy a required unit.  The
    audit deliberately ignores conditional reconciliation calls: those are
    internal events, not planned evaluation units.
    """

    expected_keys = [confirmatory_unit_key(row) for row in expected_units]
    observed_keys = [confirmatory_unit_key(row) for row in observed_rows]
    expected_set = set(expected_keys)
    observed_set = set(observed_keys)
    duplicates = sorted({key for key in observed_keys if observed_keys.count(key) > 1}, key=str)
    failed = sorted(
        {confirmatory_unit_key(row) for row in observed_rows if row.get("trial_status") == "failed"},
        key=str,
    )
    missing = sorted(expected_set - observed_set | (set(failed) & expected_set), key=str)
    unexpected = sorted(observed_set - expected_set, key=str)
    if len(expected_keys) != len(expected_set):
        raise ValueError("Confirmatory expected evaluation matrix contains duplicate planned units.")
    complete = not missing and not duplicates and not unexpected
    result = {
        "complete": complete,
        "expected_unit_count": len(expected_keys),
        "observed_unit_count": len(observed_keys),
        "unique_observed_unit_count": len(observed_set),
        "missing_units": [list(key) for key in missing],
        "failed_units": [list(key) for key in failed],
        "duplicate_units": [list(key) for key in duplicates],
        "unexpected_units": [list(key) for key in unexpected],
        "unit_identity_fields": [
            "model_condition_id", "llm_repetition_id", "benchmark_case",
            "perturbation_id", "split_seed", "ablation_name", "evaluation_variant",
        ],
    }
    def describe(key: tuple[str, ...]) -> str:
        return ", ".join(
            f"{field}={value}" for field, value in zip(result["unit_identity_fields"], key)
        )
    if not complete:
        details: list[str] = []
        if missing:
            details.append(f"missing {describe(missing[0])}")
        if duplicates:
            details.append(f"duplicate {describe(duplicates[0])}")
        if unexpected:
            details.append(f"unexpected {describe(unexpected[0])}")
        raise ValueError("Confirmatory matrix incomplete: " + "; ".join(details))
    return result


# Compatibility-friendly descriptive alias.
audit_confirmatory_matrix = validate_confirmatory_completeness


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
    expected_experiment_code_sha256: str | None = None,
    source_git_commit: str | None = None,
    deterministic_policy_sha256: str | None = None,
    empirical_probe_policy_sha256: str | None = None,
    benchmark_manifest_sha256: str | None = None,
    benchmark_task_ids: list[int] | tuple[int, ...] | None = None,
    benchmark_tranches: Mapping[str, Any] | None = None,
    benchmark_tier: str | None = None,
    preprocessing_option_space: list[str] | tuple[str, ...] | None = None,
    model_conditions: list[Mapping[str, Any]] | None = None,
    generation_settings: Mapping[str, Any] | None = None,
    llm_repetition_ids: list[str] | tuple[str, ...] | None = None,
    selected_model_condition_id: str | None = None,
) -> dict[str, Any]:
    """Build the runtime projection compared with the frozen manifest."""

    return {
        "manifest_schema_version": CONFIRMATORY_MANIFEST_SCHEMA_VERSION,
        "experiment_name": experiment_name,
        "experiment_config_version": experiment_config_version,
        "expected_experiment_code_sha256": expected_experiment_code_sha256,
        "source_git_commit": source_git_commit,
        "planner_model": planner_model,
        "reconciler_model": reconciler_model,
        "model_conditions": [dict(item) for item in model_conditions] if model_conditions is not None else [{"condition_id": "default", "planner_model": planner_model, "reconciler_model": reconciler_model, "llm_repetitions": int(llm_repetitions), "generation_settings": {}}],
        "generation_settings": dict(generation_settings or {}),
        "llm_repetition_ids": list(llm_repetition_ids) if llm_repetition_ids is not None else [f"rep_{index:03d}" for index in range(1, int(llm_repetitions) + 1)],
        "selected_model_condition_id": selected_model_condition_id,
        "split_seeds": list(split_seeds),
        "llm_repetitions": int(llm_repetitions),
        "holdout_fraction": float(holdout_fraction),
        "selected_ablations": list(selected_ablations) if selected_ablations is not None else None,
        "deterministic_policy_version": deterministic_policy_version,
        "deterministic_policy_sha256": deterministic_policy_sha256,
        "empirical_probe_policy_version": empirical_probe_policy_version,
        "empirical_probe_policy_sha256": empirical_probe_policy_sha256,
        "planner_prompt_schema_version": planner_prompt_schema_version,
        "reconciler_prompt_schema_version": reconciler_prompt_schema_version,
        "candidate_model_families": list(candidate_model_families),
        "preprocessing_option_space": list(preprocessing_option_space) if preprocessing_option_space is not None else None,
        "classification_neutral_tolerance": float(classification_neutral_tolerance),
        "regression_neutral_tolerance": float(regression_neutral_tolerance),
        "bootstrap_settings": dict(bootstrap_settings),
        "benchmark_manifest_version": benchmark_manifest_version,
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "benchmark_task_ids": list(benchmark_task_ids) if benchmark_task_ids is not None else None,
        "benchmark_tranches": dict(benchmark_tranches) if benchmark_tranches is not None else None,
        "benchmark_tier": benchmark_tier,
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
    probe_policy = loaded.get("empirical_probe_policy") or {}
    allowed_probe_fields = {
        "policy_version", "configuration_sha256", "enabled", "cv_folds",
        "minimum_rows", "max_rows", "max_configs_per_family",
        "tie_relative_threshold", "moderate_relative_threshold",
        "strong_relative_threshold", "variability_tie_multiplier",
        "variability_weak_multiplier", "minimum_consistent_win_rate",
    }
    unknown_probe_fields = sorted(set(probe_policy) - allowed_probe_fields)
    if unknown_probe_fields:
        raise ValueError(
            "Confirmatory empirical_probe_policy contains unknown/deprecated fields: "
            + ", ".join(unknown_probe_fields)
        )
    expected = _manifest_values(loaded)
    mismatches: list[str] = []
    expected_code_sha256 = loaded.get("expected_experiment_code_sha256")
    observed_code_sha256 = experiment_code_sha256()
    if not isinstance(expected_code_sha256, str) or not expected_code_sha256.strip():
        mismatches.append("expected_experiment_code_sha256 missing from frozen manifest")
    elif observed_code_sha256 != expected_code_sha256:
        mismatches.append(
            "experiment code SHA-256 mismatch: expected "
            f"{expected_code_sha256!r}, observed {observed_code_sha256!r}; "
            "result-affecting experiment code/configuration differs from the frozen manifest"
        )
    for key, expected_value in expected.items():
        if expected_value is None:
            mismatches.append(f"{key} missing from frozen manifest")
            continue
        actual_value = runtime_values.get(key)
        if key in {"llm_repetitions", "llm_repetition_ids"} and loaded.get("model_conditions") is not None:
            selected_id = runtime_values.get("selected_model_condition_id")
            selected_condition = next(
                (item for item in model_conditions(loaded) if item["condition_id"] == selected_id), None
            )
            if selected_condition is not None:
                expected_value = (
                    selected_condition["llm_repetitions"]
                    if key == "llm_repetitions"
                    else condition_repetition_ids(loaded, selected_condition)
                )
            if not _same(expected_value, actual_value):
                mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
            continue
        if key in {"planner_model", "reconciler_model"} and loaded.get("model_conditions") is not None:
            declared_values = {
                str(condition[key]) for condition in model_conditions(loaded)
            }
            if str(actual_value) not in declared_values:
                mismatches.append(
                    f"{key}: runtime model {actual_value!r} is not declared in frozen model_conditions"
                )
            continue
        if key == "model_conditions":
            if not _same(
                _comparable_model_conditions(expected_value),
                _comparable_model_conditions(actual_value),
            ):
                mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
            continue
        if key == "selected_ablations" and actual_value is None:
            continue
        if not _same(expected_value, actual_value):
            if key == "expected_experiment_code_sha256":
                mismatches.append(
                    "experiment code SHA-256 mismatch: expected "
                    f"{expected_value!r}, observed {actual_value!r}; "
                    "result-affecting experiment code/configuration differs from the frozen manifest"
                )
            else:
                mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    frozen_ids = loaded.get("external_benchmark", {}).get("task_ids")
    frozen_tranches = loaded.get("external_benchmark", {}).get("tranches")
    selected_ids = runtime_values.get("benchmark_task_ids")
    selected_tier = runtime_values.get("benchmark_tier")
    if not isinstance(frozen_ids, list) or not isinstance(frozen_tranches, dict):
        mismatches.append("external benchmark membership is incomplete in frozen manifest")
    elif not isinstance(selected_ids, list):
        mismatches.append("benchmark_task_ids missing from runtime")
    else:
        expected_ids = frozen_tranches.get(selected_tier) if selected_tier else frozen_ids
        if not isinstance(expected_ids, list) or selected_ids != expected_ids:
            label = selected_tier or "full"
            mismatches.append(f"benchmark membership for {label!r}: expected {expected_ids!r}, got {selected_ids!r}")
    if mismatches:
        raise ValueError("Confirmatory manifest mismatch; refusing to run: " + "; ".join(mismatches))
    return {
        "status": "frozen",
        "experiment_config_version": loaded["experiment_config_version"],
        "experiment_config_sha256": manifest_sha256(loaded),
        "manifest_schema_version": loaded["manifest_schema_version"],
        "expected_experiment_code_sha256": expected_code_sha256,
        "observed_experiment_code_sha256": observed_code_sha256,
        "source_git_commit": repository_commit(),
        "benchmark_manifest_matches": not any(item.startswith(("benchmark_manifest_sha256", "benchmark_manifest_version", "benchmark_tranches", "benchmark membership", "external benchmark membership")) for item in mismatches),
    }


# Descriptive aliases for callers that use "config" rather than "manifest".
validate_confirmatory_config = validate_confirmatory_manifest
