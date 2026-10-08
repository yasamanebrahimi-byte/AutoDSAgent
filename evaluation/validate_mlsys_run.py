"""Validate a prospective MLSys run before conventional baseline derivation.

The validator is intentionally strict about missing scientific evidence.  It
never reconstructs a missing pairwise comparison or four-family reference
from holdout summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evaluation.mlsys_prospective import (
    MLSYS_SOURCE_ARMS,
    SUPPORTED_MLSYS_TRIAL_SCHEMA_VERSIONS,
    canonical_sha256,
    has_four_family_reference,
    has_holdout,
    has_initial_plan,
    has_pairwise_evidence,
    identity_label,
    source_arm,
    trial_identity,
)


PROSPECTIVE_ANALYSIS_ROLE = "prospective_generalization"


def _config_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    nested = config.get("run_metadata")
    merged = dict(nested) if isinstance(nested, Mapping) else {}
    merged.update(config)
    return merged


def _is_prospective_config(config: Mapping[str, Any]) -> bool:
    metadata = _config_metadata(config)
    return (
        metadata.get("analysis_role") == PROSPECTIVE_ANALYSIS_ROLE
        or metadata.get("schema_version") == "mlsys-prospective-config-v1"
        or metadata.get("study_role") == "mlsys_four_policy_comparison"
    )


def _declared_repetition_ids(condition: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    supplied = condition.get("llm_repetition_ids") or condition.get("repetition_ids")
    if isinstance(supplied, list) and supplied:
        return [str(value) for value in supplied]
    count = condition.get("llm_repetitions")
    if count is None:
        count = config.get("llm_repetitions", config.get("repetitions"))
    try:
        count = int(count)
    except (TypeError, ValueError):
        return []
    return [f"rep_{index + 1:03d}" for index in range(count)] if count > 0 else []


def _declared_matrix(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand the declared prospective panel/config matrix.

    This is deliberately based only on declarations.  Observed trial rows are
    used later to classify each expected point as missing, incomplete, or
    complete; they never determine the expected count.
    """

    metadata = _config_metadata(config)
    tasks = manifest.get("tasks")
    split_seeds = metadata.get("split_seeds") or metadata.get("prospective_split_seeds")
    conditions = metadata.get("model_conditions")
    declaration_errors: list[str] = []
    if not isinstance(tasks, list) or not tasks:
        declaration_errors.append("prospective expected logical matrix has no panel tasks")
        tasks = []
    if not isinstance(split_seeds, list) or not split_seeds:
        declaration_errors.append("prospective expected logical matrix has no declared split seeds")
        split_seeds = []
    if not isinstance(conditions, list) or not conditions:
        condition_id = metadata.get("model_condition_id", "default")
        conditions = [{
            "condition_id": condition_id,
            "provider": metadata.get("provider"),
            "planner_model": metadata.get("planner_model") or metadata.get("model"),
            "llm_repetitions": metadata.get("llm_repetitions", metadata.get("repetitions")),
        }]
    matrix: list[dict[str, Any]] = []
    for condition in conditions:
        if not isinstance(condition, Mapping):
            declaration_errors.append("prospective model_conditions contains a non-object")
            continue
        condition_id = condition.get("condition_id") or condition.get("id")
        repetition_ids = _declared_repetition_ids(condition, metadata)
        if condition_id is None or not repetition_ids:
            declaration_errors.append(
                "prospective model_conditions must declare condition_id and positive repetitions"
            )
            continue
        for task in tasks:
            if not isinstance(task, Mapping):
                declaration_errors.append("prospective panel tasks contain a non-object")
                continue
            for split_seed in split_seeds:
                for repetition_id in repetition_ids:
                    matrix.append({
                        "task_id": task.get("task_id"),
                        "dataset_id": task.get("dataset_id"),
                        "dataset_name": task.get("dataset_name"),
                        "dataset_version": task.get("dataset_version"),
                        "task_type": task.get("task_type"),
                        "target": task.get("target"),
                        "split_seed": split_seed,
                        "split_random_state": split_seed,
                        "model_condition_id": str(condition_id),
                        "provider": condition.get("provider") or metadata.get("provider"),
                        "planner_model": (
                            condition.get("planner_model")
                            or condition.get("model")
                            or metadata.get("planner_model")
                            or metadata.get("model")
                        ),
                        "llm_repetition_id": repetition_id,
                    })
    return matrix, sorted(set(declaration_errors))


def _same_identifier(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    if str(left) == str(right):
        return True
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return False


def _row_matches_declared_group(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if row.get("task_id") is not None and expected.get("task_id") is not None:
        if not _same_identifier(row.get("task_id"), expected.get("task_id")):
            return False
    elif row.get("dataset_id") is not None and expected.get("dataset_id") is not None:
        if not _same_identifier(row.get("dataset_id"), expected.get("dataset_id")):
            return False
    else:
        observed_name = row.get("benchmark_case") or row.get("dataset_name")
        if observed_name != expected.get("dataset_name"):
            return False
    checks = (
        ("dataset_id", "dataset_id"),
        ("dataset_version", "dataset_version"),
        ("task_type", "task_type"),
        ("split_seed", "split_seed"),
        ("split_random_state", "split_random_state"),
        ("model_condition_id", "model_condition_id"),
        ("llm_repetition_id", "llm_repetition_id"),
    )
    for row_key, expected_key in checks:
        expected_value = expected.get(expected_key)
        observed_value = row.get(row_key)
        if expected_value is None:
            continue
        if row_key in {"dataset_id", "split_seed", "split_random_state"}:
            if not _same_identifier(observed_value, expected_value):
                return False
        elif observed_value is not None and str(observed_value) != str(expected_value):
            return False
        elif observed_value is None and row_key not in {"dataset_version"}:
            return False
    expected_name = expected.get("dataset_name")
    observed_name = row.get("benchmark_case") or row.get("dataset_name")
    if expected_name is not None and observed_name is not None and str(observed_name) != str(expected_name):
        return False
    return True


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        for key in ("valid", "status", "overall_status", "actionability_status"):
            if value.get(key) is not None:
                parsed = _bool_value(value.get(key))
                if parsed is not None:
                    return parsed
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "valid", "passed", "pass", "yes"}:
            return True
        if value in {"false", "invalid", "failed", "fail", "no"}:
            return False
    return None


def _pairwise_probe_required(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    nested_initial = row.get("agent_initial")
    initial_family = row.get("agent_initial_method") or (
        nested_initial.get("method") if isinstance(nested_initial, Mapping) else None
    )
    nested_challenger = row.get("deterministic_recommendation")
    challenger_family = row.get("deterministic_method") or (
        nested_challenger.get("recommended_method") if isinstance(nested_challenger, Mapping) else None
    )
    initial_valid = _bool_value(
        row.get("agent_initial_valid")
        if row.get("agent_initial_valid") is not None
        else row.get("initial_plan_valid")
    )
    if initial_valid is None and isinstance(nested_initial, Mapping):
        initial_valid = _bool_value(nested_initial.get("valid"))
    challenger_valid = _bool_value(
        row.get("deterministic_valid")
        if row.get("deterministic_valid") is not None
        else row.get("deterministic_challenger_valid", row.get("challenger_valid"))
    )
    if challenger_valid is None:
        challenger_valid = _bool_value(row.get("deterministic_challenger_valid"))
    if challenger_valid is None and isinstance(nested_challenger, Mapping):
        challenger_valid = _bool_value(nested_challenger.get("valid"))
    hard_validation = row.get("hard_validation")
    if isinstance(hard_validation, Mapping):
        if initial_valid is None and isinstance(hard_validation.get("initial_proposal"), Mapping):
            initial_valid = _bool_value(hard_validation["initial_proposal"].get("valid"))
        if challenger_valid is None and isinstance(hard_validation.get("deterministic_challenger"), Mapping):
            challenger_valid = _bool_value(hard_validation["deterministic_challenger"].get("valid"))
    return bool(
        initial_valid is True
        and challenger_valid is True
        and initial_family
        and challenger_family
        and initial_family != challenger_family
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object in {path} line {number}")
        value["_source_file"] = str(path)
        rows.append(value)
    return rows


def _supported_schema(row: dict[str, Any]) -> bool:
    value = (
        row.get("trial_schema_version")
        or row.get("schema_version")
        or row.get("mlsys_trial_schema_version")
    )
    if value is None:
        # Existing rows remain useful as input when the run was created before
        # the prospective schema stamp was introduced.  The new runner writes
        # the explicit value; this fallback keeps the validator diagnostic.
        return bool(row.get("result_schema_version") or row.get("ablation_schema_version"))
    return str(value) in SUPPORTED_MLSYS_TRIAL_SCHEMA_VERSIONS


def _row_hashes(row: dict[str, Any], config: dict[str, Any], panel_hash: str | None) -> list[str]:
    errors: list[str] = []
    metadata = _config_metadata(config)
    expected_panel = metadata.get("prospective_panel_content_sha256") or metadata.get("panel_hash") or panel_hash
    if expected_panel:
        observed = (
            row.get("prospective_panel_content_sha256")
            or row.get("panel_hash")
            or row.get("prospective_panel_manifest_sha256")
        )
        if not observed:
            errors.append("panel hash missing")
        elif observed not in {expected_panel, panel_hash}:
            errors.append("panel hash mismatch")
    expected_config = (
        metadata.get("experiment_config_sha256")
        or metadata.get("prospective_experiment_config_sha256")
        or metadata.get("config_sha256")
    )
    row_metadata = row.get("run_metadata")
    observed_config = (
        row.get("experiment_config_sha256")
        or row.get("prospective_experiment_config_sha256")
        or row.get("config_sha256")
        or (row_metadata.get("experiment_config_sha256") if isinstance(row_metadata, Mapping) else None)
        or (row_metadata.get("prospective_experiment_config_sha256") if isinstance(row_metadata, Mapping) else None)
    )
    if expected_config:
        if not observed_config:
            errors.append("config hash missing")
        elif observed_config != expected_config:
            errors.append("config hash mismatch")
    return errors


def _manifest_info(run_dir: Path) -> tuple[dict[str, Any], str | None, list[str]]:
    candidates = [
        run_dir / "prospective_task_panel_manifest.json",
        run_dir / "panel_manifest.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        manifest = _read_json(path)
        if not isinstance(manifest, dict):
            return {}, None, ["panel manifest is not a JSON object"]
        panel_hash = manifest.get("content_sha256") or manifest.get("panel_hash")
        return manifest, str(panel_hash) if panel_hash else None, []
    return {}, None, ["prospective panel manifest copy is missing"]


def _config_info(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    path = run_dir / "config.json"
    if not path.is_file():
        return {}, ["run config.json is missing"]
    value = _read_json(path)
    if not isinstance(value, dict):
        return {}, ["run config.json is not an object"]
    return value, []


def validate_mlsys_run(run_dir: str | Path, *, strict: bool = False) -> dict[str, Any]:
    """Return a machine-readable completeness report for one prospective run."""

    root = Path(run_dir).resolve()
    config, errors = _config_info(root)
    manifest, manifest_hash, manifest_errors = _manifest_info(root)
    errors.extend(manifest_errors)
    trial_files = sorted(root.rglob("trials.jsonl")) if root.exists() else []
    if not trial_files:
        errors.append("no trials.jsonl files found")

    rows: list[dict[str, Any]] = []
    trial_id_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in trial_files:
        for row in _read_rows(path):
            rows.append(row)
            trial_id = row.get("trial_id")
            if trial_id:
                trial_id_rows[str(trial_id)].append(row)

    conflicting_trial_ids: list[str] = []
    for trial_id, values in trial_id_rows.items():
        if len(values) < 2:
            continue
        fingerprints = {
            canonical_sha256({key: value for key, value in row.items() if key != "_source_file"})
            for row in values
        }
        if len(fingerprints) > 1 or sum(row.get("trial_status") != "failed" for row in values) > 1:
            conflicting_trial_ids.append(trial_id)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("trial_status") == "failed":
            continue
        groups[trial_identity(row)].append(row)

    prospective_mode = _is_prospective_config(config)
    config_metadata = _config_metadata(config)
    if prospective_mode and not (
        config_metadata.get("experiment_config_sha256")
        or config_metadata.get("prospective_experiment_config_sha256")
    ):
        errors.append("prospective experiment config hash is missing")
    declared_matrix, matrix_errors = _declared_matrix(config, manifest)
    if prospective_mode:
        errors.extend(matrix_errors)

    group_reports: list[dict[str, Any]] = []
    missing_source_arms = 0
    missing_pairwise = 0
    groups_requiring_pairwise = 0
    groups_with_pairwise = 0
    legitimate_no_probe = 0
    incomplete_four = 0
    hash_mismatches = 0
    missing_holdout = 0
    unsupported_schema = 0
    missing_preprocessing = 0
    split_mismatches = (
        sum(
            len({(row.get("split_seed"), row.get("split_random_state")) for row in values}) > 1
            for values in groups.values()
        )
        if prospective_mode and declared_matrix
        else 0
    )

    if prospective_mode and declared_matrix:
        expected_group_inputs = []
        for expected in declared_matrix:
            matching_rows = [
                row for row in rows
                if row.get("trial_status") != "failed" and _row_matches_declared_group(row, expected)
            ]
            expected_identity = (
                "declared",
                expected.get("model_condition_id"),
                expected.get("dataset_id"),
                expected.get("dataset_version"),
                expected.get("split_seed"),
                expected.get("llm_repetition_id"),
            )
            expected_group_inputs.append((expected_identity, matching_rows, expected))
        matched_observed_identities = {
            trial_identity(row)
            for _, matching_rows, _ in expected_group_inputs
            for row in matching_rows
        }
        unexpected_observed = sorted(
            (identity_label(identity) for identity in groups if identity not in matched_observed_identities),
            key=str,
        )
        if unexpected_observed:
            errors.append(
                "observed logical groups outside declared matrix: "
                + ", ".join(unexpected_observed)
            )
    else:
        expected_group_inputs = [
            (identity, group_rows, None)
            for identity, group_rows in sorted(groups.items(), key=lambda item: str(item[0]))
        ]

    for identity, group_rows, expected in expected_group_inputs:
        by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            arm = source_arm(row)
            if arm:
                by_arm[arm].append(row)
        group_missing = [arm for arm in MLSYS_SOURCE_ARMS if not by_arm.get(arm)]
        missing_source_arms += len(group_missing)
        probe_direct_row = next(iter(by_arm.get("probe_direct", [])), None)
        pair_row = next((row for row in group_rows if has_pairwise_evidence(row)), None)
        four_row = next((row for row in group_rows if has_four_family_reference(row)), None)
        pairwise_required = _pairwise_probe_required(probe_direct_row)
        missing_pair = pairwise_required and pair_row is None
        groups_requiring_pairwise += int(pairwise_required)
        groups_with_pairwise += int(pairwise_required and pair_row is not None)
        legitimate_no_probe += int(bool(probe_direct_row) and not pairwise_required and pair_row is None)
        missing_four = four_row is None
        missing_pairwise += int(missing_pair)
        incomplete_four += int(missing_four)
        group_hash_errors = [error for row in group_rows for error in _row_hashes(row, config, manifest_hash)]
        hash_mismatches += len(group_hash_errors)
        schemas_ok = all(_supported_schema(row) for row in group_rows)
        unsupported_schema += int(not schemas_ok)
        holdout_missing_arms = [arm for arm in MLSYS_SOURCE_ARMS if not any(has_holdout(row) for row in by_arm.get(arm, []))]
        missing_holdout += len(holdout_missing_arms)
        plans_ok = any(has_initial_plan(row) for row in group_rows)
        if not plans_ok:
            missing_preprocessing += 1
        split_values = {
            (row.get("split_seed"), row.get("split_random_state"))
            for row in group_rows
        }
        split_mismatch = len(split_values) > 1
        if not (prospective_mode and declared_matrix):
            split_mismatches += int(split_mismatch)
        group_reports.append({
            "logical_trial_id": identity_label(identity),
            "identity": list(identity),
            "expected_group": expected,
            "observed_rows": len(group_rows),
            "completely_missing": not bool(group_rows),
            "source_arms": {arm: len(by_arm.get(arm, [])) for arm in MLSYS_SOURCE_ARMS},
            "missing_source_arms": group_missing,
            "initial_plan": plans_ok,
            "pairwise_probe_required": pairwise_required,
            "pairwise_evidence_observed": pair_row is not None,
            "pairwise_evidence": not missing_pair,
            "all_four_family_reference": not missing_four,
            "holdout_missing_arms": holdout_missing_arms,
            "schema_supported": schemas_ok,
            "split_mismatch": split_mismatch,
            "hash_errors": sorted(set(group_hash_errors)),
        })

    expected_groups = len(declared_matrix) if prospective_mode and declared_matrix else len(group_reports)
    complete_groups = sum(
        not report["missing_source_arms"]
        and report["initial_plan"]
        and report["pairwise_evidence"]
        and report["all_four_family_reference"]
        and not report["holdout_missing_arms"]
        and report["schema_supported"]
        and not report["split_mismatch"]
        and not report["hash_errors"]
        for report in group_reports
    )
    completely_missing_groups = sum(report["completely_missing"] for report in group_reports)
    incomplete_observed_groups = sum(
        bool(report["observed_rows"]) and not (
            not report["missing_source_arms"]
            and report["initial_plan"]
            and report["pairwise_evidence"]
            and report["all_four_family_reference"]
            and not report["holdout_missing_arms"]
            and report["schema_supported"]
            and not report["split_mismatch"]
            and not report["hash_errors"]
        )
        for report in group_reports
    )
    if conflicting_trial_ids:
        errors.append(f"conflicting duplicate trial IDs: {', '.join(sorted(conflicting_trial_ids))}")
    if config.get("run_status") in {"failed", "incomplete/interrupted"}:
        errors.append(f"run status is {config['run_status']!r}")
    ready = bool(
        expected_groups > 0
        and complete_groups == expected_groups
        and not conflicting_trial_ids
        and not errors
        and missing_source_arms == 0
        and missing_pairwise == 0
        and incomplete_four == 0
        and hash_mismatches == 0
        and missing_holdout == 0
        and unsupported_schema == 0
        and split_mismatches == 0
    )
    report = {
        "validator_schema_version": "mlsys-run-validator-v1",
        "run_directory": str(root),
        "run_status": config.get("run_status"),
        "expected_logical_trial_groups": expected_groups,
        "observed_logical_trial_groups": len(groups),
        "completely_missing_logical_groups": completely_missing_groups,
        "incomplete_observed_logical_groups": incomplete_observed_groups,
        "complete_logical_trial_groups": complete_groups,
        "missing_source_arms": missing_source_arms,
        "missing_pairwise_evidence": missing_pairwise,
        "groups_requiring_pairwise_evidence": groups_requiring_pairwise,
        "groups_with_valid_pairwise_evidence": groups_with_pairwise,
        "legitimate_no_probe_groups": legitimate_no_probe,
        "incomplete_four_family_references": incomplete_four,
        "hash_mismatches": hash_mismatches,
        "conflicting_trial_ids": len(conflicting_trial_ids),
        "conflicting_trial_id_values": sorted(conflicting_trial_ids),
        "missing_holdout_results": missing_holdout,
        "unsupported_schema_versions": unsupported_schema,
        "missing_initial_plan_or_preprocessing": missing_preprocessing,
        "split_identity_mismatches": split_mismatches,
        "panel_hash": manifest_hash,
        "panel_manifest_present": bool(manifest),
        "errors": sorted(set(errors)),
        "groups": group_reports,
        "ready_for_baseline_derivation": ready,
        "READY_FOR_BASELINE_DERIVATION": "YES" if ready else "NO",
    }
    report_path = root / "mlsys_validation_report.json"
    if root.exists():
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        (root / "mlsys_validation_report.txt").write_text(render_summary(report), encoding="utf-8")
    if strict and not ready:
        raise MlsysValidationError(report)
    return report


class MlsysValidationError(ValueError):
    """Raised by strict validation when baseline derivation is not safe."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(render_summary(report))


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Expected logical trial groups: {report.get('expected_logical_trial_groups', 0)}",
        f"Observed logical trial groups: {report.get('observed_logical_trial_groups', 0)}",
        f"Completely missing groups: {report.get('completely_missing_logical_groups', 0)}",
        f"Incomplete observed groups: {report.get('incomplete_observed_logical_groups', 0)}",
        f"Complete groups: {report.get('complete_logical_trial_groups', 0)}",
        f"Missing source arms: {report.get('missing_source_arms', 0)}",
        f"Groups requiring pairwise evidence: {report.get('groups_requiring_pairwise_evidence', 0)}",
        f"Groups with valid pairwise evidence: {report.get('groups_with_valid_pairwise_evidence', 0)}",
        f"Legitimate no-probe groups: {report.get('legitimate_no_probe_groups', 0)}",
        f"Missing pairwise evidence: {report.get('missing_pairwise_evidence', 0)}",
        f"Missing required pairwise evidence: {report.get('missing_pairwise_evidence', 0)}",
        f"Incomplete four-family references: {report.get('incomplete_four_family_references', 0)}",
        f"Hash mismatches: {report.get('hash_mismatches', 0)}",
        f"Conflicting trial IDs: {report.get('conflicting_trial_ids', 0)}",
        f"READY_FOR_BASELINE_DERIVATION: {report.get('READY_FOR_BASELINE_DERIVATION', 'NO')}",
    ]
    if report.get("errors"):
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a prospective MLSys run for baseline derivation.")
    parser.add_argument("run_dir")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = validate_mlsys_run(args.run_dir, strict=args.strict)
    except MlsysValidationError as exc:
        print(render_summary(exc.report), end="")
        return 1
    print(render_summary(report), end="")
    return 0 if report["ready_for_baseline_derivation"] or not args.strict else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["MlsysValidationError", "main", "render_summary", "validate_mlsys_run"]
