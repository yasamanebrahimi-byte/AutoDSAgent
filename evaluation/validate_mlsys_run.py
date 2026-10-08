"""Validate a prospective MLSys run before conventional baseline derivation.

The validator is intentionally strict about missing scientific evidence.  It
never reconstructs a missing pairwise comparison or four-family reference
from holdout summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    expected_panel = config.get("prospective_panel_content_sha256") or config.get("panel_hash") or panel_hash
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
    expected_config = config.get("config_sha256") or config.get("experiment_config_sha256")
    observed_config = row.get("config_sha256") or row.get("experiment_config_sha256")
    if expected_config and observed_config and observed_config != expected_config:
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

    group_reports: list[dict[str, Any]] = []
    missing_source_arms = 0
    missing_pairwise = 0
    incomplete_four = 0
    hash_mismatches = 0
    missing_holdout = 0
    unsupported_schema = 0
    missing_preprocessing = 0
    split_mismatches = 0

    for identity, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            arm = source_arm(row)
            if arm:
                by_arm[arm].append(row)
        group_missing = [arm for arm in MLSYS_SOURCE_ARMS if not by_arm.get(arm)]
        missing_source_arms += len(group_missing)
        pair_row = next((row for row in group_rows if has_pairwise_evidence(row)), None)
        four_row = next((row for row in group_rows if has_four_family_reference(row)), None)
        missing_pair = pair_row is None
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
        split_mismatches += int(split_mismatch)
        group_reports.append({
            "logical_trial_id": identity_label(identity),
            "identity": list(identity),
            "source_arms": {arm: len(by_arm.get(arm, [])) for arm in MLSYS_SOURCE_ARMS},
            "missing_source_arms": group_missing,
            "initial_plan": plans_ok,
            "pairwise_evidence": not missing_pair,
            "all_four_family_reference": not missing_four,
            "holdout_missing_arms": holdout_missing_arms,
            "schema_supported": schemas_ok,
            "split_mismatch": split_mismatch,
            "hash_errors": sorted(set(group_hash_errors)),
        })

    expected_groups = len(group_reports)
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
        "complete_logical_trial_groups": complete_groups,
        "missing_source_arms": missing_source_arms,
        "missing_pairwise_evidence": missing_pairwise,
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
        f"Complete groups: {report.get('complete_logical_trial_groups', 0)}",
        f"Missing source arms: {report.get('missing_source_arms', 0)}",
        f"Missing pairwise evidence: {report.get('missing_pairwise_evidence', 0)}",
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
