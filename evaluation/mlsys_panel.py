"""Deterministic draft-panel construction for the prospective MLSys study."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from evaluation.external_benchmarks import (
    EXTERNAL_BENCHMARK_MANIFEST,
    PROSPECTIVE_ANALYSIS_ROLE,
    PROSPECTIVE_PANEL_MANIFEST_SCHEMA_VERSION,
    prospective_panel_content_sha256,
)
from evaluation.mlsys_prospective import (
    MLSYS_PROSPECTIVE_CONFIG_SCHEMA_VERSION,
    manifest_hash_without_hash_fields,
)


@dataclass(frozen=True)
class PanelEligibility:
    min_rows: int = 100
    max_rows: int | None = 100_000
    min_features: int = 2
    max_features: int | None = 500
    max_classes: int | None = 50
    max_missing_fraction: float | None = 0.50
    allowed_task_types: tuple[str, ...] = ("classification", "regression")

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "min_features": self.min_features,
            "max_features": self.max_features,
            "max_classes": self.max_classes,
            "max_missing_fraction": self.max_missing_fraction,
            "allowed_task_types": list(self.allowed_task_types),
        }


@dataclass(frozen=True)
class PanelBuildOptions:
    classification_count: int = 20
    regression_count: int = 20
    selection_seed: int = 20261008
    panel_id: str = "mlsys-prospective-draft"
    eligibility: PanelEligibility = field(default_factory=PanelEligibility)


def _value(item: Any, *names: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if item.get(name) is not None:
                return item[name]
    else:
        for name in names:
            value = getattr(item, name, None)
            if value is not None:
                return value
    return default


def _import_openml() -> Any:
    try:
        import openml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError("The panel builder requires the optional 'openml' dependency.") from exc
    return openml


def _historical_exclusions(openml_module: Any | None = None) -> dict[str, set[Any]]:
    exclusions = {
        "task_ids": {spec.task_id for spec in EXTERNAL_BENCHMARK_MANIFEST},
        "dataset_names": {spec.name.strip().casefold() for spec in EXTERNAL_BENCHMARK_MANIFEST},
        "dataset_ids": set(),
    }
    if openml_module is not None:
        for spec in EXTERNAL_BENCHMARK_MANIFEST:
            try:
                task = openml_module.tasks.get_task(spec.task_id, download_splits=False)
                dataset_id = _as_int(_value(task, "dataset_id", "did"))
                if dataset_id is not None:
                    exclusions["dataset_ids"].add(dataset_id)
            except Exception:
                # The task ID and historical dataset name remain hard
                # exclusions even when an old task is no longer resolvable.
                continue
    return exclusions


def _records_from_openml(openml_module: Any) -> list[Any]:
    tasks = openml_module.tasks
    try:
        values = tasks.list_tasks(output_format="dataframe")
    except TypeError:
        values = tasks.list_tasks()
    if hasattr(values, "to_dict"):
        try:
            return values.to_dict(orient="records")
        except TypeError:
            pass
    if isinstance(values, Mapping):
        return list(values.values()) if all(isinstance(v, Mapping) for v in values.values()) else [values]
    return list(values)


def _task_type(record: Any) -> str | None:
    value = _value(record, "task_type", "type", "task_type_name", "supervised_task_type")
    if value is not None:
        text = str(value).casefold()
        if "class" in text:
            return "classification"
        if "regress" in text:
            return "regression"
    task_type_id = _value(record, "task_type_id")
    if task_type_id in {"1", 1}:
        return "classification"
    if task_type_id in {"2", 2}:
        return "regression"
    return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        result = int(value)
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None


def _enrich_record(record: Any, openml_module: Any | None) -> dict[str, Any]:
    task_id = _as_int(_value(record, "task_id", "tid", "id"))
    task = record
    if task_id is not None and openml_module is not None and not _value(record, "dataset_id", "did"):
        try:
            task = openml_module.tasks.get_task(task_id, download_splits=False)
        except TypeError:
            task = openml_module.tasks.get_task(task_id)
    dataset = None
    get_dataset = getattr(task, "get_dataset", None)
    if callable(get_dataset):
        try:
            dataset = get_dataset()
        except Exception as exc:  # pragma: no cover - depends on remote task availability
            return {"task_id": task_id, "rejection_reasons": [f"dataset_metadata_unavailable:{type(exc).__name__}"]}
    dataset_id = _as_int(_value(record, "dataset_id", "did")) or _as_int(_value(task, "dataset_id", "did"))
    if dataset_id is None:
        dataset_id = _as_int(_value(dataset, "dataset_id", "did", "id"))
    name = _value(record, "dataset_name", "name") or _value(task, "dataset_name", "name") or _value(dataset, "name")
    version = _value(record, "dataset_version", "version") or _value(dataset, "version")
    target = _value(record, "target", "target_name", "target_attribute") or _value(task, "target_name") or _value(dataset, "default_target_attribute")
    task_type = _task_type(record) or _task_type(task)
    rows = _as_int(_value(record, "rows", "num_instances", "NumberOfInstances", "n_rows"))
    features = _as_int(_value(record, "features", "num_features", "NumberOfFeatures", "n_features"))
    classes = _as_int(_value(record, "classes", "num_classes", "NumberOfClasses", "n_classes"))
    frame = _value(record, "frame", "dataframe")
    target_values = _value(record, "target_values", "y")
    if (rows is None or features is None or classes is None and task_type == "classification") and dataset is not None:
        get_data = getattr(dataset, "get_data", None)
        if callable(get_data):
            try:
                values = get_data(target=target, dataset_format="dataframe")
                frame = values[0] if frame is None else frame
                target_values = values[1] if target_values is None else target_values
            except TypeError:
                try:
                    values = get_data(target=target)
                    frame = values[0] if frame is None else frame
                    target_values = values[1] if target_values is None else target_values
                except Exception:
                    pass
            except Exception:
                pass
    if frame is not None:
        rows = rows or _as_int(len(frame))
        features = features or _as_int(getattr(frame, "shape", [0, 0])[1])
        if target is not None and hasattr(frame, "columns") and target in frame.columns:
            features = features - 1 if features and features > 1 else features
        if classes is None and target_values is not None and task_type == "classification":
            try:
                classes = _as_int(target_values.nunique(dropna=True))
            except AttributeError:
                classes = _as_int(len(set(target_values)))
    if version is None:
        version = "unknown"
    return {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "dataset_name": str(name or ""),
        "dataset_version": version,
        "task_type": task_type,
        "target": str(target or ""),
        "expected_rows": rows,
        "expected_features": features,
        "expected_classes": classes if task_type == "classification" else None,
        "frame": frame,
        "target_values": target_values,
        "source_record": record,
        "rejection_reasons": [],
    }


def _reject(record: dict[str, Any], reason: str) -> dict[str, Any]:
    result = {
        key: value for key, value in record.items()
        if key not in {"frame", "target_values", "source_record"}
    }
    result.setdefault("rejection_reasons", []).append(reason)
    return result


def build_draft_panel(
    *,
    options: PanelBuildOptions | None = None,
    task_records: Iterable[Any] | None = None,
    openml_module: Any | None = None,
    historical_exclusions: Mapping[str, Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, never-frozen prospective panel manifest."""

    configured = options or PanelBuildOptions()
    if configured.classification_count < 0 or configured.regression_count < 0:
        raise ValueError("Requested panel counts must be non-negative.")
    if task_records is None:
        openml_module = openml_module or _import_openml()
        task_records = _records_from_openml(openml_module)
    exclusions = _historical_exclusions(openml_module)
    if historical_exclusions:
        for key, values in historical_exclusions.items():
            exclusions[key] = set(values)
    normalized_exclusions = {
        "task_ids": sorted(int(value) for value in exclusions.get("task_ids", set())),
        "dataset_ids": sorted(int(value) for value in exclusions.get("dataset_ids", set())),
        "dataset_names": sorted(str(value).casefold() for value in exclusions.get("dataset_names", set())),
    }
    records = sorted(
        (_enrich_record(item, openml_module) for item in task_records),
        key=lambda value: (
            str(value.get("task_id")),
            str(value.get("dataset_id")),
            str(value.get("dataset_version")),
            str(value.get("dataset_name")),
        ),
    )
    eligible: dict[str, list[dict[str, Any]]] = {"classification": [], "regression": []}
    rejected: list[dict[str, Any]] = []
    seen_versions: set[tuple[Any, Any]] = set()
    eligibility = configured.eligibility
    for record in records:
        reasons: list[str] = list(record.get("rejection_reasons", []))
        task_id = record.get("task_id")
        dataset_id = record.get("dataset_id")
        name = str(record.get("dataset_name", ""))
        task_type = record.get("task_type")
        version_key = (dataset_id, record.get("dataset_version"))
        if task_id in exclusions.get("task_ids", set()):
            reasons.append("historical_task_id")
        if dataset_id in exclusions.get("dataset_ids", set()):
            reasons.append("historical_dataset_id")
        if name.casefold() in exclusions.get("dataset_names", set()):
            reasons.append("historical_dataset_name")
        if task_type not in eligibility.allowed_task_types:
            reasons.append("not_supervised_classification_or_regression")
        if task_id is None or dataset_id is None:
            reasons.append("missing_openml_task_or_dataset_id")
        if not str(record.get("target", "")).strip():
            reasons.append("missing_target")
        if record.get("dataset_version") in (None, "", "unknown"):
            reasons.append("missing_dataset_version")
        rows = record.get("expected_rows")
        features = record.get("expected_features")
        if rows is None or rows < eligibility.min_rows:
            reasons.append("rows_below_minimum")
        if eligibility.max_rows is not None and rows is not None and rows > eligibility.max_rows:
            reasons.append("rows_above_maximum")
        if features is None or features < eligibility.min_features:
            reasons.append("features_below_minimum")
        if eligibility.max_features is not None and features is not None and features > eligibility.max_features:
            reasons.append("features_above_maximum")
        if task_type == "classification":
            classes = record.get("expected_classes")
            if classes is None or classes < 2:
                reasons.append("classification_class_count_unavailable")
            elif eligibility.max_classes is not None and classes > eligibility.max_classes:
                reasons.append("too_many_classes")
        if eligibility.max_missing_fraction is not None:
            frame = record.get("frame")
            missing_fraction = getattr(frame, "isna", lambda: None)()
            if missing_fraction is not None:
                try:
                    observed_missing = float(missing_fraction.to_numpy().mean())
                    if observed_missing > eligibility.max_missing_fraction:
                        reasons.append("missing_fraction_above_maximum")
                except (AttributeError, TypeError, ValueError):
                    pass
        if version_key in seen_versions:
            reasons.append("duplicate_dataset_version")
        if not reasons:
            seen_versions.add(version_key)
            eligible[task_type].append(record)
        else:
            rejected.append(_reject({**record, "rejection_reasons": reasons}, "ineligible"))

    selected: list[dict[str, Any]] = []
    rng = np.random.default_rng(configured.selection_seed)
    requested = {"classification": configured.classification_count, "regression": configured.regression_count}
    for task_type in ("classification", "regression"):
        candidates = sorted(
            eligible[task_type],
            key=lambda value: (int(value["task_id"]), int(value["dataset_id"]), str(value["dataset_version"])),
        )
        order = rng.permutation(len(candidates))
        take = min(requested[task_type], len(candidates))
        chosen_indexes = set(int(index) for index in order[:take])
        for index, record in enumerate(candidates):
            if index in chosen_indexes:
                selected.append(record)
            else:
                rejected.append(_reject({**record, "rejection_reasons": ["not_selected_after_seeded_sampling"]}, "not_selected"))
    selected.sort(key=lambda value: (value["task_type"], int(value["task_id"])))
    task_entries: list[dict[str, Any]] = []
    for draw, record in enumerate(selected, start=1):
        task_entries.append({
            "schema_version": MLSYS_PROSPECTIVE_CONFIG_SCHEMA_VERSION,
            "task_id": int(record["task_id"]),
            "dataset_id": int(record["dataset_id"]),
            "dataset_name": record["dataset_name"],
            "dataset_version": record["dataset_version"],
            "task_type": record["task_type"],
            "target": record["target"],
            "expected_rows": int(record["expected_rows"]),
            "expected_features": int(record["expected_features"]),
            "expected_classes": record.get("expected_classes"),
            "selection_metadata": {
                "selection_seed": configured.selection_seed,
                "draw_index": draw,
                "sampling_order": "seeded_permutation_after_canonical_sort",
            },
            "provenance": {
                "source": "OpenML task listing and task metadata",
                "builder": "scripts/build_mlsys_prospective_panel.py",
                "historical_panel_excluded": True,
            },
        })
    manifest: dict[str, Any] = {
        "manifest_schema_version": PROSPECTIVE_PANEL_MANIFEST_SCHEMA_VERSION,
        "schema_version": MLSYS_PROSPECTIVE_CONFIG_SCHEMA_VERSION,
        "status": "draft",
        "analysis_role": PROSPECTIVE_ANALYSIS_ROLE,
        "panel_id": configured.panel_id,
        "panel_selection": {
            "sampling_frame": "OpenML supervised tabular classification and regression tasks available at build time",
            "selection_seed": configured.selection_seed,
            "requested_counts": requested,
            "selected_counts": {
                "classification": sum(item["task_type"] == "classification" for item in task_entries),
                "regression": sum(item["task_type"] == "regression" for item in task_entries),
            },
            "eligibility_filters": eligibility.as_dict(),
            "inclusion_criteria": [
                "OpenML supervised tabular classification or regression task",
                "task and dataset metadata are available",
                "dataset version is unique within the draft panel",
                "rows/features/classes satisfy the declared resource filters",
            ],
            "exclusion_criteria": [
                "every task/dataset name or identifier used by the historical panel",
                "missing target or unsupported task type",
                "duplicate dataset version",
                "resource or class-count filter failure",
            ],
            "historical_exclusions": normalized_exclusions,
            "selection_method": "canonical metadata sort followed by deterministic seeded permutation, independently by task type",
        },
        "tasks": task_entries,
        "rejected_tasks": sorted(
            rejected,
            key=lambda value: (str(value.get("task_type")), str(value.get("task_id")), str(value.get("dataset_version"))),
        ),
        "rejection_audit": {
            "candidate_count": len(records),
            "rejected_count": len(rejected),
            "reason_counts": {
                reason: sum(reason in item.get("rejection_reasons", []) for item in rejected)
                for reason in sorted({reason for item in rejected for reason in item.get("rejection_reasons", [])})
            },
        },
        "provenance": {
            "openml_api": "lazy; resolved at build time",
            "builder_schema_version": MLSYS_PROSPECTIVE_CONFIG_SCHEMA_VERSION,
            "historical_manifest": "evaluation.external_benchmarks.EXTERNAL_BENCHMARK_MANIFEST",
        },
    }
    manifest["content_sha256"] = prospective_panel_content_sha256(manifest)
    manifest["panel_hash"] = manifest["content_sha256"]
    manifest["canonical_manifest_sha256"] = manifest_hash_without_hash_fields(manifest)
    return manifest


def write_draft_panel(manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically write a draft manifest; never changes its status."""

    if manifest.get("status") != "draft":
        raise ValueError("The panel builder only writes draft manifests.")
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp")
    temp.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic draft OpenML panel for the MLSys study.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--classification-count", type=int, default=20)
    parser.add_argument("--regression-count", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--panel-id", default="mlsys-prospective-draft")
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--min-features", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--max-classes", type=int, default=50)
    parser.add_argument("--max-missing-fraction", type=float, default=0.50)
    args = parser.parse_args(argv)
    manifest = build_draft_panel(options=PanelBuildOptions(
        classification_count=args.classification_count,
        regression_count=args.regression_count,
        selection_seed=args.selection_seed,
        panel_id=args.panel_id,
        eligibility=PanelEligibility(
            min_rows=args.min_rows,
            max_rows=args.max_rows,
            min_features=args.min_features,
            max_features=args.max_features,
            max_classes=args.max_classes,
            max_missing_fraction=args.max_missing_fraction,
        ),
    ))
    path = write_draft_panel(manifest, args.output)
    print(json.dumps({
        "output": str(path),
        "status": manifest["status"],
        "content_sha256": manifest["content_sha256"],
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "classification_count": manifest["panel_selection"]["selected_counts"]["classification"],
        "regression_count": manifest["panel_selection"]["selected_counts"]["regression"],
    }, indent=2, sort_keys=True))
    return 0


__all__ = [
    "PanelBuildOptions",
    "PanelEligibility",
    "build_draft_panel",
    "main",
    "write_draft_panel",
]
