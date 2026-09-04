"""Prefetch and validate the frozen external OpenML benchmark manifest.

This utility only downloads data and records schema metadata.  It never
creates an agent, trains a model, or calls OpenAI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.external_benchmarks import (
    CANONICAL_TARGET_COLUMN,
    EXTERNAL_BENCHMARK_SUITE_VERSION,
    OpenMLBenchmarkData,
    OpenMLBenchmarkSpec,
    external_benchmark_specs,
    load_openml_task_data,
)


def _repository_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _column_counts(features: pd.DataFrame) -> tuple[int, int]:
    numeric = sum(pd.api.types.is_numeric_dtype(features[column]) for column in features.columns)
    categorical_or_object = sum(
        pd.api.types.is_object_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(features[column])
        for column in features.columns
    )
    return int(numeric), int(categorical_or_object)


def _success_entry(spec: OpenMLBenchmarkSpec, data: OpenMLBenchmarkData) -> dict[str, Any]:
    frame = data.frame
    features = frame.drop(columns=[CANONICAL_TARGET_COLUMN])
    numeric_columns, categorical_columns = _column_counts(features)
    target = frame[CANONICAL_TARGET_COLUMN]
    return {
        "external_suite_version": EXTERNAL_BENCHMARK_SUITE_VERSION,
        "task_id": spec.task_id,
        "openml_dataset_id": data.dataset_id,
        "dataset_name": data.dataset_name,
        "manifest_name": spec.name,
        "original_openml_target_name": data.original_target_name,
        "expected_rows": spec.expected_rows,
        "actual_rows": int(len(frame)),
        "expected_features": spec.expected_features,
        "actual_features": int(
            frame.shape[1] if spec.feature_count_includes_target else features.shape[1]
        ),
        "task_type": spec.expected_task_type,
        "expected_classes": spec.expected_classes,
        "actual_target_class_count": data.observed_classes,
        "numeric_column_count": numeric_columns,
        "categorical_or_object_column_count": categorical_columns,
        "missing_feature_value_count": int(features.isna().sum().sum()),
        "target_missing_count": int(target.isna().sum()),
        "dataframe_dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "validation_success": True,
        "error": None,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _failure_entry(spec: OpenMLBenchmarkSpec, error: Exception) -> dict[str, Any]:
    return {
        "external_suite_version": EXTERNAL_BENCHMARK_SUITE_VERSION,
        "task_id": spec.task_id,
        "openml_dataset_id": None,
        "dataset_name": spec.name,
        "manifest_name": spec.name,
        "original_openml_target_name": None,
        "expected_rows": spec.expected_rows,
        "actual_rows": None,
        "expected_features": spec.expected_features,
        "actual_features": None,
        "task_type": spec.expected_task_type,
        "expected_classes": spec.expected_classes,
        "actual_target_class_count": None,
        "numeric_column_count": None,
        "categorical_or_object_column_count": None,
        "missing_feature_value_count": None,
        "target_missing_count": None,
        "dataframe_dtypes": None,
        "validation_success": False,
        "error": f"{type(error).__name__}: {error}",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def prefetch(
    output_path: str | Path,
    *,
    case_names: list[str] | None = None,
    tier: str | None = None,
    fail_fast: bool = False,
) -> tuple[dict[str, Any], int]:
    """Prefetch selected tasks and write a non-performance manifest."""

    wanted = set(case_names or [])
    specs = [
        spec
        for spec in external_benchmark_specs()
        if (not wanted or spec.name in wanted) and (tier is None or spec.tier == tier)
    ]
    if not specs:
        raise ValueError("No external benchmark tasks selected.")
    entries: list[dict[str, Any]] = []
    for spec in specs:
        try:
            entries.append(_success_entry(spec, load_openml_task_data(spec)))
            print(f"validated task {spec.task_id} ({spec.name})")
        except Exception as exc:
            entries.append(_failure_entry(spec, exc))
            print(f"FAILED task {spec.task_id} ({spec.name}): {type(exc).__name__}: {exc}")
            if fail_fast:
                break
    failures = [entry for entry in entries if not entry["validation_success"]]
    payload = {
        "external_suite_version": EXTERNAL_BENCHMARK_SUITE_VERSION,
        "source": {
            "name": "AMLB/OpenML",
            "classification_suite_id": 271,
            "regression_suite_id": 269,
            "canonical_identifier": "OpenML task ID",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _repository_commit(),
        "task_count_requested": len(specs),
        "task_count_recorded": len(entries),
        "success_count": len(entries) - len(failures),
        "failure_count": len(failures),
        "tasks": entries,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        print(f"{len(failures)} external task(s) failed; see {destination}")
    else:
        print(f"Validated {len(entries)} external task(s); manifest written to {destination}")
    return payload, len(failures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="runs/external_dataset_manifest.json",
        help="Manifest JSON output path.",
    )
    parser.add_argument("--case", action="append", dest="case_names", help="Task name; repeatable.")
    parser.add_argument("--tier", choices=("core", "stress"), help="Optional tier filter.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first download or validation failure.",
    )
    args = parser.parse_args()
    _, failures = prefetch(
        args.output,
        case_names=args.case_names,
        tier=args.tier,
        fail_fast=args.fail_fast,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
