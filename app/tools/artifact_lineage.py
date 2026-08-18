"""Local artifact lineage and invalidation helpers.

The workflow stores artifacts in stable run subdirectories.  These helpers keep
file reuse honest by pairing artifacts with deterministic lineage metadata and
by invalidating only the known downstream outputs for a changed stage.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from app.tools.file_utils import load_json, save_json
from app.workflows.workflow_steps import (
    CLEANING_PLAN_STEP,
    CLEANING_STEP,
    EDA_STEP,
    MODELING_STEP,
    PROFILE_STEP,
    REPORT_STEP,
    STEP_DEFINITIONS,
)


LINEAGE_VERSION = 1
LINEAGE_SUFFIX = ".lineage.json"
WORKFLOW_STATE_RELATIVE_PATH = "logs/workflow_state.json"

TARGET_SENSITIVE_ARTIFACTS = {
    "cleaning_plan",
    "cleaned_data",
    "cleaning_summary",
    "eda_summary",
    "eda_findings",
    "eda_report",
    "modeling_summary",
    "evaluation_summary",
    "model_results",
    "baseline_model",
    "selected_model",
    "best_model",
    "final_report",
    "executive_summary",
    "technical_summary",
    "limitations_report",
    "report_metadata",
    "report_index",
}
TASK_SENSITIVE_ARTIFACTS = {
    "modeling_summary",
    "evaluation_summary",
    "model_results",
    "baseline_model",
    "selected_model",
    "best_model",
    "final_report",
    "executive_summary",
    "technical_summary",
    "limitations_report",
    "report_metadata",
    "report_index",
}
ANALYSIS_INPUT_ARTIFACTS = {
    "eda_summary",
    "eda_findings",
    "eda_report",
    "modeling_summary",
    "evaluation_summary",
    "model_results",
    "baseline_model",
    "selected_model",
    "best_model",
}
REPORT_ARTIFACT_TYPES = {
    "final_report",
    "executive_summary",
    "technical_summary",
    "limitations_report",
    "report_metadata",
    "report_index",
}

ARTIFACT_RELATIVE_PATHS_BY_STEP: dict[str, dict[str, str]] = {
    PROFILE_STEP: {
        "profile": "intermediate/profile.json",
    },
    CLEANING_PLAN_STEP: {
        "cleaning_plan": "intermediate/cleaning_plan.json",
    },
    CLEANING_STEP: {
        "cleaned_data": "intermediate/cleaned_data.csv",
        "cleaning_summary": "intermediate/cleaning_summary.json",
    },
    EDA_STEP: {
        "eda_summary": "intermediate/eda_summary.json",
        "eda_findings": "intermediate/eda_findings.json",
        "eda_report": "reports/eda_summary.md",
    },
    MODELING_STEP: {
        "modeling_summary": "intermediate/modeling_summary.json",
        "evaluation_summary": "intermediate/evaluation_summary.json",
        "model_results": "models/model_results.json",
        "baseline_model": "models/baseline_model.pkl",
        "selected_model": "models/selected_model.pkl",
        "best_model": "models/best_model.pkl",
    },
    REPORT_STEP: {
        "final_report": "reports/final_report.md",
        "executive_summary": "reports/executive_summary.md",
        "technical_summary": "reports/technical_summary.md",
        "limitations_report": "reports/limitations.md",
        "final_report_html": "reports/final_report.html",
        "report_metadata": "intermediate/report_metadata.json",
        "report_index": "reports/report_index.json",
    },
}

PLOT_DIRECTORIES_BY_STEP: dict[str, tuple[str, ...]] = {
    EDA_STEP: ("plots/eda",),
    MODELING_STEP: ("plots/evaluation",),
}

DOWNSTREAM_STEPS_BY_CHANGED_STEP: dict[str, tuple[str, ...]] = {
    PROFILE_STEP: (
        PROFILE_STEP,
        CLEANING_PLAN_STEP,
        CLEANING_STEP,
        EDA_STEP,
        MODELING_STEP,
        REPORT_STEP,
    ),
    CLEANING_PLAN_STEP: (
        CLEANING_PLAN_STEP,
        CLEANING_STEP,
        EDA_STEP,
        MODELING_STEP,
        REPORT_STEP,
    ),
    CLEANING_STEP: (
        CLEANING_STEP,
        EDA_STEP,
        MODELING_STEP,
        REPORT_STEP,
    ),
    EDA_STEP: (
        EDA_STEP,
        REPORT_STEP,
    ),
    MODELING_STEP: (
        MODELING_STEP,
        REPORT_STEP,
    ),
    REPORT_STEP: (REPORT_STEP,),
}

STEP_FOR_SOURCE_ARTIFACT: dict[str, str] = {
    "profile": PROFILE_STEP,
    "cleaning_plan": CLEANING_PLAN_STEP,
    "cleaning_summary": CLEANING_STEP,
    "eda_summary": EDA_STEP,
    "eda_findings": EDA_STEP,
    "modeling_summary": MODELING_STEP,
    "evaluation_summary": MODELING_STEP,
    "model_results": MODELING_STEP,
}


@dataclass(frozen=True)
class ArtifactValidation:
    """Result of checking whether an artifact is current."""

    is_current: bool
    reason: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class AnalysisInputSelection:
    """Resolved dataset input for EDA or modeling."""

    path: Path
    dataset_used: str
    fingerprint: str
    source_fingerprint: str
    lineage: dict[str, Any] | None
    warnings: list[str]


def new_generation_id() -> str:
    """Return a unique identifier for one logical workflow generation."""

    return uuid4().hex


def utc_now_iso() -> str:
    """Return a timezone-aware timestamp suitable for lineage metadata."""

    return datetime.now(timezone.utc).isoformat()


def normalize_optional_text(value: Any) -> str | None:
    """Normalize optional string-like configuration values."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def file_sha256(path: str | Path) -> str:
    """Compute a SHA-256 fingerprint for a file's bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_payload(value: Any) -> Any:
    """Convert a value into a deterministic JSON-serializable structure."""

    if is_dataclass(value):
        return stable_payload(asdict(value))
    if hasattr(value, "model_dump"):
        return stable_payload(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): stable_payload(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [stable_payload(item) for item in value]
    if isinstance(value, set):
        return [stable_payload(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def fingerprint_payload(payload: Any) -> str:
    """Return a stable SHA-256 fingerprint for a configuration payload."""

    encoded = json.dumps(
        stable_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_lineage_path(artifact_path: str | Path) -> Path:
    """Return the sidecar metadata path for an artifact file."""

    path = Path(artifact_path)
    return path.with_name(f"{path.name}{LINEAGE_SUFFIX}")


def load_artifact_lineage(artifact_path: str | Path) -> dict[str, Any] | None:
    """Load sidecar lineage metadata when present."""

    path = artifact_lineage_path(artifact_path)
    if not path.exists():
        return None
    return load_json(path)


def write_artifact_lineage(
    artifact_path: str | Path,
    *,
    run_root: str | Path,
    run_id: str,
    artifact_type: str,
    generation_id: str | None = None,
    source_fingerprint: str | None = None,
    target_column: str | None = None,
    task_type: str | None = None,
    config_fingerprint: str | None = None,
    upstream_fingerprints: Mapping[str, Any] | None = None,
    relevant_config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write sidecar lineage for a saved artifact and return the payload."""

    artifact = Path(artifact_path)
    root = Path(run_root).resolve()
    resolved_artifact = artifact.resolve()
    if root != resolved_artifact and root not in resolved_artifact.parents:
        raise ValueError("Artifact lineage paths must resolve inside the run directory.")
    if not artifact.exists() or not artifact.is_file():
        raise FileNotFoundError(artifact)

    payload: dict[str, Any] = {
        "lineage_version": LINEAGE_VERSION,
        "run_id": run_id,
        "generation_id": generation_id,
        "artifact_type": artifact_type,
        "artifact_path": resolved_artifact.relative_to(root).as_posix(),
        "artifact_fingerprint": file_sha256(artifact),
        "source_fingerprint": source_fingerprint,
        "target_column": normalize_optional_text(target_column),
        "task_type": normalize_optional_text(task_type),
        "config_fingerprint": config_fingerprint,
        "upstream_fingerprints": stable_payload(upstream_fingerprints or {}),
        "relevant_config": stable_payload(relevant_config or {}),
        "created_at": utc_now_iso(),
    }
    if extra:
        payload.update(stable_payload(dict(extra)))

    save_json(artifact_lineage_path(artifact), payload)
    return payload


def load_active_workflow_state(run_manager: Any, run_id: str) -> dict[str, Any] | None:
    """Load workflow state for a run when one exists."""

    path = run_manager.get_paths(run_id).logs / "workflow_state.json"
    if not path.exists():
        return None
    try:
        state = load_json(path)
    except ValueError:
        return None
    return state if isinstance(state, dict) else None


def lineage_context(
    run_manager: Any,
    run_id: str,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return active lineage context derived from state and raw data."""

    paths = run_manager.get_paths(run_id)
    active_state = dict(state) if state is not None else load_active_workflow_state(run_manager, run_id)
    raw_path = paths.input / "raw_data.csv"
    source_fingerprint = None
    if active_state:
        source_fingerprint = active_state.get("source_fingerprint")
    if source_fingerprint is None and raw_path.exists():
        source_fingerprint = file_sha256(raw_path)

    return {
        "run_id": run_id,
        "generation_id": active_state.get("generation_id") if active_state else None,
        "source_fingerprint": source_fingerprint,
        "target_column": normalize_optional_text(
            active_state.get("target_column") if active_state else None
        ),
        "task_type": normalize_optional_text(
            active_state.get("task_type") if active_state else None
        ),
        "state": active_state,
    }


def validate_artifact_current(
    artifact_path: str | Path,
    *,
    artifact_type: str,
    expected: Mapping[str, Any] | None = None,
    require_lineage: bool = True,
) -> ArtifactValidation:
    """Validate artifact existence and lineage metadata against expected values."""

    path = Path(artifact_path)
    if not path.exists() or not path.is_file():
        return ArtifactValidation(False, "artifact file is missing")

    metadata = load_artifact_lineage(path)
    if metadata is None:
        if require_lineage:
            return ArtifactValidation(False, "lineage metadata is missing")
        return ArtifactValidation(True, None, None)

    if metadata.get("artifact_type") != artifact_type:
        return ArtifactValidation(
            False,
            f"artifact type is {metadata.get('artifact_type')!r}, expected {artifact_type!r}",
            metadata,
        )

    expected_values = dict(expected or {})
    for field in (
        "generation_id",
        "source_fingerprint",
        "target_column",
        "task_type",
        "config_fingerprint",
    ):
        if field not in expected_values:
            continue
        expected_value = expected_values[field]
        actual_value = metadata.get(field)
        if field in {"target_column", "task_type"}:
            actual_value = normalize_optional_text(actual_value)
            expected_value = normalize_optional_text(expected_value)
        if actual_value != expected_value:
            return ArtifactValidation(
                False,
                f"{field} does not match active workflow",
                metadata,
            )

    expected_upstream = expected_values.get("upstream_fingerprints") or {}
    actual_upstream = metadata.get("upstream_fingerprints") or {}
    for key, expected_value in expected_upstream.items():
        if expected_value is None:
            continue
        if actual_upstream.get(key) != expected_value:
            return ArtifactValidation(
                False,
                f"upstream fingerprint {key!r} does not match active workflow",
                metadata,
            )

    return ArtifactValidation(True, None, metadata)


def expected_lineage_for_state(
    artifact_type: str,
    state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build expected lineage fields for an artifact in the active state."""

    if not state:
        return {}

    expected: dict[str, Any] = {}
    if state.get("source_fingerprint") is not None:
        expected["source_fingerprint"] = state.get("source_fingerprint")
    if artifact_type != "metadata":
        if state.get("generation_id") is not None:
            expected["generation_id"] = state.get("generation_id")
    if artifact_type in TARGET_SENSITIVE_ARTIFACTS:
        expected["target_column"] = state.get("target_column")
    if artifact_type in TASK_SENSITIVE_ARTIFACTS and "task_type" in state:
        expected["task_type"] = state.get("task_type")

    analysis_input = state.get("analysis_input") or {}
    input_fingerprint = analysis_input.get("fingerprint")
    if artifact_type in ANALYSIS_INPUT_ARTIFACTS and input_fingerprint:
        expected["upstream_fingerprints"] = {
            str(analysis_input.get("dataset_used") or "analysis_input"): input_fingerprint,
        }

    return expected


def validate_artifact_for_state(
    artifact_path: str | Path,
    *,
    artifact_type: str,
    state: Mapping[str, Any] | None,
    require_lineage_if_stateful: bool = True,
) -> ArtifactValidation:
    """Validate an artifact against active workflow state when state exists."""

    require_lineage = bool(state and require_lineage_if_stateful)
    expected = expected_lineage_for_state(artifact_type, state)
    return validate_artifact_current(
        artifact_path,
        artifact_type=artifact_type,
        expected=expected,
        require_lineage=require_lineage,
    )


def source_fingerprint_for_run(run_manager: Any, run_id: str) -> str:
    """Return the current raw source fingerprint for a run."""

    raw_path = run_manager.get_paths(run_id).input / "raw_data.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    return file_sha256(raw_path)


def select_analysis_input(
    run_manager: Any,
    run_id: str,
    *,
    target_column: str | None = None,
    require_cleaned: bool = False,
    state: Mapping[str, Any] | None = None,
) -> AnalysisInputSelection:
    """Resolve the current dataset for downstream analysis.

    When workflow state exists, that state is authoritative.  A stale
    ``cleaned_data.csv`` never overrides an explicit raw-data selection.
    """

    paths = run_manager.get_paths(run_id)
    raw_path = paths.input / "raw_data.csv"
    active_state = state or load_active_workflow_state(run_manager, run_id)
    cleaned_path = paths.intermediate / "cleaned_data.csv"
    requested_target = normalize_optional_text(target_column)
    if not raw_path.exists():
        if active_state:
            raise FileNotFoundError(raw_path)
        if cleaned_path.exists():
            cleaned_fingerprint = file_sha256(cleaned_path)
            return AnalysisInputSelection(
                path=cleaned_path,
                dataset_used="cleaned",
                fingerprint=cleaned_fingerprint,
                source_fingerprint=cleaned_fingerprint,
                lineage=load_artifact_lineage(cleaned_path),
                warnings=[],
            )
        if require_cleaned:
            raise ValueError("Cleaned dataset was not found. Apply safe cleaning before modeling.")
        raise FileNotFoundError(raw_path)

    source_fingerprint = source_fingerprint_for_run(run_manager, run_id)

    if active_state:
        analysis_input = active_state.get("analysis_input") or {}
        if analysis_input.get("dataset_used") == "raw":
            if require_cleaned:
                raise ValueError(
                    "The active workflow selected raw data; current cleaned data is unavailable."
                )
            return AnalysisInputSelection(
                path=raw_path,
                dataset_used="raw",
                fingerprint=source_fingerprint,
                source_fingerprint=source_fingerprint,
                lineage=None,
                warnings=[],
            )

        if analysis_input.get("dataset_used") == "cleaned":
            expected = expected_lineage_for_state("cleaned_data", active_state)
            if requested_target is not None:
                expected["target_column"] = requested_target
            validation = validate_artifact_current(
                cleaned_path,
                artifact_type="cleaned_data",
                expected=expected,
                require_lineage=True,
            )
            if validation.is_current:
                return AnalysisInputSelection(
                    path=cleaned_path,
                    dataset_used="cleaned",
                    fingerprint=file_sha256(cleaned_path),
                    source_fingerprint=source_fingerprint,
                    lineage=validation.metadata,
                    warnings=[],
                )
            if require_cleaned:
                raise ValueError(f"Current cleaned data is unavailable: {validation.reason}.")
            return AnalysisInputSelection(
                path=raw_path,
                dataset_used="raw",
                fingerprint=source_fingerprint,
                source_fingerprint=source_fingerprint,
                lineage=None,
                warnings=[f"Current cleaned data was ignored: {validation.reason}."],
            )

        cleaning_step = (active_state.get("steps") or {}).get(CLEANING_STEP, {})
        if cleaning_step.get("status") in {"skipped", "failed", "waiting_for_approval"}:
            if require_cleaned:
                raise ValueError("The active workflow does not have completed cleaned data.")
            return AnalysisInputSelection(
                path=raw_path,
                dataset_used="raw",
                fingerprint=source_fingerprint,
                source_fingerprint=source_fingerprint,
                lineage=None,
                warnings=[],
            )

        expected = expected_lineage_for_state("cleaned_data", active_state)
        if requested_target is not None:
            expected["target_column"] = requested_target
        validation = validate_artifact_current(
            cleaned_path,
            artifact_type="cleaned_data",
            expected=expected,
            require_lineage=True,
        )
        if validation.is_current:
            return AnalysisInputSelection(
                path=cleaned_path,
                dataset_used="cleaned",
                fingerprint=file_sha256(cleaned_path),
                source_fingerprint=source_fingerprint,
                lineage=validation.metadata,
                warnings=[],
            )
        if require_cleaned:
            raise ValueError(f"Current cleaned data is unavailable: {validation.reason}.")

    if cleaned_path.exists():
        metadata = load_artifact_lineage(cleaned_path)
        if metadata is None and active_state is None:
            return AnalysisInputSelection(
                path=cleaned_path,
                dataset_used="cleaned",
                fingerprint=file_sha256(cleaned_path),
                source_fingerprint=source_fingerprint,
                lineage=None,
                warnings=[],
            )
        expected = {
            "source_fingerprint": source_fingerprint,
            "target_column": requested_target,
        }
        validation = validate_artifact_current(
            cleaned_path,
            artifact_type="cleaned_data",
            expected=expected,
            require_lineage=True,
        )
        if validation.is_current:
            return AnalysisInputSelection(
                path=cleaned_path,
                dataset_used="cleaned",
                fingerprint=file_sha256(cleaned_path),
                source_fingerprint=source_fingerprint,
                lineage=validation.metadata,
                warnings=[],
            )
        if require_cleaned:
            raise ValueError(f"Current cleaned data is unavailable: {validation.reason}.")

    if require_cleaned:
        raise ValueError("Cleaned dataset was not found. Apply safe cleaning before modeling.")

    return AnalysisInputSelection(
        path=raw_path,
        dataset_used="raw",
        fingerprint=source_fingerprint,
        source_fingerprint=source_fingerprint,
        lineage=None,
        warnings=[],
    )


def state_allows_source_artifact(
    artifact_key: str,
    state: Mapping[str, Any] | None,
) -> ArtifactValidation:
    """Return whether workflow step status allows a source artifact to be used."""

    if not state:
        return ArtifactValidation(True)
    step = STEP_FOR_SOURCE_ARTIFACT.get(artifact_key)
    if step is None:
        return ArtifactValidation(True)
    step_state = (state.get("steps") or {}).get(step, {})
    if step_state.get("status") == "completed":
        return ArtifactValidation(True)
    return ArtifactValidation(
        False,
        f"workflow step {step!r} is not completed",
    )


def invalidate_downstream_artifacts(
    run_manager: Any,
    run_id: str,
    changed_step: str,
    state: dict[str, Any] | None = None,
) -> list[str]:
    """Remove known artifacts for a stage and its dependents.

    Deletion is deliberately scoped to explicit files and plot directories inside
    the run root.  The workflow state and trace logs are not removed here.
    """

    paths = run_manager.get_paths(run_id)
    root = paths.root.resolve()
    steps = DOWNSTREAM_STEPS_BY_CHANGED_STEP.get(changed_step, (changed_step,))
    invalidated: list[str] = []
    if state is not None:
        _clear_invalidated_state(state, steps, changed_step)

    for step in steps:
        for relative_path in ARTIFACT_RELATIVE_PATHS_BY_STEP.get(step, {}).values():
            path = _known_path(root, relative_path)
            invalidated.extend(_remove_artifact_file(path, root))
        for relative_dir in PLOT_DIRECTORIES_BY_STEP.get(step, ()):
            directory = _known_path(root, relative_dir)
            if directory.exists():
                shutil.rmtree(directory)
                invalidated.append(directory.relative_to(root).as_posix())

    return invalidated


def invalidate_downstream_for_manual_mutation(
    run_manager: Any,
    run_id: str,
    changed_step: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Atomically reset state before deleting artifacts for a manual mutation.

    The workflow-state file is replaced before any artifact is removed.  If the
    state write fails, deletion never starts; if deletion later fails, the
    persisted state has already stopped referencing every affected artifact.
    """

    state = load_active_workflow_state(run_manager, run_id)
    if state is None:
        return None, invalidate_downstream_artifacts(run_manager, run_id, changed_step)

    paths = run_manager.get_paths(run_id)
    updated_state = deepcopy(state)
    affected_steps = DOWNSTREAM_STEPS_BY_CHANGED_STEP.get(changed_step, (changed_step,))
    updated_state["generation_id"] = new_generation_id()
    updated_state["status"] = "pending"
    updated_state["current_step"] = changed_step
    updated_state["errors"] = [
        error
        for error in updated_state.get("errors", [])
        if error.get("step") not in affected_steps
    ]

    for step in affected_steps:
        updated_state.setdefault("steps", {})[step] = _reset_step_state(
            updated_state,
            step,
        )
    _clear_invalidated_state(updated_state, affected_steps, changed_step)
    updated_state["updated_at"] = utc_now_iso()
    save_json(paths.logs / "workflow_state.json", updated_state)

    invalidated = invalidate_downstream_artifacts(run_manager, run_id, changed_step)
    return updated_state, invalidated


def relative_path_for_artifact(artifact_key: str) -> str | None:
    """Return a known relative path for an artifact key."""

    for mapping in ARTIFACT_RELATIVE_PATHS_BY_STEP.values():
        if artifact_key in mapping:
            return mapping[artifact_key]
    return None


def _known_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("Artifact invalidation path escaped the run directory.")
    return candidate


def _remove_artifact_file(path: Path, root: Path) -> list[str]:
    removed: list[str] = []
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path.relative_to(root).as_posix())

    sidecar = artifact_lineage_path(path)
    if sidecar.exists():
        sidecar.unlink()
        removed.append(sidecar.relative_to(root).as_posix())
    return removed


def _clear_invalidated_state(
    state: dict[str, Any],
    steps: tuple[str, ...],
    changed_step: str,
) -> None:
    artifacts = state.setdefault("artifacts", {})
    artifact_lineage = state.setdefault("artifact_lineage", {})
    config_fingerprints = state.setdefault("config_fingerprints", {})
    for step in steps:
        for artifact_key in ARTIFACT_RELATIVE_PATHS_BY_STEP.get(step, {}):
            artifacts[artifact_key] = None
            artifact_lineage.pop(artifact_key, None)
            config_fingerprints.pop(artifact_key, None)
        if step == EDA_STEP:
            artifacts["plots"] = []

    if CLEANING_STEP in steps:
        source_fingerprint = state.get("source_fingerprint")
        state["analysis_input"] = {
            "dataset_used": "raw",
            "path": "input/raw_data.csv",
            "fingerprint": source_fingerprint,
            "source_fingerprint": source_fingerprint,
            "selection_reason": f"{changed_step}_invalidated_cleaned_data",
        }


def _reset_step_state(state: Mapping[str, Any], step: str) -> dict[str, Any]:
    approval_settings = state.get("approval_settings") or {}
    requires_approval = (
        step == CLEANING_STEP
        and bool(approval_settings.get("require_cleaning_approval", True))
    ) or (
        step == MODELING_STEP
        and bool(approval_settings.get("require_modeling_approval", True))
    )
    definition = STEP_DEFINITIONS[step]
    return {
        "status": "pending",
        "started_at": None,
        "completed_at": None,
        "attempts": 0,
        "max_attempts": definition.max_attempts,
        "requires_approval": requires_approval,
        "approval_status": "pending" if requires_approval else "not_required",
        "approval_reason": None,
        "approval_details": {},
        "error": None,
        "outputs": {},
    }
