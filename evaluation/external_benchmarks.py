"""Frozen AMLB/OpenML benchmark definitions and lazy task loading.

The manifest in this module is deliberately static.  It is an evaluation
boundary, not a source of policy-development data, and no OpenML import or
network request occurs until a case is loaded explicitly.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from evaluation.benchmarks import BenchmarkCase, BenchmarkRole


EXTERNAL_BENCHMARK_SUITE_VERSION = "1.0.0"
AMLB_CLASSIFICATION_SUITE_ID = 271
AMLB_REGRESSION_SUITE_ID = 269
OPENML_CACHE_ENV_VAR = "AUTODS_OPENML_CACHE"
CANONICAL_TARGET_COLUMN = "__target__"

ExternalTaskType = Literal["classification", "regression"]
BenchmarkTier = Literal["core", "stress"]


@dataclass(frozen=True)
class OpenMLBenchmarkSpec:
    """One immutable entry in the external benchmark manifest."""

    task_id: int
    name: str
    expected_task_type: ExternalTaskType
    expected_rows: int
    expected_features: int
    expected_classes: int | None = None
    tier: BenchmarkTier = "core"
    notes: str = ""
    source_suite: int | None = None
    # The frozen AMLB dimensions were recorded from the complete supervised
    # table, while OpenML's X/y APIs return X without the target.  Keep this
    # compatibility detail out of the serialized manifest and default direct
    # test/custom specs to the historical raw-feature interpretation.
    feature_count_includes_target: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.task_id <= 0:
            raise ValueError("OpenML task IDs must be positive.")
        if self.expected_rows <= 0 or self.expected_features < 1:
            raise ValueError("OpenML benchmark dimensions must be positive.")
        if self.expected_task_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported external task type: {self.expected_task_type!r}")
        if self.tier not in {"core", "stress"}:
            raise ValueError(f"Unsupported external benchmark tier: {self.tier!r}")
        if self.expected_task_type == "classification":
            if self.expected_classes is None or self.expected_classes < 2:
                raise ValueError("Classification specs must declare at least two expected classes.")
            expected_suite = AMLB_CLASSIFICATION_SUITE_ID
        else:
            if self.expected_classes is not None:
                raise ValueError("Regression specs must not declare expected_classes.")
            expected_suite = AMLB_REGRESSION_SUITE_ID
        if self.source_suite is None:
            object.__setattr__(self, "source_suite", expected_suite)
        elif self.source_suite != expected_suite:
            raise ValueError(
                f"Task {self.task_id} has source suite {self.source_suite}, "
                f"expected {expected_suite} for {self.expected_task_type}."
            )

    @property
    def source_suite_label(self) -> str:
        kind = "classification" if self.expected_task_type == "classification" else "regression"
        return f"AMLB {kind} suite {self.source_suite}"

    @property
    def expected_input_features(self) -> int:
        """Expected columns in OpenML's feature matrix before the target is added."""

        return self.expected_features - int(self.feature_count_includes_target)

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "expected_task_type": self.expected_task_type,
            "expected_rows": self.expected_rows,
            "expected_features": self.expected_features,
            "expected_classes": self.expected_classes,
            "tier": self.tier,
            "notes": self.notes,
            "source_suite": self.source_suite,
            "source_suite_label": self.source_suite_label,
            "benchmark_suite_version": EXTERNAL_BENCHMARK_SUITE_VERSION,
        }


@dataclass(frozen=True)
class OpenMLBenchmarkData:
    """Loaded raw data plus non-performance metadata used by prefetching."""

    frame: pd.DataFrame
    task_id: int
    dataset_id: int | None
    dataset_name: str
    original_target_name: str | None
    observed_classes: int | None


def _classification(
    task_id: int,
    name: str,
    rows: int,
    features: int,
    classes: int,
    *,
    tier: BenchmarkTier = "core",
    notes: str = "",
) -> OpenMLBenchmarkSpec:
    return OpenMLBenchmarkSpec(
        task_id,
        name,
        "classification",
        rows,
        features,
        classes,
        tier,
        notes,
        AMLB_CLASSIFICATION_SUITE_ID,
        True,
    )


def _regression(
    task_id: int,
    name: str,
    rows: int,
    features: int,
    *,
    tier: BenchmarkTier = "core",
    notes: str = "",
) -> OpenMLBenchmarkSpec:
    return OpenMLBenchmarkSpec(
        task_id,
        name,
        "regression",
        rows,
        features,
        None,
        tier,
        notes,
        AMLB_REGRESSION_SUITE_ID,
        True,
    )


# This is the frozen AMLB/OpenML task manifest.  Keep task IDs and dimensions
# immutable once external research begins; never select entries by results.
EXTERNAL_BENCHMARK_MANIFEST: tuple[OpenMLBenchmarkSpec, ...] = (
    _classification(359983, "adult", 48842, 15, 2),
    _classification(359979, "Amazon_employee_access", 32769, 10, 2),
    _classification(168868, "APSFailure", 76000, 171, 2, tier="stress", notes="Larger and highly imbalanced."),
    _classification(146818, "Australian", 690, 15, 2),
    _classification(359982, "bank-marketing", 45211, 17, 2),
    _classification(359967, "Bioresponse", 3751, 1777, 2, tier="stress", notes="High-dimensional."),
    _classification(359955, "blood-transfusion-service-center", 748, 5, 2),
    _classification(359960, "car", 1728, 7, 4, tier="stress", notes="Multiclass and imbalanced."),
    _classification(359968, "churn", 5000, 21, 2),
    _classification(359992, "Click_prediction_small", 39948, 12, 2),
    _classification(168757, "credit-g", 1000, 21, 2),
    _classification(359964, "dna", 3186, 181, 3),
    _classification(359954, "eucalyptus", 736, 20, 5),
    _classification(359970, "GesturePhaseSegmentationProcessed", 9873, 33, 5),
    _classification(359966, "Internet-Advertisements", 3279, 1559, 2, tier="stress", notes="High-dimensional."),
    _classification(359962, "kc1", 2109, 22, 2),
    _classification(190137, "ozone-level-8hr", 2534, 73, 2),
    _classification(359971, "PhishingWebsites", 11055, 31, 2),
    _classification(168350, "phoneme", 5404, 6, 2),
    _classification(359956, "qsar-biodeg", 1055, 42, 2),
    _classification(168784, "steel-plates-fault", 1941, 28, 7),
    _classification(359974, "wine-quality-white", 4898, 12, 7, tier="stress", notes="Multiclass with rare classes."),
    _regression(359944, "abalone", 4177, 9),
    _regression(359938, "Brazilian_houses", 10692, 13),
    _regression(359942, "colleges", 7063, 45),
    _regression(233211, "diamonds", 53940, 10, tier="stress", notes="Larger regression dataset."),
    _regression(359936, "elevators", 16599, 19),
    _regression(359952, "house_16H", 22784, 17),
    _regression(359951, "house_prices_nominal", 1460, 80, tier="stress", notes="Wider small regression dataset."),
    _regression(359949, "house_sales", 21613, 22),
    _regression(233215, "Mercedes_Benz_Greener_Manufacturing", 4209, 377, tier="stress", notes="Relatively wide."),
    _regression(360945, "MIP-2016-regression", 1090, 145),
    _regression(167210, "Moneyball", 1232, 15),
    _regression(359941, "OnlineNewsPopularity", 39644, 60, tier="stress", notes="Larger and mid-dimensional."),
    _regression(359930, "quake", 2178, 4),
    _regression(359931, "sensory", 576, 12),
    _regression(359932, "socmob", 1156, 6),
    _regression(359933, "space_ga", 3107, 7),
    _regression(359934, "tecator", 240, 125, tier="stress", notes="Low-N and high-P."),
    _regression(359935, "wine_quality", 6497, 12),
)

# Common aliases make the checked-in manifest easy to discover without
# creating a second mutable representation of the task list.
EXTERNAL_BENCHMARK_SPECS = EXTERNAL_BENCHMARK_MANIFEST
EXTERNAL_TASK_MANIFEST = EXTERNAL_BENCHMARK_MANIFEST


def external_benchmark_specs() -> tuple[OpenMLBenchmarkSpec, ...]:
    """Return the immutable external task definitions in manifest order."""

    return EXTERNAL_BENCHMARK_MANIFEST


def _import_openml() -> Any:
    try:
        import openml
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "The external OpenML benchmark requires the optional dependency. "
            "Install it with `python -m pip install -e \".[benchmark]\"`."
        ) from exc
    return openml


def configure_openml_cache(openml_module: Any | None = None) -> Path | None:
    """Configure OpenML's cache from ``AUTODS_OPENML_CACHE`` when provided."""

    cache_value = os.environ.get(OPENML_CACHE_ENV_VAR)
    if not cache_value:
        return None
    cache_path = Path(cache_value).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    module = openml_module or _import_openml()
    module.config.set_root_cache_directory(cache_path)
    return cache_path


def _first_text(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return str(value) if value is not None else None


def _task_target_name(task: Any, dataset: Any | None = None) -> str | None:
    for owner in (task, dataset):
        if owner is None:
            continue
        for attribute in ("target_name", "target_names", "default_target_attribute"):
            value = _first_text(getattr(owner, attribute, None))
            if value:
                return value
    return None


def _task_dataset_id(task: Any, dataset: Any | None = None) -> int | None:
    for owner in (task, dataset):
        value = getattr(owner, "dataset_id", None) if owner is not None else None
        if value is None and owner is not None:
            value = getattr(owner, "id", None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _get_raw_task_data(task: Any) -> tuple[Any, Any, str | None, Any | None]:
    """Read raw task data across the current and older OpenML APIs."""

    get_x_and_y = getattr(task, "get_X_and_y", None)
    if callable(get_x_and_y):
        try:
            parameters = inspect.signature(get_x_and_y).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "dataset_format" in parameters:
            values = get_x_and_y(dataset_format="dataframe")
        else:
            values = get_x_and_y()
        if not isinstance(values, tuple) or len(values) < 2:
            raise ValueError("OpenML task get_X_and_y() did not return an X/y pair.")
        return values[0], values[1], _task_target_name(task), None

    get_dataset = getattr(task, "get_dataset", None)
    if not callable(get_dataset):
        raise ValueError("OpenML task does not expose get_dataset() or get_X_and_y().")
    dataset = get_dataset()
    target_name = _task_target_name(task, dataset)
    get_data = getattr(dataset, "get_data", None)
    if not callable(get_data):
        raise ValueError("OpenML dataset does not expose get_data().")
    try:
        values = get_data(target=target_name, dataset_format="dataframe")
    except TypeError:
        # Small test doubles and older package versions may only accept the
        # target parameter.  The fallback still requests the raw dataset.
        values = get_data(target=target_name)
    if not isinstance(values, tuple) or len(values) < 2:
        raise ValueError("OpenML dataset get_data() did not return an X/y pair.")
    return values[0], values[1], target_name, dataset


def _shape_error(spec: OpenMLBenchmarkSpec, actual_rows: int, actual_features: int) -> ValueError:
    return ValueError(
        f"OpenML task ID {spec.task_id} ({spec.name}) shape validation failed: "
        f"expected shape rows={spec.expected_rows}, features={spec.expected_input_features}; "
        f"actual shape rows={actual_rows}, features={actual_features}; "
        f"dataset name={spec.name!r}."
    )


def load_openml_task_data(spec: OpenMLBenchmarkSpec) -> OpenMLBenchmarkData:
    """Download or read one raw OpenML task and validate its frozen schema."""

    openml = _import_openml()
    configure_openml_cache(openml)
    task = openml.tasks.get_task(spec.task_id, download_splits=False)
    X, y, target_name, dataset = _get_raw_task_data(task)
    if isinstance(X, pd.DataFrame):
        features = X.copy()
    else:
        features = pd.DataFrame(X)
    if CANONICAL_TARGET_COLUMN in features.columns:
        raise ValueError(
            f"OpenML task ID {spec.task_id} ({spec.name}) has a feature named "
            f"{CANONICAL_TARGET_COLUMN!r}, which collides with the canonical target column."
        )
    if y is None:
        raise ValueError(
            f"OpenML task ID {spec.task_id} ({spec.name}) returned no target for "
            f"original target {target_name!r}."
        )
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(
                f"OpenML task ID {spec.task_id} ({spec.name}) returned {y.shape[1]} target columns; "
                "exactly one supervised target is required."
            )
        target = y.iloc[:, 0].copy()
    elif isinstance(y, pd.Series):
        target = y.copy()
    else:
        target = pd.Series(y)
    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)
    actual_rows = len(features)
    actual_features = features.shape[1]
    if actual_rows != spec.expected_rows or actual_features != spec.expected_input_features:
        raise _shape_error(spec, actual_rows, actual_features)
    if len(target) != spec.expected_rows:
        raise ValueError(
            f"OpenML task ID {spec.task_id} ({spec.name}) shape validation failed: "
            f"expected shape rows={spec.expected_rows}, features={spec.expected_input_features}; "
            f"actual shape rows={actual_rows}, features={actual_features}; "
            f"actual target length={len(target)}; dataset name={spec.name!r}."
        )
    if spec.expected_task_type == "classification":
        observed_classes = int(target.nunique(dropna=True))
        if observed_classes != spec.expected_classes:
            raise ValueError(
                f"OpenML task ID {spec.task_id} ({spec.name}) class-count validation failed: "
                f"expected shape rows={spec.expected_rows}, features={spec.expected_input_features}; "
                f"actual shape rows={actual_rows}, features={actual_features}; "
                f"expected classes={spec.expected_classes}, actual classes={observed_classes}; "
                f"dataset name={spec.name!r}."
            )
    else:
        if not pd.api.types.is_numeric_dtype(target):
            raise ValueError(
                f"OpenML task ID {spec.task_id} ({spec.name}) regression target validation failed: "
                f"target {target_name!r} is not numeric (dtype={target.dtype})."
            )
        observed_classes = None
    frame = features.copy()
    frame[CANONICAL_TARGET_COLUMN] = target
    dataset_name = str(
        getattr(dataset, "name", None)
        or getattr(task, "dataset_name", None)
        or spec.name
    )
    return OpenMLBenchmarkData(
        frame=frame,
        task_id=spec.task_id,
        dataset_id=_task_dataset_id(task, dataset),
        dataset_name=dataset_name,
        original_target_name=target_name,
        observed_classes=observed_classes,
    )


def load_openml_task(spec: OpenMLBenchmarkSpec) -> pd.DataFrame:
    """Return one validated raw OpenML task as a BenchmarkCase-compatible frame."""

    return load_openml_task_data(spec).frame


def _case_for_spec(spec: OpenMLBenchmarkSpec) -> BenchmarkCase:
    question = (
        "Predict the target class from the provided tabular features."
        if spec.expected_task_type == "classification"
        else "Estimate the continuous target from the provided tabular features."
    )
    return BenchmarkCase(
        name=spec.name,
        target_column=CANONICAL_TARGET_COLUMN,
        question=question,
        expected_task_type=spec.expected_task_type,
        dataset_source=f"OpenML task {spec.task_id} ({spec.source_suite_label})",
        dataframe_loader=lambda spec=spec: load_openml_task(spec),
        category="openml_amlb_external",
        notes=spec.notes,
        random_seed=42,
        role=BenchmarkRole.EXTERNAL_EVALUATION,
        openml_task_id=spec.task_id,
        source_suite=spec.source_suite_label,
        source_suite_id=spec.source_suite,
        benchmark_suite_version=EXTERNAL_BENCHMARK_SUITE_VERSION,
        tier=spec.tier,
    )


def external_benchmark_cases() -> list[BenchmarkCase]:
    """Return external cases without importing OpenML or touching the network."""

    return [_case_for_spec(spec) for spec in EXTERNAL_BENCHMARK_MANIFEST]


__all__ = [
    "AMLB_CLASSIFICATION_SUITE_ID",
    "AMLB_REGRESSION_SUITE_ID",
    "BenchmarkTier",
    "CANONICAL_TARGET_COLUMN",
    "EXTERNAL_BENCHMARK_MANIFEST",
    "EXTERNAL_BENCHMARK_SPECS",
    "EXTERNAL_BENCHMARK_SUITE_VERSION",
    "EXTERNAL_TASK_MANIFEST",
    "ExternalTaskType",
    "OpenMLBenchmarkData",
    "OpenMLBenchmarkSpec",
    "configure_openml_cache",
    "external_benchmark_cases",
    "external_benchmark_specs",
    "load_openml_task",
    "load_openml_task_data",
]
