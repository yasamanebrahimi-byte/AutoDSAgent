"""Local, reproducible benchmark cases for development and final evaluation.

The registry is intentionally checked in as code.  A case has one permanent
role so policy development cannot silently consume a final benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
    make_classification,
    make_friedman1,
    make_moons,
    make_regression,
)


TaskType = Literal["classification", "regression"]
FrameLoader = Callable[[], pd.DataFrame]
BENCHMARK_SUITE_VERSION = "2"


class BenchmarkRole(str, Enum):
    """The immutable methodological role of a benchmark case."""

    POLICY_DEVELOPMENT = "policy_development"
    FINAL_EVALUATION = "final_evaluation"
    EXTERNAL_EVALUATION = "external_evaluation"


@dataclass(frozen=True)
class BenchmarkCase:
    """One auditable benchmark definition.

    The loader is deliberately part of the case rather than the runner.  New
    local datasets can therefore be added without embedding dataset-specific
    evaluation behavior in the harness.
    """

    name: str
    target_column: str
    question: str
    expected_task_type: TaskType
    dataset_source: str
    dataframe_loader: FrameLoader | None = None
    dataframe: pd.DataFrame | None = None
    category: str = "sklearn_local"
    notes: str = ""
    random_seed: int = 42
    role: BenchmarkRole = BenchmarkRole.POLICY_DEVELOPMENT
    openml_task_id: int | None = None
    source_suite: str | None = None
    source_suite_id: int | None = None
    benchmark_suite_version: str | None = None
    tier: Literal["core", "stress"] = "core"

    def load(self) -> pd.DataFrame:
        if self.dataframe is not None:
            frame = self.dataframe.copy()
        elif self.dataframe_loader is not None:
            frame = self.dataframe_loader().copy()
        else:
            raise ValueError(f"Benchmark case {self.name!r} has no dataframe or loader.")
        if self.target_column not in frame.columns:
            raise ValueError(f"Benchmark case {self.name!r} has no target column {self.target_column!r}.")
        return frame

    def as_dict(self) -> dict[str, object]:
        frame = self.load()
        payload: dict[str, object] = {
            "name": self.name,
            "dataset_source": self.dataset_source,
            "target_column": self.target_column,
            "question": self.question,
            "expected_task_type": self.expected_task_type,
            "category": self.category,
            "notes": self.notes,
            "random_seed": self.random_seed,
            "role": self.role.value,
            "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
        }
        if self.openml_task_id is not None:
            payload.update(
                {
                    "openml_task_id": self.openml_task_id,
                    "source_suite": self.source_suite,
                    "source_suite_id": self.source_suite_id,
                    "benchmark_suite_version": (
                        self.benchmark_suite_version or BENCHMARK_SUITE_VERSION
                    ),
                    "tier": self.tier,
                }
            )
        return payload

    def provenance(self) -> dict[str, object]:
        """Return external provenance without loading the case's dataframe."""

        if self.openml_task_id is None:
            return {}
        return {
            "openml_task_id": self.openml_task_id,
            "source_suite": self.source_suite,
            "source_suite_id": self.source_suite_id,
            "benchmark_suite_version": self.benchmark_suite_version,
            "benchmark_tier": self.tier,
        }


def _breast_cancer() -> pd.DataFrame:
    loaded = load_breast_cancer(as_frame=True)
    frame = loaded.frame.copy()
    frame["target"] = loaded.target
    return frame


def _wine() -> pd.DataFrame:
    loaded = load_wine(as_frame=True)
    frame = loaded.frame.copy()
    frame["target"] = loaded.target
    return frame


def _diabetes() -> pd.DataFrame:
    loaded = load_diabetes(as_frame=True, scaled=False)
    frame = loaded.frame.copy()
    frame["target"] = loaded.target
    return frame


def _synthetic_regression() -> pd.DataFrame:
    features, target = make_regression(
        n_samples=240,
        n_features=6,
        n_informative=4,
        noise=15.0,
        random_state=123,
    )
    frame = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(features.shape[1])])
    frame["target"] = target
    return frame


def _make_numeric_frame(features: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(features.shape[1])])
    frame["target"] = target
    return frame


def _synthetic_linear_regression() -> pd.DataFrame:
    features, target = make_regression(
        n_samples=280,
        n_features=8,
        n_informative=8,
        noise=8.0,
        random_state=17,
    )
    return _make_numeric_frame(features, target)


def _synthetic_nonlinear_regression() -> pd.DataFrame:
    features, target = make_friedman1(n_samples=300, n_features=8, noise=1.5, random_state=23)
    return _make_numeric_frame(features, target)


def _synthetic_high_dim_regression() -> pd.DataFrame:
    features, target = make_regression(
        n_samples=240,
        n_features=36,
        n_informative=8,
        noise=18.0,
        random_state=31,
    )
    return _make_numeric_frame(features, target)


def _synthetic_binary_linear() -> pd.DataFrame:
    features, target = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=37,
    )
    return _make_numeric_frame(features, target)


def _synthetic_binary_nonlinear() -> pd.DataFrame:
    features, target = make_moons(n_samples=300, noise=0.20, random_state=41)
    return _make_numeric_frame(features, target)


def _synthetic_imbalanced_classification() -> pd.DataFrame:
    features, target = make_classification(
        n_samples=360,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        weights=[0.90, 0.10],
        flip_y=0.02,
        random_state=43,
    )
    return _make_numeric_frame(features, target)


def _synthetic_multiclass() -> pd.DataFrame:
    features, target = make_classification(
        n_samples=360,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=47,
    )
    return _make_numeric_frame(features, target)


def _synthetic_missingness() -> pd.DataFrame:
    frame = _synthetic_binary_linear()
    for column, stride in (("feature_0", 11), ("feature_2", 17), ("feature_5", 23)):
        frame.loc[frame.index % stride == 0, column] = np.nan
    return frame


def _synthetic_outlier_regression() -> pd.DataFrame:
    frame = _synthetic_linear_regression()
    outlier_rows = np.arange(0, len(frame), 37)
    frame.loc[outlier_rows, "feature_0"] *= 12.0
    frame.loc[outlier_rows, "target"] += 160.0
    return frame


def _digits_subset() -> pd.DataFrame:
    loaded = load_digits(as_frame=True)
    frame = loaded.frame.iloc[:900].copy()
    frame["target"] = loaded.target.iloc[:900].to_numpy()
    return frame


def _final_interaction_regression() -> pd.DataFrame:
    rng = np.random.default_rng(101)
    features = rng.normal(size=(300, 6))
    target = (
        2.0 * np.sin(features[:, 0] * features[:, 1])
        + 1.5 * features[:, 2] * features[:, 3]
        + 0.5 * features[:, 4]
        + rng.normal(scale=0.35, size=len(features))
    )
    return _make_numeric_frame(features, target)


def _final_low_n_high_p_classification() -> pd.DataFrame:
    features, target = make_classification(
        n_samples=180,
        n_features=32,
        n_informative=7,
        n_redundant=5,
        n_repeated=2,
        class_sep=0.9,
        random_state=107,
    )
    return _make_numeric_frame(features, target)


def _final_mixed_type_classification() -> pd.DataFrame:
    rng = np.random.default_rng(109)
    rows = 300
    numeric = rng.normal(size=(rows, 3))
    segment = np.where(numeric[:, 0] > 0.8, "premium", np.where(numeric[:, 0] < -0.8, "basic", "standard"))
    channel = rng.choice(["web", "store", "partner", "phone"], size=rows, p=[0.45, 0.25, 0.20, 0.10])
    target = ((numeric[:, 1] + 0.8 * (segment == "premium") - 0.5 * (channel == "phone")) > 0.35).astype(int)
    return pd.DataFrame(
        {
            "numeric_0": numeric[:, 0],
            "numeric_1": numeric[:, 1],
            "numeric_2": numeric[:, 2],
            "segment": segment,
            "channel": channel,
            "target": target,
        }
    )


def _final_shifted_nonlinear_regression() -> pd.DataFrame:
    features, target = make_friedman1(n_samples=300, n_features=10, noise=2.5, random_state=113)
    return _make_numeric_frame(features, target)


def default_benchmark_cases() -> list[BenchmarkCase]:
    """Return the frozen suite; roles are never selected from observed scores."""

    return [
        BenchmarkCase(
            name="breast_cancer",
            dataframe_loader=_breast_cancer,
            target_column="target",
            question="Classify whether the breast-cancer observation belongs to each target class.",
            expected_task_type="classification",
            dataset_source="sklearn.datasets.load_breast_cancer(as_frame=True)",
            notes="Wisconsin Diagnostic breast-cancer benchmark; software evaluation only.",
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="wine",
            dataframe_loader=_wine,
            target_column="target",
            question="Classify the wine cultivar from its measured chemical features.",
            expected_task_type="classification",
            dataset_source="sklearn.datasets.load_wine(as_frame=True)",
            role=BenchmarkRole.FINAL_EVALUATION,
        ),
        BenchmarkCase(
            name="diabetes",
            dataframe_loader=_diabetes,
            target_column="target",
            question="Estimate the disease-progression measure from the supplied patient measurements.",
            expected_task_type="regression",
            dataset_source="sklearn.datasets.load_diabetes(as_frame=True, scaled=False)",
            notes="Software evaluation only; not a medical device or clinical analysis.",
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="synthetic_regression",
            dataframe_loader=_synthetic_regression,
            target_column="target",
            question="Estimate the continuous target from the synthetic numeric features.",
            expected_task_type="regression",
            dataset_source="sklearn.datasets.make_regression(random_state=123)",
            category="synthetic_local",
            random_seed=123,
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="synthetic_linear_regression",
            dataframe_loader=_synthetic_linear_regression,
            target_column="target",
            question="Estimate the continuous target from a mostly additive linear structure.",
            expected_task_type="regression",
            dataset_source="sklearn.datasets.make_regression(random_state=17)",
            category="synthetic_local",
            random_seed=17,
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="synthetic_nonlinear_regression",
            dataframe_loader=_synthetic_nonlinear_regression,
            target_column="target",
            question="Estimate the continuous target from nonlinear feature relationships.",
            expected_task_type="regression",
            dataset_source="sklearn.datasets.make_friedman1(random_state=23)",
            category="synthetic_local",
            random_seed=23,
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="synthetic_high_dim_regression",
            dataframe_loader=_synthetic_high_dim_regression,
            target_column="target",
            question="Estimate a continuous target in a high-dimensional numeric table.",
            expected_task_type="regression",
            dataset_source="sklearn.datasets.make_regression(random_state=31)",
            category="synthetic_local",
            random_seed=31,
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="synthetic_binary_linear",
            dataframe_loader=_synthetic_binary_linear,
            target_column="target",
            question="Classify a binary target with a primarily linear decision boundary.",
            expected_task_type="classification",
            dataset_source="sklearn.datasets.make_classification(random_state=37)",
            category="synthetic_local",
            random_seed=37,
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="synthetic_binary_nonlinear",
            dataframe_loader=_synthetic_binary_nonlinear,
            target_column="target",
            question="Classify a binary target with a nonlinear two-moons boundary.",
            expected_task_type="classification",
            dataset_source="sklearn.datasets.make_moons(random_state=41)",
            category="synthetic_local",
            random_seed=41,
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="synthetic_imbalanced_classification",
            dataframe_loader=_synthetic_imbalanced_classification,
            target_column="target",
            question="Classify an imbalanced binary target from mixed signal and noise features.",
            expected_task_type="classification",
            dataset_source="sklearn.datasets.make_classification(weights=0.90, random_state=43)",
            category="synthetic_local",
            random_seed=43,
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="synthetic_multiclass",
            dataframe_loader=_synthetic_multiclass,
            target_column="target",
            question="Classify a three-class target from redundant numeric predictors.",
            expected_task_type="classification",
            dataset_source="sklearn.datasets.make_classification(n_classes=3, random_state=47)",
            category="synthetic_local",
            random_seed=47,
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="synthetic_missingness",
            dataframe_loader=_synthetic_missingness,
            target_column="target",
            question="Classify a binary target from numeric features with deterministic missingness.",
            expected_task_type="classification",
            dataset_source="synthetic_binary_linear_with_fixed_missingness",
            category="synthetic_local",
            random_seed=37,
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="synthetic_outlier_regression",
            dataframe_loader=_synthetic_outlier_regression,
            target_column="target",
            question="Estimate a continuous target in a regression table with outlier contamination.",
            expected_task_type="regression",
            dataset_source="synthetic_linear_regression_with_fixed_outliers",
            category="synthetic_local",
            random_seed=17,
            role=BenchmarkRole.POLICY_DEVELOPMENT,
        ),
        BenchmarkCase(
            name="digits_subset",
            dataframe_loader=_digits_subset,
            target_column="target",
            question="Classify the handwritten-digit class from pixel features.",
            expected_task_type="classification",
            dataset_source="sklearn.datasets.load_digits(as_frame=True).iloc[:900]",
            category="sklearn_local",
            random_seed=42,
            role=BenchmarkRole.FINAL_EVALUATION,
        ),
        BenchmarkCase(
            name="final_interaction_regression",
            dataframe_loader=_final_interaction_regression,
            target_column="target",
            question="Estimate a continuous target generated by interaction-heavy structure.",
            expected_task_type="regression",
            dataset_source="deterministic_synthetic_interaction_generator(random_state=101)",
            category="synthetic_final",
            random_seed=101,
            role=BenchmarkRole.FINAL_EVALUATION,
        ),
        BenchmarkCase(
            name="final_low_n_high_p_classification",
            dataframe_loader=_final_low_n_high_p_classification,
            target_column="target",
            question="Classify a binary target in a low-sample, high-dimensional table.",
            expected_task_type="classification",
            dataset_source="sklearn.datasets.make_classification(random_state=107)",
            category="synthetic_final",
            random_seed=107,
            role=BenchmarkRole.FINAL_EVALUATION,
        ),
        BenchmarkCase(
            name="final_mixed_type_classification",
            dataframe_loader=_final_mixed_type_classification,
            target_column="target",
            question="Classify a binary target from mixed numeric and categorical predictors.",
            expected_task_type="classification",
            dataset_source="deterministic_synthetic_mixed_type_generator(random_state=109)",
            category="synthetic_final",
            random_seed=109,
            role=BenchmarkRole.FINAL_EVALUATION,
        ),
        BenchmarkCase(
            name="final_shifted_nonlinear_regression",
            dataframe_loader=_final_shifted_nonlinear_regression,
            target_column="target",
            question="Estimate a nonlinear continuous target under a held-out generator configuration.",
            expected_task_type="regression",
            dataset_source="sklearn.datasets.make_friedman1(random_state=113)",
            category="synthetic_final",
            random_seed=113,
            role=BenchmarkRole.FINAL_EVALUATION,
        ),
    ]


def stable_frame_signature(frame: pd.DataFrame) -> dict[str, object]:
    """Return compact deterministic metadata useful in configs and tests."""

    return {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "numeric_columns": [
            str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
        ],
        "missing_values": int(frame.isna().sum().sum()),
        "finite_numeric_values": bool(
            np.isfinite(
                frame.select_dtypes(include=["number"]).to_numpy(dtype=float, na_value=np.nan)
            ).all()
        ),
    }
