"""Small, local, reproducible benchmark cases for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, make_regression


TaskType = Literal["classification", "regression"]
FrameLoader = Callable[[], pd.DataFrame]


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
        return {
            "name": self.name,
            "dataset_source": self.dataset_source,
            "target_column": self.target_column,
            "question": self.question,
            "expected_task_type": self.expected_task_type,
            "category": self.category,
            "notes": self.notes,
            "random_seed": self.random_seed,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
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


def default_benchmark_cases() -> list[BenchmarkCase]:
    """Return the initial benchmark suite using only bundled sklearn data."""

    return [
        BenchmarkCase(
            name="breast_cancer",
            dataframe_loader=_breast_cancer,
            target_column="target",
            question="Classify whether the breast-cancer observation belongs to each target class.",
            expected_task_type="classification",
            dataset_source="sklearn.datasets.load_breast_cancer(as_frame=True)",
            notes="Wisconsin Diagnostic breast-cancer benchmark; software evaluation only.",
        ),
        BenchmarkCase(
            name="wine",
            dataframe_loader=_wine,
            target_column="target",
            question="Classify the wine cultivar from its measured chemical features.",
            expected_task_type="classification",
            dataset_source="sklearn.datasets.load_wine(as_frame=True)",
        ),
        BenchmarkCase(
            name="diabetes",
            dataframe_loader=_diabetes,
            target_column="target",
            question="Estimate the disease-progression measure from the supplied patient measurements.",
            expected_task_type="regression",
            dataset_source="sklearn.datasets.load_diabetes(as_frame=True, scaled=False)",
            notes="Software evaluation only; not a medical device or clinical analysis.",
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
