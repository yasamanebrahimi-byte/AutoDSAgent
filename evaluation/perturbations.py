"""Small deterministic data-quality scenarios for exercising current invariants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

from evaluation.benchmarks import BenchmarkCase


ScenarioKind = Literal[
    "clean",
    "agent_preprocessing_challenge",
    "deterministic_invariant_violation",
    "deterministic_safe_exclusion",
]


@dataclass(frozen=True)
class Perturbation:
    id: str
    description: str
    kind: ScenarioKind
    applies_to: tuple[str, ...]
    expected_validation_codes: tuple[str, ...]
    _transform: Callable[[pd.DataFrame, int, BenchmarkCase], tuple[pd.DataFrame, list[str]]]

    def applies(self, case: BenchmarkCase) -> bool:
        return case.expected_task_type in self.applies_to

    def apply(self, frame: pd.DataFrame, seed: int, case: BenchmarkCase) -> tuple[pd.DataFrame, list[str]]:
        transformed, changes = self._transform(frame.copy(), seed, case)
        return transformed, changes

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "kind": self.kind,
            "applies_to": list(self.applies_to),
            "expected_validation_codes": list(self.expected_validation_codes),
        }


def _numeric_feature(frame: pd.DataFrame, case: BenchmarkCase) -> str:
    candidates = [
        str(column)
        for column in frame.columns
        if str(column) != case.target_column
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not candidates:
        raise ValueError(f"Perturbation requires a numeric feature in {case.name!r}.")
    return candidates[0]


def _missing(frame: pd.DataFrame, seed: int, case: BenchmarkCase) -> tuple[pd.DataFrame, list[str]]:
    column = _numeric_feature(frame, case)
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(frame), size=max(4, min(12, len(frame) // 20)), replace=False))
    frame.loc[positions, column] = np.nan
    return frame, [f"set {len(positions)} values in numeric feature {column!r} to missing"]


def _infinity(frame: pd.DataFrame, seed: int, case: BenchmarkCase) -> tuple[pd.DataFrame, list[str]]:
    column = _numeric_feature(frame, case)
    positions = [0, 1] if len(frame) > 2 else [0]
    frame.loc[positions[0], column] = np.inf
    if len(positions) > 1:
        frame.loc[positions[1], column] = -np.inf
    return frame, [f"set rows {positions} in numeric feature {column!r} to positive/negative infinity"]


def _identifier(frame: pd.DataFrame, seed: int, case: BenchmarkCase) -> tuple[pd.DataFrame, list[str]]:
    del seed
    column = "synthetic_row_id"
    frame[column] = np.arange(len(frame), dtype=np.int64)
    return frame, [f"added unique integer identifier column {column!r}"]


def _target_copy(frame: pd.DataFrame, seed: int, case: BenchmarkCase) -> tuple[pd.DataFrame, list[str]]:
    del seed
    column = "synthetic_target_copy"
    frame[column] = frame[case.target_column]
    return frame, [f"added exact copy of target column {case.target_column!r} as {column!r}"]


def _invalid_regression_target(
    frame: pd.DataFrame, seed: int, case: BenchmarkCase
) -> tuple[pd.DataFrame, list[str]]:
    del seed
    if case.expected_task_type != "regression":
        return frame, ["not applicable to classification"]
    values = frame[case.target_column].astype(object).copy()
    values.iloc[0] = "not-a-number"
    values.iloc[1] = np.inf
    frame[case.target_column] = values
    return frame, ["replaced one regression target with text and one with positive infinity"]


def _rare_classification(
    frame: pd.DataFrame, seed: int, case: BenchmarkCase
) -> tuple[pd.DataFrame, list[str]]:
    del seed
    if case.expected_task_type != "classification":
        return frame, ["not applicable to regression"]
    values = np.repeat("common", len(frame)).astype(object)
    values[:2] = "rare"
    frame[case.target_column] = values
    return frame, ["replaced the classification target with two rare-class rows and one common class"]


def default_perturbations() -> list[Perturbation]:
    return [
        Perturbation(
            id="missing_values",
            description="Inject missing numeric feature values; tests whether proposed preprocessing handles them safely.",
            kind="agent_preprocessing_challenge",
            applies_to=("classification", "regression"),
            expected_validation_codes=("numeric_missing_values_are_handled",),
            _transform=_missing,
        ),
        Perturbation(
            id="infinity_values",
            description="Inject positive and negative infinity into a numeric feature.",
            kind="agent_preprocessing_challenge",
            applies_to=("classification", "regression"),
            expected_validation_codes=("numeric_infinity_values_are_handled",),
            _transform=_infinity,
        ),
        Perturbation(
            id="identifier_column",
            description="Add a synthetic unique identifier that the validator should exclude.",
            kind="deterministic_safe_exclusion",
            applies_to=("classification", "regression"),
            expected_validation_codes=("identifier_handling_is_safe",),
            _transform=_identifier,
        ),
        Perturbation(
            id="target_copy_leakage",
            description="Add a direct target-copy feature that must fail closed.",
            kind="deterministic_invariant_violation",
            applies_to=("classification", "regression"),
            expected_validation_codes=("no_direct_target_copy_features",),
            _transform=_target_copy,
        ),
        Perturbation(
            id="invalid_regression_target",
            description="Inject nonnumeric and nonfinite regression targets.",
            kind="deterministic_invariant_violation",
            applies_to=("regression",),
            expected_validation_codes=(
                "regression_target_is_numeric_or_coercible",
                "regression_target_is_finite",
            ),
            _transform=_invalid_regression_target,
        ),
        Perturbation(
            id="classification_feasibility",
            description="Create a two-row rare class so stratified holdout/CV feasibility fails.",
            kind="deterministic_invariant_violation",
            applies_to=("classification",),
            expected_validation_codes=(
                "classification_split_and_stratification_feasible",
                "classification_training_supports_cross_validation",
            ),
            _transform=_rare_classification,
        ),
    ]
