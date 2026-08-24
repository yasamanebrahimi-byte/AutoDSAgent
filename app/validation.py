"""Fail-closed deterministic checks that must pass before model fitting.

This module deliberately contains no LLM decisions.  It validates the final
target/task/method proposal against the dataframe that will be modeled and
also provides the small amount of deterministic normalization needed to make
the fitter use the exact data that was checked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from math import ceil, isfinite
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.deterministic import is_identifier, semantic_type
from app.preprocessing import (
    PreprocessingRequirements,
    records_from_dataframe,
    requirements_from_records,
)
from app.schemas import PreprocessingContract


SUPPORTED_METHODS = frozenset(
    {"linear", "regularized_linear", "tree_ensemble", "boosted_tree"}
)
SUPPORTED_TASKS = frozenset({"classification", "regression"})
MIN_TEST_SIZE = 0.10
MAX_TEST_SIZE = 0.50
MAX_CATEGORICAL_CARDINALITY = 80
MAX_CV_FOLDS = 5


@dataclass
class ValidationCheck:
    """One inspectable deterministic invariant result."""

    code: str
    passed: bool
    evidence: dict[str, Any]
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "status": "passed" if self.passed else "failed",
            "passed": self.passed,
            "severity": self.severity,
            "evidence": self.evidence,
            "message": self.message,
        }


@dataclass(frozen=True)
class FrozenSplit:
    """Canonical supervised split expressed in original row positions.

    Incoming dataframe index labels are deliberately ignored.  The dataset
    fingerprint and position digests make the split auditable and let the
    reproduction artifact fail closed if the source data changes.
    """

    target_column: str
    task_type: str
    test_size: float
    random_state: int
    strategy: str
    dataset_fingerprint: str
    valid_row_positions: tuple[int, ...]
    train_row_positions: tuple[int, ...]
    holdout_row_positions: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "task_type": self.task_type,
            "random_state": int(self.random_state),
            "test_size": float(self.test_size),
            "strategy": self.strategy,
            "row_index_policy": "zero_based_row_position",
            "dataset_fingerprint": self.dataset_fingerprint,
            "valid_rows": len(self.valid_row_positions),
            "train_rows": len(self.train_row_positions),
            "holdout_rows": len(self.holdout_row_positions),
            "train_positions_digest": _position_digest(self.train_row_positions),
            "holdout_positions_digest": _position_digest(self.holdout_row_positions),
            "valid_positions_digest": _position_digest(self.valid_row_positions),
        }


@dataclass
class ValidationResult:
    """Structured evidence for a validated or rejected modeling plan."""

    target_column: str
    task_type: str
    method: str
    checks: list[ValidationCheck] = field(default_factory=list)
    features_used: list[str] = field(default_factory=list)
    excluded_features: list[dict[str, Any]] = field(default_factory=list)
    target_rows_removed: int = 0
    valid_target_rows: int = 0
    split: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    direct_leakage_detected: bool = False
    preprocessing_contract: PreprocessingContract | None = None
    preprocessing_requirements: PreprocessingRequirements | None = None

    @property
    def status(self) -> str:
        return "passed" if not any(not check.passed for check in self.checks) else "failed"

    @property
    def failed_checks(self) -> list[ValidationCheck]:
        return [check for check in self.checks if not check.passed]

    def add_check(
        self,
        code: str,
        passed: bool,
        evidence: dict[str, Any],
        message: str,
        *,
        severity: str = "error",
    ) -> None:
        self.checks.append(
            ValidationCheck(
                code=code,
                passed=passed,
                evidence=evidence,
                message=message,
                severity=severity,
            )
        )

    def add_failure(self, code: str, message: str, evidence: dict[str, Any] | None = None) -> None:
        self.add_check(code, False, evidence or {}, message)

    def raise_if_failed(self) -> "ValidationResult":
        if self.status == "failed":
            raise InvariantViolation.from_result(self)
        return self

    def as_dict(self) -> dict[str, Any]:
        failures = [
            {"code": check.code, "message": check.message, "evidence": check.evidence}
            for check in self.failed_checks
        ]
        return {
            "status": self.status,
            "overall_status": self.status,
            "validated_target_column": self.target_column,
            "validated_task_type": self.task_type,
            "validated_method": self.method,
            "features_used": self.features_used,
            "excluded_features": self.excluded_features,
            "target_rows_removed": self.target_rows_removed,
            "valid_target_rows": self.valid_target_rows,
            "split": self.split,
            "direct_leakage_detected": self.direct_leakage_detected,
            "warnings": self.warnings,
            "approved_preprocessing": self.preprocessing_contract.model_dump(mode="json")
            if self.preprocessing_contract is not None
            else None,
            "preprocessing_requirements": self.preprocessing_requirements.as_dict()
            if self.preprocessing_requirements is not None
            else None,
            "checks": [check.as_dict() for check in self.checks],
            "failures": failures,
        }


class InvariantViolation(ValueError):
    """Raised when deterministic validation forbids a training run."""

    def __init__(self, message: str, result: ValidationResult | None = None) -> None:
        super().__init__(message)
        self.result = result

    @classmethod
    def from_result(cls, result: ValidationResult) -> "InvariantViolation":
        first = result.failed_checks[0] if result.failed_checks else None
        if first is None:
            message = "Deterministic validation failed; correct the approved modeling plan."
        else:
            message = f"[{first.code}] {first.message}"
        return cls(message, result)


class DeterministicRecommendationUnavailable(InvariantViolation):
    """Raised when the independent deterministic recommender cannot complete."""

    code = "deterministic_recommendation_unavailable"

    def __init__(self, original_exception: Exception) -> None:
        self.original_error_type = type(original_exception).__name__
        self.original_error_message = str(original_exception)
        super().__init__(
            f"[{self.code}] The independent deterministic recommender failed before validation could begin.",
            None,
        )


def validate_training_plan(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
    method: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_columns: Sequence[str] | None = None,
    preprocessing: PreprocessingContract | dict[str, Any] | list[str] | None = None,
    split: FrozenSplit | None = None,
    row_positions: Sequence[int] | None = None,
    evidence_dataframe: pd.DataFrame | None = None,
) -> ValidationResult:
    """Evaluate one complete plan against the actual modeling dataframe.

    The function returns all available evidence, including failures.  Callers
    at a training boundary must call ``raise_if_failed`` before fitting.
    """

    result = ValidationResult(
        target_column=str(target_column),
        task_type=str(task_type),
        method=str(method),
    )
    approved_preprocessing: PreprocessingContract | None = None
    if preprocessing is not None:
        try:
            approved_preprocessing = PreprocessingContract.model_validate(preprocessing)
        except Exception as exc:
            result.add_failure(
                "preprocessing_contract_is_supported",
                "The preprocessing proposal is not a complete supported executable contract.",
                {"error": str(exc)},
            )
            return result
    column_names = [str(column) for column in dataframe.columns]
    duplicate_names = sorted(
        {name for name in column_names if column_names.count(name) > 1}
    )
    result.add_check(
        "columns_have_unique_names",
        not duplicate_names,
        {"duplicate_columns": duplicate_names},
        "Column names must be unique so the approved target and feature matrix resolve unambiguously.",
    )

    target_positions = [index for index, name in enumerate(column_names) if name == str(target_column)]
    target_exists = bool(target_positions)
    target_unique = len(target_positions) == 1
    result.add_check(
        "target_exists",
        target_exists,
        {"target_column": str(target_column), "matching_columns": len(target_positions)},
        f"Target column '{target_column}' must exist; choose an existing target column.",
    )
    result.add_check(
        "target_resolves_once",
        target_unique,
        {"target_column": str(target_column), "matching_columns": len(target_positions)},
        f"Target name '{target_column}' must resolve to exactly one column; remove duplicate or ambiguous names.",
    )
    result.add_check(
        "task_is_supported",
        task_type in SUPPORTED_TASKS,
        {"task_type": str(task_type), "supported_tasks": sorted(SUPPORTED_TASKS)},
        "Use one of the supported task types: classification or regression.",
    )
    result.add_check(
        "method_is_supported",
        method in SUPPORTED_METHODS,
        {"method": str(method), "supported_methods": sorted(SUPPORTED_METHODS)},
        "Use one of the methods proposed by the workflow: linear, regularized_linear, tree_ensemble, or boosted_tree.",
    )
    valid_test_size = _valid_test_size(test_size)
    result.add_check(
        "test_size_is_supported",
        valid_test_size,
        {"test_size": _safe_float(test_size), "supported_range": [MIN_TEST_SIZE, MAX_TEST_SIZE]},
        f"test_size must be finite and between {MIN_TEST_SIZE:.2f} and {MAX_TEST_SIZE:.2f}; choose a supported holdout fraction.",
    )

    if not target_exists or not target_unique or duplicate_names:
        result.add_failure(
            "target_validation_can_proceed",
            "Target validation cannot continue until column names and the target reference are unambiguous.",
            {"target_column": str(target_column)},
        )
        return result

    target = dataframe.iloc[:, target_positions[0]].copy()
    feature_evidence_frame = dataframe if evidence_dataframe is None else evidence_dataframe.copy()
    evidence_column_names = [str(column) for column in feature_evidence_frame.columns]
    if evidence_column_names != column_names:
        result.add_failure(
            "training_evidence_schema_matches_data",
            "Training-only feature evidence must use the same ordered schema as the modeling dataframe.",
            {
                "modeling_columns": column_names,
                "evidence_columns": evidence_column_names,
            },
        )
        return result
    evidence_target = feature_evidence_frame.iloc[:, target_positions[0]].copy()
    evidence_valid_mask, evidence_normalized_target, _ = _normalize_target(
        evidence_target,
        str(task_type),
    )
    valid_mask, normalized_target, target_evidence = _normalize_target(target, str(task_type))
    result.target_rows_removed = int((~valid_mask).sum())
    result.valid_target_rows = int(valid_mask.sum())
    result.add_check(
        "target_missing_values_filtered_before_encoding",
        True,
        {
            "rows_removed": result.target_rows_removed,
            "missing_rows": target_evidence["missing_rows"],
            "artificial_null_labels_removed": target_evidence["artificial_null_labels_removed"],
        },
        "Missing target values are removed deterministically before target encoding; they cannot become artificial classes.",
    )

    invalid_regression = int(target_evidence.get("invalid_nonnumeric_rows", 0))
    result.add_check(
        "regression_target_is_numeric_or_coercible",
        task_type != "regression" or invalid_regression == 0,
        {
            "invalid_nonnumeric_rows": invalid_regression,
            "coercible_rows": int(target_evidence.get("coercible_rows", 0)),
        },
        "Regression targets may contain numeric strings, but nonnumeric values must be corrected rather than silently discarded.",
    )
    nonfinite_target = int(target_evidence.get("nonfinite_rows", 0))
    result.add_check(
        "regression_target_is_finite",
        task_type != "regression" or nonfinite_target == 0,
        {"nonfinite_rows": nonfinite_target},
        "Regression targets must contain finite numeric values; replace infinity or invalid measurements before training.",
    )

    classes: list[str] = []
    class_counts: dict[str, int] = {}
    if task_type == "classification":
        classes_series = normalized_target.loc[valid_mask]
        class_counts = {str(key): int(value) for key, value in classes_series.value_counts().items()}
        classes = list(class_counts)
        result.add_check(
            "classification_target_has_two_classes",
            len(classes) >= 2,
            {"class_count": len(classes), "class_counts": class_counts},
            "Classification requires at least two valid classes after deterministic target filtering.",
        )
        result.add_check(
            "classification_target_has_feasible_class_count",
            len(classes) <= max(2, result.valid_target_rows // 2),
            {"class_count": len(classes), "valid_rows": result.valid_target_rows},
            "The number of classes must leave enough rows for both a holdout and cross-validation.",
        )
    else:
        result.add_check(
            "regression_target_has_valid_values",
            result.valid_target_rows >= 2,
            {"valid_rows": result.valid_target_rows},
            "Regression needs at least two valid target observations after filtering missing values.",
        )

    if task_type in SUPPORTED_TASKS:
        target_kind = semantic_type(evidence_target.loc[evidence_valid_mask])
        compatible = task_type == "classification" or (
            task_type == "regression" and target_evidence.get("invalid_nonnumeric_rows", 0) == 0
        )
        result.add_check(
            "target_task_is_compatible",
            compatible,
            {"task_type": str(task_type), "target_semantic_type": target_kind},
            "The approved task type must match the validated target representation; choose classification for labels or a numeric regression target.",
        )

    requested_names = (
        [str(name) for name in feature_columns]
        if feature_columns is not None
        else [name for index, name in enumerate(column_names) if index != target_positions[0]]
    )
    requested_duplicates = sorted(
        {name for name in requested_names if requested_names.count(name) > 1}
    )
    missing_requested = sorted(set(requested_names) - set(column_names))
    target_in_features = str(target_column) in requested_names
    result.add_check(
        "target_absent_from_feature_matrix",
        not target_in_features and not requested_duplicates and not missing_requested,
        {
            "target_in_features": target_in_features,
            "duplicate_requested_features": requested_duplicates,
            "missing_requested_features": missing_requested,
        },
        "The target must not be supplied as a feature; pass an explicit feature list without the target and use existing unique columns.",
    )
    if target_in_features or requested_duplicates or missing_requested:
        result.add_failure(
            "feature_matrix_reference_is_valid",
            "The requested feature matrix cannot be resolved safely; correct its column references before training.",
            {
                "target_in_features": target_in_features,
                "duplicate_requested_features": requested_duplicates,
                "missing_requested_features": missing_requested,
            },
        )
        return result

    requested_positions = [column_names.index(name) for name in requested_names]
    evidence_mask = evidence_valid_mask.to_numpy(dtype=bool, copy=True)
    if split is not None and evidence_dataframe is None:
        current_positions_for_evidence = (
            np.arange(len(dataframe), dtype=int)
            if row_positions is None
            else np.asarray(row_positions, dtype=int)
        )
        if len(current_positions_for_evidence) == len(dataframe):
            evidence_mask &= np.isin(
                current_positions_for_evidence,
                np.asarray(split.train_row_positions, dtype=int),
            )
    target_for_comparison = evidence_target.iloc[np.flatnonzero(evidence_mask)].reset_index(drop=True)
    normalized_for_model = evidence_normalized_target.iloc[np.flatnonzero(evidence_mask)].reset_index(drop=True)
    frame_for_features = feature_evidence_frame.iloc[evidence_mask, requested_positions].reset_index(drop=True)
    feature_evidence_frame = feature_evidence_frame.iloc[evidence_mask].reset_index(drop=True)
    direct_copies: list[str] = []
    name_warnings: list[str] = []
    for name, position in zip(requested_names, requested_positions):
        feature = feature_evidence_frame.iloc[:, position]
        if _safe_series_equal(target_for_comparison, feature.reset_index(drop=True)):
            direct_copies.append(name)
            result.excluded_features.append(
                {"column": name, "reason_code": "direct_target_copy", "reason": "exact target copy"}
            )
        if _name_suggests_target_leakage(name, str(target_column)):
            warning = (
                f"Feature '{name}' contains the target name '{target_column}'; review its provenance for semantic leakage."
            )
            name_warnings.append(warning)

    result.direct_leakage_detected = bool(direct_copies)
    result.add_check(
        "no_direct_target_copy_features",
        not direct_copies,
        {"direct_target_copy_features": direct_copies},
        "Direct copies of the target are excluded deterministically; remove them or verify the feature definition before training.",
    )
    if direct_copies:
        result.direct_leakage_detected = True

    usable_names: list[str] = []
    inf_replacements: dict[str, int] = {}
    for name, position in zip(requested_names, requested_positions):
        if name in direct_copies:
            continue
        series = feature_evidence_frame.iloc[:, position]
        kind = semantic_type(series)
        if is_identifier(name, series):
            result.excluded_features.append(
                {"column": name, "reason_code": "identifier_like", "reason": "identifier-like feature"}
            )
            continue
        if kind in {"text", "datetime", "unknown"}:
            result.excluded_features.append(
                {"column": name, "reason_code": f"unsupported_{kind}", "reason": f"unsupported {kind} feature"}
            )
            continue
        if kind in {"categorical", "boolean"} and series.nunique(dropna=True) > MAX_CATEGORICAL_CARDINALITY:
            result.excluded_features.append(
                {
                    "column": name,
                    "reason_code": "high_cardinality_categorical",
                    "reason": f"categorical cardinality exceeds {MAX_CATEGORICAL_CARDINALITY}",
                    "unique_values": int(series.nunique(dropna=True)),
                }
            )
            continue
        if series.nunique(dropna=True) <= 1:
            result.excluded_features.append(
                {"column": name, "reason_code": "constant_feature", "reason": "constant or all-null feature"}
            )
            continue
        if pd.api.types.is_numeric_dtype(series):
            infinity_count = int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum())
            if infinity_count:
                inf_replacements[name] = infinity_count
                finite_count = int(np.isfinite(series.to_numpy(dtype=float, na_value=np.nan)).sum())
                if finite_count == 0:
                    result.excluded_features.append(
                        {
                            "column": name,
                            "reason_code": "all_values_nonfinite",
                            "reason": "numeric feature contains no finite values",
                        }
                    )
                    continue
        usable_names.append(name)

    result.features_used = usable_names
    result.warnings.extend(name_warnings)
    result.add_check(
        "usable_feature_remains",
        bool(usable_names),
        {
            "features_used": usable_names,
            "excluded_feature_count": len(result.excluded_features),
        },
        "At least one feature must remain after deterministic schema, identifier, cardinality, and leakage exclusions.",
    )
    preprocessing_records = records_from_dataframe(
        feature_evidence_frame,
        usable_names,
        result.excluded_features,
    )
    preprocessing_requirements = requirements_from_records(
        preprocessing_records,
        str(task_type),
        str(method),
    )
    result.preprocessing_requirements = preprocessing_requirements
    if approved_preprocessing is None:
        approved_preprocessing = preprocessing_requirements.expected_contract
    result.preprocessing_contract = approved_preprocessing
    result.add_check(
        "preprocessing_contract_is_supported",
        True,
        {
            "approved_preprocessing": approved_preprocessing.model_dump(mode="json"),
            "supported_values_are_schema_bound": True,
        },
        "The approved preprocessing contract contains only supported executable strategies.",
    )
    result.add_check(
        "preprocessing_is_training_only",
        approved_preprocessing.fit_inside_pipeline is True,
        {
            "fit_inside_pipeline": approved_preprocessing.fit_inside_pipeline,
            "learned_steps": [
                "numeric_imputation",
                "categorical_imputation",
                "numeric_scaling",
                "categorical_encoding",
            ],
        },
        "All learned imputation, scaling, and encoding must be fitted inside the scikit-learn pipeline on training folds only.",
    )
    numeric_missing = int(preprocessing_requirements.evidence["numeric_missing_values"])
    categorical_missing = int(preprocessing_requirements.evidence["categorical_missing_values"])
    infinity_count = int(preprocessing_requirements.evidence["infinity_values"])
    categorical_features = preprocessing_requirements.evidence["categorical_features"]
    numeric_features = preprocessing_requirements.evidence["numeric_features"]
    result.add_check(
        "numeric_missing_values_are_handled",
        not numeric_missing or approved_preprocessing.numeric_imputation != "none",
        {
            "numeric_missing_values": numeric_missing,
            "numeric_imputation": approved_preprocessing.numeric_imputation,
        },
        "Observed numeric missing or infinite values require a supported numeric imputation strategy.",
    )
    result.add_check(
        "categorical_missing_values_are_handled",
        not categorical_missing or approved_preprocessing.categorical_imputation != "none",
        {
            "categorical_missing_values": categorical_missing,
            "categorical_imputation": approved_preprocessing.categorical_imputation,
        },
        "Observed categorical missing values require a supported categorical imputation strategy.",
    )
    result.add_check(
        "linear_numeric_features_use_approved_scaling_policy",
        not (method in {"linear", "regularized_linear"} and numeric_features)
        or approved_preprocessing.numeric_scaling == "standard",
        {
            "method": str(method),
            "numeric_features": numeric_features,
            "numeric_scaling": approved_preprocessing.numeric_scaling,
        },
        "Linear and regularized-linear methods require standard numeric scaling under the project policy.",
    )
    result.add_check(
        "categorical_features_use_safe_encoding",
        not categorical_features or approved_preprocessing.categorical_encoding != "none",
        {
            "categorical_features": categorical_features,
            "categorical_encoding": approved_preprocessing.categorical_encoding,
        },
        "Usable categorical features require a supported categorical encoder; unencoded categories cannot enter the estimator.",
    )
    result.add_check(
        "categorical_unknown_values_are_handled_safely",
        not categorical_features
        or approved_preprocessing.categorical_unknown_handling
        in {"ignore", "use_encoded_value"},
        {
            "categorical_features": categorical_features,
            "categorical_unknown_handling": approved_preprocessing.categorical_unknown_handling,
        },
        "Categorical preprocessing must handle categories absent from a training fold without raising or leaking holdout information.",
    )
    result.add_check(
        "boosted_tree_encoding_is_compatible",
        not (method == "boosted_tree" and categorical_features)
        or approved_preprocessing.categorical_encoding == "ordinal",
        {
            "method": str(method),
            "categorical_encoding": approved_preprocessing.categorical_encoding,
        },
        "Boosted trees use bounded ordinal encoding so the estimator never receives an unsafe dense one-hot matrix.",
    )
    result.add_check(
        "numeric_infinity_values_are_handled",
        not infinity_count or approved_preprocessing.infinity_handling == "replace_with_missing",
        {
            "infinity_values": infinity_count,
            "infinity_handling": approved_preprocessing.infinity_handling,
        },
        "Observed numeric infinities must be replaced with missing values before training-only imputation or the run must fail closed.",
    )
    for field_name, names, safe_value, message in (
        (
            "identifier_handling",
            preprocessing_requirements.evidence["identifier_features"],
            "exclude",
            "Identifier-like features are mandatory exclusions and cannot be retained by an agent or reconciliation plan.",
        ),
        (
            "high_cardinality_handling",
            preprocessing_requirements.evidence["high_cardinality_features"],
            "exclude",
            "High-cardinality categorical features are mandatory exclusions under the compact safe baseline.",
        ),
        (
            "unsupported_text_handling",
            preprocessing_requirements.evidence["unsupported_text_features"],
            "exclude",
            "Unsupported text features cannot be retained without an implemented safe text transformer.",
        ),
        (
            "datetime_handling",
            preprocessing_requirements.evidence["datetime_features"],
            "exclude",
            "Datetime features cannot be retained without an implemented temporal transformer.",
        ),
    ):
        result.add_check(
            f"{field_name}_is_safe",
            not names or getattr(approved_preprocessing, field_name) == safe_value,
            {"features": names, "approved_policy": getattr(approved_preprocessing, field_name)},
            message,
        )
    estimated_one_hot = int(preprocessing_requirements.evidence["estimated_one_hot_features"])
    max_one_hot = int(preprocessing_requirements.evidence["max_one_hot_features"])
    result.add_check(
        "one_hot_matrix_is_memory_safe",
        approved_preprocessing.categorical_encoding != "one_hot"
        or estimated_one_hot <= max_one_hot,
        {
            "categorical_encoding": approved_preprocessing.categorical_encoding,
            "estimated_one_hot_features": estimated_one_hot,
            "max_one_hot_features": max_one_hot,
            "representation": "sparse_output=True",
        },
        "One-hot encoding is bounded and sparse; an unreasonable encoded matrix fails closed before fitting.",
    )
    result.add_check(
        "numeric_infinity_policy_is_deterministic",
        True,
        {
            "converted_to_missing": inf_replacements,
            "policy": approved_preprocessing.infinity_handling,
        },
        "Numeric infinity handling is fixed by the approved contract before training-only imputation.",
    )

    if name_warnings:
        result.add_check(
            "target_name_leakage_review",
            True,
            {"warnings": name_warnings},
            "Name-based leakage indicators require domain review and are advisory unless values prove a direct copy.",
            severity="warning",
        )

    if usable_names:
        conflict_groups, conflict_rows = _conflicting_feature_groups(
            frame_for_features[usable_names], normalized_for_model
        )
    else:
        conflict_groups, conflict_rows = 0, 0
    result.add_check(
        "identical_feature_rows_have_consistent_targets",
        conflict_groups == 0,
        {"conflicting_groups": conflict_groups, "conflicting_rows": conflict_rows},
        "Identical usable feature rows cannot map to conflicting targets; inspect duplicates or correct the source data.",
    )
    result.direct_leakage_detected = result.direct_leakage_detected or conflict_groups > 0

    if split is None:
        split_evidence = _validate_split_and_cv(
            normalized_for_model,
            str(task_type),
            test_size,
            random_state,
        )
    else:
        current_positions = (
            np.arange(len(dataframe), dtype=int)
            if row_positions is None
            else np.asarray(row_positions, dtype=int)
        )
        if len(current_positions) != len(dataframe):
            split_evidence = {
                "test_size": _safe_float(test_size),
                "valid_rows": int(len(normalized_for_model)),
                "checks": [
                    {
                        "code": "frozen_split_row_mapping_is_valid",
                        "passed": False,
                        "evidence": {"mapped_rows": len(current_positions), "dataframe_rows": len(dataframe)},
                        "message": "The frozen split row-position mapping must have one entry per dataframe row.",
                    }
                ],
            }
        else:
            split_positions = current_positions[valid_mask.to_numpy(dtype=bool)]
            split_evidence = _validate_frozen_split(
                normalized_target.loc[valid_mask].reset_index(drop=True),
                str(target_column),
                str(task_type),
                test_size,
                random_state,
                split,
                split_positions,
            )
    result.split = split_evidence
    for check in split_evidence.pop("checks", []):
        result.add_check(
            check["code"],
            check["passed"],
            check["evidence"],
            check["message"],
        )
    result.split = split_evidence

    if method == "boosted_tree":
        estimated_one_hot = int(
            sum(
                feature_evidence_frame.iloc[:, position].nunique(dropna=True) + 1
                for name, position in zip(requested_names, requested_positions)
                if name in usable_names
                and semantic_type(feature_evidence_frame.iloc[:, position])
                in {"categorical", "boolean"}
            )
            + sum(
                1
                for name, position in zip(requested_names, requested_positions)
                if name in usable_names
                and pd.api.types.is_numeric_dtype(feature_evidence_frame.iloc[:, position])
            )
        )
        result.add_check(
            "boosted_tree_encoding_is_memory_safe",
            True,
            {
                "estimated_one_hot_columns": estimated_one_hot,
                "categorical_encoding": "ordinal_encoding_for_boosted_tree",
                "dense_one_hot_path": "not used",
            },
            "Boosted trees use bounded ordinal categorical encoding instead of a dense one-hot expansion.",
        )

    return result


def prepare_validated_frame(dataframe: pd.DataFrame, result: ValidationResult) -> pd.DataFrame:
    """Apply only the normalization already described by a passed report."""

    result.raise_if_failed()
    column_names = [str(column) for column in dataframe.columns]
    target_positions = [index for index, name in enumerate(column_names) if name == result.target_column]
    if len(target_positions) != 1:
        raise InvariantViolation(
            f"[target_resolves_once] Target '{result.target_column}' no longer resolves to exactly one column.",
            result,
        )
    target = dataframe.iloc[:, target_positions[0]]
    valid_mask, normalized_target, _ = _normalize_target(target, result.task_type)
    frame = dataframe.loc[valid_mask].reset_index(drop=True).copy()
    frame[result.target_column] = normalized_target.loc[valid_mask].reset_index(drop=True).to_numpy()
    for name in result.features_used:
        if pd.api.types.is_numeric_dtype(frame[name]):
            frame[name] = frame[name].replace([np.inf, -np.inf], np.nan)
    return frame


def modeling_arrays(
    dataframe: pd.DataFrame, result: ValidationResult
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the exact validated feature matrix and target used by the fitter."""

    frame = prepare_validated_frame(dataframe, result)
    return frame[result.features_used].copy(), frame[result.target_column].copy()


def training_profile_frame(
    dataframe: pd.DataFrame,
    target_column: str | None,
    task_type: str | None,
    *,
    test_size: float,
    random_state: int,
    split: FrozenSplit | None = None,
) -> pd.DataFrame:
    """Return a deterministic training-only view for planning/reconciliation prompts."""

    if split is None:
        if target_column is None or task_type is None:
            raise InvariantViolation(
                "A target and task must be established before constructing a training-only planning profile."
            )
        split = freeze_supervised_split(
            dataframe,
            target_column,
            task_type,
            test_size=test_size,
            random_state=random_state,
        )
    if split.target_column != str(target_column) or split.task_type != str(task_type):
        raise InvariantViolation("The planning profile target/task does not match the frozen split contract.")
    return dataframe.iloc[list(split.train_row_positions)].reset_index(drop=True)


def training_partition_frame(
    dataframe: pd.DataFrame,
    split: FrozenSplit,
    row_positions: Sequence[int],
) -> pd.DataFrame:
    """Return only cleaned rows mapped to the frozen training partition.

    The mapping is expressed in original source-row positions because cleaning
    can remove rows and reset the dataframe index.  This helper validates the
    complete frozen membership contract before selecting rows and fails closed
    if the caller supplies an unreconcilable mapping.
    """

    raw_positions = np.asarray(row_positions)
    if raw_positions.ndim != 1 or len(raw_positions) != len(dataframe):
        raise InvariantViolation(
            "The row-position mapping must be one-dimensional with one entry per cleaned dataframe row."
        )
    try:
        positions = raw_positions.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvariantViolation("The row-position mapping must contain integer source-row positions.") from exc
    if not np.array_equal(raw_positions, positions):
        raise InvariantViolation("The row-position mapping must contain integer source-row positions.")
    if len(np.unique(positions)) != len(positions):
        raise InvariantViolation("The row-position mapping must not contain duplicate source-row positions.")

    valid_set = set(int(value) for value in split.valid_row_positions)
    train_set = set(int(value) for value in split.train_row_positions)
    holdout_set = set(int(value) for value in split.holdout_row_positions)
    if train_set & holdout_set or valid_set != train_set | holdout_set:
        raise InvariantViolation("The frozen split contract has inconsistent train/holdout membership.")
    if not set(int(value) for value in positions).issubset(valid_set):
        raise InvariantViolation(
            "Cleaned rows contain source positions outside the valid rows in the frozen split contract."
        )

    training_mask = np.isin(positions, np.asarray(tuple(train_set), dtype=np.int64))
    holdout_mask = np.isin(positions, np.asarray(tuple(holdout_set), dtype=np.int64))
    if np.any(training_mask & holdout_mask) or not np.all(training_mask | holdout_mask):
        raise InvariantViolation(
            "Every cleaned row must resolve to exactly one frozen training or holdout partition."
        )
    training_positions = positions[training_mask]
    if not set(int(value) for value in training_positions).issubset(train_set):
        raise InvariantViolation("The EDA frame contains a source position outside the frozen training partition.")

    training_frame = dataframe.loc[training_mask].reset_index(drop=True).copy()
    if len(training_frame) != len(training_positions):
        raise InvariantViolation("The training-only EDA frame could not be reconciled with source positions.")
    return training_frame


def freeze_supervised_split(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> FrozenSplit:
    """Establish one fail-closed train/holdout partition for the whole run."""

    columns = [str(column) for column in dataframe.columns]
    positions = [index for index, name in enumerate(columns) if name == str(target_column)]
    if len(positions) != 1:
        _raise_split_failure(
            target_column,
            task_type,
            "target_resolves_once",
            f"Target '{target_column}' must resolve to exactly one column.",
        )
    if task_type not in SUPPORTED_TASKS:
        _raise_split_failure(
            target_column,
            task_type,
            "task_is_supported",
            f"Unsupported task type: {task_type!r}.",
        )
    if not _valid_test_size(test_size):
        _raise_split_failure(
            target_column,
            task_type,
            "test_size_is_supported",
            f"test_size must be finite and between {MIN_TEST_SIZE:.2f} and {MAX_TEST_SIZE:.2f}.",
        )

    target = dataframe.iloc[:, positions[0]].copy()
    valid_mask, normalized_target, target_evidence = _normalize_target(target, task_type)
    if task_type == "regression" and (
        target_evidence["invalid_nonnumeric_rows"] or target_evidence["nonfinite_rows"]
    ):
        _raise_split_failure(
            target_column,
            task_type,
            "target_is_valid_before_split",
            "Regression target values must be finite and numeric before the holdout is frozen.",
        )
    valid_target = normalized_target.loc[valid_mask].reset_index(drop=True)
    split_evidence = _validate_split_and_cv(valid_target, task_type, test_size, random_state)
    failed = [check for check in split_evidence.get("checks", []) if not check["passed"]]
    if task_type == "classification" and valid_target.nunique(dropna=True) < 2:
        failed.append(
            {
                "code": "classification_target_has_two_classes",
                "message": "Classification requires at least two valid classes before the holdout is frozen.",
            }
        )
    if failed:
        first = failed[0]
        _raise_split_failure(
            target_column,
            task_type,
            first["code"],
            first["message"],
            first.get("evidence", split_evidence),
        )

    valid_positions = np.flatnonzero(valid_mask.to_numpy(dtype=bool))
    stratify = valid_target if task_type == "classification" else None
    train_indices, holdout_indices = train_test_split(
        np.arange(len(valid_positions)),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return FrozenSplit(
        target_column=str(target_column),
        task_type=str(task_type),
        test_size=float(test_size),
        random_state=int(random_state),
        strategy="stratified" if task_type == "classification" else "seeded_random",
        dataset_fingerprint=dataset_fingerprint(dataframe),
        valid_row_positions=tuple(int(value) for value in valid_positions),
        train_row_positions=tuple(int(valid_positions[value]) for value in np.sort(train_indices)),
        holdout_row_positions=tuple(int(valid_positions[value]) for value in np.sort(holdout_indices)),
    )


def dataset_fingerprint(dataframe: pd.DataFrame) -> str:
    """Return a stable lightweight integrity fingerprint for a dataframe."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"columns": [str(column) for column in dataframe.columns], "dtypes": [str(dtype) for dtype in dataframe.dtypes]},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(dataframe, index=False).to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def _raise_split_failure(
    target_column: str,
    task_type: str,
    code: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    result = ValidationResult(
        target_column=str(target_column),
        task_type=str(task_type),
        method="not_selected",
    )
    result.add_failure(code, message, evidence or {})
    raise InvariantViolation.from_result(result)


def validated_row_positions(
    dataframe: pd.DataFrame,
    result: ValidationResult,
    row_positions: Sequence[int] | None = None,
) -> np.ndarray:
    """Map the validated modeling rows back to original positional row ids."""

    positions = np.arange(len(dataframe), dtype=int) if row_positions is None else np.asarray(row_positions, dtype=int)
    if len(positions) != len(dataframe):
        raise InvariantViolation("The row-position mapping must have one entry per dataframe row.", result)
    target_positions = [index for index, name in enumerate(map(str, dataframe.columns)) if name == result.target_column]
    if len(target_positions) != 1:
        raise InvariantViolation("The validated target no longer resolves to exactly one column.", result)
    valid_mask, _, _ = _normalize_target(dataframe.iloc[:, target_positions[0]], result.task_type)
    return positions[valid_mask.to_numpy(dtype=bool)]


def _normalize_target(
    target: pd.Series, task_type: str
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    missing = target.isna()
    artificial = pd.Series(False, index=target.index)
    if task_type == "classification":
        nonmissing = target.loc[~missing]
        string_values = nonmissing.astype("string").str.strip().str.casefold()
        artificial_nonmissing = string_values.isin(
            {"", "nan", "none", "null", "nat", "<na>"}
        ).fillna(False)
        artificial.loc[artificial_nonmissing.index] = artificial_nonmissing
        valid_mask = ~(missing | artificial)
        normalized = pd.Series(pd.NA, index=target.index, dtype="string")
        normalized.loc[nonmissing.index] = nonmissing.astype("string").str.strip()
        evidence = {
            "missing_rows": int(missing.sum()),
            "artificial_null_labels_removed": int((artificial & ~missing).sum()),
            "invalid_nonnumeric_rows": 0,
            "coercible_rows": int(valid_mask.sum()),
            "nonfinite_rows": 0,
        }
        return valid_mask, normalized, evidence

    numeric = pd.to_numeric(target, errors="coerce")
    invalid = target.notna() & numeric.isna()
    valid_mask = target.notna() & ~invalid
    nonfinite = valid_mask & ~np.isfinite(numeric.fillna(0))
    evidence = {
        "missing_rows": int(target.isna().sum()),
        "artificial_null_labels_removed": 0,
        "invalid_nonnumeric_rows": int(invalid.sum()),
        "coercible_rows": int(valid_mask.sum()),
        "nonfinite_rows": int(nonfinite.sum()),
    }
    return valid_mask, numeric, evidence


def _validate_split_and_cv(
    target: pd.Series,
    task_type: str,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {"test_size": _safe_float(test_size), "valid_rows": int(len(target))}
    checks: list[dict[str, Any]] = []
    if not _valid_test_size(test_size):
        checks.append(
            {
                "code": "split_has_nonempty_partitions",
                "passed": False,
                "evidence": evidence,
                "message": "The holdout fraction is invalid; use a finite test_size between 0.10 and 0.50.",
            }
        )
        evidence["checks"] = checks
        return evidence

    test_rows = ceil(len(target) * float(test_size))
    train_rows = len(target) - test_rows
    evidence.update({"train_rows": train_rows, "holdout_rows": test_rows})
    nonempty = train_rows > 0 and test_rows > 0
    checks.append(
        {
            "code": "split_has_nonempty_partitions",
            "passed": nonempty,
            "evidence": {"train_rows": train_rows, "holdout_rows": test_rows},
            "message": "The split must leave at least one training row and one holdout row; increase the dataset or adjust test_size.",
        }
    )
    if task_type == "classification":
        counts = target.value_counts()
        class_count = int(len(counts))
        evidence["class_count"] = class_count
        evidence["class_counts"] = {str(key): int(value) for key, value in counts.items()}
        feasible_shape = (
            class_count >= 2
            and test_rows >= class_count
            and train_rows >= class_count * 2
            and int(counts.min()) >= 2
        )
        checks.append(
            {
                "code": "classification_split_and_stratification_feasible",
                "passed": feasible_shape,
                "evidence": {
                    "class_count": class_count,
                    "minimum_class_rows": int(counts.min()) if len(counts) else 0,
                    "train_rows": train_rows,
                    "holdout_rows": test_rows,
                },
                "message": "Each class must be representable in the stratified holdout and retain at least two training rows for cross-validation.",
            }
        )
        split_ok = False
        min_train_class = 0
        min_holdout_class = 0
        if feasible_shape:
            try:
                indices = np.arange(len(target))
                train_indices, holdout_indices = train_test_split(
                    indices,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=target,
                )
                train_counts = target.iloc[train_indices].value_counts()
                holdout_counts = target.iloc[holdout_indices].value_counts()
                min_train_class = int(train_counts.min())
                min_holdout_class = int(holdout_counts.min())
                split_ok = (
                    len(train_counts) == class_count
                    and len(holdout_counts) == class_count
                    and min_train_class >= 2
                    and min_holdout_class >= 1
                )
            except ValueError:
                split_ok = False
        cv_folds = min(MAX_CV_FOLDS, min_train_class) if split_ok else 0
        checks.append(
            {
                "code": "classification_training_supports_cross_validation",
                "passed": split_ok and cv_folds >= 2,
                "evidence": {
                    "minimum_training_class_rows": min_train_class,
                    "minimum_holdout_class_rows": min_holdout_class,
                    "cv_folds": cv_folds,
                    "cv_strategy": "stratified_kfold",
                },
                "message": "Every training fold must contain every class; provide at least two rows per class after the holdout split.",
            }
        )
        evidence.update(
            {
                "cv_folds": cv_folds,
                "cv_strategy": "stratified_kfold",
                "stratification": "required",
            }
        )
    else:
        cv_folds = min(MAX_CV_FOLDS, train_rows)
        feasible = len(target) >= 4 and nonempty and train_rows >= 2 and cv_folds >= 2
        checks.append(
            {
                "code": "regression_split_and_cross_validation_feasible",
                "passed": feasible,
                "evidence": {
                    "valid_rows": len(target),
                    "train_rows": train_rows,
                    "holdout_rows": test_rows,
                    "cv_folds": cv_folds,
                    "cv_strategy": "kfold",
                },
                "message": "Regression needs enough finite observations for a nonempty holdout and at least two training rows per validation strategy.",
            }
        )
        evidence.update({"cv_folds": cv_folds, "cv_strategy": "kfold"})
    evidence["checks"] = checks
    return evidence


def _validate_frozen_split(
    target: pd.Series,
    target_column: str,
    task_type: str,
    test_size: float,
    random_state: int,
    split: FrozenSplit,
    valid_row_positions: Sequence[int],
) -> dict[str, Any]:
    """Validate the already-frozen membership without reconstructing it."""

    positions = np.asarray(valid_row_positions, dtype=int)
    evidence: dict[str, Any] = {
        "test_size": _safe_float(test_size),
        "valid_rows": int(len(target)),
        "strategy": split.strategy,
        "random_state": int(random_state),
        "contract": split.as_dict(),
        "membership_verified": False,
    }
    checks: list[dict[str, Any]] = []
    config_matches = (
        split.target_column == target_column
        and split.task_type == task_type
        and float(split.test_size) == float(test_size)
        and int(split.random_state) == int(random_state)
    )
    checks.append(
        {
            "code": "frozen_split_contract_matches_run",
            "passed": config_matches,
            "evidence": {
                "contract_target": split.target_column,
                "contract_task": split.task_type,
                "task_type": task_type,
                "contract_test_size": float(split.test_size),
                "test_size": _safe_float(test_size),
                "contract_random_state": int(split.random_state),
                "random_state": int(random_state),
            },
            "message": "Planning, validation, and modeling must use the same frozen target/task, holdout fraction, and seed.",
        }
    )
    known = set(split.valid_row_positions)
    rows_known = all(int(position) in known for position in positions)
    checks.append(
        {
            "code": "frozen_split_membership_is_known",
            "passed": rows_known,
            "evidence": {"unknown_positions": [int(position) for position in positions if int(position) not in known][:10]},
            "message": "Every validated row must belong to the original supervised split contract.",
        }
    )
    train_set = set(split.train_row_positions)
    holdout_set = set(split.holdout_row_positions)
    train_mask = np.array([int(position) in train_set for position in positions], dtype=bool)
    holdout_mask = np.array([int(position) in holdout_set for position in positions], dtype=bool)
    disjoint = not bool(np.any(train_mask & holdout_mask))
    covered = bool(np.all(train_mask | holdout_mask)) if len(positions) else False
    checks.append(
        {
            "code": "frozen_split_membership_is_disjoint_and_complete",
            "passed": rows_known and disjoint and covered,
            "evidence": {
                "train_rows": int(train_mask.sum()),
                "holdout_rows": int(holdout_mask.sum()),
                "unassigned_rows": int((~(train_mask | holdout_mask)).sum()),
            },
            "message": "Validated rows must map to exactly one of the frozen training or holdout partitions.",
        }
    )
    train_rows = int(train_mask.sum())
    holdout_rows = int(holdout_mask.sum())
    nonempty = train_rows > 0 and holdout_rows > 0
    checks.append(
        {
            "code": "split_has_nonempty_partitions",
            "passed": nonempty,
            "evidence": {"train_rows": train_rows, "holdout_rows": holdout_rows},
            "message": "The frozen split must retain at least one training row and one holdout row after structural cleaning.",
        }
    )
    if task_type == "classification":
        train_target = target.iloc[np.flatnonzero(train_mask)]
        holdout_target = target.iloc[np.flatnonzero(holdout_mask)]
        all_classes = set(target.astype("string").tolist())
        train_counts = train_target.value_counts()
        holdout_counts = holdout_target.value_counts()
        class_count = len(all_classes)
        min_train = int(train_counts.min()) if len(train_counts) else 0
        min_holdout = int(holdout_counts.min()) if len(holdout_counts) else 0
        class_safe = (
            class_count >= 2
            and len(train_counts) == class_count
            and len(holdout_counts) == class_count
            and min_train >= 2
            and min_holdout >= 1
        )
        checks.append(
            {
                "code": "classification_split_and_stratification_feasible",
                "passed": class_safe,
                "evidence": {
                    "class_count": class_count,
                    "train_class_counts": {str(key): int(value) for key, value in train_counts.items()},
                    "holdout_class_counts": {str(key): int(value) for key, value in holdout_counts.items()},
                    "minimum_training_class_rows": min_train,
                    "minimum_holdout_class_rows": min_holdout,
                },
                "message": "The frozen stratified holdout must retain every class and at least two training rows per class for cross-validation.",
            }
        )
        cv_folds = min(MAX_CV_FOLDS, min_train) if class_safe else 0
        checks.append(
            {
                "code": "classification_training_supports_cross_validation",
                "passed": class_safe and cv_folds >= 2,
                "evidence": {"cv_folds": cv_folds, "cv_strategy": "stratified_kfold"},
                "message": "Every training fold must contain every class; the frozen training partition must support at least two folds.",
            }
        )
        evidence.update({"cv_folds": cv_folds, "cv_strategy": "stratified_kfold", "stratification": "required"})
    else:
        cv_folds = min(MAX_CV_FOLDS, train_rows)
        regression_safe = len(target) >= 4 and nonempty and train_rows >= 2 and cv_folds >= 2
        checks.append(
            {
                "code": "regression_split_and_cross_validation_feasible",
                "passed": regression_safe,
                "evidence": {"train_rows": train_rows, "holdout_rows": holdout_rows, "cv_folds": cv_folds, "cv_strategy": "kfold"},
                "message": "The frozen regression split must support a nonempty holdout and at least two training CV folds.",
            }
        )
        evidence.update({"cv_folds": cv_folds, "cv_strategy": "kfold"})
    evidence["train_rows"] = train_rows
    evidence["holdout_rows"] = holdout_rows
    evidence["membership_verified"] = rows_known and disjoint and covered
    evidence["checks"] = checks
    return evidence


def _safe_series_equal(left: pd.Series, right: pd.Series) -> bool:
    if len(left) != len(right):
        return False
    left_missing = left.isna().to_numpy()
    right_missing = right.isna().to_numpy()
    if not np.array_equal(left_missing, right_missing):
        return False
    left_nonmissing = left.loc[~left_missing]
    right_nonmissing = right.loc[~right_missing]
    if left_nonmissing.empty:
        return True
    try:
        if left_nonmissing.reset_index(drop=True).equals(right_nonmissing.reset_index(drop=True)):
            return True
    except (TypeError, ValueError):
        pass
    left_numeric = pd.to_numeric(left_nonmissing, errors="coerce")
    right_numeric = pd.to_numeric(right_nonmissing, errors="coerce")
    if left_numeric.notna().all() and right_numeric.notna().all():
        return bool(
            np.allclose(
                left_numeric.to_numpy(dtype=float),
                right_numeric.to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-15,
            )
        )
    left_text = left_nonmissing.astype("string").str.strip().reset_index(drop=True)
    right_text = right_nonmissing.astype("string").str.strip().reset_index(drop=True)
    return bool(left_text.equals(right_text))


def _conflicting_feature_groups(features: pd.DataFrame, target: pd.Series) -> tuple[int, int]:
    if features.empty:
        return 0, 0
    key_frame = features.astype("string").fillna("<MISSING>")
    keys = key_frame.agg("\x1f".join, axis=1)
    target_keys = target.astype("string").fillna("<MISSING>").reset_index(drop=True)
    grouped = pd.DataFrame({"key": keys, "target": target_keys}).groupby("key", sort=False)["target"]
    conflicts = grouped.nunique(dropna=False)
    conflicting_keys = conflicts[conflicts > 1].index
    if len(conflicting_keys) == 0:
        return 0, 0
    rows = int(keys.isin(conflicting_keys).sum())
    return int(len(conflicting_keys)), rows


def _name_suggests_target_leakage(feature_name: str, target_name: str) -> bool:
    if feature_name.casefold() == target_name.casefold() or len(target_name.strip()) < 3:
        return False
    return target_name.casefold() in feature_name.casefold()


def _valid_test_size(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(numeric) and MIN_TEST_SIZE <= numeric <= MAX_TEST_SIZE


def _safe_float(value: Any) -> float | str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return numeric if isfinite(numeric) else str(value)


def _position_digest(positions: Sequence[int]) -> str:
    values = np.asarray(tuple(int(position) for position in positions), dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _head_training_frame(dataframe: pd.DataFrame, test_size: float) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    test_rows = min(max(1, ceil(len(dataframe) * float(test_size))), len(dataframe) - 1)
    return dataframe.iloc[: len(dataframe) - test_rows].reset_index(drop=True)
