"""Small, pairwise, training-only evidence for challenged model-family disagreements.

This module is deliberately separate from ``evaluation.empirical_reference``.
The runtime probe evaluates exactly two already-proposed plans after hard
validation, before any soft reconciliation decision. It never sees a holdout
frame and it never returns a final modeling decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from app.modeling import _estimator
from app.preprocessing import build_preprocessor
from app.schemas import Method, PreprocessingContract, TaskType
from app.validation import (
    ValidationResult,
    modeling_arrays,
    prepare_validated_frame,
    validate_training_plan,
)


EMPIRICAL_PROBE_POLICY_VERSION = "v1"
_EPSILON = 1e-12


@dataclass(frozen=True)
class EmpiricalProbePolicy:
    """Frozen, bounded controls for the runtime pairwise comparison."""

    enabled: bool = True
    policy_version: str = EMPIRICAL_PROBE_POLICY_VERSION
    cv_folds: int = 3
    minimum_rows: int = 12
    max_rows: int | None = None
    max_configs_per_family: int = 1
    tie_relative_threshold: float = 0.02
    moderate_relative_threshold: float = 0.10
    strong_relative_threshold: float = 0.20
    variability_tie_multiplier: float = 0.50
    variability_weak_multiplier: float = 0.50
    minimum_consistent_win_rate: float = 2.0 / 3.0
    random_state: int | None = None

    def __post_init__(self) -> None:
        if self.cv_folds < 2 or self.cv_folds > 3:
            raise ValueError("Empirical probe cv_folds must be 2 or 3.")
        if self.minimum_rows < 4:
            raise ValueError("Empirical probe minimum_rows must be at least four.")
        if self.max_rows is not None and self.max_rows < self.minimum_rows:
            raise ValueError("max_rows must be at least minimum_rows when supplied.")
        if self.max_configs_per_family < 1 or self.max_configs_per_family > 3:
            raise ValueError("max_configs_per_family must be between one and three.")
        thresholds = (
            self.tie_relative_threshold,
            self.moderate_relative_threshold,
            self.strong_relative_threshold,
        )
        if any(float(value) < 0 for value in thresholds):
            raise ValueError("Evidence thresholds must be non-negative.")
        if not (
            self.tie_relative_threshold
            <= self.moderate_relative_threshold
            <= self.strong_relative_threshold
        ):
            raise ValueError("Evidence thresholds must be ordered from tie to strong.")
        if self.variability_tie_multiplier < 0 or self.variability_weak_multiplier < 0:
            raise ValueError("Variability multipliers must be non-negative.")
        if not 0.0 < self.minimum_consistent_win_rate <= 1.0:
            raise ValueError("minimum_consistent_win_rate must be in (0, 1].")

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "policy_version": self.policy_version,
            "cv_folds": self.cv_folds,
            "minimum_rows": self.minimum_rows,
            "max_rows": self.max_rows,
            "max_configs_per_family": self.max_configs_per_family,
            "tie_relative_threshold": self.tie_relative_threshold,
            "moderate_relative_threshold": self.moderate_relative_threshold,
            "strong_relative_threshold": self.strong_relative_threshold,
            "variability_tie_multiplier": self.variability_tie_multiplier,
            "variability_weak_multiplier": self.variability_weak_multiplier,
            "minimum_consistent_win_rate": self.minimum_consistent_win_rate,
            "random_state": self.random_state,
        }


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _proposal_method(proposal: Any) -> Method:
    method = _value(proposal, "recommended_method")
    if method not in {"linear", "regularized_linear", "tree_ensemble", "boosted_tree"}:
        raise ValueError(f"Unsupported empirical probe model family: {method!r}")
    return method


def _proposal_preprocessing(proposal: Any) -> PreprocessingContract:
    value = _value(proposal, "preprocessing", {})
    return value if isinstance(value, PreprocessingContract) else PreprocessingContract.model_validate(value)


def _unavailable(
    *,
    task_type: str,
    policy: EmpiricalProbePolicy,
    reason: str,
    status: str = "unavailable",
    error: str | None = None,
    training_rows: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "policy_version": policy.policy_version,
        "task_type": task_type,
        "metric": "macro_f1" if task_type == "classification" else "rmse",
        "higher_is_better": task_type == "classification",
        "cv_folds": 0,
        "cv_strategy": None,
        "training_rows": training_rows,
        "data_used": "frozen_training_partition_only",
        "holdout_used": False,
        "fit_count": 0,
        "candidate_configurations": {},
        "winner": "tie",
        "relative_advantage": 0.0,
        "normalized_advantage": 0.0,
        "difference": None,
        "evidence_strength": "tie",
        "reason": reason,
    }
    if error:
        result["error"] = error
    return result


def _limit_rows(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    task_type: TaskType,
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, bool]:
    if max_rows is None or len(features) <= max_rows:
        return features, target, False
    if task_type == "classification":
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_rows, random_state=random_state)
        positions, _ = next(splitter.split(features, target))
    else:
        rng = np.random.RandomState(random_state)
        positions = np.sort(rng.choice(len(features), size=max_rows, replace=False))
    # Preserve the selected labels so the second proposal receives exactly
    # the same deterministic subsample rather than merely the first rows.
    return features.iloc[positions], target.iloc[positions], True


def _prepare_candidate_data(
    training_frame: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    method: Method,
    preprocessing: PreprocessingContract,
    *,
    random_state: int,
    validation: ValidationResult | None,
) -> tuple[pd.DataFrame, pd.Series, ValidationResult]:
    if validation is None:
        validation = validate_training_plan(
            training_frame,
            target_column,
            task_type,
            method,
            test_size=0.2,
            random_state=random_state,
            preprocessing=preprocessing,
            training_only=True,
        )
    if validation.status != "passed":
        raise ValueError(
            f"{method} hard validation was not passed: "
            + ", ".join(check.code for check in validation.failed_checks)
        )
    # The validation contract supplies the approved feature set and target
    # normalization.  No learned transformation is fitted here; the returned
    # feature matrix is only the input to the fold-local sklearn pipeline.
    validated = prepare_validated_frame(training_frame, validation)
    features, target = modeling_arrays(validated, validation)
    return features, target, validation


def _metric_scores(
    task_type: TaskType,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    splitter: Any,
) -> list[float]:
    metric_name = "macro_f1" if task_type == "classification" else "rmse"
    scoring = "f1_macro" if task_type == "classification" else "neg_root_mean_squared_error"
    scores = cross_validate(
        pipeline,
        features,
        target,
        cv=splitter,
        scoring={"primary": scoring},
        error_score="raise",
        return_train_score=False,
    )["test_primary"]
    values = np.asarray(scores, dtype=float)
    if metric_name == "rmse":
        values = -values
    return [float(value) for value in values]


def _fold_wins(
    scores_a: list[float],
    scores_b: list[float],
    *,
    higher_is_better: bool,
) -> tuple[int, int, int]:
    a_wins = b_wins = ties = 0
    for left, right in zip(scores_a, scores_b):
        if np.isclose(left, right, rtol=1e-12, atol=1e-12):
            ties += 1
        elif (left > right) == higher_is_better:
            a_wins += 1
        else:
            b_wins += 1
    return a_wins, b_wins, ties


def _evidence_strength(
    relative_advantage: float,
    difference: float,
    std_a: float,
    std_b: float,
    winner: str,
    fold_wins: int,
    fold_count: int,
    policy: EmpiricalProbePolicy,
) -> str:
    if winner == "tie":
        return "tie"
    variability = max(std_a, std_b, _EPSILON)
    if (
        relative_advantage <= policy.tie_relative_threshold
        or abs(difference) <= policy.variability_tie_multiplier * variability
    ):
        return "tie"
    win_rate = fold_wins / max(fold_count, 1)
    if (
        relative_advantage >= policy.strong_relative_threshold
        and win_rate >= policy.minimum_consistent_win_rate
        and abs(difference) >= variability
    ):
        return "strong"
    if (
        relative_advantage >= policy.moderate_relative_threshold
        and win_rate >= policy.minimum_consistent_win_rate
        and abs(difference) >= policy.variability_weak_multiplier * variability
    ):
        return "moderate"
    # Any non-tie result below the moderate boundary is weak.  A separate
    # weak threshold would be dead configuration because both branches return
    # the same evidence class.
    return "weak"


def run_pairwise_model_probe(
    training_frame: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    proposal_a: Any,
    proposal_b: Any,
    *,
    policy: EmpiricalProbePolicy | None = None,
    random_state: int | None = None,
    validation_by_method: Mapping[str, ValidationResult] | None = None,
) -> dict[str, Any]:
    """Compare exactly Proposal A and Proposal B on training rows only.

    ``training_frame`` is intentionally the only dataframe argument.  The
    modeling gate constructs it from the frozen training partition before
    calling this function, which makes final holdout isolation explicit at the
    call boundary.
    """

    configured = policy or EmpiricalProbePolicy()
    if not configured.enabled:
        return _unavailable(task_type=task_type, policy=configured, reason="policy_disabled")
    seed = int(random_state if random_state is not None else configured.random_state or 42)
    try:
        method_a = _proposal_method(proposal_a)
        method_b = _proposal_method(proposal_b)
        if method_a == method_b:
            return _unavailable(
                task_type=task_type,
                policy=configured,
                reason="pairwise_probe_requires_two_distinct_model_families",
                training_rows=len(training_frame),
            )
        preprocessing_a = _proposal_preprocessing(proposal_a)
        preprocessing_b = _proposal_preprocessing(proposal_b)
        if target_column not in training_frame.columns:
            return _unavailable(
                task_type=task_type,
                policy=configured,
                reason="target_column_not_in_training_frame",
                training_rows=len(training_frame),
            )
        validation_by_method = validation_by_method or {}
        features_a, target_a, validation_a = _prepare_candidate_data(
            training_frame,
            target_column,
            task_type,
            method_a,
            preprocessing_a,
            random_state=seed,
            validation=validation_by_method.get(method_a),
        )
        features_b, target_b, validation_b = _prepare_candidate_data(
            training_frame,
            target_column,
            task_type,
            method_b,
            preprocessing_b,
            random_state=seed,
            validation=validation_by_method.get(method_b),
        )
        if len(features_a) != len(features_b) or not target_a.reset_index(drop=True).equals(target_b.reset_index(drop=True)):
            return _unavailable(
                task_type=task_type,
                policy=configured,
                reason="candidate_training_rows_are_not_identical",
                training_rows=min(len(features_a), len(features_b)),
            )
        features_a, target_a, row_subsampled = _limit_rows(
            features_a,
            target_a,
            task_type=task_type,
            max_rows=configured.max_rows,
            random_state=seed,
        )
        selected_index = target_a.index
        features_a = features_a.reset_index(drop=True)
        target_a = target_a.reset_index(drop=True)
        features_b = features_b.loc[selected_index].reset_index(drop=True)
        target_b = target_b.loc[selected_index].reset_index(drop=True)
        if len(features_a) < configured.minimum_rows:
            return _unavailable(
                task_type=task_type,
                policy=configured,
                reason="training_rows_below_probe_minimum",
                training_rows=len(features_a),
            )
        if task_type == "classification":
            counts = target_a.value_counts()
            if len(counts) < 2 or int(counts.min()) < 2:
                return _unavailable(
                    task_type=task_type,
                    policy=configured,
                    reason="insufficient_class_support_for_stratified_probe",
                    training_rows=len(features_a),
                )
            feasible_folds = int(counts.min())
            cv_folds = min(configured.cv_folds, feasible_folds)
            splitter: Any = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            cv_strategy = "stratified_kfold"
        else:
            cv_folds = min(configured.cv_folds, len(features_a))
            splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            cv_strategy = "kfold"
        if cv_folds < 2:
            return _unavailable(
                task_type=task_type,
                policy=configured,
                reason="insufficient_rows_for_two_fold_probe",
                training_rows=len(features_a),
            )
        numeric_a = [column for column in features_a.columns if pd.api.types.is_numeric_dtype(features_a[column])]
        categorical_a = [column for column in features_a.columns if column not in numeric_a]
        numeric_b = [column for column in features_b.columns if pd.api.types.is_numeric_dtype(features_b[column])]
        categorical_b = [column for column in features_b.columns if column not in numeric_b]
        pipeline_a = Pipeline(
            [
                ("preprocessor", build_preprocessor(preprocessing_a, numeric_a, categorical_a, method_a)),
                ("model", _estimator(task_type, method_a, seed)),
            ]
        )
        pipeline_b = Pipeline(
            [
                ("preprocessor", build_preprocessor(preprocessing_b, numeric_b, categorical_b, method_b)),
                ("model", _estimator(task_type, method_b, seed)),
            ]
        )
        scores_a = _metric_scores(task_type, pipeline_a, features_a, target_a, splitter)
        scores_b = _metric_scores(task_type, pipeline_b, features_b, target_b, splitter)
        mean_a = float(np.mean(scores_a))
        mean_b = float(np.mean(scores_b))
        std_a = float(np.std(scores_a))
        std_b = float(np.std(scores_b))
        higher_is_better = task_type == "classification"
        if np.isclose(mean_a, mean_b, rtol=1e-12, atol=1e-12):
            winner = "tie"
            relative_advantage = 0.0
        elif (mean_a > mean_b) == higher_is_better:
            winner = "A"
            better, worse = mean_a, mean_b
            relative_advantage = (better - worse) / max(abs(worse), _EPSILON) if higher_is_better else (worse - better) / max(abs(worse), _EPSILON)
        else:
            winner = "B"
            better, worse = mean_b, mean_a
            relative_advantage = (better - worse) / max(abs(worse), _EPSILON) if higher_is_better else (worse - better) / max(abs(worse), _EPSILON)
        a_wins, b_wins, fold_ties = _fold_wins(scores_a, scores_b, higher_is_better=higher_is_better)
        winning_fold_count = a_wins if winner == "A" else b_wins
        difference = mean_a - mean_b
        strength = _evidence_strength(
            float(max(relative_advantage, 0.0)),
            difference,
            std_a,
            std_b,
            winner,
            winning_fold_count,
            cv_folds,
            configured,
        )
        if strength == "tie":
            winner = "tie"
        return {
            "status": "completed",
            "policy_version": configured.policy_version,
            "policy": configured.as_dict(),
            "task_type": task_type,
            "metric": "macro_f1" if task_type == "classification" else "rmse",
            "higher_is_better": higher_is_better,
            "cv_folds": cv_folds,
            "cv_strategy": cv_strategy,
            "training_rows": int(len(features_a)),
            "row_subsampled": row_subsampled,
            "max_rows": configured.max_rows,
            "data_used": "frozen_training_partition_only",
            "holdout_used": False,
            "fit_count": int(2 * cv_folds),
            "candidate_configurations": {
                "proposal_a": [{"configuration": "canonical_family_estimator", "model_family": method_a}],
                "proposal_b": [{"configuration": "canonical_family_estimator", "model_family": method_b}],
            },
            "proposal_a": {
                "model_family": method_a,
                "mean_score": mean_a,
                "std_score": std_a,
                "fold_scores": scores_a,
                "fold_wins": a_wins,
                "fold_ties": fold_ties,
                "preprocessing": preprocessing_a.model_dump(mode="json"),
                "validation_status": validation_a.status,
            },
            "proposal_b": {
                "model_family": method_b,
                "mean_score": mean_b,
                "std_score": std_b,
                "fold_scores": scores_b,
                "fold_wins": b_wins,
                "fold_ties": fold_ties,
                "preprocessing": preprocessing_b.model_dump(mode="json"),
                "validation_status": validation_b.status,
            },
            "difference": float(difference),
            "relative_advantage": float(max(relative_advantage, 0.0)),
            "normalized_advantage": float(max(relative_advantage, 0.0)),
            "winner": winner,
            "evidence_strength": strength,
            "scope": "directional_training_only_pairwise_evidence_not_final_evaluation",
        }
    except Exception as exc:
        return _unavailable(
            task_type=task_type,
            policy=configured,
            reason="probe_fit_or_preprocessing_failed",
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            training_rows=len(training_frame),
        )


# Descriptive alias for callers that use the terminology from the modeling gate.
run_empirical_challenge_probe = run_pairwise_model_probe
