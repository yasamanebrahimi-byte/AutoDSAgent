"""Deterministic profiling, recommendation, cleaning, and EDA utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.schemas import (
    CleaningSpecification,
    DeterministicFormulation,
    DeterministicRecommendation,
    Method,
    TaskType,
)
from app.deterministic_policy import DeterministicPolicy


def semantic_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if non_null.empty:
        return "unknown"
    numeric_ratio = pd.to_numeric(non_null, errors="coerce").notna().mean()
    if numeric_ratio >= 0.95 and non_null.nunique() > 1:
        return "numeric_like"
    average_length = non_null.astype(str).str.len().mean()
    unique_ratio = non_null.nunique() / max(len(non_null), 1)
    if average_length >= 40 and unique_ratio >= 0.5:
        return "text"
    return "categorical"


def is_identifier(name: str, series: pd.Series) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if normalized in {"id", "uuid", "guid"} or normalized.endswith("_id"):
        return True
    non_null = series.dropna()
    if len(non_null) >= 20 and non_null.nunique() / len(non_null) >= 0.98:
        numeric = pd.to_numeric(non_null, errors="coerce")
        source_integer_like = pd.api.types.is_integer_dtype(series)
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            source_integer_like = bool(
                non_null.astype(str).str.fullmatch(r"[+-]?\d+").all()
            )
        integer_like = bool(
            numeric.notna().all()
            and np.isclose(numeric % 1, 0).all()
            and source_integer_like
        )
        monotonic = bool(numeric.notna().all() and (numeric.is_monotonic_increasing or numeric.is_monotonic_decreasing))
        if integer_like and monotonic:
            return True
    return False


def profile_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a compact JSON-safe profile suitable for an LLM prompt."""

    columns: list[dict[str, Any]] = []
    for name in dataframe.columns:
        series = dataframe[name]
        non_null = series.dropna()
        record: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "semantic_type": semantic_type(series),
            "missing": int(series.isna().sum()),
            "missing_fraction": round(float(series.isna().mean()), 4),
            "unique": int(series.nunique(dropna=True)),
            "identifier_like": is_identifier(str(name), series),
            "sample_values": [to_json_value(value) for value in non_null.head(3).tolist()],
        }
        if pd.api.types.is_numeric_dtype(series) and not non_null.empty:
            infinity = int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum())
            record.update(
                {
                    "min": to_json_value(non_null.min()),
                    "max": to_json_value(non_null.max()),
                    "mean": round(float(non_null.mean()), 4),
                    "infinity": infinity,
                }
            )
        columns.append(record)
    return {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "column_details": columns,
    }


def choose_target(
    dataframe: pd.DataFrame,
    question: str,
    target_hint: str | None = None,
) -> str:
    if target_hint:
        if target_hint not in dataframe.columns:
            raise ValueError(f"Target column '{target_hint}' is not present in the dataset.")
        return target_hint
    normalized_question = _normalize_name(question)
    question_tokens = set(normalized_question.split())
    padded_question = f" {normalized_question} "
    candidates: list[tuple[int, str]] = []
    for column in dataframe.columns:
        name = str(column)
        normalized_name = _normalize_name(name)
        if not normalized_name:
            continue
        # A normalized full-column match is the strongest defensible signal.
        # A single-token match is also accepted, but only when it is unique.
        if f" {normalized_name} " in padded_question:
            score = 3
        elif len(normalized_name.split()) == 1 and normalized_name in question_tokens:
            score = 2
        else:
            continue
        # Identifier-like columns are safe only when the question explicitly
        # names them; a full normalized match above is such an explicit request.
        candidates.append((score, name))
    if not candidates:
        raise ValueError(
            "Could not infer a defensible target from the question and schema. "
            "Provide target_column explicitly."
        )
    best_score = max(score for score, _ in candidates)
    best = [name for score, name in candidates if score == best_score]
    if len(best) != 1:
        raise ValueError(
            "Target inference is ambiguous; multiple columns match the question: "
            f"{', '.join(best)}. Provide target_column explicitly."
        )
    return best[0]


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def infer_task(dataframe: pd.DataFrame, target_column: str) -> TaskType:
    target = dataframe[target_column].dropna()
    if len(target) < 8 or target.nunique() < 2:
        raise ValueError("The target needs at least eight non-null rows and two values.")
    target_kind = semantic_type(dataframe[target_column])
    if target_kind in {"categorical", "boolean", "text"}:
        if target.nunique() > min(50, max(10, len(target) // 3)):
            raise ValueError("The target has too many categories for this compact workflow.")
        return "classification"
    if target_kind == "numeric_like":
        numeric = pd.to_numeric(target, errors="coerce")
        if numeric.notna().mean() < 0.95:
            return "classification"
        target = numeric
    if target.nunique() <= max(8, int(np.sqrt(len(target)))) and target.nunique() <= 20:
        return "classification"
    return "regression"


def establish_target_task(
    dataframe: pd.DataFrame,
    question: str,
    target_hint: str | None = None,
) -> tuple[str, TaskType]:
    """Perform only the target/task work needed before a supervised split."""

    formulation = deterministic_formulation(dataframe, question, target_hint)
    if formulation.target_column is None or formulation.task_type is None:
        raise ValueError(formulation.reasoning)
    return formulation.target_column, formulation.task_type


def deterministic_formulation(
    dataframe: pd.DataFrame,
    question: str,
    target_hint: str | None = None,
) -> DeterministicFormulation:
    """Independently formulate target/task before any supervised split.

    This function intentionally performs no model-family diagnostics, fitting,
    CV, or holdout construction.  Its evidence is limited to schema and target
    value facts needed to choose between the two supported task types.
    """

    target_source = "user_supplied" if target_hint else "inferred"
    try:
        target_column = choose_target(dataframe, question, target_hint)
        task_type = infer_task(dataframe, target_column)
    except Exception as exc:
        return DeterministicFormulation(
            target_column=target_hint if target_hint in dataframe.columns else None,
            task_type=None,
            status="failed" if target_hint and target_hint not in dataframe.columns else "uncertain",
            reasoning=f"Deterministic formulation failed closed: {exc}",
            evidence=["no_arbitrary_last_column_fallback"],
            confidence=0.0,
            target_source="user_supplied" if target_hint else "uncertain",
        )

    target = dataframe[target_column]
    non_null = target.dropna()
    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_fraction = float(numeric.notna().mean()) if len(non_null) else 0.0
    unique = int(non_null.nunique())
    unique_fraction = unique / max(len(non_null), 1)
    semantic = semantic_type(target)
    evidence = [
        f"target_column={target_column}",
        f"target_source={target_source}",
        f"target_dtype={target.dtype}",
        f"target_semantic_type={semantic}",
        f"valid_target_rows={len(non_null)}",
        f"unique_values={unique}",
        f"unique_value_fraction={unique_fraction:.4f}",
        f"numeric_or_coercible_fraction={numeric_fraction:.4f}",
        "model_selection_diagnostics_not_used",
    ]
    if task_type == "classification":
        reasoning = (
            f"The target '{target_column}' is treated as classification because its "
            f"semantic type is {semantic}, or its numeric values are low-cardinality "
            f"label-like ({unique} unique values across {len(non_null)} valid rows)."
        )
    else:
        reasoning = (
            f"The target '{target_column}' is treated as regression because it is "
            f"numeric/coercible ({numeric_fraction:.3f} valid numeric fraction) and has "
            f"{unique} distinct values across {len(non_null)} valid rows, rather than "
            "appearing to be a low-cardinality label."
        )
    return DeterministicFormulation(
        target_column=target_column,
        task_type=task_type,
        status="proposed",
        reasoning=reasoning,
        evidence=evidence,
        confidence=0.9 if target_hint else 0.75,
        target_source=target_source,
    )


def deterministic_recommendation(
    dataframe: pd.DataFrame,
    question: str,
    target_hint: str | None = None,
    task_type: TaskType | None = None,
    *,
    policy: DeterministicPolicy | None = None,
) -> DeterministicRecommendation:
    """Recommend a model family from the supplied training-only dataframe.

    The public caller must provide the frozen training partition.  This
    function never receives or consults holdout values, empirical CV results,
    or prior agent decisions.
    """

    target_column = choose_target(dataframe, question, target_hint)
    task_type = task_type or infer_task(dataframe, target_column)
    from app.deterministic_diagnostics import compute_deterministic_diagnostics
    from app.deterministic_policy import (
        SUPPORTED_METHOD_ORDER,
        DeterministicPolicy,
        score_model_families,
    )
    from app.preprocessing import requirements_from_records

    policy = policy or DeterministicPolicy()
    diagnostics = compute_deterministic_diagnostics(
        dataframe,
        target_column,
        task_type,
        policy=policy,
    )
    feature_records = [
        record
        for record in profile_dataframe(dataframe)["column_details"]
        if record["name"] != target_column
    ]
    requirements_by_method = {
        candidate: requirements_from_records(feature_records, task_type, candidate)
        for candidate in SUPPORTED_METHOD_ORDER
    }
    assessments = score_model_families(diagnostics, policy=policy)
    eligible_scores = {
        method: int(assessment.score)
        for method, assessment in assessments.items()
        if assessment.eligible and assessment.score is not None
    }
    if eligible_scores:
        ranked = sorted(
            eligible_scores,
            key=lambda candidate: (-eligible_scores[candidate], SUPPORTED_METHOD_ORDER.index(candidate)),
        )
        method: Method = ranked[0]
    else:
        # The fail-closed validation gate remains the authority when no family
        # is executable.  Keeping a supported fallback makes the failed
        # recommendation itself inspectable rather than hiding the evidence.
        ranked = list(SUPPORTED_METHOD_ORDER)
        method = "linear"
    top_score = eligible_scores.get(method)
    runner_up_score = eligible_scores.get(ranked[1]) if len(ranked) > 1 else None
    score_margin = float(top_score - runner_up_score) if top_score is not None and runner_up_score is not None else None
    if score_margin is None or score_margin < policy.low_confidence_margin:
        confidence = "low"
    elif score_margin < policy.high_confidence_margin:
        confidence = "medium"
    else:
        confidence = "high"
    preprocessing_requirements = requirements_by_method[method]
    preprocessing = preprocessing_requirements.expected_contract
    selected_assessment = assessments[method]
    positive_reasons = [
        contribution.observation
        for contribution in selected_assessment.contributions
        if contribution.points > 0
    ][:3]
    reason_detail = "; ".join(positive_reasons) or "no positive compatibility factor dominated"
    if task_type == "classification":
        boundary = diagnostics.classification_boundary_signals
        relationship_summary = (
            f"nominal class association measured with {diagnostics.association_measure}; "
            f"maximum class-separation strength was {diagnostics.class_separation_strength:.2f}; "
            f"boundary complexity was {boundary.boundary_complexity} "
            f"({boundary.boundary_complexity_score:.2f}) with "
            f"{boundary.boundary_diagnostic_confidence} diagnostic confidence"
        )
    else:
        relationship_summary = (
            f"{diagnostics.nonlinearity_signal} numeric-target nonlinearity and "
            f"{diagnostics.association_measure}; "
            f"{diagnostics.interaction_signals.interaction_strength} interaction evidence "
            f"({diagnostics.interaction_signals.interaction_score:.2f})"
        )
    reason = (
        f"The training-only deterministic policy ranked {method} highest with "
        f"compatibility score {top_score if top_score is not None else 'unavailable'} "
        f"({confidence} confidence). It considered {diagnostics.usable_features} usable features "
        f"across {diagnostics.rows} training rows, estimated {diagnostics.effective_features_estimate} "
        f"post-one-hot features, and observed {relationship_summary}. "
        f"The structural-complexity signal was {diagnostics.structural_complexity_signal} "
        f"({diagnostics.structural_complexity_score:.2f}). "
        f"Key positive factors: {reason_detail}. Compatibility scores are policy rankings, not probabilities."
    )
    evidence = [
        f"training_rows={diagnostics.rows}",
        f"usable_features={diagnostics.usable_features}",
        f"numeric_features={diagnostics.numeric_feature_count}",
        f"categorical_features={diagnostics.categorical_feature_count}",
        f"effective_one_hot_features={diagnostics.effective_features_estimate}",
        f"sample_to_feature_ratio={diagnostics.sample_to_feature_ratio:.3f}",
        f"overall_missing_fraction={diagnostics.overall_missing_fraction:.3f}",
        f"max_abs_numeric_correlation={diagnostics.max_abs_numeric_correlation:.3f}",
        (
            f"association_measure={diagnostics.association_measure}; "
            f"marginal_association_strength={diagnostics.marginal_association_strength:.3f}; "
            f"class_separation_strength={diagnostics.class_separation_strength:.3f}"
        ),
        (
            f"nonlinearity_signal={diagnostics.nonlinearity_signal}; "
            f"nonlinearity_applicable={diagnostics.nonlinearity_applicable}; "
            f"nonlinear_feature_fraction={diagnostics.nonlinear_feature_fraction:.3f}; "
            f"nonlinearity_heterogeneity={diagnostics.nonlinearity_heterogeneity:.3f}; "
            f"structural_complexity={diagnostics.structural_complexity_signal}"
            f"({diagnostics.structural_complexity_score:.3f})"
        ),
        (
            f"classification_boundary={diagnostics.classification_boundary_signals.boundary_complexity}; "
            f"boundary_score={diagnostics.classification_boundary_signals.boundary_complexity_score:.3f}; "
            f"linear_boundary_probe_score={diagnostics.classification_boundary_signals.linear_boundary_probe_score:.3f}; "
            f"linear_separability={diagnostics.classification_boundary_signals.linear_separability_score:.3f}; "
            f"local_class_consistency={diagnostics.classification_boundary_signals.local_class_consistency:.3f}; "
            f"nonlinear_advantage={diagnostics.classification_boundary_signals.nonlinear_advantage_score:.3f}; "
            f"boundary_confidence={diagnostics.classification_boundary_signals.boundary_diagnostic_confidence}"
        )
        if task_type == "classification"
        else "classification_boundary=not_applicable_for_regression",
        (
            f"interaction_strength={diagnostics.interaction_signals.interaction_strength}; "
            f"interaction_score={diagnostics.interaction_signals.interaction_score:.3f}; "
            f"pairs_evaluated={diagnostics.interaction_signals.interaction_pairs_evaluated}; "
            f"strong_pair_count={diagnostics.interaction_signals.strong_interaction_pair_count}; "
            f"strong_pair_fraction={diagnostics.interaction_signals.strong_interaction_pair_fraction:.3f}"
        ),
    ]
    return DeterministicRecommendation(
        target_column=target_column,
        task_type=task_type,
        recommended_method=method,
        preprocessing=preprocessing,
        reasoning=reason,
        evidence=evidence,
        policy_version=policy.version,
        method_scores={
            candidate: assessment.score
            for candidate, assessment in assessments.items()
        },
        ranked_methods=ranked,
        method_assessments=assessments,
        diagnostics=diagnostics,
        top_score=float(top_score) if top_score is not None else None,
        runner_up_score=float(runner_up_score) if runner_up_score is not None else None,
        score_margin=score_margin,
        confidence=confidence,
    )


def _trim_string_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"The frozen cleaning specification references missing column '{column}'.")
        frame[column] = frame[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        frame[column] = frame[column].replace(r"^\s*$", np.nan, regex=True)
    return frame


def fit_cleaning_spec(
    training_dataframe: pd.DataFrame,
    target_column: str,
    actions: list[str],
    row_position_column: str | None = None,
) -> CleaningSpecification:
    """Fit structural-cleaning decisions from one partition only.

    This function is intentionally the only place where structural properties
    such as numeric-like ratios, all-null columns, constants, and training
    duplicate membership are derived.  Callers must pass the frozen training
    partition, never a frame containing holdout rows.
    """

    frame = training_dataframe.copy()
    if row_position_column is not None and row_position_column not in frame.columns:
        raise ValueError(f"The row-position column '{row_position_column}' is missing from the training frame.")
    if target_column not in frame.columns:
        raise ValueError(f"The target column '{target_column}' is missing from the training frame.")

    source_positions = (
        frame[row_position_column].to_numpy(dtype=int, copy=True)
        if row_position_column is not None
        else np.arange(len(frame), dtype=int)
    )
    evidence: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns if column != row_position_column],
        "string_columns_considered": [],
        "numeric_coercion_ratios": {},
        "all_null_columns": [],
        "constant_columns": {},
        "training_duplicate_row_positions": [],
    }

    trim_columns: list[str] = []
    if "trim_strings" in actions:
        trim_columns = [
            str(column)
            for column in frame.select_dtypes(include=["object", "string"]).columns
            if column != row_position_column
        ]
        evidence["string_columns_considered"] = trim_columns
        _trim_string_columns(frame, trim_columns)

    numeric_coercion_columns: list[str] = []
    if "coerce_numeric_strings" in actions:
        for column in frame.columns:
            if column == target_column or column == row_position_column or frame[column].dtype != "object":
                continue
            non_null = frame[column].dropna()
            ratio = (
                float(pd.to_numeric(non_null, errors="coerce").notna().mean())
                if len(non_null)
                else 0.0
            )
            evidence["numeric_coercion_ratios"][str(column)] = ratio
            if len(non_null) and ratio >= 0.95:
                numeric_coercion_columns.append(str(column))
        for column in numeric_coercion_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    training_duplicate_row_positions: list[int] = []
    if "drop_exact_duplicates" in actions:
        duplicate_columns = [column for column in frame.columns if column != row_position_column]
        duplicate_mask = frame.duplicated(subset=duplicate_columns, keep="first")
        training_duplicate_row_positions = [
            int(value) for value in source_positions[duplicate_mask.to_numpy(dtype=bool)]
        ]
        evidence["training_duplicate_row_positions"] = training_duplicate_row_positions
        frame = frame.loc[~duplicate_mask].copy()

    all_null_columns: list[str] = []
    if "drop_all_null_columns" in actions:
        all_null_columns = [
            str(column)
            for column in frame.columns
            if column != target_column
            and column != row_position_column
            and frame[column].isna().all()
        ]
        evidence["all_null_columns"] = all_null_columns
        if all_null_columns:
            frame = frame.drop(columns=all_null_columns)

    constant_columns: list[str] = []
    if "drop_constant_features" in actions:
        constant_columns = [
            str(column)
            for column in frame.columns
            if column != target_column
            and column != row_position_column
            and frame[column].nunique(dropna=True) <= 1
        ]
        evidence["constant_columns"] = {
            column: {
                "non_null_unique_values": int(frame[column].nunique(dropna=True)),
                "all_null": bool(frame[column].isna().all()),
            }
            for column in constant_columns
        }

    if "drop_rows_missing_target" in actions:
        evidence["training_missing_target_rows"] = int(frame[target_column].isna().sum())

    return CleaningSpecification(
        target_column=str(target_column),
        row_position_column=row_position_column,
        requested_actions=list(actions),
        trim_columns=trim_columns,
        numeric_coercion_columns=numeric_coercion_columns,
        all_null_columns=all_null_columns,
        constant_columns=constant_columns,
        drop_rows_missing_target="drop_rows_missing_target" in actions,
        training_duplicate_row_positions=training_duplicate_row_positions,
        training_only_evidence=evidence,
    )


def transform_cleaning(
    dataframe: pd.DataFrame,
    specification: CleaningSpecification | dict[str, Any],
    *,
    partition: str = "training",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a frozen specification without fitting structural decisions.

    Duplicate removal is deliberately the sole partition-local operation:
    training duplicates use positions frozen by :func:`fit_cleaning_spec`,
    while holdout (and otherwise-unassigned) duplicates are detected only
    within that partition.  No row is ever compared across partitions.
    """

    spec = CleaningSpecification.model_validate(specification)
    if partition not in {"training", "holdout", "unassigned"}:
        raise ValueError("Cleaning transforms require partition='training', 'holdout', or 'unassigned'.")
    frame = dataframe.copy()
    original_shape = [int(frame.shape[0]), int(frame.shape[1])]
    row_position_column = spec.row_position_column
    if row_position_column is not None and row_position_column not in frame.columns:
        raise ValueError(f"The row-position column '{row_position_column}' is missing from the transform frame.")
    if spec.target_column not in frame.columns:
        raise ValueError(f"The target column '{spec.target_column}' is missing from the transform frame.")

    if "trim_strings" in spec.requested_actions:
        _trim_string_columns(frame, list(spec.trim_columns))

    if "coerce_numeric_strings" in spec.requested_actions:
        for column in spec.numeric_coercion_columns:
            if column not in frame.columns:
                raise ValueError(f"The frozen cleaning specification references missing column '{column}'.")
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    removed_row_positions: list[int] = []
    if "drop_exact_duplicates" in spec.requested_actions:
        if partition == "training":
            if row_position_column is not None:
                position_values = frame[row_position_column].to_numpy(dtype=int, copy=False)
                duplicate_mask = np.isin(position_values, np.asarray(spec.training_duplicate_row_positions, dtype=int))
            else:
                duplicate_mask = np.isin(
                    np.arange(len(frame), dtype=int),
                    np.asarray(spec.training_duplicate_row_positions, dtype=int),
                )
        else:
            duplicate_columns = [column for column in frame.columns if column != row_position_column]
            duplicate_mask = frame.duplicated(subset=duplicate_columns, keep="first").to_numpy(dtype=bool)
        if row_position_column is not None:
            removed_row_positions.extend(
                int(value)
                for value in frame.loc[duplicate_mask, row_position_column].to_numpy(dtype=int)
            )
        frame = frame.loc[~duplicate_mask].copy()

    removed_columns: list[str] = []
    frozen_drop_columns = list(dict.fromkeys(spec.all_null_columns + spec.constant_columns))
    if frozen_drop_columns:
        missing = [column for column in frozen_drop_columns if column not in frame.columns]
        if missing:
            raise ValueError(
                "The frozen cleaning specification cannot be applied because columns are missing: "
                + ", ".join(missing)
            )
        frame = frame.drop(columns=frozen_drop_columns)
        removed_columns.extend(frozen_drop_columns)

    if spec.drop_rows_missing_target:
        missing_target_mask = frame[spec.target_column].isna().to_numpy(dtype=bool)
        if row_position_column is not None:
            removed_row_positions.extend(
                int(value)
                for value in frame.loc[missing_target_mask, row_position_column].to_numpy(dtype=int)
            )
        frame = frame.loc[~missing_target_mask].copy()

    removed_row_positions = list(dict.fromkeys(removed_row_positions))
    applied = list(spec.requested_actions)
    return frame.reset_index(drop=True), {
        "partition": partition,
        "original_shape": original_shape,
        "cleaned_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "requested_actions": list(spec.requested_actions),
        "applied_actions": applied,
        "removed_rows": int(original_shape[0] - len(frame)),
        "removed_row_positions": removed_row_positions,
        "removed_columns": removed_columns,
        "frozen_columns_used": {
            "trim_columns": list(spec.trim_columns),
            "numeric_coercion_columns": list(spec.numeric_coercion_columns),
            "all_null_columns": list(spec.all_null_columns),
            "constant_columns": list(spec.constant_columns),
        },
        "duplicate_scope": "within_partition_only",
    }


def apply_cleaning(
    dataframe: pd.DataFrame,
    target_column: str,
    actions: list[str] | None = None,
    row_position_column: str | None = None,
    *,
    specification: CleaningSpecification | dict[str, Any] | None = None,
    partition: str = "training",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compatibility entry point for explicit fit/transform cleaning.

    New pipeline code passes a fitted ``specification``.  When omitted, the
    supplied dataframe is treated as the single fitting partition; callers
    handling a frozen split should use :func:`fit_cleaning_spec` on training
    rows and :func:`transform_cleaning` on each partition explicitly.
    """

    if specification is None:
        if actions is None:
            raise ValueError("Either a cleaning specification or requested actions must be provided.")
        specification = fit_cleaning_spec(
            dataframe,
            target_column,
            actions,
            row_position_column=row_position_column,
        )
    return transform_cleaning(dataframe, specification, partition=partition)


def eda_summary(dataframe: pd.DataFrame, target_column: str) -> dict[str, Any]:
    numeric = dataframe.select_dtypes(include=[np.number])
    summary: dict[str, Any] = {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "missing_by_column": {
            str(k): int(v) for k, v in dataframe.isna().sum().items() if int(v)
        },
        "numeric_summary": {},
        "target": {"column": target_column, "unique": int(dataframe[target_column].nunique(dropna=True))},
    }
    for column in numeric.columns[:40]:
        desc = numeric[column].describe()
        summary["numeric_summary"][str(column)] = {
            key: round(float(desc[key]), 5)
            for key in ["mean", "std", "min", "25%", "50%", "75%", "max"]
            if key in desc
        }
    target = dataframe[target_column]
    if pd.api.types.is_numeric_dtype(target):
        summary["target"].update(
            {"mean": round(float(target.mean()), 5), "std": round(float(target.std()), 5)}
        )
    else:
        counts = target.astype(str).value_counts().head(12)
        summary["target"]["value_counts"] = {str(k): int(v) for k, v in counts.items()}
    if len(numeric.columns) >= 2:
        corr = numeric.corr(numeric_only=True).abs()
        pairs: list[dict[str, Any]] = []
        for left_index, left in enumerate(corr.columns):
            for right in corr.columns[left_index + 1 :]:
                pairs.append(
                    {
                        "feature_a": str(left),
                        "feature_b": str(right),
                        "abs_correlation": round(float(corr.loc[left, right]), 4),
                    }
                )
        summary["strongest_numeric_relationships"] = sorted(
            pairs, key=lambda item: item["abs_correlation"], reverse=True
        )[:8]
    else:
        summary["strongest_numeric_relationships"] = []
    return summary


def make_plots(dataframe: pd.DataFrame, target_column: str, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    target = dataframe[target_column]
    fig, ax = plt.subplots(figsize=(7, 4))
    if pd.api.types.is_numeric_dtype(target):
        ax.hist(target.dropna(), bins=20, color="#2563eb", alpha=0.85)
        ax.set_ylabel("Rows")
    else:
        target.astype(str).value_counts().head(12).plot.bar(ax=ax, color="#2563eb")
        ax.set_ylabel("Rows")
        ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Target distribution: {target_column}")
    fig.tight_layout()
    target_path = output_dir / "target_distribution.png"
    fig.savefig(target_path, dpi=140)
    plt.close(fig)
    paths.append(str(target_path))

    numeric = dataframe.select_dtypes(include=[np.number])
    if len(numeric.columns) >= 2:
        matrix = numeric.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.8)
        ax.set_title("Numeric correlation heatmap")
        fig.tight_layout()
        correlation_path = output_dir / "correlation_heatmap.png"
        fig.savefig(correlation_path, dpi=140)
        plt.close(fig)
        paths.append(str(correlation_path))
    return paths


def to_json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
