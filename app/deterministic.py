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

from app.schemas import DeterministicRecommendation, Method, TaskType


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
        if pd.api.types.is_integer_dtype(series) and (
            non_null.is_monotonic_increasing or non_null.is_monotonic_decreasing
        ):
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
    normalized_question = re.sub(r"[^a-z0-9]+", " ", question.lower())
    exact = [
        str(column)
        for column in dataframe.columns
        if str(column).lower() in normalized_question
    ]
    if exact:
        return exact[0]
    return str(dataframe.columns[-1])


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

    target_column = choose_target(dataframe, question, target_hint)
    return target_column, infer_task(dataframe, target_column)


def deterministic_recommendation(
    dataframe: pd.DataFrame,
    question: str,
    target_hint: str | None = None,
    task_type: TaskType | None = None,
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

    policy = DeterministicPolicy()
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
        relationship_summary = (
            f"nominal class association measured with {diagnostics.association_measure}; "
            f"maximum class-separation strength was {diagnostics.class_separation_strength:.2f}"
        )
    else:
        relationship_summary = (
            f"{diagnostics.nonlinearity_signal} numeric-target nonlinearity and "
            f"{diagnostics.association_measure}"
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
        f"selected_score={top_score if top_score is not None else 'ineligible'}",
        f"runner_up={ranked[1] if len(ranked) > 1 else 'none'}",
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


def apply_cleaning(
    dataframe: pd.DataFrame,
    target_column: str,
    actions: list[str],
    row_position_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply only allow-listed structural operations selected by the agent."""

    frame = dataframe.copy()
    original_shape = [int(frame.shape[0]), int(frame.shape[1])]
    applied: list[str] = []
    removed_columns: list[str] = []
    removed_rows = 0

    if "trim_strings" in actions:
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
            frame[column] = frame[column].replace(r"^\s*$", np.nan, regex=True)
        applied.append("trim_strings")

    if "coerce_numeric_strings" in actions:
        for column in frame.columns:
            if column == target_column or frame[column].dtype != "object":
                continue
            non_null = frame[column].dropna()
            if len(non_null) and pd.to_numeric(non_null, errors="coerce").notna().mean() >= 0.95:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        applied.append("coerce_numeric_strings")

    if "drop_exact_duplicates" in actions:
        before = len(frame)
        duplicate_columns = [column for column in frame.columns if column != row_position_column]
        frame = frame.drop_duplicates(subset=duplicate_columns).reset_index(drop=True)
        removed_rows += before - len(frame)
        applied.append("drop_exact_duplicates")

    if "drop_all_null_columns" in actions:
        all_null = [
            str(column)
            for column in frame.columns
            if frame[column].isna().all()
            and column != target_column
            and column != row_position_column
        ]
        if all_null:
            frame = frame.drop(columns=all_null)
            removed_columns.extend(all_null)
        applied.append("drop_all_null_columns")

    if "drop_constant_features" in actions:
        constant = [
            str(column)
            for column in frame.columns
            if column != target_column
            and column != row_position_column
            and frame[column].nunique(dropna=True) <= 1
        ]
        if constant:
            frame = frame.drop(columns=constant)
            removed_columns.extend(constant)
        applied.append("drop_constant_features")

    if "drop_rows_missing_target" in actions:
        before = len(frame)
        frame = frame.dropna(subset=[target_column]).reset_index(drop=True)
        removed_rows += before - len(frame)
        applied.append("drop_rows_missing_target")

    return frame, {
        "original_shape": original_shape,
        "cleaned_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "requested_actions": actions,
        "applied_actions": applied,
        "removed_rows": removed_rows,
        "removed_columns": removed_columns,
    }


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
