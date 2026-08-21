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
    target_column = choose_target(dataframe, question, target_hint)
    task_type = task_type or infer_task(dataframe, target_column)
    feature_records = [
        record
        for record in profile_dataframe(dataframe)["column_details"]
        if record["name"] != target_column
    ]
    usable = [record for record in feature_records if not record["identifier_like"]]
    numeric_count = sum(record["semantic_type"] in {"numeric", "numeric_like"} for record in usable)
    categorical_count = sum(record["semantic_type"] in {"categorical", "boolean"} for record in usable)
    text_count = sum(record["semantic_type"] == "text" for record in usable)
    missing_fraction = max((record["missing_fraction"] for record in usable), default=0.0)
    rows = len(dataframe)

    if text_count or categorical_count >= 2 or missing_fraction >= 0.25:
        method: Method = "tree_ensemble"
        reason = "The schema contains categorical/text structure or substantial missingness, so a tree ensemble is a conservative non-linear baseline after safe preprocessing."
    elif numeric_count >= 3 and rows >= 200 and numeric_count < rows / 8:
        method = "regularized_linear"
        reason = "The dataset is numeric, has enough rows for stable validation, and has a feature count where regularization controls coefficient variance."
    else:
        method = "linear"
        reason = "The compact deterministic baseline favors an interpretable linear family when the schema is mostly numeric and no stronger structural signal is available before fitting."

    from app.preprocessing import requirements_from_records

    preprocessing_requirements = requirements_from_records(
        feature_records,
        task_type,
        method,
    )
    preprocessing = preprocessing_requirements.expected_contract
    evidence = [
        f"rows={rows}",
        f"usable_numeric_features={numeric_count}",
        f"usable_categorical_features={categorical_count}",
        f"text_features={text_count}",
        f"max_missing_fraction={missing_fraction:.3f}",
    ]
    return DeterministicRecommendation(
        target_column=target_column,
        task_type=task_type,
        recommended_method=method,
        preprocessing=preprocessing,
        reasoning=reason,
        evidence=evidence
        + [
            f"required_preprocessing={','.join(preprocessing_requirements.required_steps)}",
            f"irrelevant_preprocessing={','.join(preprocessing_requirements.irrelevant_steps) or 'none'}",
        ],
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
