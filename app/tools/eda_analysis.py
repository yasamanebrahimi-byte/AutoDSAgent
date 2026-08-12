"""Deterministic exploratory data analysis helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.tools.statistics_utils import (
    calculate_skewness,
    format_percentage,
    iqr_outlier_count,
    numeric_non_null,
    safe_correlation,
    safe_float,
    to_json_safe,
)


def summarize_numeric_columns(
    dataframe: pd.DataFrame,
    schema: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Return JSON-safe numeric statistics keyed by column name."""

    summaries: dict[str, dict[str, Any]] = {}
    for column in _columns_by_type(dataframe, schema, {"numeric"}):
        numeric = numeric_non_null(dataframe[column])
        if numeric.empty:
            summaries[column] = {
                "count": 0,
                "missing_values": int(dataframe[column].isna().sum()),
                "unique_values": int(dataframe[column].nunique(dropna=True)),
            }
            continue

        summaries[column] = {
            "count": int(numeric.count()),
            "missing_values": int(dataframe[column].isna().sum()),
            "unique_values": int(dataframe[column].nunique(dropna=True)),
            "mean": safe_float(numeric.mean()),
            "median": safe_float(numeric.median()),
            "standard_deviation": safe_float(numeric.std()),
            "minimum": safe_float(numeric.min()),
            "maximum": safe_float(numeric.max()),
            "percentile_25": safe_float(numeric.quantile(0.25)),
            "percentile_75": safe_float(numeric.quantile(0.75)),
            "skewness": calculate_skewness(numeric),
            "possible_outlier_count": iqr_outlier_count(numeric),
        }

    return to_json_safe(summaries)


def summarize_categorical_columns(
    dataframe: pd.DataFrame,
    schema: dict[str, str],
    top_n: int = 10,
) -> dict[str, dict[str, Any]]:
    """Return top-value summaries for categorical and boolean columns."""

    summaries: dict[str, dict[str, Any]] = {}
    for column in _columns_by_type(dataframe, schema, {"categorical", "boolean"}):
        series = dataframe[column]
        value_counts = series.value_counts(dropna=False).head(top_n)
        summaries[column] = {
            "count": int(series.notna().sum()),
            "missing_values": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
            "top_values": [
                {"value": _display_value(value), "count": int(count)}
                for value, count in value_counts.items()
            ],
        }

    return to_json_safe(summaries)


def detect_skewness_patterns(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    threshold: float = 0.5,
) -> list[str]:
    """Create plain-language findings for visibly skewed numeric columns."""

    findings: list[str] = []
    for column in numeric_columns:
        if column not in dataframe.columns:
            continue

        numeric = numeric_non_null(dataframe[column])
        if numeric.empty:
            continue

        skewness = calculate_skewness(numeric)
        mean = safe_float(numeric.mean())
        median = safe_float(numeric.median())
        if skewness is None or mean is None or median is None:
            continue

        if abs(skewness) < threshold:
            continue

        direction = "right-skewed" if skewness > 0 else "left-skewed"
        if mean > median:
            comparison = "mean being greater than median"
        elif mean < median:
            comparison = "mean being less than median"
        else:
            comparison = "a non-zero skewness score"

        findings.append(
            f"Column `{column}` appears {direction} based on {comparison}."
        )

    return findings


def detect_outlier_patterns(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
) -> list[str]:
    """Create findings for numeric columns with possible IQR outliers."""

    findings: list[str] = []
    for column in numeric_columns:
        if column not in dataframe.columns:
            continue

        numeric = numeric_non_null(dataframe[column])
        if numeric.empty:
            continue

        outlier_count = iqr_outlier_count(numeric)
        if outlier_count <= 0:
            continue

        findings.append(
            f"Column `{column}` contains {outlier_count} possible outliers "
            f"using the IQR rule ({format_percentage(outlier_count, len(numeric))} "
            "of non-null rows)."
        )

    return findings


def compute_correlation_summary(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return pairwise correlation summaries for useful numeric columns."""

    available_columns = [column for column in numeric_columns if column in dataframe.columns]
    pairs: list[dict[str, Any]] = []

    for index, column_a in enumerate(available_columns):
        for column_b in available_columns[index + 1 :]:
            correlation = safe_correlation(dataframe[column_a], dataframe[column_b])
            if correlation is None or abs(correlation) < threshold:
                continue

            strength = "strong" if abs(correlation) >= 0.7 else "moderate"
            direction = "positive" if correlation > 0 else "negative"
            pairs.append(
                {
                    "column_a": column_a,
                    "column_b": column_b,
                    "correlation": correlation,
                    "strength": strength,
                    "direction": direction,
                }
            )

    pairs = sorted(pairs, key=lambda item: abs(float(item["correlation"])), reverse=True)
    return {
        "pairs": pairs,
        "strong_pairs": [
            pair for pair in pairs if pair["strength"] == "strong"
        ],
    }


def analyze_target_distribution(
    dataframe: pd.DataFrame,
    target_column: str,
    schema: dict[str, str],
) -> dict[str, Any]:
    """Analyze a target column without performing modeling."""

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    semantic_type = schema.get(target_column, "unknown")
    series = dataframe[target_column]
    missing_values = int(series.isna().sum())
    unique_values = int(series.nunique(dropna=True))
    findings: list[str] = []

    payload: dict[str, Any] = {
        "column": target_column,
        "semantic_type": semantic_type,
        "missing_values": missing_values,
        "unique_values": unique_values,
        "findings": findings,
    }

    if missing_values:
        findings.append(
            f"Target column `{target_column}` has {missing_values} missing values."
        )

    if semantic_type == "numeric":
        numeric = numeric_non_null(series)
        payload["numeric_summary"] = {
            "count": int(numeric.count()),
            "mean": safe_float(numeric.mean()) if not numeric.empty else None,
            "median": safe_float(numeric.median()) if not numeric.empty else None,
            "minimum": safe_float(numeric.min()) if not numeric.empty else None,
            "maximum": safe_float(numeric.max()) if not numeric.empty else None,
            "skewness": calculate_skewness(numeric),
            "possible_outlier_count": iqr_outlier_count(numeric),
        }
        if payload["numeric_summary"]["possible_outlier_count"]:
            findings.append(
                f"Target column `{target_column}` contains possible outliers using the IQR rule."
            )
    else:
        counts = series.value_counts(dropna=False)
        total = int(counts.sum())
        top_count = int(counts.iloc[0]) if total else 0
        top_share = round(top_count / total, 4) if total else 0.0
        payload["class_distribution"] = [
            {
                "value": _display_value(value),
                "count": int(count),
                "percentage": round(int(count) / total * 100, 4) if total else 0.0,
            }
            for value, count in counts.items()
        ]
        payload["is_imbalanced"] = bool(unique_values > 1 and top_share >= 0.75)
        payload["majority_class_percentage"] = round(top_share * 100, 4)

        if payload["is_imbalanced"]:
            findings.append(
                f"Target column `{target_column}` appears imbalanced; the largest class "
                f"contains {top_share * 100:.1f}% of rows."
            )

    return to_json_safe(payload)


def generate_recommended_next_steps(
    profile: dict[str, Any] | None,
    eda_summary: dict[str, Any],
    findings: dict[str, list[str]],
) -> list[str]:
    """Generate deterministic next-step suggestions from EDA artifacts."""

    del profile

    recommendations: list[str] = []
    missing_values = {
        column: count
        for column, count in eda_summary.get("missing_values_remaining", {}).items()
        if int(count) > 0
    }
    if missing_values:
        recommendations.append(
            "Review columns with remaining missing values before hypothesis generation or modeling."
        )

    if int(eda_summary.get("duplicate_rows_remaining", 0)) > 0:
        recommendations.append(
            "Review remaining duplicate rows and decide whether they are expected observations."
        )

    if findings.get("correlation_findings"):
        recommendations.append(
            "Use strong correlation findings as candidates for deeper bivariate analysis."
        )

    if eda_summary.get("target_column"):
        recommendations.append(
            "Use target-specific findings to guide Week 4 feature preparation without assuming causality."
        )
    else:
        recommendations.append(
            "Select a target column before supervised modeling in Week 4, if the project has one."
        )

    recommendations.append(
        "Keep `eda_summary.json` and `eda_findings.json` as inputs for the next hypothesis agent."
    )

    return _deduplicate(recommendations)


def _columns_by_type(
    dataframe: pd.DataFrame,
    schema: dict[str, str],
    semantic_types: set[str],
) -> list[str]:
    return [
        str(column)
        for column in dataframe.columns
        if schema.get(str(column), "unknown") in semantic_types
    ]


def _display_value(value: Any) -> str:
    if pd.isna(value):
        return "Missing"
    return str(value)


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduplicated: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduplicated.append(value)
    return deduplicated
