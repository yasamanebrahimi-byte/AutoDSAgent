"""Conservative cleaning plan generation and execution helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CleaningConfig:
    """Configurable thresholds for safe cleaning."""

    high_missing_percentage: float = 80.0
    moderate_missing_percentage: float = 50.0
    max_auto_drop_column_fraction: float = 0.5
    categorical_missing_value: str = "Unknown"


def generate_cleaning_plan_payload(
    profile: dict[str, Any],
    config: CleaningConfig | None = None,
) -> dict[str, Any]:
    """Generate a conservative cleaning plan from a saved profile payload."""

    selected_config = config or CleaningConfig()
    columns = profile.get("column_profiles", [])
    duplicate_rows = int(profile.get("duplicate_rows", 0))
    total_columns = max(int(profile.get("columns", len(columns))), 1)

    constant_columns = [
        str(column["column_name"])
        for column in columns
        if bool(column.get("is_constant", False))
    ]
    auto_drop_constant_columns = (
        len(constant_columns) / total_columns
        <= selected_config.max_auto_drop_column_fraction
    )

    missing_value_strategies: list[dict[str, Any]] = []
    type_conversion_recommendations: list[dict[str, Any]] = []
    encoding_recommendations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    columns_recommended_for_dropping = list(constant_columns)
    warnings_requiring_review: list[str] = []

    duplicate_action = {
        "action_type": "duplicate_rows",
        "column": None,
        "strategy": "remove_exact_duplicates",
        "reason": (
            f"{duplicate_rows} exact duplicate rows were found."
            if duplicate_rows
            else "No exact duplicate rows were found."
        ),
        "apply": duplicate_rows > 0,
        "details": {"duplicate_rows": duplicate_rows},
    }
    actions.append(duplicate_action)

    if constant_columns and not auto_drop_constant_columns:
        warnings_requiring_review.append(
            "More constant columns were found than the automatic drop limit allows; "
            "review them before dropping."
        )

    for column in columns:
        column_name = str(column["column_name"])
        semantic_type = str(column.get("semantic_type", "unknown"))
        missing_percentage = float(column.get("missing_percentage", 0.0))

        if column_name in constant_columns:
            action = {
                "action_type": "drop_constant_column",
                "column": column_name,
                "strategy": "drop",
                "reason": "The column has one or fewer distinct non-null values.",
                "apply": auto_drop_constant_columns,
                "details": {
                    "missing_percentage": missing_percentage,
                    "auto_drop_limit_fraction": (
                        selected_config.max_auto_drop_column_fraction
                    ),
                },
            }
            actions.append(action)

        if missing_percentage > 0:
            missing_action = _missing_value_action(
                column_name=column_name,
                semantic_type=semantic_type,
                missing_percentage=missing_percentage,
                config=selected_config,
            )
            missing_value_strategies.append(missing_action)
            actions.append(missing_action)

            if missing_percentage > selected_config.high_missing_percentage:
                columns_recommended_for_dropping.append(column_name)
                warnings_requiring_review.append(
                    f"Column '{column_name}' is {missing_percentage:.1f}% missing; "
                    "safe cleaning will not drop it automatically."
                )

        if semantic_type == "datetime":
            action = {
                "action_type": "type_conversion",
                "column": column_name,
                "strategy": "parse_datetime_to_iso_string",
                "reason": "Datetime-like values should be normalized for later analysis.",
                "apply": True,
                "details": {},
            }
            type_conversion_recommendations.append(action)
            actions.append(action)

        encoding_recommendation = _encoding_recommendation(column_name, semantic_type)
        if encoding_recommendation is not None:
            encoding_recommendations.append(encoding_recommendation)

    for issue in profile.get("data_quality_issues", []):
        if issue.get("severity") in {"warning", "critical"}:
            warnings_requiring_review.append(str(issue.get("message")))

    columns_recommended_for_dropping = sorted(set(columns_recommended_for_dropping))
    columns_recommended_for_keeping = [
        str(column["column_name"])
        for column in columns
        if str(column["column_name"]) not in set(columns_recommended_for_dropping)
    ]

    return {
        "run_id": profile["run_id"],
        "duplicate_row_handling": duplicate_action,
        "missing_value_strategies": missing_value_strategies,
        "columns_recommended_for_dropping": columns_recommended_for_dropping,
        "columns_recommended_for_keeping": columns_recommended_for_keeping,
        "type_conversion_recommendations": type_conversion_recommendations,
        "encoding_recommendations": encoding_recommendations,
        "warnings_requiring_review": _deduplicate(warnings_requiring_review),
        "actions": actions,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def apply_safe_cleaning(
    dataframe: pd.DataFrame,
    profile: dict[str, Any],
    plan: dict[str, Any],
    config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply only conservative cleaning actions from a cleaning plan."""

    selected_config = config or CleaningConfig()
    cleaned = dataframe.copy()
    original_shape = [int(dataframe.shape[0]), int(dataframe.shape[1])]
    missing_values_before = int(dataframe.isna().sum().sum())
    warnings = list(plan.get("warnings_requiring_review", []))
    imputation_strategies_used: dict[str, str] = {}
    type_conversions_applied: dict[str, str] = {}

    duplicate_rows_before = int(cleaned.duplicated().sum())
    if bool(plan.get("duplicate_row_handling", {}).get("apply", False)):
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicate_rows_removed = duplicate_rows_before - int(cleaned.duplicated().sum())

    columns_to_drop = _columns_to_drop_from_plan(plan, cleaned.columns)
    if columns_to_drop:
        drop_fraction = len(columns_to_drop) / max(len(cleaned.columns), 1)
        if drop_fraction > selected_config.max_auto_drop_column_fraction:
            warnings.append(
                "Constant columns were not dropped because they exceeded the automatic "
                "column-drop limit."
            )
            columns_to_drop = []

    if columns_to_drop:
        cleaned = cleaned.drop(columns=columns_to_drop)

    for action in plan.get("missing_value_strategies", []):
        if not bool(action.get("apply", False)):
            continue

        column = action.get("column")
        if column not in cleaned.columns:
            continue

        strategy = str(action.get("strategy"))
        if strategy == "median_imputation":
            numeric = pd.to_numeric(cleaned[column], errors="coerce")
            median = numeric.median()
            if _is_finite_number(median):
                cleaned[column] = numeric.fillna(median)
                imputation_strategies_used[str(column)] = "median"
            else:
                warnings.append(
                    f"Column '{column}' could not be median-imputed because no median was available."
                )
        elif strategy == "fill_unknown":
            cleaned[column] = cleaned[column].fillna(selected_config.categorical_missing_value)
            imputation_strategies_used[str(column)] = selected_config.categorical_missing_value
        elif strategy == "mode_imputation":
            modes = cleaned[column].mode(dropna=True)
            if not modes.empty:
                cleaned[column] = cleaned[column].fillna(modes.iloc[0])
                imputation_strategies_used[str(column)] = "mode"
            else:
                warnings.append(
                    f"Column '{column}' could not be mode-imputed because no mode was available."
                )

    for action in plan.get("type_conversion_recommendations", []):
        if not bool(action.get("apply", False)):
            continue

        column = action.get("column")
        if column not in cleaned.columns:
            continue

        strategy = str(action.get("strategy"))
        if strategy == "parse_datetime_to_iso_string":
            parsed = pd.to_datetime(cleaned[column], errors="coerce", format="mixed")
            cleaned[column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")
            type_conversions_applied[str(column)] = "parsed_datetime_to_iso_string"

    summary = {
        "run_id": profile["run_id"],
        "original_shape": original_shape,
        "cleaned_shape": [int(cleaned.shape[0]), int(cleaned.shape[1])],
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "columns_dropped": columns_to_drop,
        "missing_values_before": missing_values_before,
        "missing_values_after": int(cleaned.isna().sum().sum()),
        "imputation_strategies_used": imputation_strategies_used,
        "type_conversions_applied": type_conversions_applied,
        "warnings": _deduplicate(warnings),
    }

    return cleaned, summary


def _missing_value_action(
    column_name: str,
    semantic_type: str,
    missing_percentage: float,
    config: CleaningConfig,
) -> dict[str, Any]:
    if semantic_type == "numeric":
        if missing_percentage > config.high_missing_percentage:
            return {
                "action_type": "missing_values",
                "column": column_name,
                "strategy": "review_high_missingness",
                "reason": "Numeric missingness is too high for automatic imputation.",
                "apply": False,
                "details": {"missing_percentage": missing_percentage},
            }

        return {
            "action_type": "missing_values",
            "column": column_name,
            "strategy": "median_imputation",
            "reason": "Median imputation is a conservative numeric default.",
            "apply": True,
            "details": {"missing_percentage": missing_percentage},
        }

    if semantic_type in {"categorical", "text"}:
        return {
            "action_type": "missing_values",
            "column": column_name,
            "strategy": "fill_unknown",
            "reason": "Missing labels can be preserved as an explicit Unknown category.",
            "apply": True,
            "details": {"missing_percentage": missing_percentage},
        }

    if semantic_type == "boolean":
        return {
            "action_type": "missing_values",
            "column": column_name,
            "strategy": "mode_imputation",
            "reason": "Mode imputation is a conservative boolean default.",
            "apply": True,
            "details": {"missing_percentage": missing_percentage},
        }

    return {
        "action_type": "missing_values",
        "column": column_name,
        "strategy": "review_missing_values",
        "reason": "The column type is not safe to impute automatically.",
        "apply": False,
        "details": {"missing_percentage": missing_percentage},
    }


def _encoding_recommendation(column_name: str, semantic_type: str) -> dict[str, Any] | None:
    strategies = {
        "categorical": "one_hot_or_target_encoding",
        "boolean": "binary_encoding",
        "text": "text_vectorization",
        "id": "exclude_from_modeling_features",
    }
    strategy = strategies.get(semantic_type)
    if strategy is None:
        return None

    return {
        "action_type": "future_encoding",
        "column": column_name,
        "strategy": strategy,
        "reason": "Recommendation for future modeling; not applied in Week 2.",
        "apply": False,
        "details": {},
    }


def _columns_to_drop_from_plan(plan: dict[str, Any], available_columns: pd.Index) -> list[str]:
    available = {str(column) for column in available_columns}
    return [
        str(action["column"])
        for action in plan.get("actions", [])
        if action.get("action_type") == "drop_constant_column"
        and bool(action.get("apply", False))
        and action.get("column") in available
    ]


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduplicated: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduplicated.append(value)
    return deduplicated
