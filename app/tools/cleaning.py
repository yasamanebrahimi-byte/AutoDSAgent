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
    target_column: str | None = None,
) -> dict[str, Any]:
    """Generate a conservative cleaning plan from a saved profile payload."""

    selected_config = config or CleaningConfig()
    columns = profile.get("column_profiles", [])
    available_columns = {str(column["column_name"]) for column in columns}
    selected_target = _normalize_optional_target(target_column)
    if selected_target is not None and selected_target not in available_columns:
        raise ValueError(f"Target column '{selected_target}' was not found in the profile.")

    duplicate_rows = int(profile.get("duplicate_rows", 0))
    total_columns = max(int(profile.get("columns", len(columns))), 1)

    constant_columns = [
        str(column["column_name"])
        for column in columns
        if bool(column.get("is_constant", False))
        and str(column["column_name"]) != selected_target
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
        is_target = column_name == selected_target

        if is_target and bool(column.get("is_constant", False)):
            warnings_requiring_review.append(
                f"Target column '{column_name}' is constant; it was preserved during cleaning."
            )

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
            if is_target:
                missing_action = _target_missing_value_action(
                    column_name=column_name,
                    missing_percentage=missing_percentage,
                )
                warnings_requiring_review.append(
                    f"Target column '{column_name}' has {missing_percentage:.1f}% missing values; "
                    "safe cleaning will preserve them and supervised modeling will exclude those rows."
                )
            else:
                missing_action = _missing_value_action(
                    column_name=column_name,
                    semantic_type=semantic_type,
                    missing_percentage=missing_percentage,
                    config=selected_config,
                )
            missing_value_strategies.append(missing_action)
            actions.append(missing_action)

            if (
                not is_target
                and missing_percentage >= selected_config.high_missing_percentage
            ):
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
                "apply": not is_target,
                "details": {},
            }
            if is_target:
                action["reason"] = (
                    "Datetime-like target values are preserved during cleaning unless a "
                    "supervised target policy explicitly transforms them."
                )
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
        "target_column": selected_target,
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
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply only conservative cleaning actions from a cleaning plan."""

    selected_config = config or CleaningConfig()
    selected_target = _normalize_optional_target(target_column) or _normalize_optional_target(
        plan.get("target_column")
    )
    if selected_target is not None and selected_target not in dataframe.columns:
        raise ValueError(f"Target column '{selected_target}' was not found in the dataset.")

    cleaned = dataframe.copy()
    original_shape = [int(dataframe.shape[0]), int(dataframe.shape[1])]
    missing_values_before = int(dataframe.isna().sum().sum())
    target_missing_values_before = (
        int(dataframe[selected_target].isna().sum())
        if selected_target is not None
        else None
    )
    warnings = list(plan.get("warnings_requiring_review", []))
    imputation_strategies_used: dict[str, str] = {}
    type_conversions_applied: dict[str, str] = {}
    datetime_parse_failures: dict[str, dict[str, Any]] = {}

    duplicate_rows_before = int(cleaned.duplicated().sum())
    if bool(plan.get("duplicate_row_handling", {}).get("apply", False)):
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicate_rows_removed = duplicate_rows_before - int(cleaned.duplicated().sum())

    columns_to_drop = _columns_to_drop_from_plan(plan, cleaned.columns, selected_target)
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
        if selected_target is not None and column == selected_target:
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
            cleaned[column] = _fill_missing_preserving_future_behavior(
                cleaned[column],
                selected_config.categorical_missing_value,
            )
            imputation_strategies_used[str(column)] = selected_config.categorical_missing_value
        elif strategy == "mode_imputation":
            modes = cleaned[column].mode(dropna=True)
            if not modes.empty:
                cleaned[column] = _fill_missing_preserving_future_behavior(
                    cleaned[column],
                    modes.iloc[0],
                )
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
        if selected_target is not None and column == selected_target:
            continue

        strategy = str(action.get("strategy"))
        if strategy == "parse_datetime_to_iso_string":
            original = cleaned[column].copy()
            parsed = pd.to_datetime(original, errors="coerce", format="mixed")
            failed_mask = original.notna() & parsed.isna()
            failed_count = int(failed_mask.sum())
            if failed_count:
                examples = (
                    original[failed_mask]
                    .astype(str)
                    .drop_duplicates()
                    .head(5)
                    .tolist()
                )
                datetime_parse_failures[str(column)] = {
                    "failed_values": failed_count,
                    "examples": examples,
                }
                warnings.append(
                    f"Column '{column}' was not converted to datetime because "
                    f"{failed_count} non-null value(s) failed parsing."
                )
                continue
            cleaned[column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")
            type_conversions_applied[str(column)] = "parsed_datetime_to_iso_string"

    target_missing_values_after = (
        int(cleaned[selected_target].isna().sum())
        if selected_target is not None and selected_target in cleaned.columns
        else None
    )
    target_action = None
    if selected_target is not None:
        target_action = (
            "missing target values preserved; supervised modeling excludes those rows"
            if target_missing_values_before
            else "target preserved"
        )

    summary = {
        "run_id": profile["run_id"],
        "target_column": selected_target,
        "original_shape": original_shape,
        "cleaned_shape": [int(cleaned.shape[0]), int(cleaned.shape[1])],
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "columns_dropped": columns_to_drop,
        "missing_values_before": missing_values_before,
        "missing_values_after": int(cleaned.isna().sum().sum()),
        "imputation_strategies_used": imputation_strategies_used,
        "type_conversions_applied": type_conversions_applied,
        "target_missing_values_before": target_missing_values_before,
        "target_missing_values_after": target_missing_values_after,
        "target_action": target_action,
        "datetime_parse_failures": datetime_parse_failures,
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
        if missing_percentage >= config.high_missing_percentage:
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
        "reason": "Recommendation for future modeling; not applied during safe cleaning.",
        "apply": False,
        "details": {},
    }


def _target_missing_value_action(
    column_name: str,
    missing_percentage: float,
) -> dict[str, Any]:
    return {
        "action_type": "target_missing_values",
        "column": column_name,
        "strategy": "preserve_missing_target",
        "reason": (
            "Supervised labels are not imputed during cleaning; modeling preparation "
            "will exclude rows with missing target values."
        ),
        "apply": False,
        "details": {"missing_percentage": missing_percentage},
    }


def _columns_to_drop_from_plan(
    plan: dict[str, Any],
    available_columns: pd.Index,
    target_column: str | None = None,
) -> list[str]:
    available = {str(column) for column in available_columns}
    return [
        str(action["column"])
        for action in plan.get("actions", [])
        if action.get("action_type") == "drop_constant_column"
        and bool(action.get("apply", False))
        and action.get("column") in available
        and action.get("column") != target_column
    ]


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fill_missing_preserving_future_behavior(series: pd.Series, value: Any) -> pd.Series:
    with pd.option_context("future.no_silent_downcasting", True):
        filled = series.fillna(value)
    return filled.infer_objects(copy=False)


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduplicated: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduplicated.append(value)
    return deduplicated


def _normalize_optional_target(target_column: Any) -> str | None:
    if target_column is None:
        return None
    target = str(target_column).strip()
    return target or None
