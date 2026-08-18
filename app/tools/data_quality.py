"""Data quality issue detection for dataset profiles."""

from __future__ import annotations

from typing import Any

import pandas as pd


def detect_data_quality_issues(
    dataframe: pd.DataFrame,
    column_profiles: list[dict[str, Any]],
    target_column: str | None = None,
) -> list[dict[str, Any]]:
    """Detect structured data quality issues from a DataFrame and column profiles."""

    issues: list[dict[str, Any]] = []
    rows, columns = dataframe.shape
    duplicate_rows = int(dataframe.duplicated().sum())

    if rows == 0 or columns == 0:
        issues.append(
            _issue(
                severity="critical",
                issue_type="empty_dataset",
                column=None,
                message="The dataset appears to be empty.",
                recommendation="Upload a dataset with rows and columns before analysis.",
            )
        )

    if duplicate_rows > 0:
        issues.append(
            _issue(
                severity="warning",
                issue_type="duplicate_rows",
                column=None,
                message=f"The dataset contains {duplicate_rows} exact duplicate rows.",
                recommendation="Remove exact duplicate rows during safe cleaning.",
            )
        )

    if 0 < rows < 30:
        issues.append(
            _issue(
                severity="info",
                issue_type="very_few_rows",
                column=None,
                message=f"The dataset has only {rows} rows.",
                recommendation="Treat statistics and future modeling results as unstable.",
            )
        )

    if rows > 0 and columns > max(20, rows):
        issues.append(
            _issue(
                severity="warning",
                issue_type="many_columns_relative_to_rows",
                column=None,
                message="The dataset has many columns relative to the number of rows.",
                recommendation="Review feature relevance before future modeling.",
            )
        )

    for profile in column_profiles:
        column = str(profile["column_name"])
        missing_percentage = float(profile.get("missing_percentage", 0.0))
        semantic_type = str(profile.get("semantic_type", "unknown"))

        if missing_percentage > 50:
            severity = "critical" if missing_percentage >= 80 else "warning"
            issues.append(
                _issue(
                    severity=severity,
                    issue_type="high_missingness",
                    column=column,
                    message=f"Column '{column}' is {missing_percentage:.1f}% missing.",
                    recommendation=(
                        "Review whether the column should be dropped, imputed, or collected again."
                    ),
                )
            )

        if bool(profile.get("is_constant", False)):
            issues.append(
                _issue(
                    severity="warning",
                    issue_type="constant_column",
                    column=column,
                    message=f"Column '{column}' has one or fewer distinct non-null values.",
                    recommendation="Drop the column unless it has business meaning outside modeling.",
                )
            )

        if bool(profile.get("is_id", False)):
            issues.append(
                _issue(
                    severity="info",
                    issue_type="likely_id_column",
                    column=column,
                    message=f"Column '{column}' appears to be an identifier.",
                    recommendation="Keep it for traceability, but exclude it from future modeling features.",
                )
            )

        if bool(profile.get("is_high_cardinality", False)):
            issues.append(
                _issue(
                    severity="warning",
                    issue_type="high_cardinality",
                    column=column,
                    message=f"Column '{column}' has very high cardinality.",
                    recommendation="Review before encoding for future modeling.",
                )
            )

        numeric_stats = profile.get("numeric_stats") or {}
        outlier_count = int(numeric_stats.get("possible_outlier_count") or 0)
        if outlier_count > 0:
            issues.append(
                _issue(
                    severity="info",
                    issue_type="possible_numeric_outliers",
                    column=column,
                    message=f"Column '{column}' has {outlier_count} possible IQR outliers.",
                    recommendation="Inspect outliers during EDA before clipping or removing values.",
                )
            )

        if semantic_type == "datetime":
            inconsistent_values = _count_inconsistent_datetimes(dataframe[column])
            if inconsistent_values > 0:
                issues.append(
                    _issue(
                        severity="warning",
                        issue_type="inconsistent_datetime_parsing",
                        column=column,
                        message=(
                            f"Column '{column}' has {inconsistent_values} non-null values "
                            "that could not be parsed as datetimes."
                        ),
                        recommendation="Review invalid date values before time-based analysis.",
                    )
                )

    if not str(target_column or "").strip():
        issues.append(
            _issue(
                severity="info",
                issue_type="target_not_selected",
                column=None,
                message="No target column has been selected yet.",
                recommendation="Select a target later if the project moves into supervised modeling.",
            )
        )

    return issues


def _count_inconsistent_datetimes(series: pd.Series) -> int:
    non_null = series.dropna()
    if non_null.empty:
        return 0

    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return int(parsed.isna().sum())


def _issue(
    severity: str,
    issue_type: str,
    column: str | None,
    message: str,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "issue_type": issue_type,
        "column": column,
        "message": message,
        "recommendation": recommendation,
    }
