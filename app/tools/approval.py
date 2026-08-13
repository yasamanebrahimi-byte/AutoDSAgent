"""Human approval gate rules for workflow orchestration."""

from __future__ import annotations

from typing import Any


ApprovalDecision = dict[str, Any]


def cleaning_approval_decision(
    cleaning_plan: dict[str, Any],
    require_approval: bool = True,
) -> ApprovalDecision:
    """Evaluate whether a cleaning plan should pause for human approval."""

    if not require_approval:
        return {
            "required": False,
            "reasons": [],
            "details": {"approval_disabled": True},
        }

    reasons: list[str] = []
    details: dict[str, Any] = {}

    drop_columns = list(cleaning_plan.get("columns_recommended_for_dropping", []))
    if drop_columns:
        reasons.append(
            "The cleaning plan recommends dropping columns: "
            + ", ".join(str(column) for column in drop_columns)
        )
        details["columns_recommended_for_dropping"] = drop_columns

    duplicate_action = cleaning_plan.get("duplicate_row_handling", {}) or {}
    if bool(duplicate_action.get("apply")):
        reasons.append("The cleaning plan would remove duplicate rows.")
        details["duplicate_row_handling"] = duplicate_action

    review_warnings = list(cleaning_plan.get("warnings_requiring_review", []))
    if review_warnings:
        reasons.append("The cleaning plan contains warnings that require review.")
        details["warnings_requiring_review"] = review_warnings

    imputation_actions = [
        action
        for action in cleaning_plan.get("missing_value_strategies", [])
        if bool(action.get("apply"))
    ]
    imputation_columns = [
        str(action.get("column"))
        for action in imputation_actions
        if action.get("column") is not None
    ]
    if len(imputation_columns) >= 2:
        reasons.append(
            "The cleaning plan would impute missing values in multiple columns: "
            + ", ".join(imputation_columns)
        )
        details["imputation_columns"] = imputation_columns

    return {
        "required": bool(reasons),
        "reasons": reasons,
        "details": details,
    }


def modeling_approval_decision(
    state: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    require_approval: bool = True,
) -> ApprovalDecision:
    """Evaluate whether modeling should pause for human approval."""

    target_column = state.get("target_column")
    if not target_column:
        return {
            "required": False,
            "reasons": ["Modeling is skipped because no target column was provided."],
            "details": {},
        }

    if not require_approval:
        return {
            "required": False,
            "reasons": [],
            "details": {"approval_disabled": True},
        }

    reasons = [
        f"Modeling is about to train deterministic models using `{target_column}` as the target."
    ]
    details: dict[str, Any] = {"target_column": target_column}

    row_count = int((metadata or {}).get("rows") or 0)
    if row_count >= 10_000:
        reasons.append(
            f"The dataset has {row_count} rows, so training may take extra time."
        )
        details["row_count"] = row_count

    quality_warnings = list(state.get("warnings", []))
    if quality_warnings:
        details["workflow_warnings"] = quality_warnings

    return {
        "required": True,
        "reasons": reasons,
        "details": details,
    }
