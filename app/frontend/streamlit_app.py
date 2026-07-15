"""Streamlit frontend for AutoDS Agent."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv("AUTODS_BACKEND_URL", "http://localhost:8000").rstrip("/")


def main() -> None:
    st.set_page_config(
        page_title="AutoDS Agent",
        page_icon="AD",
        layout="wide",
    )

    st.title("AutoDS Agent: Autonomous Data Science Analyst")
    st.write(
        "Upload a CSV, profile the dataset, generate a conservative cleaning plan, "
        "and apply safe cleaning while preserving the raw input."
    )
    st.caption(f"Backend: {BACKEND_URL}")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        if st.button("Create analysis run", type="primary"):
            metadata = upload_csv(uploaded_file)
            if metadata:
                st.session_state["metadata"] = metadata
                st.session_state.pop("profile", None)
                st.session_state.pop("cleaning_plan", None)
                st.session_state.pop("cleaning_summary", None)

    metadata = st.session_state.get("metadata")
    if metadata:
        render_metadata(metadata)
        render_week2_workflow(metadata["run_id"])
    else:
        st.info("Create an analysis run to begin profiling and cleaning.")


def upload_csv(uploaded_file) -> dict | None:
    """Send the uploaded CSV to the FastAPI backend."""

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "text/csv",
        )
    }

    try:
        with st.spinner("Creating run and reading metadata..."):
            response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
            response.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the backend. Start it with: "
            "uvicorn app.backend.main:app --reload"
        )
        return None
    except requests.exceptions.HTTPError:
        detail = _extract_error_detail(response)
        st.error(f"Upload failed: {detail}")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Upload failed: {exc}")
        return None

    return response.json()


def render_metadata(metadata: dict) -> None:
    """Render uploaded dataset metadata."""

    st.success("Analysis run ready.")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Rows", metadata["rows"])
    metric_cols[1].metric("Columns", metadata["columns"])
    metric_cols[2].metric("Duplicate rows", metadata["duplicate_rows"])

    st.text_input("Run ID", value=metadata["run_id"], disabled=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Column Data Types")
        st.dataframe(
            pd.DataFrame(
                metadata["dtypes"].items(),
                columns=["Column", "Data type"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Missing Values")
        st.dataframe(
            pd.DataFrame(
                metadata["missing_values"].items(),
                columns=["Column", "Missing values"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Column Names")
    st.write(", ".join(metadata["column_names"]))

    st.subheader("Dataset Preview")
    st.dataframe(pd.DataFrame(metadata["preview"]), use_container_width=True)


def render_week2_workflow(run_id: str) -> None:
    """Render profiling and cleaning controls for one run."""

    st.divider()
    profile_tab, plan_tab, clean_tab, next_tab = st.tabs(
        ["Dataset Profile", "Cleaning Plan", "Safe Cleaning", "Week 3"]
    )

    with profile_tab:
        if st.button("Generate Dataset Profile", key="generate_profile"):
            profile = post_json(
                f"/runs/{run_id}/profile",
                "Generating dataset profile...",
            )
            if profile:
                st.session_state["profile"] = profile

        profile = st.session_state.get("profile")
        if profile:
            render_profile(profile)

    with plan_tab:
        if st.button("Generate Cleaning Plan", key="generate_cleaning_plan"):
            plan = post_json(
                f"/runs/{run_id}/cleaning-plan",
                "Generating cleaning plan...",
            )
            if plan:
                st.session_state["cleaning_plan"] = plan

        plan = st.session_state.get("cleaning_plan")
        if plan:
            render_cleaning_plan(plan)

    with clean_tab:
        st.caption("Safe cleaning is conservative. The raw CSV is preserved unchanged.")
        if st.button("Apply Safe Cleaning", key="apply_safe_cleaning"):
            summary = post_json(
                f"/runs/{run_id}/clean",
                "Applying safe cleaning...",
            )
            if summary:
                st.session_state["cleaning_summary"] = summary

        summary = st.session_state.get("cleaning_summary")
        if summary:
            render_cleaning_summary(summary)

    with next_tab:
        st.subheader("Next: EDA generation and visualization")
        st.write(
            "Week 3 can build on `cleaned_data.csv` with deterministic charts, "
            "EDA summaries, and richer analyst-style observations."
        )


def render_profile(profile: dict[str, Any]) -> None:
    """Render a rich dataset profile."""

    st.subheader("Dataset Summary")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", profile["rows"])
    metric_cols[1].metric("Columns", profile["columns"])
    metric_cols[2].metric("Missing values", profile["total_missing_values"])
    metric_cols[3].metric("Duplicate rows", profile["duplicate_rows"])

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Column Types")
        st.dataframe(
            pd.DataFrame(
                profile["column_type_counts"].items(),
                columns=["Semantic type", "Count"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Data Quality Warnings")
        issues = profile.get("data_quality_issues", [])
        if issues:
            st.dataframe(
                pd.DataFrame(issues)[
                    ["severity", "issue_type", "column", "message", "recommendation"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No data quality warnings were generated.")

    st.subheader("Column Profile")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "column": column["column_name"],
                    "pandas dtype": column["pandas_dtype"],
                    "semantic type": column["semantic_type"],
                    "missing %": column["missing_percentage"],
                    "unique values": column["unique_values"],
                    "unique %": column["unique_percentage"],
                    "constant": column["is_constant"],
                    "high cardinality": column["is_high_cardinality"],
                    "id-like": column["is_id"],
                }
                for column in profile.get("column_profiles", [])
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_cleaning_plan(plan: dict[str, Any]) -> None:
    """Render a cleaning plan."""

    st.subheader("Duplicate Handling")
    duplicate_action = plan["duplicate_row_handling"]
    st.write(duplicate_action["reason"])

    st.subheader("Missing Value Strategies")
    missing_actions = plan.get("missing_value_strategies", [])
    if missing_actions:
        st.dataframe(
            pd.DataFrame(missing_actions)[
                ["column", "strategy", "apply", "reason"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No missing value strategies are needed.")

    left, right = st.columns(2)
    with left:
        st.subheader("Recommended Drops")
        drops = plan.get("columns_recommended_for_dropping", [])
        st.write(", ".join(drops) if drops else "None")

    with right:
        st.subheader("Type Conversions")
        conversions = plan.get("type_conversion_recommendations", [])
        if conversions:
            st.dataframe(
                pd.DataFrame(conversions)[["column", "strategy", "apply"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("None")

    st.subheader("Warnings Requiring Review")
    warnings = plan.get("warnings_requiring_review", [])
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No review warnings were added to the plan.")


def render_cleaning_summary(summary: dict[str, Any]) -> None:
    """Render the safe cleaning summary."""

    st.subheader("Cleaning Summary")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Original shape", _shape_text(summary["original_shape"]))
    metric_cols[1].metric("Cleaned shape", _shape_text(summary["cleaned_shape"]))
    metric_cols[2].metric("Duplicates removed", summary["duplicate_rows_removed"])
    metric_cols[3].metric(
        "Missing before/after",
        f"{summary['missing_values_before']} -> {summary['missing_values_after']}",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Columns Dropped")
        dropped = summary.get("columns_dropped", [])
        st.write(", ".join(dropped) if dropped else "None")

    with right:
        st.subheader("Imputations")
        imputations = summary.get("imputation_strategies_used", {})
        if imputations:
            st.dataframe(
                pd.DataFrame(
                    imputations.items(),
                    columns=["Column", "Strategy"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("None")

    warnings = summary.get("warnings", [])
    if warnings:
        st.subheader("Warnings")
        for warning in warnings:
            st.warning(warning)


def post_json(path: str, spinner_message: str) -> dict[str, Any] | None:
    """POST to the backend and return JSON."""

    try:
        with st.spinner(spinner_message):
            response = requests.post(f"{BACKEND_URL}{path}", timeout=120)
            response.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the backend. Start it with: "
            "uvicorn app.backend.main:app --reload"
        )
        return None
    except requests.exceptions.HTTPError:
        detail = _extract_error_detail(response)
        st.error(f"Backend request failed: {detail}")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Backend request failed: {exc}")
        return None

    return response.json()


def _shape_text(shape: list[int]) -> str:
    return f"{shape[0]} x {shape[1]}"


def _extract_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Unknown backend error."

    return str(payload.get("detail", payload))


if __name__ == "__main__":
    main()
