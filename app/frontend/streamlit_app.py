"""Streamlit frontend for AutoDS Agent."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

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
        "apply safe cleaning, generate deterministic EDA, and train lightweight "
        "baseline models while preserving the raw input."
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
                st.session_state.pop("eda_response", None)
                st.session_state.pop("modeling_response", None)
                st.session_state.pop("workflow_state", None)

    metadata = st.session_state.get("metadata")
    if metadata:
        render_metadata(metadata)
        render_autonomous_workflow(metadata)
        st.header("Advanced Manual Controls")
        render_week2_workflow(metadata)
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


def render_week2_workflow(metadata: dict[str, Any]) -> None:
    """Render profiling and cleaning controls for one run."""

    run_id = metadata["run_id"]

    st.divider()
    profile_tab, plan_tab, clean_tab, eda_tab, modeling_tab, next_tab = st.tabs(
        [
            "Dataset Profile",
            "Cleaning Plan",
            "Safe Cleaning",
            "Exploratory Data Analysis",
            "Modeling and Evaluation",
            "Next",
        ]
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
                st.session_state.pop("eda_response", None)
                st.session_state.pop("modeling_response", None)

        summary = st.session_state.get("cleaning_summary")
        if summary:
            render_cleaning_summary(summary)

    with eda_tab:
        render_eda_workflow(run_id=run_id, column_names=metadata.get("column_names", []))

    with modeling_tab:
        render_modeling_workflow(
            run_id=run_id,
            column_names=metadata.get("column_names", []),
        )

    with next_tab:
        st.subheader("Next: Agent orchestration, retries, and human approval gates")
        st.write(
            "Week 5 can connect these deterministic services through an agent workflow "
            "with retry behavior and user review steps. Week 4 does not use LLM calls."
        )


def render_autonomous_workflow(metadata: dict[str, Any]) -> None:
    """Render Week 5 autonomous workflow controls and state."""

    run_id = metadata["run_id"]
    column_names = list(metadata.get("column_names", []))

    st.divider()
    st.header("Autonomous Workflow")
    st.caption(
        "Runs the deterministic Week 1-4 services through an auditable agent workflow. "
        "No LLM API calls or paid credits are used."
    )

    target_options = ["No target column"] + column_names
    selected_target = st.selectbox(
        "Optional target column",
        options=target_options,
        index=0,
        key="workflow_target_column",
    )
    target_column = None if selected_target == "No target column" else selected_target

    control_cols = st.columns(3)
    with control_cols[0]:
        task_type_label = st.selectbox(
            "Task type",
            options=["Auto-detect", "Regression", "Classification"],
            key="workflow_task_type",
        )
    with control_cols[1]:
        require_cleaning_approval = st.checkbox(
            "Require approval before cleaning",
            value=True,
            key="workflow_require_cleaning_approval",
        )
    with control_cols[2]:
        require_modeling_approval = st.checkbox(
            "Require approval before modeling",
            value=True,
            key="workflow_require_modeling_approval",
        )

    action_cols = st.columns([1, 1, 4])
    with action_cols[0]:
        if st.button("Start Autonomous Workflow", type="primary"):
            payload = {
                "target_column": target_column,
                "task_type": None
                if task_type_label == "Auto-detect"
                else task_type_label.lower(),
                "require_cleaning_approval": require_cleaning_approval,
                "require_modeling_approval": require_modeling_approval,
            }
            state = post_json(
                f"/runs/{run_id}/workflow/start",
                "Running autonomous workflow...",
                payload=payload,
            )
            if state:
                st.session_state["workflow_state"] = state

    with action_cols[1]:
        if st.button("Refresh Workflow"):
            state = get_json(f"/runs/{run_id}/workflow/state")
            if state:
                st.session_state["workflow_state"] = state

    state = st.session_state.get("workflow_state")
    if state:
        render_workflow_state(run_id, state)
    else:
        st.info("Start the autonomous workflow to profile, clean, analyze, and optionally model this run.")


def render_workflow_state(run_id: str, state: dict[str, Any]) -> None:
    """Render workflow progress, approval gates, retry controls, artifacts, and trace."""

    status_cols = st.columns(3)
    status_cols[0].metric("Workflow status", state.get("status", "unknown"))
    status_cols[1].metric("Current step", state.get("current_step") or "None")
    status_cols[2].metric("Target", state.get("target_column") or "None")

    st.subheader("Step Status")
    st.dataframe(
        pd.DataFrame(_workflow_step_rows(state)),
        use_container_width=True,
        hide_index=True,
    )

    render_workflow_approval_controls(run_id, state)
    render_workflow_retry_controls(run_id, state)
    render_workflow_artifacts(run_id, state)
    render_workflow_trace(run_id)


def render_workflow_approval_controls(run_id: str, state: dict[str, Any]) -> None:
    """Render approval controls for a waiting workflow gate."""

    waiting_steps = [
        step
        for step, step_state in state.get("steps", {}).items()
        if step_state.get("status") == "waiting_for_approval"
    ]
    if not waiting_steps:
        return

    step = waiting_steps[0]
    step_state = state["steps"][step]
    st.subheader("Approval Needed")
    st.warning(step_state.get("approval_reason") or f"{step} needs approval.")

    reasons = step_state.get("approval_details", {}).get("reasons", [])
    if reasons:
        for reason in reasons:
            st.write(f"- {reason}")

    approval_cols = st.columns([1, 1, 4])
    with approval_cols[0]:
        if st.button("Approve and Continue", type="primary", key=f"approve_{step}"):
            updated = post_json(
                f"/runs/{run_id}/workflow/approve",
                "Applying approval and continuing workflow...",
                payload={"step": step, "action": "approve"},
            )
            if updated:
                st.session_state["workflow_state"] = updated
                st.rerun()
    with approval_cols[1]:
        if st.button("Reject Step", key=f"reject_{step}"):
            updated = post_json(
                f"/runs/{run_id}/workflow/approve",
                "Recording rejection...",
                payload={"step": step, "action": "reject"},
            )
            if updated:
                st.session_state["workflow_state"] = updated
                st.rerun()


def render_workflow_retry_controls(run_id: str, state: dict[str, Any]) -> None:
    """Render retry controls for failed steps."""

    failed_steps = [
        (step, step_state)
        for step, step_state in state.get("steps", {}).items()
        if step_state.get("status") == "failed"
    ]
    if not failed_steps:
        return

    st.subheader("Failed Step")
    for step, step_state in failed_steps:
        st.error(step_state.get("error") or f"{step} failed.")
        attempts = int(step_state.get("attempts", 0))
        max_attempts = int(step_state.get("max_attempts", 0))
        if attempts < max_attempts:
            if st.button(f"Retry {step}", key=f"retry_{step}"):
                updated = post_json(
                    f"/runs/{run_id}/workflow/retry",
                    f"Retrying {step}...",
                    payload={"step": step},
                )
                if updated:
                    st.session_state["workflow_state"] = updated
                    st.rerun()
        else:
            st.warning("No retry attempts remain for this step.")


def render_workflow_artifacts(run_id: str, state: dict[str, Any]) -> None:
    """Render generated artifact pointers and summaries as they appear."""

    artifacts = state.get("artifacts", {})
    st.subheader("Generated Artifacts")
    rows = [
        {"Artifact": key, "Path": value if value else "Not generated"}
        for key, value in artifacts.items()
        if key != "plots"
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if artifacts.get("profile"):
        with st.expander("Profile", expanded=False):
            profile = get_json(f"/runs/{run_id}/profile", show_error=False)
            if profile:
                render_profile(profile)

    if artifacts.get("cleaning_plan"):
        with st.expander("Cleaning Plan", expanded=False):
            plan = get_json(f"/runs/{run_id}/cleaning-plan", show_error=False)
            if plan:
                render_cleaning_plan(plan)

    if artifacts.get("cleaning_summary"):
        with st.expander("Cleaning Summary", expanded=False):
            summary = get_json(f"/runs/{run_id}/cleaning-summary", show_error=False)
            if summary:
                render_cleaning_summary(summary)

    if artifacts.get("eda_summary"):
        with st.expander("EDA Summary", expanded=False):
            eda_response = get_json(f"/runs/{run_id}/eda", show_error=False)
            if eda_response:
                render_eda_response(run_id, eda_response)

    if artifacts.get("modeling_summary") or artifacts.get("evaluation_summary"):
        with st.expander("Modeling and Evaluation Summary", expanded=False):
            modeling_summary = get_json(
                f"/runs/{run_id}/modeling-summary",
                show_error=False,
            )
            evaluation_summary = get_json(
                f"/runs/{run_id}/evaluation-summary",
                show_error=False,
            )
            if modeling_summary:
                metric_cols = st.columns(4)
                metric_cols[0].metric("Target", modeling_summary["target_column"])
                metric_cols[1].metric("Task", modeling_summary["task_type"])
                metric_cols[2].metric("Best model", modeling_summary["best_model_name"])
                metric_cols[3].metric(
                    "Primary metric",
                    modeling_summary["primary_metric"].upper(),
                )
            if evaluation_summary:
                st.dataframe(
                    _metrics_dataframe(evaluation_summary.get("best_model_metrics", {})),
                    use_container_width=True,
                    hide_index=True,
                )


def render_workflow_trace(run_id: str) -> None:
    """Render ordered agent trace events."""

    with st.expander("Agent Trace", expanded=False):
        trace = get_json(f"/runs/{run_id}/workflow/trace", show_error=False)
        if not trace:
            st.write("No trace events have been saved yet.")
            return

        st.dataframe(
            pd.DataFrame(trace)[
                ["timestamp", "agent", "step", "event_type", "message"]
            ],
            use_container_width=True,
            hide_index=True,
        )


def _workflow_step_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, step_state in state.get("steps", {}).items():
        rows.append(
            {
                "Step": step,
                "Status": step_state.get("status"),
                "Attempts": f"{step_state.get('attempts', 0)} / {step_state.get('max_attempts', 0)}",
                "Approval": step_state.get("approval_status"),
                "Error": step_state.get("error") or "",
            }
        )
    return rows


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


def render_eda_workflow(run_id: str, column_names: list[str]) -> None:
    """Render Week 3 EDA controls and outputs."""

    st.subheader("Exploratory Data Analysis")
    st.caption(
        "EDA uses `cleaned_data.csv` when available and falls back to the raw upload with a warning."
    )

    target_options = ["No target column"] + list(column_names)
    selected_target = st.selectbox(
        "Optional target column",
        options=target_options,
        index=0,
        key="eda_target_column",
    )
    target_column = None if selected_target == "No target column" else selected_target

    with st.expander("Plot limits"):
        max_numeric_plots = st.number_input(
            "Numeric histograms",
            min_value=0,
            max_value=25,
            value=10,
            step=1,
        )
        max_categorical_plots = st.number_input(
            "Categorical bar charts",
            min_value=0,
            max_value=25,
            value=10,
            step=1,
        )
        max_target_relationship_plots = st.number_input(
            "Target relationship plots",
            min_value=0,
            max_value=15,
            value=5,
            step=1,
        )

    if st.button("Generate EDA", type="primary", key="generate_eda"):
        payload = {
            "target_column": target_column,
            "max_numeric_plots": int(max_numeric_plots),
            "max_categorical_plots": int(max_categorical_plots),
            "max_target_relationship_plots": int(max_target_relationship_plots),
        }
        response = post_json(
            f"/runs/{run_id}/eda",
            "Generating EDA summaries and plots...",
            payload=payload,
        )
        if response:
            st.session_state["eda_response"] = response

    eda_response = st.session_state.get("eda_response")
    if eda_response:
        render_eda_response(run_id, eda_response)


def render_modeling_workflow(run_id: str, column_names: list[str]) -> None:
    """Render Week 4 modeling controls and outputs."""

    st.subheader("Modeling and Evaluation")
    st.caption(
        "Modeling requires `cleaned_data.csv`, trains deterministic sklearn models, "
        "and does not use LLM API calls."
    )

    if not column_names:
        st.info("Upload a dataset before selecting a modeling target.")
        return

    target_column = st.selectbox(
        "Target column",
        options=column_names,
        key="modeling_target_column",
    )
    task_type_label = st.selectbox(
        "Task type",
        options=["Auto-detect", "Regression", "Classification"],
        key="modeling_task_type",
    )
    test_size = st.slider(
        "Test set size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        key="modeling_test_size",
    )

    if st.button("Train and Evaluate Models", type="primary", key="train_models"):
        payload = {
            "target_column": target_column,
            "task_type": None
            if task_type_label == "Auto-detect"
            else task_type_label.lower(),
            "test_size": float(test_size),
            "random_state": 42,
        }
        response = post_json(
            f"/runs/{run_id}/model",
            "Training baseline and candidate models...",
            payload=payload,
        )
        if response:
            st.session_state["modeling_response"] = response

    modeling_response = st.session_state.get("modeling_response")
    if modeling_response:
        render_modeling_response(run_id, modeling_response)


def render_modeling_response(run_id: str, response: dict[str, Any]) -> None:
    """Render modeling and evaluation summaries."""

    modeling_summary = response["modeling_summary"]
    evaluation_summary = response["evaluation_summary"]
    model_results = response.get("model_results", {})

    st.success("Modeling and evaluation artifacts were saved.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Target", modeling_summary["target_column"])
    metric_cols[1].metric("Task", modeling_summary["task_type"])
    metric_cols[2].metric("Best model", modeling_summary["best_model_name"])
    metric_cols[3].metric("Primary metric", modeling_summary["primary_metric"].upper())

    split_cols = st.columns(3)
    split_cols[0].metric("Rows used", modeling_summary["rows_used"])
    split_cols[1].metric("Train rows", modeling_summary["train_rows"])
    split_cols[2].metric("Test rows", modeling_summary["test_rows"])

    st.subheader("Model Metrics")
    comparison_table = _model_comparison_dataframe(model_results, evaluation_summary)
    if not comparison_table.empty:
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)
    else:
        st.write("No model metrics were returned.")

    left, right = st.columns(2)
    with left:
        st.subheader("Baseline Metrics")
        st.dataframe(
            _metrics_dataframe(evaluation_summary.get("baseline_metrics", {})),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Best Model Metrics")
        st.dataframe(
            _metrics_dataframe(evaluation_summary.get("best_model_metrics", {})),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Baseline Comparison")
    st.write(evaluation_summary.get("baseline_comparison", {}).get("interpretation", ""))

    warnings = evaluation_summary.get("warnings", [])
    if warnings:
        st.subheader("Evaluation Warnings")
        for warning in warnings:
            st.warning(warning)

    feature_cols = st.columns(2)
    with feature_cols[0]:
        with st.expander("Features used", expanded=False):
            features_used = modeling_summary.get("features_used", [])
            st.write(", ".join(features_used) if features_used else "None")
    with feature_cols[1]:
        with st.expander("Features excluded", expanded=False):
            excluded = modeling_summary.get("excluded_feature_reasons", {})
            if excluded:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Column": column, "Reason": reason}
                            for column, reason in excluded.items()
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("None")

    with st.expander("Models attempted", expanded=False):
        st.write(", ".join(modeling_summary.get("models_attempted", [])))
        succeeded = modeling_summary.get("models_succeeded", [])
        failed = modeling_summary.get("models_failed", [])
        st.write(f"Succeeded: {', '.join(succeeded) if succeeded else 'None'}")
        st.write(f"Failed: {', '.join(failed) if failed else 'None'}")

    failed_records = [
        result
        for result in model_results.get("results", [])
        if result.get("status") == "failed"
    ]
    if failed_records:
        with st.expander("Model failure details", expanded=False):
            st.dataframe(
                pd.DataFrame(failed_records)[["model_name", "role", "error"]],
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Evaluation Plots")
    plots = evaluation_summary.get("generated_plots", [])
    render_plot_section(
        "Model comparison",
        _plots_by_category(plots, "evaluation_model_comparison"),
        run_id,
    )
    render_plot_section(
        "Confusion matrix",
        _plots_by_category(plots, "evaluation_confusion_matrix"),
        run_id,
    )
    render_plot_section(
        "Predicted vs actual",
        _plots_by_category(plots, "evaluation_predicted_vs_actual"),
        run_id,
    )
    render_plot_section(
        "Residuals",
        _plots_by_category(plots, "evaluation_residuals"),
        run_id,
    )
    render_plot_section(
        "Feature signal",
        _plots_by_category(plots, "evaluation_feature_importance"),
        run_id,
    )


def render_eda_response(run_id: str, eda_response: dict[str, Any]) -> None:
    """Render EDA summary, findings, next steps, and generated plots."""

    summary = eda_response["summary"]
    findings = eda_response["findings"]

    if summary["dataset_used"] == "raw":
        st.warning("EDA was generated from the raw uploaded dataset.")
    else:
        st.success("EDA was generated from the cleaned dataset.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Dataset used", summary["dataset_used"])
    metric_cols[1].metric("Rows", summary["rows"])
    metric_cols[2].metric("Columns", summary["columns"])
    metric_cols[3].metric("Duplicate rows", summary["duplicate_rows_remaining"])

    st.write(f"Target column: `{summary.get('target_column') or 'None selected'}`")

    type_counts = {
        "numeric": len(summary.get("numeric_columns", [])),
        "categorical": len(summary.get("categorical_columns", [])),
        "boolean": len(summary.get("boolean_columns", [])),
        "datetime": len(summary.get("datetime_columns", [])),
        "text": len(summary.get("text_columns", [])),
        "id": len(summary.get("id_columns", [])),
    }
    st.subheader("Column Type Summary")
    st.dataframe(
        pd.DataFrame(type_counts.items(), columns=["Type", "Count"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Remaining Missing Values")
    remaining_missing = {
        column: count
        for column, count in summary.get("missing_values_remaining", {}).items()
        if int(count) > 0
    }
    if remaining_missing:
        st.dataframe(
            pd.DataFrame(
                remaining_missing.items(),
                columns=["Column", "Missing values"],
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No missing values remain in the EDA dataset.")

    warnings = summary.get("warnings", [])
    if warnings:
        st.subheader("Warnings And Notes")
        for warning in warnings:
            st.warning(warning)

    st.subheader("Key Findings")
    render_findings_group("Univariate findings", findings.get("univariate_findings", []))
    render_findings_group("Bivariate findings", findings.get("bivariate_findings", []))
    render_findings_group("Target findings", findings.get("target_findings", []))
    render_findings_group("Data quality notes", findings.get("data_quality_notes", []))

    st.subheader("Recommended Next Steps")
    for step in findings.get("recommended_next_steps", []):
        st.write(f"- {step}")

    st.subheader("Generated Plots")
    plots = summary.get("generated_plots", [])
    render_plot_section(
        "Missing values",
        _plots_by_category(plots, "missing_values"),
        run_id,
    )
    render_plot_section(
        "Numeric distributions",
        _plots_by_category(plots, "numeric_distribution"),
        run_id,
    )
    render_plot_section(
        "Categorical distributions",
        _plots_by_category(plots, "categorical_distribution"),
        run_id,
    )
    render_plot_section(
        "Correlation heatmap",
        _plots_by_category(plots, "correlation_heatmap"),
        run_id,
    )
    render_plot_section(
        "Target relationships",
        _plots_by_category(plots, "target_relationship"),
        run_id,
    )


def render_findings_group(title: str, values: list[str]) -> None:
    """Render a group of findings inside a compact expander."""

    with st.expander(title, expanded=bool(values)):
        if values:
            for value in values:
                st.write(f"- {value}")
        else:
            st.write("None")


def render_plot_section(title: str, plots: list[dict[str, Any]], run_id: str) -> None:
    """Render generated plot images in a two-column grid."""

    with st.expander(title, expanded=bool(plots)):
        if not plots:
            st.write("No plot generated.")
            return

        image_columns = st.columns(2)
        for index, plot in enumerate(plots):
            with image_columns[index % 2]:
                st.image(
                    _plot_image_url(run_id, plot["path"]),
                    caption=plot["label"],
                    use_column_width=True,
                )


def post_json(
    path: str,
    spinner_message: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """POST to the backend and return JSON."""

    try:
        with st.spinner(spinner_message):
            request_kwargs: dict[str, Any] = {"timeout": 120}
            if payload is not None:
                request_kwargs["json"] = payload
            response = requests.post(f"{BACKEND_URL}{path}", **request_kwargs)
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


def get_json(path: str, show_error: bool = True) -> dict[str, Any] | list[Any] | None:
    """GET from the backend and return JSON."""

    try:
        response = requests.get(f"{BACKEND_URL}{path}", timeout=60)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        if show_error:
            st.error(
                "Could not connect to the backend. Start it with: "
                "uvicorn app.backend.main:app --reload"
            )
        return None
    except requests.exceptions.HTTPError:
        if show_error:
            detail = _extract_error_detail(response)
            st.error(f"Backend request failed: {detail}")
        return None
    except requests.exceptions.RequestException as exc:
        if show_error:
            st.error(f"Backend request failed: {exc}")
        return None

    return response.json()


def _model_comparison_dataframe(
    model_results: dict[str, Any],
    evaluation_summary: dict[str, Any],
) -> pd.DataFrame:
    records = model_results.get("results", [])
    if records:
        rows: list[dict[str, Any]] = []
        for record in records:
            row = {
                "Model": record.get("model_name"),
                "Role": record.get("role"),
                "Status": record.get("status"),
            }
            row.update(record.get("metrics", {}))
            if record.get("primary_metric_value") is not None:
                row["Primary metric value"] = record.get("primary_metric_value")
            if record.get("error"):
                row["Error"] = record.get("error")
            rows.append(row)
        return pd.DataFrame(rows)

    all_metrics = evaluation_summary.get("all_model_metrics", {})
    if not all_metrics:
        return pd.DataFrame()

    rows = []
    for model_name, metrics in all_metrics.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame([{"Metric": "None", "Value": None}])

    return pd.DataFrame(
        [
            {"Metric": metric.upper(), "Value": value}
            for metric, value in metrics.items()
        ]
    )


def _shape_text(shape: list[int]) -> str:
    return f"{shape[0]} x {shape[1]}"


def _plots_by_category(plots: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return [plot for plot in plots if plot.get("category") == category]


def _plot_image_url(run_id: str, plot_path: str) -> str:
    path = plot_path.removeprefix("plots/")
    return f"{BACKEND_URL}/runs/{quote(run_id, safe='')}/plots/{quote(path, safe='/')}"


def _extract_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Unknown backend error."

    return str(payload.get("detail", payload))


if __name__ == "__main__":
    main()
