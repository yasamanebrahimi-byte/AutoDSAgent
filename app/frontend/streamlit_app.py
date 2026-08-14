"""Streamlit frontend for AutoDS Agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv("AUTODS_BACKEND_URL", "http://localhost:8000").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_DATASETS = {
    "Synthetic Customer Churn Classification": {
        "filename": "classification_churn.csv",
        "path": PROJECT_ROOT / "examples" / "sample_data" / "classification_churn.csv",
        "target": "churn",
        "task_type": "Classification",
        "description": "Binary churn prediction with numeric, categorical, boolean, missing-value, duplicate-row, and ID-like columns.",
    },
    "Synthetic Housing Regression": {
        "filename": "regression_housing.csv",
        "path": PROJECT_ROOT / "examples" / "sample_data" / "regression_housing.csv",
        "target": "sale_price",
        "task_type": "Regression",
        "description": "Home-price prediction with mixed feature types and conservative cleaning opportunities.",
    },
}

RUN_STATE_KEYS = [
    "profile",
    "cleaning_plan",
    "cleaning_summary",
    "eda_response",
    "modeling_response",
    "workflow_state",
    "reports_response",
]

TARGET_WIDGET_KEYS = [
    "workflow_target_column",
    "workflow_task_type",
    "modeling_target_column",
    "modeling_task_type",
    "eda_target_column",
]


def main() -> None:
    st.set_page_config(
        page_title="AutoDS Agent",
        page_icon="AD",
        layout="wide",
    )

    st.title("AutoDS Agent: Automated Data Science Workflow")
    st.caption(f"Backend: {BACKEND_URL}")
    render_project_overview()
    render_runtime_status()
    render_start_run_controls()

    metadata = st.session_state.get("metadata")
    if metadata:
        render_metadata(metadata)
        with st.expander("Advanced Manual Controls", expanded=False):
            render_week2_workflow(metadata)
        render_automated_workflow(metadata)
        render_final_reports(metadata)
    else:
        st.info("Create an analysis run from an upload or sample dataset to begin.")


def render_project_overview() -> None:
    """Render the portfolio-friendly project overview."""

    st.write(
        "AutoDS Agent turns a raw tabular CSV into an inspectable analysis run: "
        "metadata, profiling, conservative cleaning, EDA plots, baseline and "
        "candidate models, evaluation outputs, workflow trace logs, and final "
        "Markdown reports."
    )
    overview_cols = st.columns(4)
    overview_cols[0].metric("Workflow", "Profile to Report")
    overview_cols[1].metric("Raw data", "Preserved")
    overview_cols[2].metric("LLM cost", "$0 required")
    overview_cols[3].metric("Tracking", "Optional MLflow")


def render_start_run_controls() -> None:
    """Render upload and bundled sample dataset controls."""

    st.header("Start An Analysis Run")
    st.caption(
        "Use your own CSV or load a bundled sample dataset. Either path creates "
        "a normal run folder and keeps the uploaded raw file unchanged."
    )

    upload_tab, sample_tab = st.tabs(["Upload CSV", "Try A Sample Dataset"])

    with upload_tab:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            if st.button("Create Analysis Run", type="primary"):
                metadata = upload_csv(uploaded_file)
                if metadata:
                    set_active_run(metadata)

    with sample_tab:
        selected_label = st.radio(
            "Sample dataset",
            options=list(SAMPLE_DATASETS),
            horizontal=True,
        )
        selected_sample = SAMPLE_DATASETS[selected_label]
        st.write(selected_sample["description"])
        sample_cols = st.columns(3)
        sample_cols[0].metric("Target", selected_sample["target"])
        sample_cols[1].metric("Task", selected_sample["task_type"])
        sample_cols[2].metric("File", selected_sample["filename"])
        if st.button("Load Sample Dataset", type="primary"):
            metadata = upload_sample_dataset(selected_label)
            if metadata:
                set_active_run(metadata, selected_sample)


def set_active_run(
    metadata: dict[str, Any],
    sample_config: dict[str, Any] | None = None,
) -> None:
    """Store a new active run and clear stale derived artifacts."""

    st.session_state["metadata"] = metadata
    for key in RUN_STATE_KEYS:
        st.session_state.pop(key, None)

    for key in TARGET_WIDGET_KEYS:
        st.session_state.pop(key, None)

    if sample_config is None:
        st.session_state.pop("recommended_target_column", None)
        st.session_state.pop("recommended_task_type", None)
        return

    target = str(sample_config["target"])
    task_type = str(sample_config["task_type"])
    st.session_state["recommended_target_column"] = target
    st.session_state["recommended_task_type"] = task_type
    st.session_state["workflow_target_column"] = target
    st.session_state["workflow_task_type"] = task_type
    st.session_state["modeling_target_column"] = target
    st.session_state["modeling_task_type"] = task_type
    st.session_state["eda_target_column"] = target


def upload_sample_dataset(sample_label: str) -> dict[str, Any] | None:
    """Upload one bundled sample dataset through the regular backend endpoint."""

    sample = SAMPLE_DATASETS[sample_label]
    path = Path(sample["path"])
    if not path.exists():
        st.error(f"Sample dataset was not found: {path}")
        return None
    return upload_csv_bytes(path.name, path.read_bytes())


def ensure_widget_value(key: str, options: list[Any], default_value: Any) -> None:
    """Keep Streamlit selectboxes aligned with the current dataset columns."""

    if st.session_state.get(key) in options:
        return
    st.session_state[key] = default_value if default_value in options else options[0]


def render_runtime_status() -> None:
    """Render non-secret runtime configuration and demo hints."""

    config_status = get_json("/config/status", show_error=False)
    with st.expander("Runtime and Demo Notes", expanded=False):
        if isinstance(config_status, dict):
            status_cols = st.columns(3)
            status_cols[0].metric("Environment", config_status.get("environment", "unknown"))
            status_cols[1].metric(
                "MLflow",
                "enabled" if config_status.get("mlflow_enabled") else "disabled",
            )
            status_cols[2].metric("Runs directory", config_status.get("runs_dir", "runs"))
            if config_status.get("mlflow_enabled"):
                st.write(f"MLflow tracking URI: `{config_status.get('mlflow_tracking_uri')}`")
                st.write("Open the MLflow UI to inspect model parameters, metrics, and artifacts.")
        else:
            st.write("Runtime status is unavailable until the backend is reachable.")

        st.write("Bundled demo datasets are available under `examples/sample_data/`.")
        st.write("Use `classification_churn.csv` with target `churn` or `regression_housing.csv` with target `sale_price`.")


def upload_csv(uploaded_file) -> dict | None:
    """Send the uploaded CSV to the FastAPI backend."""

    return upload_csv_bytes(uploaded_file.name, uploaded_file.getvalue())


def upload_csv_bytes(filename: str, content: bytes) -> dict[str, Any] | None:
    """Send CSV bytes to the FastAPI backend."""

    files = {
        "file": (
            filename,
            content,
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
    recommended_target = st.session_state.get("recommended_target_column")
    recommended_task = st.session_state.get("recommended_task_type")
    if recommended_target in metadata.get("column_names", []):
        st.info(
            f"Sample defaults loaded: target `{recommended_target}`, "
            f"task `{recommended_task}`."
        )

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
        st.caption(
            "Profiling reads the preserved raw CSV and saves schema, missingness, "
            "duplicates, sample values, and quality warnings."
        )
        if st.button("Generate Dataset Profile", key="generate_profile"):
            profile = post_json(
                f"/runs/{run_id}/profile",
                "Generating dataset profile...",
            )
            if profile:
                st.session_state["profile"] = profile
                st.session_state.pop("reports_response", None)

        profile = st.session_state.get("profile")
        if profile:
            render_profile(profile)

    with plan_tab:
        st.caption(
            "The cleaning plan is conservative: it recommends safe actions and "
            "surfaces anything that deserves human review."
        )
        if st.button("Generate Cleaning Plan", key="generate_cleaning_plan"):
            plan = post_json(
                f"/runs/{run_id}/cleaning-plan",
                "Generating cleaning plan...",
            )
            if plan:
                st.session_state["cleaning_plan"] = plan
                st.session_state.pop("reports_response", None)

        plan = st.session_state.get("cleaning_plan")
        if plan:
            render_cleaning_plan(plan)

    with clean_tab:
        st.caption(
            "Safe cleaning writes a new cleaned CSV while leaving the raw upload "
            "unchanged in the run input folder."
        )
        if st.button("Apply Safe Cleaning", key="apply_safe_cleaning"):
            summary = post_json(
                f"/runs/{run_id}/clean",
                "Applying safe cleaning...",
            )
            if summary:
                st.session_state["cleaning_summary"] = summary
                st.session_state.pop("eda_response", None)
                st.session_state.pop("modeling_response", None)
                st.session_state.pop("reports_response", None)

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
        st.subheader("Next: Final report generation")
        st.write(
            "Final reports are generated from the saved workflow artifacts. "
            "The reports remain deterministic and do not use LLM calls."
        )


def render_automated_workflow(metadata: dict[str, Any]) -> None:
    """Render automated workflow controls and state."""

    run_id = metadata["run_id"]
    column_names = list(metadata.get("column_names", []))

    st.divider()
    st.header("Automated Workflow")
    st.caption(
        "Runs deterministic analysis services through an auditable agent workflow. "
        "No LLM API calls or paid credits are used."
    )

    target_options = ["No target column"] + column_names
    default_target = st.session_state.get("recommended_target_column")
    default_target_value = (
        default_target if default_target in column_names else "No target column"
    )
    ensure_widget_value("workflow_target_column", target_options, default_target_value)
    selected_target = st.selectbox(
        "Optional target column",
        options=target_options,
        key="workflow_target_column",
    )
    target_column = None if selected_target == "No target column" else selected_target

    control_cols = st.columns(3)
    with control_cols[0]:
        task_options = ["Auto-detect", "Regression", "Classification"]
        recommended_task = st.session_state.get("recommended_task_type", "Auto-detect")
        ensure_widget_value(
            "workflow_task_type",
            task_options,
            recommended_task if recommended_task in task_options else "Auto-detect",
        )
        task_type_label = st.selectbox(
            "Task type",
            options=task_options,
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
        if st.button("Start Automated Workflow", type="primary"):
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
                "Running automated workflow...",
                payload=payload,
            )
            if state:
                st.session_state["workflow_state"] = state
                st.session_state.pop("reports_response", None)

    with action_cols[1]:
        if st.button("Refresh Workflow"):
            state = get_json(f"/runs/{run_id}/workflow/state")
            if state:
                st.session_state["workflow_state"] = state

    state = st.session_state.get("workflow_state")
    if state:
        render_workflow_state(run_id, state)
    else:
        st.info("Start the automated workflow to profile, clean, analyze, and optionally model this run.")


def render_final_reports(metadata: dict[str, Any]) -> None:
    """Render final report controls, previews, and downloads."""

    run_id = metadata["run_id"]

    st.divider()
    st.header("Final Reports")
    st.caption(
        "Generate deterministic Markdown reports from the artifacts saved for this run. "
        "No LLM API calls or paid credits are used."
    )

    control_cols = st.columns([1, 1, 2])
    with control_cols[0]:
        include_html = st.checkbox(
            "Create simple HTML export",
            value=False,
            key="reports_include_html",
        )
    with control_cols[1]:
        force_regenerate = st.checkbox(
            "Regenerate reports",
            value=True,
            key="reports_force_regenerate",
        )

    action_cols = st.columns([1, 1, 4])
    with action_cols[0]:
        if st.button("Generate Reports", type="primary", key="generate_reports"):
            response = post_json(
                f"/runs/{run_id}/reports/generate",
                "Generating final reports...",
                payload={
                    "include_html": bool(include_html),
                    "force_regenerate": bool(force_regenerate),
                },
            )
            if response:
                st.session_state["reports_response"] = response

    with action_cols[1]:
        if st.button("Refresh Reports", key="refresh_reports"):
            response = get_json(f"/runs/{run_id}/reports", show_error=False)
            if response:
                st.session_state["reports_response"] = response
            else:
                st.info("Reports have not been generated for this run yet.")

    reports_response = st.session_state.get("reports_response")
    if not reports_response:
        reports_response = get_json(f"/runs/{run_id}/reports", show_error=False)
        if reports_response:
            st.session_state["reports_response"] = reports_response

    if not reports_response:
        st.info("Generate reports after profiling, cleaning, EDA, or modeling artifacts are available.")
        return

    render_reports_response(run_id, reports_response)


def render_reports_response(run_id: str, response: dict[str, Any]) -> None:
    """Render report metadata, index, previews, and downloads."""

    metadata = response.get("metadata", {})
    index = response.get("index", {})

    render_report_metadata_summary(metadata)
    render_report_index(index)

    report_tabs = st.tabs(
        [
            "Executive Summary",
            "Final Report",
            "Technical Summary",
            "Limitations",
            "Downloads",
        ]
    )
    preview_specs = [
        ("executive_summary", "Executive Summary"),
        ("final_report", "Final Report"),
        ("technical_summary", "Technical Summary"),
        ("limitations", "Limitations"),
    ]
    for tab, (report_name, label) in zip(report_tabs[:4], preview_specs):
        with tab:
            content = get_report_content(run_id, report_name)
            if content:
                st.markdown(content)
            else:
                st.info(f"{label} is not available yet.")

    with report_tabs[4]:
        render_report_downloads(run_id, preview_specs)


def render_report_metadata_summary(metadata: dict[str, Any]) -> None:
    """Render compact report metadata."""

    status = metadata.get("report_status", "unknown")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Report status", status)
    metric_cols[1].metric("Reports", len(metadata.get("reports_generated", [])))
    metric_cols[2].metric("Sources used", len(metadata.get("source_artifacts_used", [])))
    metric_cols[3].metric("Sources missing", len(metadata.get("source_artifacts_missing", [])))

    if status == "partial":
        st.warning("This is a partial report. Missing or skipped sections are listed below.")
    elif status == "completed":
        st.success("Reports were generated from the available analysis artifacts.")

    detail_tabs = st.tabs(["Sources", "Sections", "Warnings"])
    with detail_tabs[0]:
        source_cols = st.columns(2)
        with source_cols[0]:
            st.subheader("Used")
            st.write(_bullet_text(metadata.get("source_artifacts_used", [])))
        with source_cols[1]:
            st.subheader("Missing")
            st.write(_bullet_text(metadata.get("source_artifacts_missing", [])))

    with detail_tabs[1]:
        section_cols = st.columns(2)
        with section_cols[0]:
            st.subheader("Generated")
            st.write(_bullet_text(metadata.get("sections_generated", [])))
        with section_cols[1]:
            st.subheader("Skipped")
            st.write(_bullet_text(metadata.get("sections_skipped", [])))

    with detail_tabs[2]:
        warnings = metadata.get("warnings", [])
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.write("No report warnings were saved.")


def render_report_index(index: dict[str, Any]) -> None:
    """Render generated report index."""

    reports = index.get("reports", [])
    with st.expander("Report Index", expanded=False):
        if reports:
            st.dataframe(pd.DataFrame(reports), use_container_width=True, hide_index=True)
        else:
            st.write("No report files are listed yet.")


def render_report_downloads(
    run_id: str,
    report_specs: list[tuple[str, str]],
) -> None:
    """Render Markdown download buttons for generated reports."""

    st.subheader("Download Markdown Reports")
    for report_name, label in report_specs:
        content = get_report_content(run_id, report_name)
        if not content:
            st.write(f"{label}: unavailable")
            continue
        st.download_button(
            label=f"Download {label}",
            data=content,
            file_name=f"{report_name}.md",
            mime="text/markdown",
            key=f"download_{run_id}_{report_name}",
        )


def get_report_content(run_id: str, report_name: str) -> str | None:
    """Load one report's Markdown content."""

    response = get_json(
        f"/runs/{run_id}/reports/{report_name}",
        show_error=False,
    )
    if not isinstance(response, dict):
        return None
    return response.get("content")


def render_workflow_state(run_id: str, state: dict[str, Any]) -> None:
    """Render workflow progress, approval gates, retry controls, artifacts, and trace."""

    status_cols = st.columns(3)
    status_cols[0].metric("Workflow status", state.get("status", "unknown"))
    status_cols[1].metric("Current step", state.get("current_step") or "None")
    status_cols[2].metric("Target", state.get("target_column") or "None")
    st.caption(
        "Workflow state is saved after each transition, so the run remains "
        "auditable even after the UI session ends."
    )

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
                    _metrics_dataframe(_final_test_metrics(evaluation_summary)),
                    use_container_width=True,
                    hide_index=True,
                )

    if artifacts.get("final_report") or artifacts.get("report_index"):
        with st.expander("Final Reports", expanded=False):
            reports_response = get_json(f"/runs/{run_id}/reports", show_error=False)
            if reports_response:
                render_report_metadata_summary(reports_response.get("metadata", {}))
                render_report_index(reports_response.get("index", {}))
            else:
                st.write("Report metadata is not available yet.")


def render_workflow_trace(run_id: str) -> None:
    """Render ordered agent trace events."""

    with st.expander("Agent Trace", expanded=False):
        st.caption(
            "Trace events show which agent boundary started, completed, paused, "
            "retried, skipped, or failed each step."
        )
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
    """Render EDA controls and outputs."""

    st.subheader("Exploratory Data Analysis")
    st.caption(
        "EDA uses the workflow's current analysis input and falls back to the raw upload with a warning."
    )

    target_options = ["No target column"] + list(column_names)
    default_target = st.session_state.get("recommended_target_column")
    default_target_value = (
        default_target if default_target in column_names else "No target column"
    )
    ensure_widget_value("eda_target_column", target_options, default_target_value)
    selected_target = st.selectbox(
        "Optional target column",
        options=target_options,
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
            st.session_state.pop("reports_response", None)

    eda_response = st.session_state.get("eda_response")
    if eda_response:
        render_eda_response(run_id, eda_response)


def render_modeling_workflow(run_id: str, column_names: list[str]) -> None:
    """Render modeling controls and outputs."""

    st.subheader("Modeling and Evaluation")
    st.caption(
        "Modeling requires current approved cleaned data, trains deterministic sklearn models, "
        "and does not use LLM API calls."
    )

    if not column_names:
        st.info("Upload a dataset before selecting a modeling target.")
        return

    default_target = st.session_state.get("recommended_target_column")
    target_options = ["Select a target"] + column_names
    ensure_widget_value(
        "modeling_target_column",
        target_options,
        default_target if default_target in column_names else "Select a target",
    )
    selected_target = st.selectbox(
        "Target column",
        options=target_options,
        key="modeling_target_column",
    )
    target_column = None if selected_target == "Select a target" else selected_target
    task_options = ["Auto-detect", "Regression", "Classification"]
    recommended_task = st.session_state.get("recommended_task_type", "Auto-detect")
    ensure_widget_value(
        "modeling_task_type",
        task_options,
        recommended_task if recommended_task in task_options else "Auto-detect",
    )
    task_type_label = st.selectbox(
        "Task type",
        options=task_options,
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

    if target_column is None:
        st.info("Select a target before modeling.")

    if st.button(
        "Train and Evaluate Models",
        type="primary",
        key="train_models",
        disabled=target_column is None,
    ):
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
            st.session_state.pop("reports_response", None)

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

    st.subheader("CV Model Selection Metrics")
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
        st.subheader("Final Test Metrics")
        st.dataframe(
            _metrics_dataframe(_final_test_metrics(evaluation_summary)),
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
            row.update(record.get("cv_metrics") or record.get("metrics", {}))
            if record.get("primary_metric_value") is not None:
                row["Selection metric value"] = record.get("primary_metric_value")
            if record.get("error"):
                row["Error"] = record.get("error")
            rows.append(row)
        return pd.DataFrame(rows)

    all_metrics = (
        evaluation_summary.get("cv_model_metrics")
        or evaluation_summary.get("candidate_cv_results")
        or evaluation_summary.get("all_model_metrics")
        or {}
    )
    if not all_metrics:
        return pd.DataFrame()

    rows = []
    for model_name, metrics in all_metrics.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _final_test_metrics(evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    return (
        evaluation_summary.get("final_test_metrics")
        or evaluation_summary.get("holdout_metrics")
        or evaluation_summary.get("best_model_metrics")
        or {}
    )


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


def _bullet_text(values: list[Any]) -> str:
    if not values:
        return "None"
    return "\n".join(f"- `{value}`" for value in values)


def _extract_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Unknown backend error."

    return str(payload.get("detail", payload))


if __name__ == "__main__":
    main()
