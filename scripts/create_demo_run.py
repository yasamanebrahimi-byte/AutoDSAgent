"""Create a local demo run from the bundled example datasets."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.schemas.eda import EDARequest
from app.backend.schemas.modeling import ModelingRequest
from app.backend.schemas.reports import ReportGenerateRequest
from app.backend.services.cleaning_service import CleaningService
from app.backend.services.dataset_service import generate_dataset_metadata, load_csv
from app.backend.services.eda_service import EDAService
from app.backend.services.modeling_service import ModelingService
from app.backend.services.profiling_service import ProfilingService
from app.backend.services.report_service import ReportService
from app.backend.services.run_manager import RunManager
from app.tools.app_logging import configure_logging
from app.tools.trace_logger import TraceLogger
from app.workflows.workflow_state import (
    create_initial_workflow_state,
    relative_to_run,
    save_workflow_state,
    state_path_for_logs_dir,
)
from app.workflows.workflow_steps import CANONICAL_WORKFLOW_STEPS


DATASETS = {
    "regression": {
        "path": PROJECT_ROOT / "examples" / "sample_data" / "regression_housing.csv",
        "target": "sale_price",
        "task_type": "regression",
    },
    "classification": {
        "path": PROJECT_ROOT / "examples" / "sample_data" / "classification_churn.csv",
        "target": "churn",
        "task_type": "classification",
    },
}


def main() -> None:
    args = parse_args()
    configure_logging()

    dataset_info = DATASETS[args.dataset]
    dataset_path = Path(args.path or dataset_info["path"]).resolve()
    target_column = args.target or str(dataset_info["target"])
    task_type = None if args.task_type == "auto" else args.task_type

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    manager = RunManager()
    paths = manager.create_run()
    raw_path = paths.input / "raw_data.csv"
    shutil.copyfile(dataset_path, raw_path)

    dataframe = load_csv(raw_path)
    metadata = generate_dataset_metadata(
        dataframe=dataframe,
        filename=dataset_path.name,
        run_id=paths.root.name,
    )
    manager.save_metadata(paths.root.name, metadata.model_dump(mode="json"))

    profiling_service = ProfilingService(manager)
    cleaning_service = CleaningService(manager, profiling_service)
    eda_service = EDAService(manager)
    modeling_service = ModelingService(manager)
    report_service = ReportService(manager)

    profile = profiling_service.generate_profile(paths.root.name)
    cleaning_plan = cleaning_service.generate_cleaning_plan(paths.root.name)
    cleaning_summary = cleaning_service.apply_cleaning(paths.root.name)
    eda_response = eda_service.generate_eda(
        paths.root.name,
        EDARequest(target_column=target_column),
    )

    modeling_response = None
    modeling_error = None
    if target_column:
        try:
            modeling_response = modeling_service.train_and_evaluate(
                paths.root.name,
                ModelingRequest(
                    target_column=target_column,
                    task_type=task_type,
                    random_state=args.random_state,
                    test_size=args.test_size,
                ),
            )
        except Exception as exc:
            modeling_error = str(exc)

    write_demo_workflow_logs(
        manager=manager,
        run_id=paths.root.name,
        target_column=target_column,
        task_type=(
            modeling_response.modeling_summary.task_type
            if modeling_response
            else task_type
        ),
        modeling_error=modeling_error,
    )

    report_response = report_service.generate_reports(
        paths.root.name,
        ReportGenerateRequest(include_html=args.include_html, force_regenerate=True),
    )

    print_summary(
        run_id=paths.root.name,
        artifacts={
            "raw_data": raw_path,
            "profile": profiling_service.profile_path(paths.root.name),
            "cleaning_plan": cleaning_service.cleaning_plan_path(paths.root.name),
            "cleaned_data": cleaning_service.cleaned_data_path(paths.root.name),
            "cleaning_summary": cleaning_service.cleaning_summary_path(paths.root.name),
            "eda_summary": eda_service.eda_summary_path(paths.root.name),
            "eda_findings": eda_service.eda_findings_path(paths.root.name),
            "final_report": report_service.report_path(paths.root.name, "final_report"),
            "executive_summary": report_service.report_path(paths.root.name, "executive_summary"),
            "technical_summary": report_service.report_path(paths.root.name, "technical_summary"),
            "limitations": report_service.report_path(paths.root.name, "limitations"),
            "report_metadata": report_service.report_metadata_path(paths.root.name),
            "report_index": report_service.report_index_path(paths.root.name),
        },
        best_model_name=(
            modeling_response.modeling_summary.best_model_name
            if modeling_response
            else None
        ),
        modeling_error=modeling_error,
        report_status=report_response.metadata.report_status,
        profile_warnings=len(profile.data_quality_issues),
        cleaning_actions=len(cleaning_plan.missing_value_strategies),
        plots=len(eda_response.summary.generated_plots),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an AutoDS Agent demo run.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="classification",
        help="Bundled example dataset to use.",
    )
    parser.add_argument("--path", help="Optional custom CSV path.")
    parser.add_argument("--target", help="Target column. Defaults to the dataset target.")
    parser.add_argument(
        "--task-type",
        choices=["auto", "regression", "classification"],
        default="auto",
        help="Modeling task type override.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-html", action="store_true")
    return parser.parse_args()


def print_summary(
    run_id: str,
    artifacts: dict[str, Path],
    best_model_name: str | None,
    modeling_error: str | None,
    report_status: str,
    profile_warnings: int,
    cleaning_actions: int,
    plots: int,
) -> None:
    print(f"Created demo run: {run_id}")
    print(f"Profile warnings: {profile_warnings}")
    print(f"Cleaning strategies planned: {cleaning_actions}")
    print(f"EDA plots generated: {plots}")
    print(f"Report status: {report_status}")
    if best_model_name:
        print(f"Best model: {best_model_name}")
    if modeling_error:
        print(f"Modeling skipped or failed: {modeling_error}")
    print("Artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {_display_path(path)}")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def write_demo_workflow_logs(
    manager: RunManager,
    run_id: str,
    target_column: str | None,
    task_type: str | None,
    modeling_error: str | None,
) -> None:
    """Write lightweight workflow artifacts for direct-service demo runs."""

    paths = manager.get_paths(run_id)
    state = create_initial_workflow_state(
        run_id=run_id,
        target_column=target_column,
        task_type=task_type,
        require_cleaning_approval=False,
        require_modeling_approval=False,
    )
    state["status"] = "completed"
    state["current_step"] = None
    artifact_paths = {
        "metadata": paths.intermediate / "metadata.json",
        "profile": paths.intermediate / "profile.json",
        "cleaning_plan": paths.intermediate / "cleaning_plan.json",
        "cleaned_data": paths.intermediate / "cleaned_data.csv",
        "cleaning_summary": paths.intermediate / "cleaning_summary.json",
        "eda_summary": paths.intermediate / "eda_summary.json",
        "eda_findings": paths.intermediate / "eda_findings.json",
        "eda_report": paths.reports / "eda_summary.md",
        "modeling_summary": paths.intermediate / "modeling_summary.json",
        "evaluation_summary": paths.intermediate / "evaluation_summary.json",
        "model_results": paths.models / "model_results.json",
        "baseline_model": paths.models / "baseline_model.pkl",
        "best_model": paths.models / "best_model.pkl",
        "final_report": paths.reports / "final_report.md",
        "executive_summary": paths.reports / "executive_summary.md",
        "technical_summary": paths.reports / "technical_summary.md",
        "limitations_report": paths.reports / "limitations.md",
        "report_metadata": paths.intermediate / "report_metadata.json",
        "report_index": paths.reports / "report_index.json",
    }
    for key, path in artifact_paths.items():
        state["artifacts"][key] = (
            relative_to_run(path, paths.root)
            if path.exists() or key.startswith("report") or key in {"final_report", "executive_summary", "technical_summary", "limitations_report"}
            else None
        )

    for step_name in CANONICAL_WORKFLOW_STEPS:
        step_state = state["steps"][step_name]
        step_state["attempts"] = 1
        step_state["started_at"] = state["created_at"]
        step_state["completed_at"] = state["updated_at"]
        step_state["approval_status"] = "not_required"
        step_state["requires_approval"] = False
        if step_name == "modeling" and modeling_error:
            step_state["status"] = "failed"
            step_state["error"] = modeling_error
            state["errors"].append({"step": "modeling", "message": modeling_error})
        else:
            step_state["status"] = "completed"

    save_workflow_state(state_path_for_logs_dir(paths.logs), state)

    trace_logger = TraceLogger(manager)
    trace_logger.reset(run_id)
    trace_logger.append_event(
        run_id=run_id,
        agent="DemoScript",
        step=None,
        event_type="workflow_started",
        message="Demo run created with direct service calls.",
    )
    for step_name in CANONICAL_WORKFLOW_STEPS:
        trace_logger.append_event(
            run_id=run_id,
            agent="DemoScript",
            step=step_name,
            event_type="step_completed" if not (step_name == "modeling" and modeling_error) else "step_failed",
            message=f"Demo step '{step_name}' recorded.",
        )
    trace_logger.append_event(
        run_id=run_id,
        agent="DemoScript",
        step=None,
        event_type="workflow_completed",
        message="Demo run completed.",
    )


if __name__ == "__main__":
    main()
