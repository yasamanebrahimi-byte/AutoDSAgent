"""Run a complete local AutoDS Agent demo from a bundled dataset.

This script uses internal backend services instead of HTTP calls. That keeps the
demo repeatable in CI and interviews without requiring the FastAPI server to be
running. The raw example CSV is still copied into a normal run folder, and the
automated workflow service executes the same profiling, cleaning, EDA,
modeling, evaluation, trace, and report-generation steps used by the API.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.version_info < (3, 11):
    raise SystemExit(
        "AutoDS Agent requires Python 3.11 or newer. "
        f"Current interpreter: Python {sys.version.split()[0]} at {sys.executable}. "
        "Activate the project .venv or run .\\.venv\\Scripts\\python.exe "
        "scripts\\run_full_demo.py --dataset classification."
    )

from app.backend.schemas.reports import ReportGenerateRequest
from app.backend.schemas.workflow import WorkflowStartRequest
from app.backend.services.dataset_service import generate_dataset_metadata, load_csv
from app.backend.services.report_service import ReportService
from app.backend.services.run_manager import RunManager
from app.backend.services.workflow_service import WorkflowService
from app.tools.app_logging import configure_logging


@dataclass(frozen=True)
class DemoDataset:
    """Configuration for one bundled demo dataset."""

    filename: str
    target: str
    task_type: str
    description: str

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / "examples" / "sample_data" / self.filename


@dataclass(frozen=True)
class DemoRunResult:
    """Structured output returned by the demo runner for tests and scripts."""

    run_id: str
    run_root: Path
    dataset_key: str
    target_column: str
    task_type: str | None
    workflow_status: str
    selected_model_name: str | None
    best_candidate_name: str | None
    primary_metric: str | None
    primary_metric_value: float | None
    final_report_path: Path
    artifacts: dict[str, Path]

    @property
    def best_model_name(self) -> str | None:
        """Backward-compatible alias for the selected model name."""

        return self.selected_model_name


DATASETS: dict[str, DemoDataset] = {
    "regression": DemoDataset(
        filename="diabetes_progression.csv",
        target="disease_progression",
        task_type="regression",
        description="Diabetes disease-progression benchmark",
    ),
    "classification": DemoDataset(
        filename="breast_cancer_wisconsin.csv",
        target="diagnosis",
        task_type="classification",
        description="Breast Cancer Wisconsin diagnostic benchmark",
    ),
}


ARTIFACTS: dict[str, tuple[str, str]] = {
    "raw_data": ("input", "raw_data.csv"),
    "metadata": ("intermediate", "metadata.json"),
    "profile": ("intermediate", "profile.json"),
    "cleaning_plan": ("intermediate", "cleaning_plan.json"),
    "cleaned_data": ("intermediate", "cleaned_data.csv"),
    "cleaning_summary": ("intermediate", "cleaning_summary.json"),
    "eda_summary": ("intermediate", "eda_summary.json"),
    "eda_findings": ("intermediate", "eda_findings.json"),
    "modeling_summary": ("intermediate", "modeling_summary.json"),
    "evaluation_summary": ("intermediate", "evaluation_summary.json"),
    "report_metadata": ("intermediate", "report_metadata.json"),
    "model_results": ("models", "model_results.json"),
    "baseline_model": ("models", "baseline_model.pkl"),
    "selected_model": ("models", "selected_model.pkl"),
    "best_model": ("models", "best_model.pkl"),
    "eda_report": ("reports", "eda_summary.md"),
    "final_report": ("reports", "final_report.md"),
    "executive_summary": ("reports", "executive_summary.md"),
    "technical_summary": ("reports", "technical_summary.md"),
    "limitations": ("reports", "limitations.md"),
    "report_index": ("reports", "report_index.json"),
    "workflow_state": ("logs", "workflow_state.json"),
    "agent_trace": ("logs", "agent_trace.json"),
}


def run_demo(
    dataset_key: str,
    runs_dir: str | Path | None = None,
    target_column: str | None = None,
    task_type: str | None = None,
    include_html: bool = False,
) -> DemoRunResult:
    """Create a run and execute the full automated workflow."""

    if dataset_key not in DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_key}'.")

    dataset = DATASETS[dataset_key]
    dataset_path = dataset.path
    selected_target = target_column or dataset.target
    selected_task_type = task_type if task_type != "auto" else None

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    manager = RunManager(runs_dir=runs_dir)
    paths = manager.create_run()
    raw_path = paths.input / "raw_data.csv"
    shutil.copyfile(dataset_path, raw_path)

    dataframe = load_csv(raw_path)
    metadata = generate_dataset_metadata(
        dataframe=dataframe,
        filename=dataset.filename,
        run_id=paths.root.name,
    )
    manager.save_metadata(paths.root.name, metadata.model_dump(mode="json"))

    workflow_service = WorkflowService(run_manager=manager)
    state = workflow_service.start_workflow(
        paths.root.name,
        WorkflowStartRequest(
            target_column=selected_target,
            task_type=selected_task_type,
            require_cleaning_approval=False,
            require_modeling_approval=False,
        ),
    )

    if include_html:
        ReportService(manager).generate_reports(
            paths.root.name,
            ReportGenerateRequest(include_html=True, force_regenerate=True),
        )

    artifacts = _artifact_paths(paths)
    modeling_summary = _load_json_if_exists(artifacts["modeling_summary"])
    evaluation_summary = _load_json_if_exists(artifacts["evaluation_summary"])
    primary_metric = evaluation_summary.get("primary_metric") or modeling_summary.get(
        "primary_metric"
    )
    primary_metric_value = None
    if primary_metric:
        final_test_metrics = (
            evaluation_summary.get("final_test_metrics")
            or evaluation_summary.get("holdout_metrics")
            or evaluation_summary.get("selected_model_holdout_metrics")
            or evaluation_summary.get("best_model_metrics", {})
        )
        value = final_test_metrics.get(primary_metric)
        if value is not None:
            primary_metric_value = float(value)

    return DemoRunResult(
        run_id=paths.root.name,
        run_root=paths.root,
        dataset_key=dataset_key,
        target_column=selected_target,
        task_type=state.get("task_type") or selected_task_type,
        workflow_status=str(state.get("status")),
        selected_model_name=(
            modeling_summary.get("selected_model_name")
            or modeling_summary.get("best_model_name")
        ),
        best_candidate_name=modeling_summary.get("best_candidate_name"),
        primary_metric=primary_metric,
        primary_metric_value=primary_metric_value,
        final_report_path=artifacts["final_report"],
        artifacts=artifacts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a complete AutoDS Agent demo.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        required=True,
        help="Bundled dataset to run: regression or classification.",
    )
    parser.add_argument(
        "--target",
        help="Optional target-column override. Defaults to the sample dataset target.",
    )
    parser.add_argument(
        "--task-type",
        choices=["auto", "regression", "classification"],
        default="auto",
        help="Optional task type. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--runs-dir",
        help="Optional output directory for run artifacts. Defaults to AUTODS_RUNS_DIR or runs/.",
    )
    parser.add_argument(
        "--include-html",
        action="store_true",
        help="Also generate reports/final_report.html.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    result = run_demo(
        dataset_key=args.dataset,
        runs_dir=args.runs_dir,
        target_column=args.target,
        task_type=args.task_type,
        include_html=args.include_html,
    )
    print_summary(result)

    if result.workflow_status != "completed":
        raise SystemExit(1)


def print_summary(result: DemoRunResult) -> None:
    """Print a concise, recruiter-demo-friendly summary."""

    dataset = DATASETS[result.dataset_key]
    print("AutoDS Agent full demo completed.")
    print(f"Run ID: {result.run_id}")
    print(f"Dataset: {dataset.filename} ({dataset.description})")
    print(f"Target column: {result.target_column}")
    print(f"Workflow status: {result.workflow_status}")
    if result.task_type:
        print(f"Task type: {result.task_type}")
    if result.selected_model_name and result.primary_metric:
        metric_text = result.primary_metric.upper()
        if result.primary_metric_value is None:
            print(f"Selected model: {result.selected_model_name} ({metric_text})")
        else:
            print(
                f"Selected model: {result.selected_model_name} "
                f"({metric_text}={result.primary_metric_value:.4f})"
            )
    if result.best_candidate_name and result.best_candidate_name != result.selected_model_name:
        print(f"Best candidate: {result.best_candidate_name}")
    print(f"Final report: {_display_path(result.final_report_path)}")
    print("Key artifacts:")
    for name, path in result.artifacts.items():
        status = "ok" if path.exists() else "missing"
        print(f"  [{status}] {name}: {_display_path(path)}")


def _artifact_paths(paths: Any) -> dict[str, Path]:
    by_name = paths.as_dict()
    return {
        name: by_name[directory_name] / filename
        for name, (directory_name, filename) in ARTIFACTS.items()
    }


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
