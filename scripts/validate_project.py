"""Run portfolio-readiness validation checks for AutoDS Agent."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_DIRECTORIES = [
    ".github/workflows",
    "app",
    "app/backend",
    "app/frontend",
    "app/tools",
    "app/workflows",
    "examples/sample_data",
    "examples/demo_outputs",
    "runs",
    "docs",
    "docs/screenshots",
    "tests",
]

EXPECTED_FILES = [
    "README.md",
    "pyproject.toml",
    "constraints-dev.txt",
    ".env.example",
    ".gitignore",
    ".dockerignore",
    "Dockerfile",
    "docker-compose.yml",
    "LICENSE",
    "CHANGELOG.md",
    "PROJECT_STATUS.md",
    "CONTRIBUTING.md",
    ".github/workflows/tests.yml",
    "scripts/create_demo_run.py",
    "scripts/run_full_demo.py",
    "scripts/smoke_test.py",
    "scripts/validate_project.py",
    "docs/architecture.md",
    "docs/api_reference.md",
    "docs/demo_walkthrough.md",
    "docs/final_demo_script.md",
    "docs/recruiter_walkthrough.md",
    "docs/interview_talking_points.md",
    "docs/resume_bullets.md",
    "docs/project_showcase.md",
    "docs/screenshots/README.md",
    "examples/demo_outputs/README.md",
    "examples/demo_outputs/regression_demo_summary.md",
    "examples/demo_outputs/classification_demo_summary.md",
]

EXPECTED_DATASETS = {
    "examples/sample_data/regression_housing.csv": "sale_price",
    "examples/sample_data/classification_churn.csv": "churn",
}

EXPECTED_MODULES = [
    "app.backend.main",
    "app.backend.config",
    "app.backend.services.run_manager",
    "app.backend.services.dataset_service",
    "app.backend.services.profiling_service",
    "app.backend.services.cleaning_service",
    "app.backend.services.eda_service",
    "app.backend.services.modeling_service",
    "app.backend.services.workflow_service",
    "app.backend.services.report_service",
    "app.frontend.streamlit_app",
]

README_SECTIONS = [
    "## Overview",
    "## Demo",
    "## Why This Project",
    "## Key Features",
    "## Deterministic Agentic Workflow",
    "## Architecture",
    "## Tech Stack",
    "## Quickstart",
    "## Run With Docker",
    "## Try The Example Datasets",
    "## Run The Full Demo",
    "## Generated Artifacts",
    "## MLflow Tracking",
    "## API Overview",
    "## Testing",
    "## Project Structure",
    "## Documentation",
    "## Current Limitations",
    "## Future Work",
    "## Resume Highlights",
]


def main() -> None:
    checks = [
        check_files(),
        check_directories(),
        check_imports(),
        check_config(),
        check_example_datasets(),
        check_readme_sections(),
        check_tests_discoverable(),
    ]

    if all(checks):
        print("Project validation passed.")
        return

    print("Project validation failed.")
    raise SystemExit(1)


def check_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        print(f"[fail] import {module_name}: {exc}")
        return False
    print(f"[ok] import {module_name}")
    return True


def check_imports() -> bool:
    ok = True
    for module_name in EXPECTED_MODULES:
        ok = check_import(module_name) and ok
    return ok


def check_files() -> bool:
    ok = True
    for relative_path in EXPECTED_FILES:
        path = PROJECT_ROOT / relative_path
        if path.exists() and path.is_file():
            print(f"[ok] file {relative_path}")
        else:
            print(f"[fail] file {relative_path}")
            ok = False
    return ok


def check_directories() -> bool:
    ok = True
    for relative_path in EXPECTED_DIRECTORIES:
        path = PROJECT_ROOT / relative_path
        if path.exists() and path.is_dir():
            print(f"[ok] directory {relative_path}")
        else:
            print(f"[fail] directory {relative_path}")
            ok = False
    return ok


def check_config() -> bool:
    from app.backend.config import load_settings

    active_settings = load_settings()
    if not active_settings.runs_dir.is_absolute():
        print("[fail] runs_dir is not absolute")
        return False
    print(f"[ok] config runs_dir={active_settings.runs_dir}")
    print(f"[ok] config mlflow_enabled={active_settings.mlflow_enabled}")
    return True


def check_example_datasets() -> bool:
    ok = True
    for relative_path, target_column in EXPECTED_DATASETS.items():
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            print(f"[fail] dataset missing {relative_path}")
            ok = False
            continue
        dataframe = pd.read_csv(path)
        if target_column not in dataframe.columns:
            print(f"[fail] dataset {relative_path} missing target {target_column}")
            ok = False
        else:
            print(f"[ok] dataset {relative_path} rows={len(dataframe)} target={target_column}")
    return ok


def check_tests_discoverable() -> bool:
    test_files = sorted((PROJECT_ROOT / "tests").glob("test_*.py"))
    if not test_files:
        print("[fail] no tests discovered")
        return False
    print(f"[ok] discovered {len(test_files)} test files")
    return True


def check_readme_sections() -> bool:
    readme_path = PROJECT_ROOT / "README.md"
    if not readme_path.exists():
        print("[fail] README.md missing")
        return False

    content = readme_path.read_text(encoding="utf-8")
    missing = [section for section in README_SECTIONS if section not in content]
    if missing:
        for section in missing:
            print(f"[fail] README missing section {section}")
        return False

    print(f"[ok] README contains {len(README_SECTIONS)} portfolio sections")
    return True


if __name__ == "__main__":
    main()
