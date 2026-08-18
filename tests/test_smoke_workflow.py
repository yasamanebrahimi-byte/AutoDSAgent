from __future__ import annotations

import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_key_modules_import():
    modules = [
        "app.backend.main",
        "app.backend.config",
        "app.backend.services.workflow_service",
        "app.backend.services.report_service",
        "app.frontend.streamlit_app",
        "scripts.run_full_demo",
        "scripts.smoke_test",
    ]

    for module_name in modules:
        importlib.import_module(module_name)


def test_portfolio_files_exist():
    expected_files = [
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "PROJECT_STATUS.md",
        "CONTRIBUTING.md",
        ".github/workflows/tests.yml",
        "docs/final_demo_script.md",
        "docs/recruiter_walkthrough.md",
        "docs/interview_talking_points.md",
        "docs/resume_bullets.md",
        "docs/project_showcase.md",
        "docs/screenshots/README.md",
        "examples/demo_outputs/README.md",
        "examples/demo_outputs/regression_demo_summary.md",
        "examples/demo_outputs/classification_demo_summary.md",
        "scripts/run_full_demo.py",
        "scripts/smoke_test.py",
    ]

    for relative_path in expected_files:
        assert (PROJECT_ROOT / relative_path).exists(), relative_path


def test_readme_contains_week_8_sections():
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    sections = [
        "## Overview",
        "## Demo",
        "## Quickstart",
        "## Run The Full Demo",
        "## Generated Artifacts",
        "## Documentation",
        "## Resume Highlights",
    ]

    for section in sections:
        assert section in content


def test_example_dataset_targets_exist():
    regression = PROJECT_ROOT / "examples" / "sample_data" / "diabetes_progression.csv"
    classification = PROJECT_ROOT / "examples" / "sample_data" / "breast_cancer_wisconsin.csv"

    assert regression.exists()
    assert classification.exists()
    assert "disease_progression" in regression.read_text(encoding="utf-8").splitlines()[0]
    assert "diagnosis" in classification.read_text(encoding="utf-8").splitlines()[0]
