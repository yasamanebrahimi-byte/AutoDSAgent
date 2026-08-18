from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_example_regression_benchmark_exists_and_loads():
    path = PROJECT_ROOT / "examples" / "sample_data" / "diabetes_progression.csv"

    dataframe = pd.read_csv(path)

    assert path.exists()
    assert dataframe.shape == (442, 11)
    assert "disease_progression" in dataframe.columns
    assert dataframe["disease_progression"].notna().all()
    assert dataframe.duplicated().sum() == 0


def test_example_classification_benchmark_exists_and_loads():
    path = PROJECT_ROOT / "examples" / "sample_data" / "breast_cancer_wisconsin.csv"

    dataframe = pd.read_csv(path)

    assert path.exists()
    assert dataframe.shape == (569, 31)
    assert "diagnosis" in dataframe.columns
    assert set(dataframe["diagnosis"].unique()) == {"malignant", "benign"}
    assert dataframe.duplicated().sum() == 0


def test_benchmark_manifest_matches_checked_in_files():
    manifest_path = PROJECT_ROOT / "examples" / "sample_data" / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["generated_by"] == "scripts/export_benchmark_datasets.py"
    assert {dataset["rows"] for dataset in manifest["datasets"]} == {442, 569}
    for dataset in manifest["datasets"]:
        path = PROJECT_ROOT / dataset["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == dataset["sha256"]


def test_tiny_synthetic_files_are_test_fixtures_only():
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "sample_data"
    churn = pd.read_csv(fixture_dir / "classification_churn.csv")
    housing = pd.read_csv(fixture_dir / "regression_housing.csv")

    assert len(churn) == 25
    assert len(housing) == 25
    assert not (PROJECT_ROOT / "examples" / "sample_data" / "classification_churn.csv").exists()
    assert not (PROJECT_ROOT / "examples" / "sample_data" / "regression_housing.csv").exists()


def test_validation_script_runs_successfully():
    result = subprocess.run(
        [sys.executable, "scripts/validate_project.py"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Project validation passed." in result.stdout
