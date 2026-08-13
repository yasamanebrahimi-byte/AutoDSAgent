from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_example_regression_dataset_exists_and_loads():
    path = PROJECT_ROOT / "examples" / "sample_data" / "regression_housing.csv"

    dataframe = pd.read_csv(path)

    assert path.exists()
    assert "sale_price" in dataframe.columns
    assert "home_id" in dataframe.columns
    assert dataframe["sale_price"].notna().all()
    assert dataframe.duplicated().sum() >= 1


def test_example_classification_dataset_exists_and_loads():
    path = PROJECT_ROOT / "examples" / "sample_data" / "classification_churn.csv"

    dataframe = pd.read_csv(path)

    assert path.exists()
    assert "churn" in dataframe.columns
    assert "customer_id" in dataframe.columns
    assert set(dataframe["churn"].dropna().unique()) == {"yes", "no"}
    assert dataframe.duplicated().sum() >= 1


def test_validation_script_runs_successfully():
    result = subprocess.run(
        [sys.executable, "scripts/validate_project.py"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Project validation passed." in result.stdout
