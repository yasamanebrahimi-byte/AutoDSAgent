"""Export the established public datasets used by the product demos.

The source data ships with scikit-learn, so regenerating the checked-in CSVs is
deterministic and does not require network access.  The manifest records the
exact output hashes and public provenance for review.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sklearn.datasets import load_breast_cancer, load_diabetes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "examples" / "sample_data"


def _column_name(value: object) -> str:
    """Return a stable snake-case CSV column name."""

    name = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return name.strip("_")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_breast_cancer() -> dict[str, Any]:
    """Export the Wisconsin Diagnostic Breast Cancer benchmark."""

    dataset = load_breast_cancer(as_frame=True)
    dataframe = dataset.data.copy()
    dataframe.columns = [_column_name(column) for column in dataframe.columns]
    dataframe["diagnosis"] = dataset.target.map(
        dict(enumerate(str(value) for value in dataset.target_names))
    )

    output_path = OUTPUT_DIR / "breast_cancer_wisconsin.csv"
    dataframe.to_csv(output_path, index=False, lineterminator="\n")
    return {
        "file": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "task": "classification",
        "target": "diagnosis",
        "rows": len(dataframe),
        "features": len(dataframe.columns) - 1,
        "source": "https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic",
        "loader": "sklearn.datasets.load_breast_cancer",
        "license": "CC BY 4.0",
        "citation": (
            "Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). "
            "Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5DW2B"
        ),
        "sha256": _sha256(output_path),
    }


def export_diabetes() -> dict[str, Any]:
    """Export scikit-learn's established diabetes regression benchmark."""

    dataset = load_diabetes(as_frame=True, scaled=False)
    dataframe = dataset.data.copy()
    dataframe.columns = [_column_name(column) for column in dataframe.columns]
    dataframe["disease_progression"] = dataset.target

    output_path = OUTPUT_DIR / "diabetes_progression.csv"
    dataframe.to_csv(output_path, index=False, float_format="%.10g", lineterminator="\n")
    return {
        "file": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "name": "Diabetes disease progression",
        "task": "regression",
        "target": "disease_progression",
        "rows": len(dataframe),
        "features": len(dataframe.columns) - 1,
        "source": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html",
        "loader": "sklearn.datasets.load_diabetes(scaled=False)",
        "license": "Distributed with scikit-learn (BSD-3-Clause)",
        "citation": (
            "Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). "
            "Least angle regression. The Annals of Statistics, 32(2), 407-499."
        ),
        "sha256": _sha256(output_path),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_by": "scripts/export_benchmark_datasets.py",
        "datasets": [export_breast_cancer(), export_diabetes()],
    }
    manifest_path = OUTPUT_DIR / "benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(manifest['datasets'])} benchmarks to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
