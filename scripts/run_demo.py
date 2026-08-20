"""Run the two bundled benchmark demos."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.pipeline import run_analysis


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()
    jobs = [
        (
            ROOT / "examples" / "sample_data" / "breast_cancer_wisconsin.csv",
            "Can we classify diagnosis from the measured cell features?",
            "diagnosis",
        ),
        (
            ROOT / "examples" / "sample_data" / "diabetes_progression.csv",
            "Can we estimate disease progression from the patient measurements?",
            "disease_progression",
        ),
    ]
    for dataset, question, target in jobs:
        result = run_analysis(
            dataset,
            question,
            target_column=target,
            output_dir=args.output_dir,
            offline=args.offline,
        )
        print(result["run_dir"])


if __name__ == "__main__":
    main()

