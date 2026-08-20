"""Command-line entry point for AutoDS Agent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an auditable agent-vs-deterministic tabular analysis."
    )
    parser.add_argument("--data", required=True, help="Path to a CSV dataset.")
    parser.add_argument("--question", required=True, help="The data science question to answer.")
    parser.add_argument(
        "--target",
        default=None,
        help="Optional target column; otherwise the workflow infers one.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Directory where the run folder is written.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model for agent calls.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip API calls and use documented local fallbacks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_analysis(
        dataset_path=Path(args.data),
        question=args.question,
        target_column=args.target,
        output_dir=args.output_dir,
        model=args.model,
        offline=args.offline,
        random_state=args.seed,
    )
    print(json.dumps(result, indent=2, default=str))
    print(f"\nReport: {result['run_dir']}\\report.md")


if __name__ == "__main__":
    main()

