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
        "--provider",
        choices=("openai", "google"),
        default=os.getenv("AUTODS_PROVIDER", "openai"),
        help="LLM provider for agent calls.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model for agent calls; defaults to the provider-specific environment or model.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip API calls and use documented local fallbacks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    default_model = (
        os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        if args.provider == "openai"
        else os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    )
    result = run_analysis(
        dataset_path=Path(args.data),
        question=args.question,
        target_column=args.target,
        output_dir=args.output_dir,
        model=args.model or default_model,
        provider=args.provider,
        offline=args.offline,
        random_state=args.seed,
    )
    print(json.dumps(result, indent=2, default=str))
    print(f"\nReport: {result['run_dir']}\\report.md")


if __name__ == "__main__":
    main()
