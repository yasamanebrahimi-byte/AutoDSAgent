"""Command-line entry point for the post-hoc evaluation harness."""

from __future__ import annotations

import argparse
import json

from evaluation.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the AutoDSAgent validation architecture on local tabular benchmarks."
    )
    parser.add_argument("--output", default="evaluation_results/offline", help="Evaluation output directory.")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--case", action="append", dest="cases", help="Benchmark case name; repeatable.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Use the documented offline fallback.")
    mode.add_argument("--live", action="store_true", help="Request live OpenAI trials (the default when not offline).")
    parser.add_argument("--resume", action="store_true", help="Resume missing trial IDs from an existing compatible output bundle.")
    parser.add_argument(
        "--include-perturbations",
        action="store_true",
        help="Add the small deterministic data-quality scenario suite.",
    )
    args = parser.parse_args()
    result = run_evaluation(
        args.output,
        repetitions=args.repetitions,
        seed=args.seed,
        model=args.model,
        offline=args.offline,
        include_perturbations=args.include_perturbations,
        case_names=args.cases,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
