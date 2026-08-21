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
    parser.add_argument("--offline", action="store_true", help="Use the documented offline fallback.")
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
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
