"""Measure deterministic recommendation overhead on a frozen training profile."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from app.deterministic import deterministic_recommendation
from app.validation import freeze_supervised_split, training_profile_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--question", default="Recommend a model family for the target.")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")

    frame = pd.read_csv(args.data)
    split = freeze_supervised_split(
        frame,
        args.target,
        args.task,
        random_state=args.seed,
    )
    training = training_profile_frame(
        frame,
        args.target,
        args.task,
        test_size=0.2,
        random_state=args.seed,
        split=split,
    )
    started = time.perf_counter()
    recommendation = None
    for _ in range(args.repetitions):
        recommendation = deterministic_recommendation(
            training,
            args.question,
            args.target,
            task_type=args.task,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(
        json.dumps(
            {
                "repetitions": args.repetitions,
                "average_runtime_ms": elapsed_ms / args.repetitions,
                "training_rows": len(training),
                "training_columns": len(training.columns),
                "selected_method": recommendation.recommended_method if recommendation else None,
                "policy_version": recommendation.policy_version if recommendation else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
