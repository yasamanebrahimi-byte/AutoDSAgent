"""Command-line entry point for the post-hoc evaluation harness."""

from __future__ import annotations

import argparse
import json

from evaluation.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the AutoDSAgent validation architecture on local tabular benchmarks."
    )
    parser.add_argument(
        "--suite",
        choices=("local", "external"),
        default="local",
        help="Benchmark suite to evaluate; local development is the default. External is frozen confirmatory infrastructure and is not a routine live smoke target.",
    )
    parser.add_argument(
        "--tier",
        choices=("core", "stress"),
        help="Optional external-suite tier filter.",
    )
    parser.add_argument("--output", default="evaluation_results/offline", help="Evaluation output directory.")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", action="append", dest="split_seeds", type=int)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--planner-model",
        help="Model used for the initial modeling proposal; defaults to --model.",
    )
    parser.add_argument(
        "--reconciler-model",
        help="Model used for modeling reconciliation; defaults to --model.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=("llm_only", "hard_validation_only", "deterministic_only", "always_reconcile", "selective", "probe_direct", "full", "probe_first"),
        default="full",
        help="Select the current evaluation decision path.",
    )
    parser.add_argument(
        "--order-swap",
        action="store_true",
        help="Run paired A/B and B/A reconciliation trials on identical evidence.",
    )
    parser.add_argument(
        "--no-empirical-probe",
        action="store_true",
        help="Disable the challenged-disagreement training-only probe for the no-probe ablation.",
    )
    parser.add_argument(
        "--soft-challenge-strategy",
        choices=("calibrated", "high_confidence_only"),
        default="calibrated",
    )
    parser.add_argument("--disable-interaction-diagnostics", action="store_true")
    parser.add_argument("--disable-boundary-diagnostics", action="store_true")
    parser.add_argument("--case", action="append", dest="cases", help="Benchmark case name; repeatable.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Use the documented offline fallback.")
    mode.add_argument("--live", action="store_true", help="Request live OpenAI trials; use local/synthetic cases for development smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Resume missing trial IDs from an existing compatible output bundle.")
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Strict research mode: fail incomplete live trials instead of substituting fallbacks; failed/incomplete rows are non-confirmatory.",
    )
    parser.add_argument(
        "--confirmatory-config",
        help="Opt into a frozen external confirmatory manifest; development runs omit this flag.",
    )
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
        split_seeds=args.split_seeds or [args.seed],
        model=args.model,
        planner_model=args.planner_model,
        reconciler_model=args.reconciler_model,
        offline=args.offline,
        include_perturbations=args.include_perturbations,
        case_names=args.cases,
        gate_mode=args.gate_mode,
        order_swap=args.order_swap,
        empirical_probe_enabled=not args.no_empirical_probe,
        require_live=args.require_live,
        soft_challenge_strategy=args.soft_challenge_strategy,
        enable_regression_interaction_diagnostics=not args.disable_interaction_diagnostics,
        enable_classification_boundary_diagnostics=not args.disable_boundary_diagnostics,
        resume=args.resume,
        suite=args.suite,
        tier=args.tier,
        confirmatory_config_path=args.confirmatory_config,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
