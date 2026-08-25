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
    parser.add_argument("--split-seed", action="append", dest="split_seeds", type=int)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--gate-mode",
        choices=("llm_only", "deterministic_only", "always_reconcile", "selective", "probe_first"),
        default="probe_first",
        help="Compare the initial agent, historical gates, or the probe-first intervention gate.",
    )
    parser.add_argument(
        "--reconciliation-mode",
        choices=("blinded", "legacy"),
        default="blinded",
        help="Use source-neutral blinded reconciliation or the legacy prompt baseline.",
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
    mode.add_argument("--live", action="store_true", help="Request live OpenAI trials (the default when not offline).")
    parser.add_argument("--resume", action="store_true", help="Resume missing trial IDs from an existing compatible output bundle.")
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Fail incomplete trials instead of substituting offline modeling or reconciliation fallbacks.",
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
        offline=args.offline,
        include_perturbations=args.include_perturbations,
        case_names=args.cases,
        gate_mode=args.gate_mode,
        reconciliation_mode=args.reconciliation_mode,
        order_swap=args.order_swap,
        empirical_probe_enabled=not args.no_empirical_probe,
        require_live=args.require_live,
        soft_challenge_strategy=args.soft_challenge_strategy,
        enable_regression_interaction_diagnostics=not args.disable_interaction_diagnostics,
        enable_classification_boundary_diagnostics=not args.disable_boundary_diagnostics,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
