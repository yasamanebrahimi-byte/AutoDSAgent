"""Post-hoc evaluation harness for the AutoDSAgent validation architecture."""

from evaluation.benchmarks import (
    BENCHMARK_SUITE_VERSION,
    BenchmarkCase,
    BenchmarkRole,
    default_benchmark_cases,
)
from evaluation.runner import EvaluationConfig, run_evaluation

__all__ = [
    "BENCHMARK_SUITE_VERSION",
    "BenchmarkCase",
    "BenchmarkRole",
    "EvaluationConfig",
    "default_benchmark_cases",
    "run_evaluation",
]
