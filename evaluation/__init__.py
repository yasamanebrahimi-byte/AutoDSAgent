"""Post-hoc evaluation harness for the AutoDSAgent validation architecture."""

from evaluation.benchmarks import (
    BENCHMARK_SUITE_VERSION,
    BenchmarkCase,
    BenchmarkRole,
    default_benchmark_cases,
)
from evaluation.runner import EvaluationConfig, run_evaluation
from evaluation.external_benchmarks import (
    EXTERNAL_BENCHMARK_MANIFEST,
    EXTERNAL_BENCHMARK_SUITE_VERSION,
    OpenMLBenchmarkSpec,
    external_benchmark_cases,
    external_benchmark_specs,
)

__all__ = [
    "BENCHMARK_SUITE_VERSION",
    "BenchmarkCase",
    "BenchmarkRole",
    "EvaluationConfig",
    "default_benchmark_cases",
    "EXTERNAL_BENCHMARK_MANIFEST",
    "EXTERNAL_BENCHMARK_SUITE_VERSION",
    "OpenMLBenchmarkSpec",
    "external_benchmark_cases",
    "external_benchmark_specs",
    "run_evaluation",
]
