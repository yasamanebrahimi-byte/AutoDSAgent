"""Small, reusable uncertainty helpers for benchmark result aggregation.

The independent unit for paper-facing benchmark inference is a dataset/task.
These helpers deliberately resample complete clusters and retain multiplicity
when a cluster is drawn more than once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any

import numpy as np


DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260824


def resolve_cluster_key(row: dict[str, Any], *, default: str = "unknown") -> str:
    """Resolve the canonical benchmark dataset/task identifier from a row."""

    for field in ("benchmark_case", "dataset_id", "openml_task_id", "task_id", "dataset_name"):
        value = row.get(field)
        if value is not None and str(value) != "":
            return str(value)
    return default


def sample_clusters(
    data: Any,
    cluster_col: str,
    sampled_clusters: Sequence[Any],
) -> Any:
    """Construct a clustered bootstrap sample, preserving duplicate draws.

    ``sampled_clusters`` is intentionally injectable so tests and callers can
    inspect a particular draw. DataFrame inputs return a DataFrame; list-like
    record inputs return a list of records.
    """

    if hasattr(data, "loc") and hasattr(data, "columns"):
        pieces = []
        for instance, cluster in enumerate(sampled_clusters):
            piece = data.loc[data[cluster_col] == cluster].copy()
            piece["_bootstrap_cluster_instance"] = instance
            pieces.append(piece)
        if not pieces:
            return data.iloc[0:0].copy()
        import pandas as pd
        return pd.concat(pieces, ignore_index=True)

    records = list(data)
    pieces: list[Any] = []
    for instance, cluster in enumerate(sampled_clusters):
        for row in records:
            if row.get(cluster_col) == cluster:
                copied = dict(row)
                copied["_bootstrap_cluster_instance"] = instance
                pieces.append(copied)
    return pieces


def cluster_bootstrap_distribution(
    data: Any,
    statistic_fn: Callable[[Any], float | None],
    cluster_col: str,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_state: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[list[float], int]:
    """Return statistic values from a dataset-cluster percentile bootstrap."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if hasattr(data, "loc") and hasattr(data, "columns"):
        clusters = list(data[cluster_col].dropna().unique())
    else:
        clusters = list(dict.fromkeys(row.get(cluster_col) for row in data if row.get(cluster_col) is not None))
    n_clusters = len(clusters)
    if n_clusters < 2:
        return [], n_clusters
    rng = np.random.default_rng(random_state)
    estimates: list[float] = []
    for sampled_indices in rng.integers(0, n_clusters, size=(n_bootstrap, n_clusters)):
        replicate = sample_clusters(data, cluster_col, [clusters[i] for i in sampled_indices])
        value = statistic_fn(replicate)
        if value is not None and math.isfinite(float(value)):
            estimates.append(float(value))
    return estimates, n_clusters


def cluster_bootstrap_ci(
    data: Any,
    statistic_fn: Callable[[Any], float | None],
    cluster_col: str,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = 0.95,
    random_state: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute a deterministic percentile CI by resampling complete clusters."""

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    estimates, n_clusters = cluster_bootstrap_distribution(
        data, statistic_fn, cluster_col, n_bootstrap=n_bootstrap, random_state=random_state
    )
    result: dict[str, Any] = {
        "lower": None, "upper": None, "ci_low": None, "ci_high": None,
        "support": n_clusters, "n_clusters": n_clusters,
        "n_bootstrap": n_bootstrap, "confidence_level": confidence_level,
        "uncertainty_method": "dataset_cluster_bootstrap_percentile",
        "cluster_column": cluster_col,
        "stable": n_clusters >= 20,
        "status": "unavailable" if n_clusters < 2 or not estimates else "ok",
    }
    if estimates:
        alpha = (1.0 - confidence_level) / 2.0
        low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
        result.update(lower=float(low), upper=float(high), ci_low=float(low), ci_high=float(high))
    return result


def paired_cluster_bootstrap_difference(
    data: Any,
    statistic_fn: Callable[[Any], float | None],
    cluster_col: str,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = 0.95,
    random_state: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """CI for a paired difference computed jointly on each cluster replicate."""

    return cluster_bootstrap_ci(
        data, statistic_fn, cluster_col, n_bootstrap=n_bootstrap,
        confidence_level=confidence_level, random_state=random_state,
    )
