import pytest

import evaluation.statistics as statistics_module
from evaluation.statistics import (
    cluster_bootstrap_ci,
    cluster_bootstrap_multi_ci,
    cluster_bootstrap_multi_distribution,
    sample_clusters,
)
from evaluation.metrics import DEFAULT_GATE_UTILITY_WEIGHTS, _dataset_macro_health


def _rows():
    return [
        {"dataset": "A", "value": 1}, {"dataset": "A", "value": 1}, {"dataset": "A", "value": 1},
        {"dataset": "B", "value": 10}, {"dataset": "B", "value": 10},
        {"dataset": "C", "value": 100}, {"dataset": "C", "value": 100},
        {"dataset": "C", "value": 100}, {"dataset": "C", "value": 100},
    ]


def test_cluster_sample_keeps_complete_clusters():
    sample = sample_clusters(_rows(), "dataset", ["A", "C"])
    assert [row["dataset"] for row in sample] == ["A"] * 3 + ["C"] * 4


def test_duplicate_cluster_draws_preserve_multiplicity():
    sample = sample_clusters(_rows(), "dataset", ["A", "A", "C"])
    assert len(sample) == 3 + 3 + 4
    assert sum(row["dataset"] == "A" for row in sample) == 6


def test_cluster_bootstrap_is_reproducible_and_counts_clusters():
    data = _rows()

    def statistic(rows):
        return sum(row["value"] for row in rows) / len(rows)

    first = cluster_bootstrap_ci(data, statistic, "dataset", n_bootstrap=200, random_state=7)
    second = cluster_bootstrap_ci(data, statistic, "dataset", n_bootstrap=200, random_state=7)
    assert first == second
    assert first["n_clusters"] == 3
    assert first["uncertainty_method"] == "dataset_cluster_bootstrap_percentile"


def test_cluster_and_row_bootstrap_have_distinct_support():
    data = _rows()
    cluster = cluster_bootstrap_ci(
        data, lambda rows: sum(row["value"] for row in rows) / len(rows), "dataset",
        n_bootstrap=200, random_state=7,
    )
    row = cluster_bootstrap_ci(
        [{"row": i, "value": row["value"]} for i, row in enumerate(data)],
        lambda rows: sum(row["value"] for row in rows) / len(rows), "row",
        n_bootstrap=200, random_state=7,
    )
    assert (cluster["lower"], cluster["upper"]) != (row["lower"], row["upper"])


def test_one_cluster_ci_is_explicitly_unavailable():
    result = cluster_bootstrap_ci(
        [{"dataset": "A", "value": 1}, {"dataset": "A", "value": 2}],
        lambda rows: 1.0, "dataset", n_bootstrap=100,
    )
    assert result["status"] == "unavailable"
    assert result["lower"] is None and result["upper"] is None


def test_batched_cluster_bootstrap_matches_single_metric_bootstrap():
    data = _rows()

    def mean(rows):
        return sum(row["value"] for row in rows) / len(rows)

    def max_value(rows):
        return max(row["value"] for row in rows)

    statistic_fns = {
        "mean": mean,
        "max": max_value,
        "missing": lambda rows: None,
        "non_finite": lambda rows: float("nan"),
    }
    old = {
        metric: cluster_bootstrap_ci(
            data,
            statistic_fn,
            "dataset",
            n_bootstrap=200,
            confidence_level=0.90,
            random_state=19,
        )
        for metric, statistic_fn in statistic_fns.items()
    }
    new = cluster_bootstrap_multi_ci(
        data,
        statistic_fns,
        "dataset",
        n_bootstrap=200,
        confidence_level=0.90,
        random_state=19,
    )

    for metric in statistic_fns:
        assert new[metric]["n_clusters"] == old[metric]["n_clusters"]
        assert new[metric]["n_bootstrap"] == old[metric]["n_bootstrap"]
        assert new[metric]["status"] == old[metric]["status"]
        assert new[metric]["lower"] == pytest.approx(old[metric]["lower"])
        assert new[metric]["upper"] == pytest.approx(old[metric]["upper"])


def test_batched_cluster_bootstrap_shares_each_sample_across_metrics(monkeypatch):
    data = _rows()
    sampled = []
    calls = {"mean": 0, "max": 0}
    original_sample_clusters = statistics_module.sample_clusters

    def sample_spy(data, cluster_col, sampled_clusters):
        sampled.append(tuple(sampled_clusters))
        return original_sample_clusters(data, cluster_col, sampled_clusters)

    def mean(rows):
        calls["mean"] += 1
        return sum(row["value"] for row in rows) / len(rows)

    def max_value(rows):
        calls["max"] += 1
        return max(row["value"] for row in rows)

    monkeypatch.setattr(statistics_module, "sample_clusters", sample_spy)
    distributions, n_clusters = cluster_bootstrap_multi_distribution(
        data,
        {"mean": mean, "max": max_value},
        "dataset",
        n_bootstrap=17,
        random_state=23,
    )

    assert n_clusters == 3
    assert len(sampled) == 17
    assert calls == {"mean": 17, "max": 17}
    assert len(distributions["mean"]) == 17
    assert len(distributions["max"]) == 17


def test_batched_cluster_bootstrap_matches_single_metric_one_cluster_behavior():
    data = [{"dataset": "A", "value": 1}, {"dataset": "A", "value": 2}]
    statistic_fns = {"value": lambda rows: 1.0, "missing": lambda rows: None}
    result = cluster_bootstrap_multi_ci(data, statistic_fns, "dataset", n_bootstrap=100)

    for metric, statistic_fn in statistic_fns.items():
        assert result[metric] == cluster_bootstrap_ci(
            data, statistic_fn, "dataset", n_bootstrap=100
        )


def test_batched_cluster_bootstrap_validates_confidence_level_and_reproduces():
    with pytest.raises(ValueError, match="confidence_level"):
        cluster_bootstrap_multi_ci(_rows(), {"value": lambda rows: 1.0}, "dataset", confidence_level=1.0)

    statistic_fns = {"value": lambda rows: sum(row["value"] for row in rows) / len(rows)}
    first = cluster_bootstrap_multi_ci(
        _rows(), statistic_fns, "dataset", n_bootstrap=100, random_state=31
    )
    second = cluster_bootstrap_multi_ci(
        _rows(), statistic_fns, "dataset", n_bootstrap=100, random_state=31
    )
    assert first == second


def test_dataset_macro_health_equalizes_unequal_trial_counts():
    records = [
        {"benchmark_case": "small", "trial_status": "completed", "agent_normalized_regret": 0.0, "gated_normalized_regret": 0.0},
        *[
            {"benchmark_case": "large", "trial_status": "completed", "agent_normalized_regret": 1.0, "gated_normalized_regret": 0.0}
            for _ in range(9)
        ],
    ]
    health = _dataset_macro_health(
        records, tolerance=0.02, catastrophic_threshold=0.9,
        weights=DEFAULT_GATE_UTILITY_WEIGHTS,
    )
    assert health["mean_regret_reduction"] == 0.5


def test_more_llm_repetitions_do_not_increase_independent_dataset_count():
    records = [
        {"benchmark_case": "A", "trial_status": "completed", "agent_normalized_regret": 0.1, "gated_normalized_regret": 0.0},
        {"benchmark_case": "B", "trial_status": "completed", "agent_normalized_regret": 0.1, "gated_normalized_regret": 0.0},
    ]
    repeated = records + [
        {**row, "llm_repetition_id": "rep_002"} for row in records
    ]
    health = _dataset_macro_health(
        repeated, tolerance=0.02, catastrophic_threshold=0.9,
        weights=DEFAULT_GATE_UTILITY_WEIGHTS,
    )
    assert health["dataset_count"] == 2
