"""Lightweight, training-only diagnostics for deterministic recommendation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.deterministic_policy import (
    DeterministicPolicy,
    MAX_CATEGORICAL_CARDINALITY,
)
from app.schemas import (
    ClassificationTargetDiagnostics,
    DeterministicDiagnostics,
    RegressionTargetDiagnostics,
    TargetDiagnostics,
    TaskType,
)


def _finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def _iqr_outlier_fraction(values: pd.Series) -> float:
    numeric = _finite_numeric(values)
    if len(numeric) < 4:
        return 0.0
    q1, q3 = numeric.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 0:
        return 0.0
    outliers = (numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)
    return float(outliers.mean())


def _target_numeric(dataframe: pd.DataFrame, target_column: str, task_type: TaskType) -> pd.Series:
    target = dataframe[target_column]
    if task_type == "regression":
        return _finite_numeric(target)
    values = target.dropna()
    categories = sorted(values.astype(str).unique().tolist())
    mapping = {category: float(index) for index, category in enumerate(categories)}
    return values.astype(str).map(mapping).astype(float)


def _binned_signal(feature: pd.Series, target: pd.Series, task_type: TaskType, min_bin_count: int = 8) -> tuple[float, float]:
    x = pd.to_numeric(feature, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if task_type == "regression":
        y = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        y = target
    valid = x.notna() & y.notna()
    if int(valid.sum()) < max(4 * min_bin_count, 24):
        return 0.0, 0.0
    try:
        bins = pd.qcut(x.loc[valid], q=min(10, max(4, int(np.sqrt(valid.sum())))), duplicates="drop")
        grouped = pd.DataFrame({"bin": bins, "target": y.loc[valid]}).groupby("bin", observed=True)["target"]
        means = grouped.mean()
        counts = grouped.size()
        means = means[counts >= min_bin_count]
    except (TypeError, ValueError):
        return 0.0, 0.0
    if len(means) < 4:
        return 0.0, 0.0
    target_scale = float(y.loc[valid].std(ddof=0)) if task_type == "regression" else 1.0
    target_scale = max(target_scale, 1e-12)
    range_signal = min(1.0, float(means.max() - means.min()) / (2.0 * target_scale))
    order = np.arange(len(means), dtype=float)
    monotonicity = abs(float(pd.Series(order).corr(pd.Series(means.to_numpy(dtype=float)), method="spearman")))
    if not np.isfinite(monotonicity):
        monotonicity = 0.0
    nonlinearity = min(1.0, range_signal * (1.0 - monotonicity))
    return nonlinearity, min(1.0, range_signal)


def _relationship_signals(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    numeric_names: list[str],
) -> tuple[float, float, float, int]:
    if not numeric_names:
        return 0.0, 0.0, 0.0, 0
    target = _target_numeric(dataframe, target_column, task_type)
    gaps: list[float] = []
    nonlinear: list[float] = []
    marginal: list[float] = []
    for name in numeric_names:
        x = pd.to_numeric(dataframe[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        paired = pd.concat([x.rename("x"), target.rename("y")], axis=1).dropna()
        if len(paired) >= 4 and paired["x"].nunique() > 1 and paired["y"].nunique() > 1:
            pearson = abs(float(paired["x"].corr(paired["y"], method="pearson")))
            spearman = abs(float(paired["x"].corr(paired["y"], method="spearman")))
            if np.isfinite(pearson) and np.isfinite(spearman):
                gaps.append(min(1.0, abs(pearson - spearman)))
                marginal.append(min(1.0, max(pearson, spearman)))
        binned, range_signal = _binned_signal(dataframe[name], target, task_type)
        nonlinear.append(binned)
        if range_signal:
            marginal.append(range_signal)
    if not nonlinear:
        return 0.0, 0.0, 0.0, 0
    mean_nonlinear = float(np.mean(nonlinear))
    top_count = max(1, int(np.ceil(len(nonlinear) * 0.25)))
    top_nonlinear = float(np.mean(sorted(nonlinear, reverse=True)[:top_count]))
    # One strong nonlinear marginal relationship should not disappear merely
    # because a dataset also contains several deliberately weak predictors.
    score = min(1.0, 0.70 * top_nonlinear + 0.30 * mean_nonlinear + 0.35 * (max(gaps) if gaps else 0.0))
    nonlinear_count = sum(value >= 0.15 for value in nonlinear)
    return score, max(gaps, default=0.0), float(np.mean(marginal)) if marginal else 0.0, int(nonlinear_count)


def _target_diagnostics(dataframe: pd.DataFrame, target_column: str, task_type: TaskType) -> TargetDiagnostics:
    target = dataframe[target_column].dropna()
    if task_type == "classification":
        counts = target.astype(str).value_counts().sort_index()
        values = counts.to_numpy(dtype=float)
        total = max(float(values.sum()), 1.0)
        minimum = int(values.min()) if len(values) else 0
        majority = float(values.max() / total) if len(values) else 0.0
        minority = float(values.min() / total) if len(values) else 0.0
        return TargetDiagnostics(
            classification=ClassificationTargetDiagnostics(
                classes=int(len(counts)),
                minority_class_fraction=minority,
                majority_class_fraction=majority,
                imbalance_ratio=float(values.max() / max(values.min(), 1.0)) if len(values) else 0.0,
                samples_per_class={str(key): int(value) for key, value in counts.items()},
                minimum_class_size=minimum,
            )
        )
    values = _finite_numeric(target)
    variance = float(values.var(ddof=1)) if len(values) > 1 else 0.0
    skewness = float(values.skew()) if len(values) >= 3 else 0.0
    if not np.isfinite(skewness):
        skewness = 0.0
    outliers = _iqr_outlier_fraction(values)
    heavy_tail = "high" if outliers >= 0.10 else "moderate" if outliers >= 0.05 or abs(skewness) >= 2 else "low"
    return TargetDiagnostics(
        regression=RegressionTargetDiagnostics(
            variance=max(0.0, variance),
            skewness=skewness,
            outlier_fraction=outliers,
            heavy_tail_signal=heavy_tail,
        )
    )


def compute_deterministic_diagnostics(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    *,
    policy: DeterministicPolicy | None = None,
) -> DeterministicDiagnostics:
    """Compute only aggregate facts from the supplied training frame.

    The caller is responsible for passing the frozen training partition.  No
    holdout, model fit, CV result, or empirical reference is consulted here.
    """

    policy = policy or DeterministicPolicy()
    from app.deterministic import profile_dataframe
    from app.preprocessing import requirements_from_records

    profile = profile_dataframe(dataframe)
    records = [record for record in profile["column_details"] if record["name"] != target_column]
    requirements = requirements_from_records(records, task_type, "linear")
    numeric_names = [str(value) for value in requirements.evidence["numeric_features"]]
    categorical_names = [str(value) for value in requirements.evidence["categorical_features"]]
    usable_names = numeric_names + categorical_names
    feature_frame = dataframe.drop(columns=[target_column])
    rows = int(len(dataframe))
    feature_count = len(feature_frame.columns)
    usable = len(usable_names)
    excluded = max(0, feature_count - usable)
    excluded_types: dict[str, int] = {}
    for record in records:
        if str(record["name"]) not in usable_names:
            reason = "identifier" if record["identifier_like"] else str(record["semantic_type"])
            if record["semantic_type"] in {"categorical", "boolean"} and int(record["unique"]) > MAX_CATEGORICAL_CARDINALITY:
                reason = "high_cardinality"
            excluded_types[reason] = excluded_types.get(reason, 0) + 1

    missing_by_feature = feature_frame.isna().mean() if feature_count else pd.Series(dtype=float)
    overall_missing = float(feature_frame.isna().to_numpy(dtype=float).mean()) if feature_count and rows else 0.0
    max_missing = float(missing_by_feature.max()) if len(missing_by_feature) else 0.0
    missing_count = int((missing_by_feature > 0).sum())
    missing_fraction = missing_count / max(feature_count, 1)
    if not missing_count:
        missing_pattern = "none"
    elif missing_fraction <= 0.25 or max_missing >= 0.50:
        missing_pattern = "concentrated"
    else:
        missing_pattern = "widespread"

    all_categorical_records = [
        record
        for record in records
        if record["semantic_type"] in {"categorical", "boolean", "numeric_like"}
    ]
    cardinalities = [int(record["unique"]) for record in all_categorical_records]
    mean_cardinality = float(np.mean(cardinalities)) if cardinalities else 0.0
    max_cardinality = max(cardinalities, default=0)
    high_cardinality_count = sum(value > MAX_CATEGORICAL_CARDINALITY for value in cardinalities)
    categorical_total = len(cardinalities)
    estimated_one_hot = int(requirements.evidence["estimated_one_hot_features"])
    boosted_effective = len(numeric_names) + len(categorical_names)
    binary_count = 0
    for name in usable_names:
        unique = int(dataframe[name].nunique(dropna=True))
        if unique <= 2:
            binary_count += 1

    correlations = dataframe[numeric_names].replace([np.inf, -np.inf], np.nan).corr() if len(numeric_names) >= 2 else pd.DataFrame()
    pair_values: list[float] = []
    if not correlations.empty:
        for left_index, left in enumerate(correlations.columns):
            for right in correlations.columns[left_index + 1 :]:
                value = abs(float(correlations.loc[left, right]))
                if np.isfinite(value):
                    pair_values.append(value)
    max_corr = max(pair_values, default=0.0)
    high_pairs = sum(value >= policy.high_correlation_threshold for value in pair_values)
    high_pair_fraction = high_pairs / max(len(pair_values), 1)
    nonlinearity, pearson_spearman_gap, univariate_signal, nonlinear_count = _relationship_signals(
        dataframe, target_column, task_type, numeric_names
    )
    if nonlinearity >= policy.nonlinear_high_threshold:
        nonlinearity_signal = "high"
    elif nonlinearity >= policy.nonlinear_moderate_threshold:
        nonlinearity_signal = "moderate"
    else:
        nonlinearity_signal = "low"
    heterogeneity = min(1.0, float(np.std([nonlinearity])) * 2.0) if numeric_names else 0.0
    mixed_signal = 1.0 if numeric_names and categorical_names else 0.0
    categorical_signal = min(1.0, len(categorical_names) / 3.0)
    weak_univariate = 1.0 if univariate_signal < 0.20 and usable >= 3 else 0.0
    nonlinear_share = nonlinear_count / max(len(numeric_names), 1)
    interaction = min(
        1.0,
        0.40 * mixed_signal
        + 0.20 * categorical_signal
        + 0.15 * heterogeneity
        + 0.10 * weak_univariate
        + 0.15 * min(1.0, nonlinear_share),
    )
    if interaction >= policy.interaction_high_threshold:
        interaction_signal = "high"
    elif interaction >= policy.interaction_moderate_threshold:
        interaction_signal = "moderate"
    else:
        interaction_signal = "low"

    outlier_fractions = [_iqr_outlier_fraction(dataframe[name]) for name in numeric_names]
    outlier_feature_fraction = float(np.mean([value >= policy.outlier_moderate_fraction for value in outlier_fractions])) if outlier_fractions else 0.0
    numeric_values = [
        _finite_numeric(dataframe[name])
        for name in numeric_names
    ]
    numeric_cells = sum(len(values) for values in numeric_values)
    outlier_cells = sum(int(round(_iqr_outlier_fraction(values) * len(values))) for values in numeric_values)
    outlier_cell_fraction = outlier_cells / max(numeric_cells, 1)
    effective_ratio = rows / max(estimated_one_hot, 1)
    return DeterministicDiagnostics(
        rows=rows,
        usable_features=usable,
        excluded_features=excluded,
        excluded_feature_types=excluded_types,
        numeric_feature_count=len(numeric_names),
        categorical_feature_count=len(categorical_names),
        binary_feature_count=binary_count,
        text_feature_count=sum(record["semantic_type"] == "text" for record in records),
        sample_to_feature_ratio=float(effective_ratio),
        effective_features_estimate=estimated_one_hot,
        linear_effective_features_estimate=estimated_one_hot,
        tree_effective_features_estimate=estimated_one_hot,
        boosted_effective_features_estimate=boosted_effective,
        overall_missing_fraction=overall_missing,
        max_feature_missing_fraction=max_missing,
        features_with_missing_count=missing_count,
        features_with_missing_fraction=float(missing_fraction),
        missingness_pattern=missing_pattern,
        mean_categorical_cardinality=mean_cardinality,
        max_categorical_cardinality=max_cardinality,
        estimated_one_hot_dimensionality=estimated_one_hot,
        high_cardinality_feature_count=high_cardinality_count,
        high_cardinality_feature_fraction=high_cardinality_count / max(categorical_total, 1),
        max_abs_numeric_correlation=max(0.0, min(1.0, max_corr)),
        high_correlation_pair_count=high_pairs,
        high_correlation_pair_fraction=float(high_pair_fraction),
        pearson_spearman_gap=max(0.0, min(1.0, pearson_spearman_gap)),
        mean_univariate_signal=max(0.0, min(1.0, univariate_signal)),
        nonlinearity_score=max(0.0, min(1.0, nonlinearity)),
        nonlinearity_signal=nonlinearity_signal,
        nonlinear_feature_count=nonlinear_count,
        interaction_potential=interaction,
        interaction_signal=interaction_signal,
        numeric_outlier_feature_fraction=outlier_feature_fraction,
        numeric_outlier_cell_fraction=outlier_cell_fraction,
        target=_target_diagnostics(dataframe, target_column, task_type),
    )
