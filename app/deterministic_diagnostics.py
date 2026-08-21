"""Lightweight, training-only diagnostics for deterministic recommendation."""

from __future__ import annotations

from dataclasses import dataclass

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


def _regression_binned_signal(
    feature: pd.Series,
    target: pd.Series,
    min_bin_count: int = 8,
) -> tuple[float, float]:
    x = pd.to_numeric(feature, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan)
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
    target_scale = float(y.loc[valid].std(ddof=0))
    target_scale = max(target_scale, 1e-12)
    range_signal = min(1.0, float(means.max() - means.min()) / (2.0 * target_scale))
    order = np.arange(len(means), dtype=float)
    monotonicity = abs(float(pd.Series(order).corr(pd.Series(means.to_numpy(dtype=float)), method="spearman")))
    if not np.isfinite(monotonicity):
        monotonicity = 0.0
    nonlinearity = min(1.0, range_signal * (1.0 - monotonicity))
    return nonlinearity, min(1.0, range_signal)


def _eta_squared(feature: pd.Series, target: pd.Series) -> float:
    """Return bounded numeric-feature/nominal-target association.

    This is the correlation-ratio (eta-squared): between-class variation in a
    numeric feature divided by its total variation.  It only groups by target
    membership, so class names, dtypes, and ordering cannot affect the value.
    """

    paired = pd.DataFrame({"x": feature, "target": target}).dropna()
    if len(paired) < 2 or paired["x"].nunique(dropna=True) <= 1:
        return 0.0
    if paired["target"].nunique(dropna=True) <= 1:
        return 0.0
    grand_mean = float(paired["x"].mean())
    ss_total = float(((paired["x"] - grand_mean) ** 2).sum())
    if not np.isfinite(ss_total) or ss_total <= 0:
        return 0.0
    grouped = paired.groupby("target", sort=False, observed=True)["x"].agg(["count", "mean"])
    ss_between = float(
        (grouped["count"] * (grouped["mean"] - grand_mean) ** 2).sum()
    )
    if not np.isfinite(ss_between):
        return 0.0
    return min(1.0, max(0.0, ss_between / ss_total))


def _cramers_v(feature: pd.Series, target: pd.Series) -> float:
    """Return bounded, label-order-invariant categorical association."""

    paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if len(paired) < 2:
        return 0.0
    table = pd.crosstab(paired["feature"], paired["target"])
    observed = table.to_numpy(dtype=float)
    if observed.shape[0] <= 1 or observed.shape[1] <= 1:
        return 0.0
    total = float(observed.sum())
    if total <= 0 or not np.isfinite(total):
        return 0.0
    row_totals = observed.sum(axis=1, keepdims=True)
    column_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / total
    valid_expected = expected > 0
    chi_squared = float(
        (((observed - expected) ** 2) / np.where(valid_expected, expected, 1.0))[valid_expected].sum()
    )
    phi_squared = chi_squared / total
    denominator = min(observed.shape[0] - 1, observed.shape[1] - 1)
    if denominator <= 0 or not np.isfinite(phi_squared):
        return 0.0
    return min(1.0, max(0.0, float(np.sqrt(phi_squared / denominator))))


@dataclass(frozen=True)
class _RelationshipSignals:
    """Aggregate relationship facts plus variation across numeric features."""

    nonlinearity_score: float
    pearson_spearman_gap: float
    univariate_signal: float
    nonlinear_feature_count: int
    nonlinear_feature_fraction: float
    nonlinearity_heterogeneity: float
    marginal_association_strength: float
    class_separation_strength: float
    association_measure: str
    nonlinearity_applicable: bool


def _empty_relationship_signals(
    *,
    association_measure: str,
    nonlinearity_applicable: bool,
) -> _RelationshipSignals:
    return _RelationshipSignals(
        nonlinearity_score=0.0,
        pearson_spearman_gap=0.0,
        univariate_signal=0.0,
        nonlinear_feature_count=0,
        nonlinear_feature_fraction=0.0,
        nonlinearity_heterogeneity=0.0,
        marginal_association_strength=0.0,
        class_separation_strength=0.0,
        association_measure=association_measure,
        nonlinearity_applicable=nonlinearity_applicable,
    )


def _regression_relationship_signals(
    dataframe: pd.DataFrame,
    target_column: str,
    numeric_names: list[str],
) -> _RelationshipSignals:
    if not numeric_names:
        return _empty_relationship_signals(
            association_measure="regression_pearson_spearman_binned",
            nonlinearity_applicable=True,
        )
    target = _finite_numeric(dataframe[target_column])
    gaps: list[float] = []
    feature_nonlinearity_scores: list[float] = []
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
        binned, range_signal = _regression_binned_signal(dataframe[name], target)
        # Keep one bounded marginal nonlinearity measurement per numeric
        # feature.  The aggregate score below is useful for policy bands, but
        # it must not be reused as if it contained feature-level variation.
        feature_nonlinearity_scores.append(binned)
        if range_signal:
            marginal.append(range_signal)
    if not feature_nonlinearity_scores:
        return _empty_relationship_signals(
            association_measure="regression_pearson_spearman_binned",
            nonlinearity_applicable=True,
        )
    mean_nonlinear = float(np.mean(feature_nonlinearity_scores))
    top_count = max(1, int(np.ceil(len(feature_nonlinearity_scores) * 0.25)))
    top_nonlinear = float(np.mean(sorted(feature_nonlinearity_scores, reverse=True)[:top_count]))
    # One strong nonlinear marginal relationship should not disappear merely
    # because a dataset also contains several deliberately weak predictors.
    score = min(1.0, 0.70 * top_nonlinear + 0.30 * mean_nonlinear + 0.35 * (max(gaps) if gaps else 0.0))
    nonlinear_count = sum(value >= 0.15 for value in feature_nonlinearity_scores)
    if len(feature_nonlinearity_scores) < 2:
        # A single feature cannot establish cross-feature heterogeneity.
        heterogeneity = 0.0
    else:
        # Each feature score is bounded to [0, 1].  Population standard
        # deviation is therefore at most 0.5; multiplying by two maps the
        # largest possible two-point spread to the policy's [0, 1] range.
        heterogeneity = min(1.0, max(0.0, float(np.std(feature_nonlinearity_scores)) * 2.0))
    return _RelationshipSignals(
        nonlinearity_score=score,
        pearson_spearman_gap=max(gaps, default=0.0),
        univariate_signal=float(np.mean(marginal)) if marginal else 0.0,
        nonlinear_feature_count=int(nonlinear_count),
        nonlinear_feature_fraction=float(nonlinear_count / len(feature_nonlinearity_scores)),
        nonlinearity_heterogeneity=heterogeneity,
        marginal_association_strength=float(np.mean(marginal)) if marginal else 0.0,
        class_separation_strength=0.0,
        association_measure="regression_pearson_spearman_binned",
        nonlinearity_applicable=True,
    )


def _classification_relationship_signals(
    dataframe: pd.DataFrame,
    target_column: str,
    numeric_names: list[str],
    categorical_names: list[str],
) -> _RelationshipSignals:
    associations: list[float] = []
    numeric_associations = False
    categorical_associations = False
    target = dataframe[target_column]
    for name in numeric_names:
        associations.append(_eta_squared(dataframe[name], target))
        numeric_associations = True
    for name in categorical_names:
        associations.append(_cramers_v(dataframe[name], target))
        categorical_associations = True
    if numeric_associations and categorical_associations:
        measure = "classification_eta_squared_and_cramers_v"
    elif numeric_associations:
        measure = "classification_eta_squared"
    elif categorical_associations:
        measure = "classification_cramers_v"
    else:
        measure = "classification_nominal_association"
    if not associations:
        return _empty_relationship_signals(
            association_measure=measure,
            nonlinearity_applicable=False,
        )
    return _RelationshipSignals(
        # Eta-squared and Cramer's V establish marginal class association, not
        # linearity or nonlinearity.  Keep the latter neutral for nominal
        # multiclass targets rather than inventing a target ordering.
        nonlinearity_score=0.0,
        pearson_spearman_gap=0.0,
        univariate_signal=float(np.mean(associations)),
        nonlinear_feature_count=0,
        nonlinear_feature_fraction=0.0,
        nonlinearity_heterogeneity=0.0,
        marginal_association_strength=float(np.mean(associations)),
        class_separation_strength=max(associations, default=0.0),
        association_measure=measure,
        nonlinearity_applicable=False,
    )


def _relationship_signals(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    numeric_names: list[str],
    categorical_names: list[str] | None = None,
) -> _RelationshipSignals:
    if task_type == "classification":
        return _classification_relationship_signals(
            dataframe,
            target_column,
            numeric_names,
            categorical_names or [],
        )
    return _regression_relationship_signals(dataframe, target_column, numeric_names)


def _target_diagnostics(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    policy: DeterministicPolicy,
) -> TargetDiagnostics:
    target = dataframe[target_column].dropna()
    if task_type == "classification":
        # Keep this aggregate distribution label-free.  The deterministic
        # policy needs class sizes, not arbitrary class names or their order.
        raw_counts = target.value_counts(sort=False)
        raw_counts = raw_counts[raw_counts > 0]
        values = np.sort(raw_counts.to_numpy(dtype=float))[::-1]
        total = max(float(values.sum()), 1.0)
        minimum = int(values.min()) if len(values) else 0
        majority = float(values.max() / total) if len(values) else 0.0
        minority = float(values.min() / total) if len(values) else 0.0
        return TargetDiagnostics(
            classification=ClassificationTargetDiagnostics(
                classes=int(len(values)),
                minority_class_fraction=minority,
                majority_class_fraction=majority,
                imbalance_ratio=float(values.max() / max(values.min(), 1.0)) if len(values) else 0.0,
                samples_per_class={
                    f"class_{index + 1}": int(value) for index, value in enumerate(values)
                },
                minimum_class_size=minimum,
            )
        )
    values = _finite_numeric(target)
    variance = float(values.var(ddof=1)) if len(values) > 1 else 0.0
    skewness = float(values.skew()) if len(values) >= 3 else 0.0
    if not np.isfinite(skewness):
        skewness = 0.0
    outliers = _iqr_outlier_fraction(values)
    heavy_tail = (
        "high"
        if outliers >= policy.target_outlier_high_fraction
        else "moderate"
        if outliers >= policy.outlier_moderate_fraction or abs(skewness) >= policy.high_target_skewness
        else "low"
    )
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
    elif missing_fraction <= policy.concentrated_missing_feature_fraction or max_missing >= 0.50:
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
    relationship = _relationship_signals(
        dataframe,
        target_column,
        task_type,
        numeric_names,
        categorical_names,
    )
    nonlinearity = relationship.nonlinearity_score
    pearson_spearman_gap = relationship.pearson_spearman_gap
    univariate_signal = relationship.univariate_signal
    nonlinear_count = relationship.nonlinear_feature_count
    if nonlinearity >= policy.nonlinear_high_threshold:
        nonlinearity_signal = "high"
    elif nonlinearity >= policy.nonlinear_moderate_threshold:
        nonlinearity_signal = "moderate"
    else:
        nonlinearity_signal = "low"
    heterogeneity = relationship.nonlinearity_heterogeneity
    mixed_signal = 1.0 if numeric_names and categorical_names else 0.0
    categorical_signal = min(1.0, len(categorical_names) / 3.0)
    weak_association_threshold = (
        policy.classification_weak_association_threshold
        if task_type == "classification"
        else policy.regression_weak_association_threshold
    )
    weak_univariate = 1.0 if univariate_signal < weak_association_threshold and usable >= 3 else 0.0
    nonlinear_share = relationship.nonlinear_feature_fraction
    # This is a bounded structural prior for model-family compatibility.  It
    # summarizes several observable reasons a simple additive linear family
    # may be inadequate; it does not identify or prove feature interactions.
    structural_complexity = min(
        1.0,
        policy.structural_complexity_mixed_weight * mixed_signal
        + policy.structural_complexity_categorical_weight * categorical_signal
        + policy.structural_complexity_nonlinear_fraction_weight * nonlinear_share
        + policy.structural_complexity_nonlinearity_strength_weight * nonlinearity
        + policy.structural_complexity_heterogeneity_weight * heterogeneity
        + policy.structural_complexity_weak_marginal_weight * weak_univariate,
    )
    if structural_complexity >= policy.structural_complexity_high_threshold:
        structural_complexity_signal = "high"
    elif structural_complexity >= policy.structural_complexity_moderate_threshold:
        structural_complexity_signal = "moderate"
    else:
        structural_complexity_signal = "low"

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
        nonlinear_feature_fraction=max(0.0, min(1.0, nonlinear_share)),
        nonlinearity_heterogeneity=max(0.0, min(1.0, heterogeneity)),
        structural_complexity_score=max(0.0, min(1.0, structural_complexity)),
        structural_complexity_signal=structural_complexity_signal,
        numeric_outlier_feature_fraction=outlier_feature_fraction,
        numeric_outlier_cell_fraction=outlier_cell_fraction,
        target=_target_diagnostics(dataframe, target_column, task_type, policy),
        marginal_association_strength=max(
            0.0,
            min(1.0, relationship.marginal_association_strength),
        ),
        class_separation_strength=max(0.0, min(1.0, relationship.class_separation_strength)),
        association_measure=relationship.association_measure,
        nonlinearity_applicable=relationship.nonlinearity_applicable,
    )
