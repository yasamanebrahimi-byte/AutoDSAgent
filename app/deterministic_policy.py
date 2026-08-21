"""Versioned, interpretable policy constants for model-family compatibility."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas import (
    DeterministicDiagnostics,
    DeterministicMethodAssessment,
    DeterministicScoreContribution,
    Method,
)


# These two limits are also the canonical preprocessing safety limits.  The
# preprocessing module re-exports them so recommendation and execution cannot
# drift to contradictory categorical policies.
MAX_CATEGORICAL_CARDINALITY = 80
MAX_ONE_HOT_FEATURES = 4000


@dataclass(frozen=True)
class DeterministicPolicy:
    """Named thresholds and score bands for policy version 4.

    Compatibility scores are bounded policy points, not probabilities.  The
    thresholds are deliberately coarse and documented so a reviewer can see
    which structural observation caused a score change.
    """

    version: str = "4"
    high_correlation_threshold: float = 0.80
    severe_correlation_threshold: float = 0.90
    low_sample_feature_ratio: float = 3.0
    moderate_sample_feature_ratio: float = 8.0
    healthy_sample_feature_ratio: float = 20.0
    high_effective_features: int = 100
    very_high_effective_features: int = 300
    moderate_missing_fraction: float = 0.10
    high_missing_fraction: float = 0.25
    widespread_missing_feature_fraction: float = 0.40
    nonlinear_moderate_threshold: float = 0.15
    nonlinear_high_threshold: float = 0.35
    # These thresholds are explicit by task because eta-squared/Cramer's V
    # and regression correlation evidence are bounded but not interchangeable.
    regression_weak_association_threshold: float = 0.20
    classification_weak_association_threshold: float = 0.20
    # The structural-complexity score is normalized by weights that sum to
    # one, so these coarse low/moderate/high bands remain interpretable.
    structural_complexity_moderate_threshold: float = 0.30
    structural_complexity_high_threshold: float = 0.60
    # Explicit policy weights keep the heuristic auditable and easy to revise.
    # Mixed types and categorical structure are modest priors; nonlinearity is
    # the strongest signal; heterogeneity and weak marginal evidence are
    # supporting signals rather than proof of interactions.
    structural_complexity_mixed_weight: float = 0.15
    structural_complexity_categorical_weight: float = 0.15
    structural_complexity_nonlinear_fraction_weight: float = 0.20
    structural_complexity_nonlinearity_strength_weight: float = 0.25
    structural_complexity_heterogeneity_weight: float = 0.15
    structural_complexity_weak_marginal_weight: float = 0.10
    outlier_moderate_fraction: float = 0.05
    boosted_tree_min_samples: int = 300
    boosted_tree_preferred_samples: int = 600
    tiny_dataset_max_samples: int = 80
    minimum_boosted_effective_features: int = 1000
    high_cardinality_fraction: float = 0.50
    high_class_imbalance_fraction: float = 0.10
    severe_class_imbalance_ratio: float = 9.0
    unstable_class_size: int = 15
    low_confidence_margin: int = 7
    high_confidence_margin: int = 15


SUPPORTED_METHOD_ORDER: tuple[Method, ...] = (
    "linear",
    "regularized_linear",
    "tree_ensemble",
    "boosted_tree",
)


def _contribution(
    method: Method,
    factor: str,
    points: int,
    observation: str,
) -> DeterministicScoreContribution:
    return DeterministicScoreContribution(
        factor=factor,
        effect="favors" if points >= 0 else "penalizes",
        method=method,
        points=points,
        observation=observation,
    )


def score_model_families(
    diagnostics: DeterministicDiagnostics,
    *,
    policy: DeterministicPolicy | None = None,
) -> dict[Method, DeterministicMethodAssessment]:
    """Score all supported families from diagnostics without fitting models."""

    policy = policy or DeterministicPolicy()
    scores: dict[Method, int] = {method: 50 for method in SUPPORTED_METHOD_ORDER}
    contributions: dict[Method, list[DeterministicScoreContribution]] = {
        method: [] for method in SUPPORTED_METHOD_ORDER
    }

    def add(method: Method, factor: str, points: int, observation: str) -> None:
        scores[method] += points
        contributions[method].append(_contribution(method, factor, points, observation))

    numeric_dominant = diagnostics.numeric_feature_count > diagnostics.categorical_feature_count
    mixed = diagnostics.numeric_feature_count > 0 and diagnostics.categorical_feature_count > 0
    if numeric_dominant:
        add("linear", "feature_composition", 8, "numeric predictors predominate")
        add("regularized_linear", "feature_composition", 6, "numeric predictors predominate")
    if mixed:
        add("linear", "feature_composition", -3, "numeric and categorical predictors are mixed")
        add("regularized_linear", "feature_composition", 2, "numeric and categorical predictors are mixed")
        add("tree_ensemble", "feature_composition", 7, "numeric and categorical predictors are mixed")
        add("boosted_tree", "feature_composition", 5, "numeric and categorical predictors are mixed")
    elif diagnostics.categorical_feature_count:
        add("linear", "feature_composition", -2, "categorical predictors are present")
        add("regularized_linear", "feature_composition", 2, "categorical predictors are present")
        add("tree_ensemble", "feature_composition", 5, "categorical predictors are present")
        add("boosted_tree", "feature_composition", 4, "categorical predictors are present")
    if diagnostics.categorical_feature_count >= 2:
        add("linear", "categorical_structure", -3, "multiple categorical predictors increase encoded structure")
        add("regularized_linear", "categorical_structure", 1, "regularization can absorb several encoded predictors")
        add("tree_ensemble", "categorical_structure", 5, "multiple categorical predictors suggest structured splits")
        add("boosted_tree", "categorical_structure", 4, "multiple categorical predictors suggest structured splits")
    if diagnostics.high_cardinality_feature_fraction >= policy.high_cardinality_fraction:
        add("linear", "categorical_cardinality", -4, "at least half of categorical predictors exceed the canonical cardinality band")
        add("regularized_linear", "categorical_cardinality", -2, "at least half of categorical predictors exceed the canonical cardinality band")
        add("tree_ensemble", "categorical_cardinality", -3, "at least half of categorical predictors exceed the canonical cardinality band")
        add("boosted_tree", "categorical_cardinality", 1, "ordinal encoding limits expansion from high-cardinality predictors")
    elif diagnostics.max_categorical_cardinality >= 40:
        add("linear", "categorical_cardinality", -2, f"maximum categorical cardinality {diagnostics.max_categorical_cardinality} increases one-hot burden")
        add("regularized_linear", "categorical_cardinality", 2, f"maximum categorical cardinality {diagnostics.max_categorical_cardinality} favors shrinkage")
        add("tree_ensemble", "categorical_cardinality", -1, f"maximum categorical cardinality {diagnostics.max_categorical_cardinality} burdens one-hot trees")
        add("boosted_tree", "categorical_cardinality", 1, f"ordinal encoding avoids one-hot expansion from cardinality {diagnostics.max_categorical_cardinality}")

    ratio = diagnostics.sample_to_feature_ratio
    if ratio < policy.low_sample_feature_ratio:
        add("linear", "sample_to_feature_ratio", -8, f"ratio {ratio:.2f} is below {policy.low_sample_feature_ratio:.1f}")
        add("regularized_linear", "sample_to_feature_ratio", -2, f"ratio {ratio:.2f} is below {policy.low_sample_feature_ratio:.1f}")
        add("tree_ensemble", "sample_to_feature_ratio", -10, f"ratio {ratio:.2f} is below {policy.low_sample_feature_ratio:.1f}")
        add("boosted_tree", "sample_to_feature_ratio", -14, f"ratio {ratio:.2f} is below {policy.low_sample_feature_ratio:.1f}")
    elif ratio < policy.moderate_sample_feature_ratio:
        add("linear", "sample_to_feature_ratio", -2, f"ratio {ratio:.2f} is between low and moderate bands")
        add("regularized_linear", "sample_to_feature_ratio", 8, f"ratio {ratio:.2f} benefits from coefficient shrinkage")
        add("tree_ensemble", "sample_to_feature_ratio", 2, f"ratio {ratio:.2f} supports only a modest ensemble preference")
        add("boosted_tree", "sample_to_feature_ratio", 0, f"ratio {ratio:.2f} is below the preferred boosted-tree band")
    elif ratio < policy.healthy_sample_feature_ratio:
        add("linear", "sample_to_feature_ratio", 4, f"ratio {ratio:.2f} is adequate")
        add("regularized_linear", "sample_to_feature_ratio", 6, f"ratio {ratio:.2f} is adequate with a lower-variance baseline")
        add("tree_ensemble", "sample_to_feature_ratio", 6, f"ratio {ratio:.2f} supports ensemble fitting")
        add("boosted_tree", "sample_to_feature_ratio", 7, f"ratio {ratio:.2f} supports boosted-tree fitting")
    else:
        add("linear", "sample_to_feature_ratio", 8, f"ratio {ratio:.2f} is healthy")
        add("regularized_linear", "sample_to_feature_ratio", 5, f"ratio {ratio:.2f} is healthy")
        add("tree_ensemble", "sample_to_feature_ratio", 7, f"ratio {ratio:.2f} is healthy")
        add("boosted_tree", "sample_to_feature_ratio", 10, f"ratio {ratio:.2f} is healthy")

    if diagnostics.rows < policy.tiny_dataset_max_samples:
        add("linear", "dataset_scale", 2, f"{diagnostics.rows} rows favor a low-variance baseline")
        add("regularized_linear", "dataset_scale", 4, f"{diagnostics.rows} rows favor a compact regularized baseline")
        add("tree_ensemble", "dataset_scale", -8, f"{diagnostics.rows} rows are small for an ensemble")
        add("boosted_tree", "dataset_scale", -12, f"{diagnostics.rows} rows are small for boosted trees")
    elif diagnostics.rows < policy.boosted_tree_min_samples:
        add("tree_ensemble", "dataset_scale", 3, f"{diagnostics.rows} rows support a modest ensemble")
        add("boosted_tree", "dataset_scale", -3, f"{diagnostics.rows} rows are below {policy.boosted_tree_min_samples}")
    elif diagnostics.rows < policy.boosted_tree_preferred_samples:
        add("tree_ensemble", "dataset_scale", 7, f"{diagnostics.rows} rows support an ensemble")
        add("boosted_tree", "dataset_scale", 5, f"{diagnostics.rows} rows support boosted trees")
    else:
        add("tree_ensemble", "dataset_scale", 8, f"{diagnostics.rows} rows support an ensemble")
        add("boosted_tree", "dataset_scale", 10, f"{diagnostics.rows} rows support a larger boosted-tree fit")

    effective = diagnostics.effective_features_estimate
    if effective <= 20:
        add("linear", "encoded_dimensionality", 7, f"estimated encoded dimension {effective} is low")
        add("regularized_linear", "encoded_dimensionality", 3, f"estimated encoded dimension {effective} is low")
        add("tree_ensemble", "encoded_dimensionality", 2, f"estimated encoded dimension {effective} is low")
        add("boosted_tree", "encoded_dimensionality", 2, f"estimated encoded dimension {effective} is low")
    elif effective <= policy.high_effective_features:
        add("linear", "encoded_dimensionality", 2, f"estimated encoded dimension {effective} is manageable")
        add("regularized_linear", "encoded_dimensionality", 6, f"estimated encoded dimension {effective} is manageable")
        add("tree_ensemble", "encoded_dimensionality", 4, f"estimated encoded dimension {effective} is manageable")
        add("boosted_tree", "encoded_dimensionality", 4, f"estimated encoded dimension {effective} is manageable")
    elif effective <= policy.very_high_effective_features:
        add("linear", "encoded_dimensionality", -5, f"estimated encoded dimension {effective} is large")
        add("regularized_linear", "encoded_dimensionality", 10, f"estimated encoded dimension {effective} favors shrinkage")
        add("tree_ensemble", "encoded_dimensionality", 0, f"estimated encoded dimension {effective} is a burden for one-hot trees")
        add("boosted_tree", "encoded_dimensionality", 1, f"ordinal dimension remains {diagnostics.boosted_effective_features_estimate}")
    else:
        add("linear", "encoded_dimensionality", -12, f"estimated encoded dimension {effective} is very large")
        add("regularized_linear", "encoded_dimensionality", 8, f"estimated encoded dimension {effective} favors regularization")
        add("tree_ensemble", "encoded_dimensionality", -8, f"estimated encoded dimension {effective} burdens one-hot trees")
        add("boosted_tree", "encoded_dimensionality", 3, f"ordinal dimension remains {diagnostics.boosted_effective_features_estimate}")

    corr = diagnostics.max_abs_numeric_correlation
    if corr >= policy.severe_correlation_threshold or diagnostics.high_correlation_pair_fraction >= 0.25:
        add("linear", "multicollinearity", -14, f"max absolute correlation {corr:.2f} is severe")
        add("regularized_linear", "multicollinearity", 14, f"max absolute correlation {corr:.2f} favors shrinkage")
        add("tree_ensemble", "multicollinearity", 2, f"max absolute correlation {corr:.2f} is tolerated by split-based models")
        add("boosted_tree", "multicollinearity", 1, f"max absolute correlation {corr:.2f} is tolerated by split-based models")
    elif corr >= policy.high_correlation_threshold:
        add("linear", "multicollinearity", -6, f"max absolute correlation {corr:.2f} is elevated")
        add("regularized_linear", "multicollinearity", 8, f"max absolute correlation {corr:.2f} favors shrinkage")
        add("tree_ensemble", "multicollinearity", 1, f"max absolute correlation {corr:.2f} is tolerated")
        add("boosted_tree", "multicollinearity", 1, f"max absolute correlation {corr:.2f} is tolerated")
    else:
        add("linear", "multicollinearity", 7, f"max absolute correlation {corr:.2f} is limited")
        add("regularized_linear", "multicollinearity", 4, f"max absolute correlation {corr:.2f} is limited")

    signal = diagnostics.nonlinearity_signal
    if diagnostics.nonlinearity_applicable:
        if signal == "high":
            add("linear", "nonlinearity", -10, f"nonlinearity score {diagnostics.nonlinearity_score:.2f} is high")
            add("regularized_linear", "nonlinearity", -8, f"nonlinearity score {diagnostics.nonlinearity_score:.2f} is high")
            add("tree_ensemble", "nonlinearity", 15, f"nonlinearity score {diagnostics.nonlinearity_score:.2f} is high")
            add("boosted_tree", "nonlinearity", 17, f"nonlinearity score {diagnostics.nonlinearity_score:.2f} is high")
        elif signal == "moderate":
            add("linear", "nonlinearity", -3, f"nonlinearity score {diagnostics.nonlinearity_score:.2f} is moderate")
            add("tree_ensemble", "nonlinearity", 10, f"nonlinearity score {diagnostics.nonlinearity_score:.2f} is moderate")
            add("boosted_tree", "nonlinearity", 8, f"nonlinearity score {diagnostics.nonlinearity_score:.2f} is moderate")
        else:
            add("linear", "nonlinearity", 8, f"nonlinearity score {diagnostics.nonlinearity_score:.2f} is low")
            add("regularized_linear", "nonlinearity", 5, f"nonlinearity score {diagnostics.nonlinearity_score:.2f} is low")
            add("tree_ensemble", "nonlinearity", 1, f"nonlinearity score {diagnostics.nonlinearity_score:.2f} is low")
            add("boosted_tree", "nonlinearity", -2, f"nonlinearity score {diagnostics.nonlinearity_score:.2f} is low")

    complexity = diagnostics.structural_complexity_score
    if complexity >= policy.structural_complexity_high_threshold:
        points = {"linear": -5, "regularized_linear": -2, "tree_ensemble": 8, "boosted_tree": 8}
    elif complexity >= policy.structural_complexity_moderate_threshold:
        points = {"linear": -2, "regularized_linear": 0, "tree_ensemble": 4, "boosted_tree": 4}
    else:
        points = {"linear": 2, "regularized_linear": 2, "tree_ensemble": 0, "boosted_tree": 0}
    if diagnostics.target.classification is not None:
        complexity_observation = (
            f"structural complexity score {complexity:.2f} reflects mixed/categorical structure "
            "and weak marginal class association; it does not assert multiclass nonlinearity"
        )
    else:
        complexity_observation = (
            f"structural complexity score {complexity:.2f} suggests heterogeneous or nonlinear "
            "feature relationships"
        )
    for method, points_for_method in points.items():
        add(
            method,
            "structural_complexity",
            points_for_method,
            complexity_observation,
        )

    if diagnostics.overall_missing_fraction >= policy.high_missing_fraction:
        missing_points = {"linear": -3, "regularized_linear": -1, "tree_ensemble": 2, "boosted_tree": 1}
    elif diagnostics.overall_missing_fraction >= policy.moderate_missing_fraction:
        missing_points = {"linear": -1, "regularized_linear": 0, "tree_ensemble": 1, "boosted_tree": 1}
    else:
        missing_points = {"linear": 0, "regularized_linear": 0, "tree_ensemble": 0, "boosted_tree": 0}
    for method, points_for_method in missing_points.items():
        add(method, "missingness", points_for_method, f"overall missing-cell fraction {diagnostics.overall_missing_fraction:.2f}; imputation remains available")

    outlier = diagnostics.numeric_outlier_cell_fraction
    if outlier >= policy.outlier_moderate_fraction:
        for method, points_for_method in {"linear": -4, "regularized_linear": 2, "tree_ensemble": 4, "boosted_tree": 2}.items():
            add(method, "feature_outliers", points_for_method, f"robust outlier cell fraction {outlier:.2f}")

    target = diagnostics.target.classification
    if target is not None:
        severe_imbalance = (
            target.minority_class_fraction < policy.high_class_imbalance_fraction
            or target.imbalance_ratio >= policy.severe_class_imbalance_ratio
        )
        if severe_imbalance:
            for method, points_for_method in {"linear": -5, "regularized_linear": 1, "tree_ensemble": -2, "boosted_tree": -3}.items():
                add(method, "class_balance", points_for_method, f"minority fraction {target.minority_class_fraction:.2f}; class ratio {target.imbalance_ratio:.1f}")
        if target.minimum_class_size < policy.unstable_class_size:
            for method, points_for_method in {"linear": 0, "regularized_linear": 1, "tree_ensemble": -5, "boosted_tree": -7}.items():
                add(method, "class_support", points_for_method, f"minimum class size {target.minimum_class_size} is small for high-capacity fitting")
    regression = diagnostics.target.regression
    if regression is not None:
        if regression.outlier_fraction >= policy.outlier_moderate_fraction:
            for method, points_for_method in {"linear": -3, "regularized_linear": 1, "tree_ensemble": 3, "boosted_tree": 1}.items():
                add(method, "target_robustness", points_for_method, f"target IQR-outlier fraction {regression.outlier_fraction:.2f}")
        if abs(regression.skewness) >= 2:
            for method, points_for_method in {"linear": -2, "regularized_linear": 0, "tree_ensemble": 1, "boosted_tree": 1}.items():
                add(method, "target_shape", points_for_method, f"target skewness {regression.skewness:.2f} is large")

    if diagnostics.nonlinearity_applicable and diagnostics.rows >= policy.boosted_tree_preferred_samples and signal in {"moderate", "high"}:
        add("boosted_tree", "boosted_tree_scale_signal", 8, "sample size and nonlinear signal support a higher-capacity structured fit")
        add("tree_ensemble", "boosted_tree_scale_signal", 3, "sample size and nonlinear signal also support an ensemble")

    eligibility: dict[Method, tuple[bool, str | None]] = {}
    for method in SUPPORTED_METHOD_ORDER:
        if diagnostics.usable_features == 0:
            eligibility[method] = (False, "no usable feature remains after canonical schema exclusions")
        elif method != "boosted_tree" and diagnostics.estimated_one_hot_dimensionality > MAX_ONE_HOT_FEATURES:
            eligibility[method] = (
                False,
                f"estimated one-hot dimensionality {diagnostics.estimated_one_hot_dimensionality} exceeds safe bound {MAX_ONE_HOT_FEATURES}",
            )
        elif method == "boosted_tree" and diagnostics.boosted_effective_features_estimate > policy.minimum_boosted_effective_features:
            eligibility[method] = (
                False,
                f"estimated ordinal dimensionality {diagnostics.boosted_effective_features_estimate} exceeds boosted-tree bound {policy.minimum_boosted_effective_features}",
            )
        else:
            eligibility[method] = (True, None)

    assessments: dict[Method, DeterministicMethodAssessment] = {}
    for method in SUPPORTED_METHOD_ORDER:
        eligible, reason = eligibility[method]
        bounded_score = max(0, min(100, scores[method]))
        assessments[method] = DeterministicMethodAssessment(
            score=bounded_score if eligible else None,
            eligible=eligible,
            eligibility_reason=reason,
            contributions=contributions[method],
        )
    return assessments
