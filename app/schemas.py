"""Small, explicit schemas shared by the agents and the deterministic engine."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


TaskType = Literal["classification", "regression"]
FormulationStatus = Literal["proposed", "uncertain", "failed"]
Method = Literal["linear", "regularized_linear", "tree_ensemble", "boosted_tree"]
ScoreEffect = Literal["favors", "penalizes", "limits"]
ConfidenceLevel = Literal["low", "medium", "high"]
CleaningAction = Literal[
    "trim_strings",
    "drop_exact_duplicates",
    "drop_all_null_columns",
    "drop_constant_features",
    "drop_rows_missing_target",
    "coerce_numeric_strings",
]
NumericImputation = Literal["median", "none"]
CategoricalImputation = Literal["most_frequent", "none"]
NumericScaling = Literal["standard", "none"]
CategoricalEncoding = Literal["one_hot", "ordinal", "none"]
CategoricalUnknownHandling = Literal["ignore", "use_encoded_value"]
FeatureHandling = Literal["exclude", "retain", "reject"]
InfinityHandling = Literal["replace_with_missing", "reject"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreprocessingContract(StrictModel):
    """The complete, executable learned-preprocessing policy.

    The small legacy alias migration is intentionally allow-listed.  It keeps
    old serialized plans readable while ensuring every new artifact contains
    this typed object rather than a decorative list of strings.
    """

    numeric_imputation: NumericImputation = "median"
    categorical_imputation: CategoricalImputation = "most_frequent"
    numeric_scaling: NumericScaling = "none"
    categorical_encoding: CategoricalEncoding = "one_hot"
    categorical_unknown_handling: CategoricalUnknownHandling = "ignore"
    identifier_handling: FeatureHandling = "exclude"
    high_cardinality_handling: FeatureHandling = "exclude"
    unsupported_text_handling: FeatureHandling = "exclude"
    datetime_handling: FeatureHandling = "exclude"
    infinity_handling: InfinityHandling = "replace_with_missing"
    fit_inside_pipeline: StrictBool = True

    @model_validator(mode="before")
    @classmethod
    def migrate_allowlisted_legacy_aliases(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        aliases = {
            "training_only_imputation": {
                "numeric_imputation": "median",
                "categorical_imputation": "most_frequent",
            },
            "scale_numeric_features": {"numeric_scaling": "standard"},
            "one_hot_encode_categories": {
                "categorical_encoding": "one_hot",
                "categorical_unknown_handling": "ignore",
            },
            "schema_aware_encoding": {
                "categorical_encoding": "one_hot",
                "categorical_unknown_handling": "ignore",
            },
            "ordinal_encode_categories": {
                "categorical_encoding": "ordinal",
                "categorical_unknown_handling": "use_encoded_value",
            },
            "ignore_high_cardinality_identifiers": {
                "identifier_handling": "exclude",
                "high_cardinality_handling": "exclude",
            },
            "replace_infinity_with_missing": {"infinity_handling": "replace_with_missing"},
        }
        migrated: dict[str, object] = {}
        for alias in value:
            if not isinstance(alias, str) or alias not in aliases:
                raise ValueError(f"Unsupported legacy preprocessing option: {alias!r}")
            migrated.update(aliases[alias])
        return migrated

    @model_validator(mode="after")
    def validate_executable_encoding(self) -> "PreprocessingContract":
        if self.categorical_encoding == "one_hot" and self.categorical_unknown_handling != "ignore":
            raise ValueError("one_hot encoding requires categorical_unknown_handling='ignore'.")
        if self.categorical_encoding == "ordinal" and self.categorical_unknown_handling != "use_encoded_value":
            raise ValueError(
                "ordinal encoding requires categorical_unknown_handling='use_encoded_value'."
            )
        if self.categorical_encoding == "none" and self.categorical_unknown_handling != "ignore":
            raise ValueError("categorical encoding='none' requires categorical_unknown_handling='ignore'.")
        if not self.fit_inside_pipeline:
            raise ValueError("All learned preprocessing must be fitted inside the training pipeline.")
        return self


class FormulationPlan(StrictModel):
    """One independent proposal for the supervised problem definition."""

    target_column: str
    task_type: TaskType
    reasoning: str = Field(min_length=20, max_length=1200)
    confidence: float = Field(ge=0, le=1)


class DeterministicFormulation(StrictModel):
    """Auditable, non-ML evidence for target and task formulation."""

    target_column: Optional[str] = None
    task_type: Optional[TaskType] = None
    status: FormulationStatus = "proposed"
    reasoning: str = Field(min_length=20, max_length=1600)
    evidence: list[str] = Field(default_factory=list, max_length=16)
    confidence: float = Field(ge=0, le=1)
    target_source: Literal["user_supplied", "inferred", "uncertain"] = "inferred"


class FormulationComparison(StrictModel):
    target_agreement: bool
    task_agreement: bool
    overall_agreement: bool
    status: Literal["agreement", "disagreement", "invalid"]
    differences: list[str] = Field(default_factory=list, max_length=8)


class FormulationResolution(StrictModel):
    selected_target_column: str
    selected_task_type: TaskType
    checks: list[str] = Field(min_length=1, max_length=12)
    justification: str = Field(min_length=20, max_length=1600)
    confidence: float = Field(ge=0, le=1)


class ModelingPlan(StrictModel):
    """Post-split model-family proposal; target/task are immutable context."""

    recommended_method: Method
    preprocessing: PreprocessingContract = Field(default_factory=PreprocessingContract)
    reasoning: str = Field(min_length=20, max_length=1200)
    confidence: float = Field(ge=0, le=1)


class ModelingResolution(StrictModel):
    selected_method: Method
    selected_preprocessing: PreprocessingContract = Field(default_factory=PreprocessingContract)
    checks: list[str] = Field(min_length=1, max_length=12)
    justification: str = Field(min_length=20, max_length=1600)
    confidence: float = Field(ge=0, le=1)


class HardValidationArtifact(StrictModel):
    """Machine-readable safety/executability result for a modeling gate.

    Hard validation is deliberately represented separately from model-family
    comparison.  The nested proposal results make it possible to audit which
    plan failed an invariant without treating a challenger disagreement as a
    validation failure.
    """

    status: Literal["passed", "failed", "unavailable"]
    intervention_required: StrictBool
    initial_hard_invalid: StrictBool = False
    checks: list[dict[str, Any]] = Field(default_factory=list)
    initial_proposal: dict[str, Any] = Field(default_factory=dict)
    deterministic_challenger: dict[str, Any] = Field(default_factory=dict)
    final_plan: dict[str, Any] = Field(default_factory=dict)


class SoftChallengeArtifact(StrictModel):
    """Advisory deterministic model-family comparison metadata."""

    status: Literal["agreement", "disagreement", "invalid", "unavailable"]
    agent_method: Optional[Method] = None
    deterministic_method: Optional[Method] = None
    deterministic_confidence: Optional[ConfidenceLevel] = None
    method_disagreement: StrictBool = False
    preprocessing_disagreement: StrictBool = False
    reconciliation_invoked: StrictBool = False
    reconciliation_status: str = "not_invoked"
    reconciliation_method_source: Optional[str] = None
    reconciliation_preprocessing_source: Optional[str] = None
    scores_are_probabilities: StrictBool = False
    scores_are_cross_validation_results: StrictBool = False
    scores_are_empirical_performance_estimates: StrictBool = False


class ModelingGateArtifact(StrictModel):
    """Explicit hard-vs-soft-vs-final modeling decision artifact."""

    hard_validation: HardValidationArtifact
    soft_challenge: SoftChallengeArtifact
    final: dict[str, Any] = Field(default_factory=dict)


class AgentPlan(StrictModel):
    """Legacy combined plan retained for evaluation and API compatibility.

    Runtime production orchestration uses FormulationPlan plus ModelingPlan.
    """

    target_column: str = Field(description="The outcome column to predict.")
    task_type: TaskType
    recommended_method: Method
    preprocessing: PreprocessingContract = Field(default_factory=PreprocessingContract)
    reasoning: str = Field(min_length=20, max_length=1200)
    confidence: float = Field(ge=0, le=1)


class ClassificationTargetDiagnostics(StrictModel):
    classes: int = Field(ge=0)
    minority_class_fraction: float = Field(ge=0, le=1)
    majority_class_fraction: float = Field(ge=0, le=1)
    imbalance_ratio: float = Field(ge=0)
    # Keys are stable size ranks (class_1 is the largest class), not source
    # labels.  Structural diagnostics should not persist arbitrary label names.
    samples_per_class: dict[str, int] = Field(default_factory=dict)
    minimum_class_size: int = Field(ge=0)


class RegressionTargetDiagnostics(StrictModel):
    variance: float = Field(ge=0)
    skewness: float
    outlier_fraction: float = Field(ge=0, le=1)
    heavy_tail_signal: Literal["low", "moderate", "high"]


class TargetDiagnostics(StrictModel):
    classification: Optional[ClassificationTargetDiagnostics] = None
    regression: Optional[RegressionTargetDiagnostics] = None


class DeterministicDiagnostics(StrictModel):
    """Compact, training-only facts used by the deterministic policy."""

    rows: int = Field(ge=0)
    usable_features: int = Field(ge=0)
    excluded_features: int = Field(ge=0)
    excluded_feature_types: dict[str, int] = Field(default_factory=dict)
    numeric_feature_count: int = Field(ge=0)
    categorical_feature_count: int = Field(ge=0)
    binary_feature_count: int = Field(ge=0)
    text_feature_count: int = Field(ge=0)
    sample_to_feature_ratio: float = Field(ge=0)
    effective_features_estimate: int = Field(ge=0)
    linear_effective_features_estimate: int = Field(ge=0)
    tree_effective_features_estimate: int = Field(ge=0)
    boosted_effective_features_estimate: int = Field(ge=0)
    overall_missing_fraction: float = Field(ge=0, le=1)
    max_feature_missing_fraction: float = Field(ge=0, le=1)
    features_with_missing_count: int = Field(ge=0)
    features_with_missing_fraction: float = Field(ge=0, le=1)
    missingness_pattern: Literal["none", "concentrated", "widespread"]
    mean_categorical_cardinality: float = Field(ge=0)
    max_categorical_cardinality: int = Field(ge=0)
    estimated_one_hot_dimensionality: int = Field(ge=0)
    high_cardinality_feature_count: int = Field(ge=0)
    high_cardinality_feature_fraction: float = Field(ge=0, le=1)
    max_abs_numeric_correlation: float = Field(ge=0, le=1)
    high_correlation_pair_count: int = Field(ge=0)
    high_correlation_pair_fraction: float = Field(ge=0, le=1)
    pearson_spearman_gap: float = Field(ge=0, le=1)
    mean_univariate_signal: float = Field(ge=0, le=1)
    nonlinearity_score: float = Field(ge=0, le=1)
    nonlinearity_signal: Literal["low", "moderate", "high"]
    nonlinear_feature_count: int = Field(ge=0)
    nonlinear_feature_fraction: float = Field(ge=0, le=1)
    nonlinearity_heterogeneity: float = Field(ge=0, le=1)
    structural_complexity_score: float = Field(ge=0, le=1)
    structural_complexity_signal: Literal["low", "moderate", "high"]
    numeric_outlier_feature_fraction: float = Field(ge=0, le=1)
    numeric_outlier_cell_fraction: float = Field(ge=0, le=1)
    target: TargetDiagnostics
    marginal_association_strength: float = Field(default=0.0, ge=0, le=1)
    class_separation_strength: float = Field(default=0.0, ge=0, le=1)
    association_measure: str = Field(default="unspecified", min_length=1)
    nonlinearity_applicable: bool = True


class DeterministicScoreContribution(StrictModel):
    factor: str
    effect: ScoreEffect
    method: Method
    points: int
    observation: str


class DeterministicMethodAssessment(StrictModel):
    score: Optional[int] = Field(default=None, ge=0, le=100)
    eligible: StrictBool
    eligibility_reason: Optional[str] = None
    contributions: list[DeterministicScoreContribution] = Field(default_factory=list, max_length=32)


class DeterministicRecommendation(StrictModel):
    target_column: str
    task_type: TaskType
    recommended_method: Method
    preprocessing: PreprocessingContract = Field(default_factory=PreprocessingContract)
    reasoning: str = Field(min_length=20, max_length=1200)
    evidence: list[str] = Field(min_length=1, max_length=12)
    policy_version: str = "4"
    method_scores: dict[Method, Optional[float]] = Field(default_factory=dict)
    ranked_methods: list[Method] = Field(default_factory=list, max_length=4)
    method_assessments: dict[Method, DeterministicMethodAssessment] = Field(default_factory=dict)
    diagnostics: Optional[DeterministicDiagnostics] = None
    top_score: Optional[float] = Field(default=None, ge=0, le=100)
    runner_up_score: Optional[float] = Field(default=None, ge=0, le=100)
    score_margin: Optional[float] = Field(default=None, ge=0, le=100)
    confidence: ConfidenceLevel = "low"


class ConflictResolution(StrictModel):
    selected_target_column: str
    selected_task_type: TaskType
    selected_method: Method
    selected_preprocessing: PreprocessingContract = Field(default_factory=PreprocessingContract)
    checks: list[str] = Field(min_length=1, max_length=8)
    justification: str = Field(min_length=20, max_length=1600)
    confidence: float = Field(ge=0, le=1)


class CleaningPlan(StrictModel):
    actions: list[CleaningAction] = Field(max_length=6)
    reasoning: str = Field(min_length=10, max_length=1000)


class CleaningSpecification(StrictModel):
    """Frozen structural-cleaning decisions fitted from training rows only."""

    schema_version: Literal["1"] = "1"
    target_column: str
    row_position_column: Optional[str] = None
    requested_actions: list[CleaningAction] = Field(max_length=6)
    trim_columns: list[str] = Field(default_factory=list)
    numeric_coercion_columns: list[str] = Field(default_factory=list)
    all_null_columns: list[str] = Field(default_factory=list)
    constant_columns: list[str] = Field(default_factory=list)
    drop_rows_missing_target: bool = False
    duplicate_policy: Literal["within_partition_keep_first"] = "within_partition_keep_first"
    training_duplicate_row_positions: list[int] = Field(default_factory=list)
    training_only_evidence: dict[str, Any] = Field(default_factory=dict)


class ReportDraft(StrictModel):
    executive_summary: str = Field(min_length=30, max_length=1800)
    key_findings: list[str] = Field(min_length=1, max_length=8)
    modeling_interpretation: str = Field(min_length=20, max_length=1600)
    limitations: list[str] = Field(min_length=1, max_length=8)
    next_steps: list[str] = Field(min_length=1, max_length=8)
