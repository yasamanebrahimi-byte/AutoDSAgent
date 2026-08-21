"""Small, explicit schemas shared by the agents and the deterministic engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


TaskType = Literal["classification", "regression"]
Method = Literal["linear", "regularized_linear", "tree_ensemble", "boosted_tree"]
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


class AgentPlan(StrictModel):
    target_column: str = Field(description="The outcome column to predict.")
    task_type: TaskType
    recommended_method: Method
    preprocessing: PreprocessingContract = Field(default_factory=PreprocessingContract)
    reasoning: str = Field(min_length=20, max_length=1200)
    confidence: float = Field(ge=0, le=1)


class DeterministicRecommendation(StrictModel):
    target_column: str
    task_type: TaskType
    recommended_method: Method
    preprocessing: PreprocessingContract = Field(default_factory=PreprocessingContract)
    reasoning: str = Field(min_length=20, max_length=1200)
    evidence: list[str] = Field(min_length=1, max_length=8)


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


class ReportDraft(StrictModel):
    executive_summary: str = Field(min_length=30, max_length=1800)
    key_findings: list[str] = Field(min_length=1, max_length=8)
    modeling_interpretation: str = Field(min_length=20, max_length=1600)
    limitations: list[str] = Field(min_length=1, max_length=8)
    next_steps: list[str] = Field(min_length=1, max_length=8)
