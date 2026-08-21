"""Canonical preprocessing contracts, requirements, and pipeline builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.deterministic import is_identifier, semantic_type
from app.schemas import Method, PreprocessingContract, TaskType


MAX_CATEGORICAL_CARDINALITY = 80
MAX_ONE_HOT_FEATURES = 4000


@dataclass(frozen=True)
class PreprocessingRequirements:
    """Deterministic evidence for what a contract must do on this schema."""

    expected_contract: PreprocessingContract
    required_steps: tuple[str, ...]
    optional_steps: tuple[str, ...]
    prohibited_steps: tuple[str, ...]
    irrelevant_steps: tuple[str, ...]
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_contract": self.expected_contract.model_dump(mode="json"),
            "required_steps": list(self.required_steps),
            "optional_steps": list(self.optional_steps),
            "prohibited_steps": list(self.prohibited_steps),
            "irrelevant_steps": list(self.irrelevant_steps),
            "evidence": self.evidence,
        }


def requirements_from_records(
    records: Sequence[dict[str, Any]],
    task_type: TaskType | str,
    method: Method | str,
) -> PreprocessingRequirements:
    """Derive the supported preprocessing policy from compact schema records.

    Records with ``used=False`` are excluded from learned preprocessing.  If
    that marker is absent, the same deterministic eligibility policy used by
    validation is applied for recommendation purposes.
    """

    used_records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record in records:
        name = str(record["name"])
        kind = str(record.get("semantic_type", "unknown"))
        unique = int(record.get("unique", 0) or 0)
        high_cardinality = kind in {"categorical", "boolean"} and unique > MAX_CATEGORICAL_CARDINALITY
        eligible = (
            bool(record.get("used", True))
            and not bool(record.get("identifier_like", False))
            and kind not in {"text", "datetime", "unknown"}
            and not high_cardinality
            and unique > 1
        )
        if eligible:
            used_records.append(record)
        else:
            excluded.append(
                {
                    "column": name,
                    "semantic_type": kind,
                    "identifier_like": bool(record.get("identifier_like", False)),
                    "high_cardinality": high_cardinality,
                    "used": False,
                }
            )

    numeric_records = [
        record
        for record in used_records
        if record.get("semantic_type") == "numeric"
        or record.get("dtype", "").startswith(("int", "uint", "float"))
    ]
    categorical_records = [
        record
        for record in used_records
        if record not in numeric_records
        and record.get("semantic_type") in {"categorical", "boolean", "numeric_like"}
    ]
    numeric_missing = int(sum(int(record.get("missing", 0) or 0) for record in numeric_records))
    categorical_missing = int(
        sum(int(record.get("missing", 0) or 0) for record in categorical_records)
    )
    infinity_count = int(sum(int(record.get("infinity", 0) or 0) for record in numeric_records))
    numeric_features = [str(record["name"]) for record in numeric_records]
    categorical_features = [str(record["name"]) for record in categorical_records]

    numeric_imputation = "median" if numeric_missing or infinity_count else "none"
    categorical_imputation = "most_frequent" if categorical_missing else "none"
    linear_method = method in {"linear", "regularized_linear"}
    numeric_scaling = "standard" if numeric_features and linear_method else "none"
    if categorical_features:
        categorical_encoding = "ordinal" if method == "boosted_tree" else "one_hot"
        unknown_handling = (
            "use_encoded_value" if categorical_encoding == "ordinal" else "ignore"
        )
    else:
        categorical_encoding = "none"
        unknown_handling = "ignore"

    expected = PreprocessingContract(
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
        numeric_scaling=numeric_scaling,
        categorical_encoding=categorical_encoding,
        categorical_unknown_handling=unknown_handling,
        identifier_handling="exclude",
        high_cardinality_handling="exclude",
        unsupported_text_handling="exclude",
        datetime_handling="exclude",
        infinity_handling="replace_with_missing",
        fit_inside_pipeline=True,
    )

    required = ["fit_inside_pipeline"]
    optional: list[str] = []
    irrelevant: list[str] = []
    prohibited: list[str] = []
    if numeric_features:
        if numeric_missing or infinity_count:
            required.append("numeric_imputation")
        else:
            optional.append("numeric_imputation")
        if linear_method:
            required.append("numeric_scaling")
        else:
            optional.append("numeric_scaling")
    else:
        irrelevant.extend(["numeric_imputation", "numeric_scaling"])
    if categorical_features:
        required.append("categorical_encoding")
        if categorical_missing:
            required.append("categorical_imputation")
        else:
            optional.append("categorical_imputation")
        required.append("categorical_unknown_handling")
    else:
        irrelevant.extend(["categorical_imputation", "categorical_encoding", "categorical_unknown_handling"])
    if infinity_count:
        required.append("infinity_handling")
    else:
        irrelevant.append("infinity_handling")

    excluded_names = {str(item["column"]) for item in excluded}
    identifier_names = [
        str(record["name"])
        for record in records
        if bool(record.get("identifier_like", False))
    ]
    high_cardinality_names = [
        str(record["name"])
        for record in records
        if record.get("semantic_type") in {"categorical", "boolean"}
        and int(record.get("unique", 0) or 0) > MAX_CATEGORICAL_CARDINALITY
    ]
    text_names = [str(record["name"]) for record in records if record.get("semantic_type") == "text"]
    datetime_names = [
        str(record["name"]) for record in records if record.get("semantic_type") == "datetime"
    ]
    for field_name, names in (
        ("identifier_handling", identifier_names),
        ("high_cardinality_handling", high_cardinality_names),
        ("unsupported_text_handling", text_names),
        ("datetime_handling", datetime_names),
    ):
        if names:
            required.append(field_name)
        else:
            irrelevant.append(field_name)

    estimated_one_hot = int(
        sum(int(record.get("unique", 0) or 0) + 1 for record in categorical_records)
        + len(numeric_records)
    )
    if categorical_encoding == "one_hot" and estimated_one_hot > MAX_ONE_HOT_FEATURES:
        prohibited.append("dense_or_oversized_one_hot_matrix")

    evidence = {
        "task_type": str(task_type),
        "method": str(method),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_missing_values": numeric_missing,
        "categorical_missing_values": categorical_missing,
        "infinity_values": infinity_count,
        "identifier_features": identifier_names,
        "high_cardinality_features": high_cardinality_names,
        "unsupported_text_features": text_names,
        "datetime_features": datetime_names,
        "excluded_features": sorted(excluded_names),
        "estimated_one_hot_features": estimated_one_hot,
        "max_one_hot_features": MAX_ONE_HOT_FEATURES,
    }
    return PreprocessingRequirements(
        expected_contract=expected,
        required_steps=tuple(dict.fromkeys(required)),
        optional_steps=tuple(dict.fromkeys(optional)),
        prohibited_steps=tuple(dict.fromkeys(prohibited)),
        irrelevant_steps=tuple(dict.fromkeys(irrelevant)),
        evidence=evidence,
    )


def material_preprocessing_view(
    contract: PreprocessingContract,
    requirements: PreprocessingRequirements | None = None,
) -> dict[str, Any]:
    """Return only behavior that can matter for this observed schema."""

    values = contract.model_dump(mode="json")
    if requirements is None:
        return values
    for field_name in requirements.irrelevant_steps:
        values.pop(field_name, None)
    for field_name in requirements.optional_steps:
        if field_name in {"numeric_imputation", "categorical_imputation"}:
            values.pop(field_name, None)
    if "numeric_scaling" in requirements.optional_steps and requirements.evidence["method"] in {
        "tree_ensemble",
        "boosted_tree",
    }:
        values.pop("numeric_scaling", None)
    return values


def compare_preprocessing_plans(
    agent: PreprocessingContract,
    deterministic: PreprocessingContract,
    requirements: PreprocessingRequirements | None = None,
) -> dict[str, Any]:
    agent_view = material_preprocessing_view(agent, requirements)
    deterministic_view = material_preprocessing_view(deterministic, requirements)
    differences: list[dict[str, Any]] = []
    immaterial_differences: list[dict[str, Any]] = []
    raw_agent = agent.model_dump(mode="json")
    raw_deterministic = deterministic.model_dump(mode="json")
    for field_name in sorted(set(raw_agent) | set(raw_deterministic)):
        agent_value = raw_agent.get(field_name)
        deterministic_value = raw_deterministic.get(field_name)
        if agent_value == deterministic_value:
            continue
        if field_name not in agent_view and field_name not in deterministic_view:
            reason = (
                "The observed schema makes this a harmless optional modeling preference."
                if requirements and field_name in requirements.optional_steps
                else "The corresponding feature type or observed condition is absent, so this option cannot change execution for this dataset."
            )
            immaterial_differences.append(
                {
                    "field": field_name,
                    "agent": agent_value,
                    "deterministic": deterministic_value,
                    "material": False,
                    "reason": reason,
                }
            )
            continue
        reason = "Executable preprocessing behavior differs."
        if requirements and field_name in requirements.optional_steps:
            reason = "The difference is an optional modeling preference for this schema/model family."
        differences.append(
            {
                "field": field_name,
                "agent": agent_value,
                "deterministic": deterministic_value,
                "material": True,
                "reason": reason,
            }
        )
    return {
        "agent_proposal": agent.model_dump(mode="json"),
        "deterministic_proposal": deterministic.model_dump(mode="json"),
        "agent_normalized": agent_view,
        "deterministic_normalized": deterministic_view,
        "material_differences": differences,
        "immaterial_differences": immaterial_differences,
        "status": "agreement" if not differences else "disagreement",
    }


def build_preprocessor(
    contract: PreprocessingContract,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    method: Method | str,
) -> ColumnTransformer:
    """Build the one canonical executable preprocessing transformer."""

    if method == "boosted_tree" and categorical_features and contract.categorical_encoding != "ordinal":
        raise ValueError("Boosted trees require ordinal categorical encoding in the canonical builder.")
    transformers: list[tuple[str, Pipeline, Sequence[str]]] = []
    if numeric_features:
        numeric_steps: list[tuple[str, Any]] = []
        if contract.numeric_imputation == "median":
            numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        if contract.numeric_scaling == "standard":
            numeric_steps.append(("scale", StandardScaler()))
        numeric_transformer = Pipeline(numeric_steps) if numeric_steps else "passthrough"
        transformers.append(("numeric", numeric_transformer, list(numeric_features)))
    if categorical_features:
        categorical_steps: list[tuple[str, Any]] = []
        if contract.categorical_imputation == "most_frequent":
            categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        if contract.categorical_encoding == "one_hot":
            categorical_steps.append(
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                )
            )
        elif contract.categorical_encoding == "ordinal":
            categorical_steps.append(
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                )
            )
        categorical_transformer = Pipeline(categorical_steps) if categorical_steps else "passthrough"
        transformers.append(("categorical", categorical_transformer, list(categorical_features)))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )


def records_from_dataframe(
    dataframe: pd.DataFrame,
    feature_names: Sequence[str],
    excluded_features: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create compact records for deterministic preprocessing validation."""

    excluded_by_name = {
        str(item.get("column")): item for item in (excluded_features or [])
    }
    records: list[dict[str, Any]] = []
    for column in dataframe.columns:
        name = str(column)
        if name not in feature_names and name not in excluded_by_name:
            continue
        series = dataframe[column]
        infinity = 0
        if pd.api.types.is_numeric_dtype(series):
            infinity = int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum())
        record = {
            "name": name,
            "dtype": str(series.dtype),
            "semantic_type": semantic_type(series),
            "missing": int(series.isna().sum()),
            "infinity": infinity,
            "unique": int(series.nunique(dropna=True)),
            "identifier_like": is_identifier(name, series),
            "used": name in feature_names,
        }
        records.append(record)
    return records
