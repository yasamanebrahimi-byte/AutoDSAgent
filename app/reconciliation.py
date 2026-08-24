"""Source-neutral inputs and bookkeeping for modeling reconciliation.

The deterministic recommender remains fully persisted for audit and policy
calibration.  This module defines the narrower view that is safe to send to a
reconciler: shared training-data evidence plus two identically shaped,
source-blinded proposals.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


BLINDED_RECONCILIATION_MODE = "blinded_evidence_comparison"
BLINDED_RECONCILIATION_PROMPT_VERSION = "2026-08-24.blinded-evidence-comparison.v1"
LEGACY_RECONCILIATION_PROMPT_VERSION = "legacy.reconciliation.v1"
ProposalSource = Literal["agent", "deterministic"]
ProposalLabel = Literal["A", "B"]


@dataclass(frozen=True)
class BlindedReconciliation:
    """Prepared prompt payload plus private source/order metadata."""

    payload: dict[str, Any]
    proposal_order_seed: int
    proposal_a_source: ProposalSource
    proposal_b_source: ProposalSource

    def source_for(self, proposal: str | None) -> ProposalSource | None:
        if proposal == "A":
            return self.proposal_a_source
        if proposal == "B":
            return self.proposal_b_source
        return None

    def proposal_for_source(self, source: ProposalSource) -> ProposalLabel:
        return "A" if self.proposal_a_source == source else "B"


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {}


def _diagnostics(value: Any) -> dict[str, Any]:
    raw = _dump(value)
    diagnostics = raw.get("diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


def _band(value: Any, *, low: float = 0.20, high: float = 0.60) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if numeric < low:
        return "low"
    if numeric < high:
        return "moderate"
    return "high"


def _neutralize_source_words(value: str) -> str:
    """Prevent source identity in free-form initial rationale from leaking."""

    return re.sub(
        r"\b(agent|llm|openai|deterministic|challenger|validator|recommender)\b",
        "proposal",
        value,
        flags=re.IGNORECASE,
    ).strip()


def _shared_dataset_evidence(
    profile: dict[str, Any],
    deterministic: Any,
    target_column: str | None,
    task_type: str | None,
) -> dict[str, Any]:
    diagnostics = _diagnostics(deterministic)
    details = profile.get("column_details", [])
    raw_feature_count = max(
        0,
        len(details) - (1 if target_column and any(
            str(item.get("name")) == target_column for item in details if isinstance(item, dict)
        ) else 0),
    )
    rows = diagnostics.get("training_row_count", diagnostics.get("rows", profile.get("rows", 0)))
    usable_features = diagnostics.get("usable_features", raw_feature_count)
    feature_count = diagnostics.get("usable_features", raw_feature_count)
    if diagnostics.get("excluded_features") is not None:
        feature_count = int(diagnostics.get("usable_features", 0)) + int(
            diagnostics.get("excluded_features", 0)
        )

    classification_boundary = diagnostics.get("classification_boundary_signals") or {}
    interaction = diagnostics.get("interaction_signals") or {}
    target = diagnostics.get("target") or {}
    classification_target = target.get("classification") or {}
    regression_target = target.get("regression") or {}
    missing_fraction = diagnostics.get("overall_missing_fraction")
    correlation = diagnostics.get("max_abs_numeric_correlation")
    structural = diagnostics.get("structural_complexity_score")

    evidence: dict[str, Any] = {
        "target_column": target_column,
        "task_type": task_type,
        "training_rows": int(rows) if rows is not None else None,
        "feature_count": int(feature_count),
        "usable_feature_count": int(usable_features),
        "numeric_feature_count": diagnostics.get("numeric_feature_count"),
        "categorical_feature_count": diagnostics.get("categorical_feature_count"),
        "sample_to_feature_ratio": diagnostics.get("sample_to_feature_ratio"),
        "missingness": {
            "overall": _band(missing_fraction),
            "pattern": diagnostics.get("missingness_pattern", "unavailable"),
            "features_with_missing": diagnostics.get("features_with_missing_count"),
        },
        "outlier_burden": _band(diagnostics.get("numeric_outlier_cell_fraction")),
        "multicollinearity": _band(correlation, low=0.50, high=0.85),
        "marginal_association": _band(diagnostics.get("marginal_association_strength")),
        "nonlinearity": {
            "signal": diagnostics.get("nonlinearity_signal", "unavailable"),
            "feature_fraction": diagnostics.get("nonlinear_feature_fraction"),
            "heterogeneity": _band(diagnostics.get("nonlinearity_heterogeneity")),
        },
        "structural_complexity": diagnostics.get("structural_complexity_signal", _band(structural)),
        "interaction_evidence": interaction.get("interaction_strength", "not_applicable"),
        "classification_boundary_complexity": classification_boundary.get(
            "boundary_complexity", "not_applicable"
        ),
        "target_balance_or_shape": {
            "classes": classification_target.get("classes"),
            "minority_fraction": classification_target.get("minority_class_fraction"),
            "imbalance": _band(
                1.0 - float(classification_target.get("minority_class_fraction", 0.5))
                if classification_target.get("minority_class_fraction") is not None
                else None
            ) if task_type == "classification" else None,
            "heavy_tail": regression_target.get("heavy_tail_signal")
            if task_type == "regression" else None,
        },
    }

    statements = [
        f"The frozen training partition contains {evidence['training_rows']} rows and {evidence['feature_count']} total features.",
        f"The usable-feature count is {evidence['usable_feature_count']}, with a sample-to-feature ratio of {evidence['sample_to_feature_ratio']!s}.",
        f"Observed missingness is {evidence['missingness']['overall']} and its pattern is {evidence['missingness']['pattern']}.",
        f"Numeric outlier burden is {evidence['outlier_burden']}; multicollinearity is {evidence['multicollinearity']}.",
        f"Marginal association is {evidence['marginal_association']}; nonlinear structure is {evidence['nonlinearity']['signal']}.",
        f"Structural complexity is {evidence['structural_complexity']}; interaction evidence is {evidence['interaction_evidence']}.",
    ]
    if task_type == "classification":
        statements.append(
            "Classification boundary complexity is "
            f"{evidence['classification_boundary_complexity']} and class balance is "
            f"{evidence['target_balance_or_shape']['imbalance']}."
        )
    else:
        statements.append(
            "Regression target shape shows "
            f"{evidence['target_balance_or_shape']['heavy_tail'] or 'unavailable'} heavy-tail evidence."
        )
    evidence["observed_statements"] = statements
    # Keep this object explicitly source-neutral.  The recommendation itself
    # is intentionally not copied here, and compatibility scores/rankings are
    # never part of this prompt evidence.
    return evidence


def _method_interpretation(method: str, evidence: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    nonlinear = evidence.get("nonlinearity", {}).get("signal")
    structural = evidence.get("structural_complexity")
    ratio = evidence.get("sample_to_feature_ratio")
    risks: list[str] = []
    support: list[str] = []
    assumptions: list[str] = []
    if method in {"linear", "regularized_linear"}:
        assumptions.append("The relationship can be represented adequately by a linear decision or response surface after preprocessing.")
        if nonlinear in {"low", "unavailable"} and structural in {"low", "unavailable"}:
            support.append("The shared diagnostics do not show strong structural pressure for nonlinear capacity.")
        else:
            risks.append("Observed nonlinear or structurally complex evidence may be underrepresented by a linear surface.")
        if method == "regularized_linear":
            support.append("Regularization can reduce sensitivity to redundant or weak features when the feature geometry is difficult.")
    else:
        assumptions.append("The available sample size and feature geometry support a flexible interaction-capable model.")
        if nonlinear in {"moderate", "high"} or structural in {"moderate", "high"}:
            support.append("The shared diagnostics contain nonlinear or structurally complex evidence that flexible boundaries can represent.")
        else:
            risks.append("Flexible capacity may be unnecessary or unstable when structural evidence is weak relative to the sample size.")
        if ratio is not None:
            try:
                if float(ratio) < 8:
                    risks.append("The sample-to-feature ratio is limited, so flexible capacity may require stability checks.")
            except (TypeError, ValueError):
                pass
    return support[:4], risks[:4], assumptions[:3]


def _proposal(
    source: ProposalSource,
    plan: Any,
    dataset_evidence: dict[str, Any],
) -> dict[str, Any]:
    values = _dump(plan)
    method = values.get("recommended_method")
    reasoning = _neutralize_source_words(str(values.get("reasoning", "")).strip())
    if source == "agent":
        support: list[str] = []
        risks: list[str] = []
        assumptions: list[str] = []
        rationale = [reasoning] if reasoning else []
    else:
        support, risks, assumptions = _method_interpretation(method, dataset_evidence)
        rationale = [
            "This proposal interprets the shared training-data evidence in relation to its selected model family."
        ]
    return {
        "model_family": method,
        "preprocessing": values.get("preprocessing", {}),
        "rationale": rationale,
        "supporting_evidence": support,
        "assumptions": assumptions,
        "risks": risks,
    }


def _blind_preprocessing_differences(
    comparison: dict[str, Any] | None,
    proposal_a_source: ProposalSource,
) -> list[dict[str, Any]]:
    if not comparison:
        return []
    differences: list[dict[str, Any]] = []
    source_for_a = proposal_a_source
    source_for_b: ProposalSource = "deterministic" if source_for_a == "agent" else "agent"
    for item in [
        *(comparison.get("material_differences") or []),
        *(comparison.get("immaterial_differences") or []),
    ]:
        values = {
            "agent": item.get("agent"),
            "deterministic": item.get("deterministic"),
        }
        differences.append({
            "field": item.get("field"),
            "proposal_a_value": values[source_for_a],
            "proposal_b_value": values[source_for_b],
            "material": bool(item.get("material")),
            "reason": item.get("reason"),
        })
    return differences


def _stable_order_seed(
    base_seed: int,
    profile: dict[str, Any],
    target_column: str | None,
    task_type: str | None,
) -> int:
    stable_profile = {
        "rows": profile.get("rows"),
        "columns": profile.get("columns"),
        "column_details": [
            {
                key: item.get(key)
                for key in ("name", "dtype", "semantic_type", "missing", "unique", "identifier_like")
            }
            for item in profile.get("column_details", [])
            if isinstance(item, dict)
        ],
    }
    raw = json.dumps(
        {
            "base_seed": int(base_seed),
            "target_column": target_column,
            "task_type": task_type,
            "profile": stable_profile,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 2_147_483_647


def build_blinded_reconciliation(
    profile: dict[str, Any],
    agent_plan: Any,
    deterministic: Any,
    *,
    target_column: str | None = None,
    task_type: str | None = None,
    preprocessing_comparison: dict[str, Any] | None = None,
    preprocessing_requirements: dict[str, Any] | None = None,
    hard_validation: dict[str, Any] | None = None,
    order_seed: int = 0,
    proposal_order: tuple[ProposalSource, ProposalSource] | None = None,
) -> BlindedReconciliation:
    agent_values = _dump(agent_plan)
    deterministic_values = _dump(deterministic)
    target_column = target_column or agent_values.get("target_column") or deterministic_values.get("target_column")
    task_type = task_type or agent_values.get("task_type") or deterministic_values.get("task_type")
    dataset_evidence = _shared_dataset_evidence(
        profile,
        deterministic,
        target_column,
        task_type,
    )
    resolved_seed = _stable_order_seed(order_seed, profile, target_column, task_type)
    sources: list[ProposalSource] = list(proposal_order or ("agent", "deterministic"))
    if sorted(sources) != ["agent", "deterministic"]:
        raise ValueError("proposal_order must contain agent and deterministic exactly once.")
    if proposal_order is None:
        random.Random(resolved_seed).shuffle(sources)
    proposal_a_source, proposal_b_source = sources
    proposals = {
        "agent": _proposal("agent", agent_plan, dataset_evidence),
        "deterministic": _proposal("deterministic", deterministic, dataset_evidence),
    }
    payload = {
        "reconciliation_mode": BLINDED_RECONCILIATION_MODE,
        # Stable aggregate aliases preserve the legacy helper contract for
        # callers/tests that inspect the profile envelope; no raw columns or
        # row values are copied into the prompt.
        "rows": dataset_evidence.get("training_rows"),
        "columns": dataset_evidence.get("feature_count"),
        "approved_context": {
            "target_column": target_column,
            "task_type": task_type,
        },
        "dataset_evidence": dataset_evidence,
        "proposal_a": proposals[proposal_a_source],
        "proposal_b": proposals[proposal_b_source],
        "shared_preprocessing_requirements": preprocessing_requirements,
        "observed_preprocessing_differences": _blind_preprocessing_differences(
            preprocessing_comparison,
            proposal_a_source,
        ),
        "proposal_hard_validation": {
            "proposal_a": (hard_validation or {}).get(proposal_a_source, {}),
            "proposal_b": (hard_validation or {}).get(proposal_b_source, {}),
        },
    }
    return BlindedReconciliation(
        payload=payload,
        proposal_order_seed=resolved_seed,
        proposal_a_source=proposal_a_source,
        proposal_b_source=proposal_b_source,
    )


def infer_selected_proposal(
    resolution: Any,
    blinded: BlindedReconciliation,
) -> tuple[ProposalLabel | None, ProposalSource | None]:
    """Map an A/B result to its source, retaining compatibility with old mocks."""

    values = _dump(resolution)
    explicit = values.get("selected_proposal")
    if explicit in {"A", "B"}:
        expected_source = blinded.source_for(explicit)
        expected_method = blinded.payload[f"proposal_{explicit.lower()}"]["model_family"]
        if values.get("selected_method") != expected_method:
            raise ValueError(
                "The selected proposal label and selected model family disagree."
            )
        return explicit, expected_source

    selected_method = values.get("selected_method")
    selected_preprocessing = values.get("selected_preprocessing")
    for label in ("A", "B"):
        proposal = blinded.payload[f"proposal_{label.lower()}"]
        if selected_method != proposal.get("model_family"):
            continue
        if selected_preprocessing is None or selected_preprocessing == proposal.get("preprocessing"):
            return label, blinded.source_for(label)
    return None, None
