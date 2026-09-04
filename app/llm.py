"""OpenAI-backed specialist agents with strict structured outputs.

The API is optional at import time so deterministic/offline runs remain useful.
When configured, every semantic decision is returned through a Pydantic schema
and the raw prompt/response is intentionally not persisted in the run folder.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from app.schemas import (
    CleaningPlan,
    FormulationPlan,
    FormulationResolution,
    ModelingPlan,
    ModelingResolution,
    ReportDraft,
    StrictModel,
)
from app.reconciliation import (
    BLINDED_RECONCILIATION_MODE,
    build_blinded_reconciliation,
)


# Bump this when the modeling/reconciliation input contract changes.  The
# evaluation harness records it beside every trial so a result bundle can be
# interpreted without preserving provider-specific request metadata.
PROMPT_SCHEMA_VERSION = "2026-08-24.blinded-evidence-comparison.v3-empirical-probe"


T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised when an OpenAI-backed agent cannot be used."""


@dataclass
class OpenAIAgents:
    """Focused modeling, cleaning, validation, EDA, and report agent roles."""

    api_key: str | None = None
    model: str = "gpt-4.1-mini"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", self.model)
        self._client: Any | None = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _client_or_raise(self) -> Any:
        if not self.api_key:
            raise LLMUnavailable("OPENAI_API_KEY is not configured.")
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise LLMUnavailable(
                    "The OpenAI package is not installed. Install the project dependencies."
                ) from exc
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _structured(
        self,
        schema_name: str,
        schema: type[T],
        instructions: str,
        payload: dict[str, Any],
    ) -> T:
        client = self._client_or_raise()
        prompt = f"{instructions}\n\nINPUT JSON:\n{json.dumps(payload, default=str)}"
        response = client.responses.parse(
            model=self.model,
            input=prompt,
            store=False,
            text_format=schema,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            return parsed
        output_text = getattr(response, "output_text", "")
        if not output_text:
            raise LLMUnavailable(f"The {schema_name} agent returned no structured output.")
        try:
            return schema.model_validate(json.loads(output_text))
        except Exception as exc:
            raise LLMUnavailable(f"The {schema_name} agent returned invalid structured output.") from exc

    def formulate_problem(
        self,
        profile: dict[str, Any],
        question: str,
        target_constraint: dict[str, Any] | None = None,
    ) -> FormulationPlan:
        """Propose only target and task from pre-split formulation evidence."""

        return self._structured(
            "formulation_agent_plan",
            FormulationPlan,
            """You are the independent problem-formulation agent. Infer the supervised
prediction target and choose only classification or regression. Reason only from
the user question, the compact raw-data schema profile, and an explicit target
constraint when supplied. Do not assume another recommender exists and do not
select a model family, preprocessing, cleaning, holdout, or future empirical
reference. If a target constraint is supplied, use that exact existing column and
never substitute another target; infer the task type independently.""",
            {
                "question": question,
                "target_constraint": target_constraint,
                "formulation_profile": profile,
            },
        )

    def modeling_plan(
        self,
        profile: dict[str, Any],
        question: str,
        target_hint: str | None,
        task_type: str | None = None,
    ) -> ModelingPlan:
        return self._structured(
            "modeling_agent_plan",
            ModelingPlan,
            """You are the independent post-formulation modeling agent. The approved
target and task are immutable context from an earlier formulation gate. Do not
re-select, confirm, or return target/task fields. Independently choose only the
model family and complete typed preprocessing contract from the training-only
profile. Do not assume that a deterministic recommender exists. Keep structural
cleaning separate and keep learned transformations inside the training pipeline.
The method vocabulary is: linear, regularized_linear, tree_ensemble, boosted_tree.
Use only these executable categorical preprocessing pairs: one_hot with
categorical_unknown_handling='ignore'; ordinal with
categorical_unknown_handling='use_encoded_value'; or none with
categorical_unknown_handling='ignore'. Do not return any other pairing.""",
            {
                "question": question,
                "approved_formulation": {
                    "target_column": target_hint or "not provided",
                    "task_type": task_type or "not provided",
                },
                "training_only_profile": profile,
            },
        )

    def reconcile_formulation(
        self,
        question: str,
        profile: dict[str, Any],
        user_target_constraint: dict[str, Any] | None,
        agent_formulation: FormulationPlan,
        deterministic_formulation: dict[str, Any],
    ) -> FormulationResolution:
        return self._structured(
            "formulation_resolution",
            FormulationResolution,
            """You are the dedicated formulation reconciliation agent. Investigate
only the disagreement between two pre-split target/task proposals using the user
question, compact formulation profile, explicit user target constraint, and the
recorded proposal evidence. Select only classification or regression and, when no
user target is fixed, select one of the proposed targets. When a user target is
fixed, it is a hard invariant and must be returned exactly. Do not select a model
family or preprocessing. Explain the disagreement and the evidence used.""",
            {
                "question": question,
                "formulation_profile": profile,
                "user_target_constraint": user_target_constraint,
                "agent_formulation": agent_formulation.model_dump(mode="json"),
                "deterministic_formulation": deterministic_formulation,
            },
        )

    def reconcile_modeling(
        self,
        question: str,
        profile: dict[str, Any],
        modeling_plan: ModelingPlan,
        deterministic: dict[str, Any],
    ) -> ModelingResolution:
        if isinstance(deterministic.get("_blinded_reconciliation_payload"), dict):
            blinded_payload = deterministic["_blinded_reconciliation_payload"]
        elif profile.get("reconciliation_mode") == BLINDED_RECONCILIATION_MODE:
            blinded_payload = profile
        else:
            blinded_payload = build_blinded_reconciliation(
                profile,
                modeling_plan,
                deterministic,
                target_column=deterministic.get("target_column"),
                task_type=deterministic.get("task_type"),
                preprocessing_comparison=deterministic.get("preprocessing_comparison"),
                preprocessing_requirements=deterministic.get("preprocessing_requirements"),
                hard_validation={
                    "agent": (
                        (deterministic.get("hard_validation") or {}).get("agent")
                        or (deterministic.get("hard_validation") or {}).get("agent_proposal")
                        or {}
                    ),
                    "deterministic": (
                        (deterministic.get("hard_validation") or {}).get("deterministic")
                        or (deterministic.get("hard_validation") or {}).get("deterministic_challenger")
                        or {}
                    ),
                },
                order_seed=int(deterministic.get("_reconciliation_order_seed") or 0),
                proposal_order=(
                    tuple(deterministic["_reconciliation_proposal_order"])
                    if deterministic.get("_reconciliation_proposal_order")
                    else None
                ),
            ).payload
        return self._structured(
            "modeling_resolution",
            ModelingResolution,
            """You are comparing two independently generated modeling proposals.
            They are deliberately presented as Proposal A and Proposal B; do not infer,
            mention, or favor their origins. Target and task are immutable approved context.
            Choose exactly one of Proposal A or Proposal B, return its model family and a
            complete supported preprocessing contract, and never invent Proposal C.

            First provide a concise, two-sided critique: strengths and weaknesses for A,
            strengths and weaknesses for B, including the strongest case against each.
            Then list the decisive observed evidence and select A or B. The output must be
            methodological justification, not hidden chain-of-thought. If evidence is close,
            still select one proposal and prefer the one whose assumptions are less fragile
            and whose complexity is more proportional to the observed evidence; do not use a
            universal simplicity or complexity rule.

            Distinguish observed dataset evidence from each proposal's interpretation. The
            compatibility diagnostics in the input are heuristic structural evidence only.
            They are not probabilities, cross-validation results, empirical performance,
            expected accuracy/RMSE, or proof that either proposal is better. When present,
            the section named LIMITED TRAINING-ONLY EMPIRICAL COMPARISON is a small directional
            comparison of only Proposal A and Proposal B using training-side folds. It is not
            final holdout performance or a guarantee of future generalization; fold variability
            matters, and preprocessing was fitted inside each fold. Weigh a strong, consistent
            comparison more heavily than heuristic point scores, but do not treat its winner as
            an automatic final decision. Do not use holdout values, empirical-reference
            rankings, or calibration reliability. Hard validation outcomes describe
            safety constraints, not comparative predictive quality. Re-check preprocessing,
            leakage, immutable context, supported methods, and the complete contract before
            returning the selected plan.""",
            {
                "question": question,
                **blinded_payload,
            },
        )

    def cleaning(self, profile: dict[str, Any], target_column: str) -> CleaningPlan:
        return self._structured(
            "cleaning_plan",
            CleaningPlan,
            """You are the data cleaning agent. Select only safe, structural actions
from the allowed list. Do not invent transformations, delete the target, or
impute learned values outside a modeling pipeline. Use an empty action list if
nothing is warranted. Allowed actions: trim_strings, drop_exact_duplicates,
drop_all_null_columns, drop_constant_features, drop_rows_missing_target,
coerce_numeric_strings.""",
            {"target_column": target_column, "profile": profile},
        )

    def eda(self, question: str, summary: dict[str, Any]) -> list[str]:
        class EDAOutput(StrictModel):
            findings: list[str]

        output = self._structured(
            "eda_findings",
            EDAOutput,
            """You are the EDA agent. Turn the computed summary into three to five
specific, cautious findings relevant to the question. Do not invent causes,
relationships, or significance tests. Mention uncertainty when appropriate.""",
            {"question": question, "eda_summary": summary},
        )
        return output.findings[:5]

    def report(self, question: str, context: dict[str, Any]) -> ReportDraft:
        return self._structured(
            "report_draft",
            ReportDraft,
            """You are the report agent. Write an analyst-style summary from the
computed evidence below. Do not claim causality or clinical validity. Keep the
validation decision and its justification visible, state limitations, and give
practical next steps. Every factual statement must be supported by the input
summary.""",
            {"question": question, "computed_context": context},
        )
