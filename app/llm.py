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
    AgentPlan,
    CleaningPlan,
    ConflictResolution,
    FormulationPlan,
    FormulationResolution,
    ModelingPlan,
    ModelingResolution,
    ReportDraft,
    StrictModel,
)


# Bump this when the modeling/reconciliation input contract changes.  The
# evaluation harness records it beside every trial so a result bundle can be
# interpreted without preserving provider-specific request metadata.
PROMPT_SCHEMA_VERSION = "2026-08-23.formulation-modeling-gates.v2"


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
The method vocabulary is: linear, regularized_linear, tree_ensemble, boosted_tree.""",
            {
                "question": question,
                "approved_formulation": {
                    "target_column": target_hint or "not provided",
                    "task_type": task_type or "not provided",
                },
                "training_only_profile": profile,
            },
        )

    def planning(
        self,
        profile: dict[str, Any],
        question: str,
        target_hint: str | None,
        task_type: str | None = None,
    ) -> AgentPlan:
        """Backward-compatible alias for the modeling agent."""

        plan = self.modeling_plan(profile, question, target_hint, task_type)
        return AgentPlan(
            target_column=target_hint or "",
            task_type=task_type or "classification",
            recommended_method=plan.recommended_method,
            preprocessing=plan.preprocessing,
            reasoning=plan.reasoning,
            confidence=plan.confidence,
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
        return self._structured(
            "modeling_resolution",
            ModelingResolution,
            """You are the modeling-gate reconciliation agent. Choose only one of
            the two proposed model families and one complete supported preprocessing contract.
            Target and task are immutable approved context and are not semantic decisions in
            this call. The hard deterministic safety/executability checks have already been
            run for the proposals; treat those checks as authoritative. In the normal
            soft-challenge case, both proposals have passed hard validation and the remaining
            question is advisory model-family compatibility. The deterministic recommendation
            is advisory, and its numerical compatibility scores are not probabilities, not
            cross-validation results, and not empirical performance estimates. Disagreement
            does not mean the initial agent is wrong. Preserve the initial plan unless the
            actual training-only evidence identifies a convincing methodological reason to
            prefer the alternative. Compare the evidence for both proposals and choose only
            one of the two proposed methods; never invent a third model family. If a proposal
            failed a hard check, correct or reject that proposal rather than treating the
            failure as a soft preference. Use training-only evidence; never use holdout values,
            CV results, or an empirical reference. Explain material preprocessing differences.""",
            {
                "question": question,
                "training_only_profile": profile,
                "modeling_plan": modeling_plan.model_dump(mode="json"),
                "deterministic_recommendation": deterministic,
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

    def reconcile(
        self,
        question: str,
        profile: dict[str, Any],
        agent_plan: AgentPlan,
        deterministic: dict[str, Any],
    ) -> ConflictResolution:
        return self._structured(
            "conflict_resolution",
            ConflictResolution,
            """You are the validation agent. An independent planning agent and an
            independent deterministic recommender disagree. Inspect both recommendations
            and the dataset evidence, then choose exactly one of the two proposed methods.
            Both proposals have already passed hard deterministic safety/executability
            validation when this is a soft challenge. The deterministic recommendation is
            an advisory model-family compatibility assessment. Its numerical compatibility
            scores are not probabilities, not cross-validation results, and not empirical
            performance estimates. Disagreement does not mean the initial agent is wrong.
            Preserve the initial plan unless the actual training-only evidence identifies a
            convincing methodological reason to prefer the alternative. Compare the evidence
            for both proposals. Use diagnostics and score-contribution evidence as auditable
            structural reasoning, not as proof that a method is best. Do not use holdout
            values, cross-validation results, or empirical-reference rankings even if they
            appear elsewhere in the surrounding application. Return a complete
            supported preprocessing contract. Re-check that the
            target exists, the task matches the target, all observed missing/infinite values
            are handled, identifiers and unsupported feature types remain excluded, unknown
            categories are safe, and all learned transformations stay inside the pipeline.
            Your justification must explicitly discuss every material preprocessing
            difference as well as any target/task/method difference. Do not invent a method
            or preprocessing strategy outside the typed schema, and choose only one of the
            two proposed model families.""",
            {
                "question": question,
                "profile": profile,
                "agent_plan": agent_plan.model_dump(mode="json"),
                "deterministic_recommendation": deterministic,
            },
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
