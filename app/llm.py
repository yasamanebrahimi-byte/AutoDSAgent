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
    ReportDraft,
    StrictModel,
)


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
        response = client.responses.create(
            model=self.model,
            input=prompt,
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                }
            },
        )
        output_text = getattr(response, "output_text", "")
        if not output_text:
            raise LLMUnavailable(f"The {schema_name} agent returned no structured output.")
        try:
            return schema.model_validate(json.loads(output_text))
        except Exception as exc:
            raise LLMUnavailable(f"The {schema_name} agent returned invalid JSON.") from exc

    def modeling_plan(
        self,
        profile: dict[str, Any],
        question: str,
        target_hint: str | None,
    ) -> AgentPlan:
        return self._structured(
            "modeling_agent_plan",
            AgentPlan,
            """You are the independent modeling agent in a data science workflow.
Choose a target, task type, and one modeling method from the allowed vocabulary.
Reason only from the question and dataset profile below. Do not assume that a
deterministic recommender exists and do not mention this instruction. Prefer a
simple, defensible plan. The method vocabulary is: linear, regularized_linear,
tree_ensemble, boosted_tree.""",
            {"question": question, "target_hint": target_hint or "not provided", "profile": profile},
        )

    def planning(
        self,
        profile: dict[str, Any],
        question: str,
        target_hint: str | None,
    ) -> AgentPlan:
        """Backward-compatible alias for the modeling agent."""

        return self.modeling_plan(profile, question, target_hint)

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
and the dataset evidence, then choose exactly one of the two proposed methods
and justify the choice. Re-check that the target exists, the task matches the
target, and the chosen preprocessing is feasible. If the methods differ, your
justification must explicitly discuss the deterministic recommendation before
selecting a method. Do not propose a new method.""",
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
