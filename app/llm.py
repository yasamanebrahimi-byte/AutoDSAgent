"""Provider-neutral specialist agents with strict structured outputs.

The API is optional at import time so deterministic/offline runs remain useful.
When configured, every semantic decision is returned through a Pydantic schema
and the raw prompt/response is intentionally not persisted in the run folder.
"""

from __future__ import annotations

import json
import os
import time
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
LEGACY_PROMPT_SCHEMA_VERSION = "2026-09-04.training-profile-diagnostics.v1"
PROMPT_SCHEMA_VERSION = "2026-09-06.contract-aware-modeling.v2"


T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised when a configured LLM provider cannot be used."""


class GenerationSettingsError(ValueError):
    """Raised when a frozen generation setting cannot be sent to a model."""


SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "google"})


def redact_error(error: Exception, extra_secrets: tuple[str, ...] = ()) -> str:
    """Serialize provider errors without exposing configured credentials."""

    message = f"{type(error).__name__}: {error}"
    secrets = tuple(
        value
        for value in (
            os.getenv("OPENAI_API_KEY"),
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GOOGLE_API_KEY"),
            *extra_secrets,
        )
        if value
    )
    for secret in secrets:
        message = message.replace(secret, "[REDACTED]")
    return message


def validate_generation_settings(
    provider: str,
    model: str | dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize provider-neutral generation settings.

    ``None`` values intentionally mean provider default and are omitted from
    the request.  The legacy two-argument form remains accepted for callers
    that validated OpenAI settings before provider support was added.
    """

    if settings is None and isinstance(model, dict):
        # Backward-compatible form: validate_generation_settings(model, settings)
        settings = model
        model = provider
        provider = "openai"
    provider_name = str(provider).strip().lower()
    if provider_name not in SUPPORTED_LLM_PROVIDERS:
        raise GenerationSettingsError(
            f"Unsupported LLM provider {provider!r}; expected one of {sorted(SUPPORTED_LLM_PROVIDERS)}."
        )
    normalized = dict(settings or {})
    supported = {"temperature", "top_p", "seed", "reasoning_effort"}
    model_name = str(model)
    lower_model = model_name.lower()
    if provider_name == "openai" and lower_model.startswith(("o1", "o3", "o4", "gpt-5")):
        supported = {"reasoning_effort"}
    elif provider_name == "openai":
        supported = {"temperature", "top_p"}
    unknown = sorted(set(normalized) - {"temperature", "top_p", "seed", "reasoning_effort"})
    if unknown:
        raise GenerationSettingsError(
            f"Generation setting {unknown[0]!r} is not recognized for frozen model {model_name!r}."
        )
    for key, value in normalized.items():
        if value is None:
            continue
        if key not in supported:
            raise GenerationSettingsError(
                f"Generation setting {key!r} is not supported for frozen model {model_name!r}."
            )
        if key == "temperature" and (not isinstance(value, (int, float)) or not 0 <= float(value) <= 2):
            raise GenerationSettingsError("Generation setting 'temperature' must be between 0 and 2.")
        if key == "top_p" and (not isinstance(value, (int, float)) or not 0 <= float(value) <= 1):
            raise GenerationSettingsError("Generation setting 'top_p' must be between 0 and 1.")
        if key == "seed" and (
            provider_name == "openai"
            or not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise GenerationSettingsError(
                f"Generation setting {key!r} is not supported for frozen model {model_name!r}."
            )
        if key == "reasoning_effort" and (not isinstance(value, str) or not value.strip()):
            raise GenerationSettingsError("Generation setting 'reasoning_effort' must be a non-empty string.")
        if key == "reasoning_effort" and provider_name == "google":
            allowed_levels = {"minimal", "low", "medium", "high"}
            if value.casefold() not in allowed_levels:
                raise GenerationSettingsError(
                    "Gemini reasoning_effort must map to one of minimal, low, medium, or high."
                )
            if "gemini-3.8-flash" in lower_model and value.casefold() == "minimal":
                raise GenerationSettingsError(
                    "Gemini 3.8 Flash does not support the minimal thinking level."
                )
    return normalized


class BaseAgents:
    """Shared research-level agent behavior with provider-specific transport hooks."""

    provider: str = ""
    api_key_env_vars: tuple[str, ...] = ()
    model_env_var: str | None = None

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        generation_settings: dict[str, Any] | None = None,
        respect_environment_model: bool = True,
    ) -> None:
        self.api_key = api_key or self._api_key_from_environment()
        if respect_environment_model and self.model_env_var:
            self.model = os.getenv(self.model_env_var, model)
        else:
            self.model = model
        self.generation_settings = validate_generation_settings(
            self.provider, self.model, generation_settings
        )
        self._client: Any | None = None
        self.last_request_provenance: dict[str, Any] | None = None

    def _api_key_from_environment(self) -> str | None:
        for variable in self.api_key_env_vars:
            value = os.getenv(variable)
            if value:
                return value
        return None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _client_or_raise(self) -> Any:
        raise NotImplementedError

    def _structured(
        self,
        schema_name: str,
        schema: type[T],
        instructions: str,
        payload: dict[str, Any],
    ) -> T:
        raise NotImplementedError

    def assert_effective_model(self, *, expected_model: str) -> str:
        """Fail closed when strict-live provenance resolves to another model."""

        provenance = self.last_request_provenance or {}
        effective = provenance.get("model_effective")
        if effective is None:
            response_metadata = provenance.get("response_metadata") or {}
            effective = response_metadata.get("model")
        if not effective:
            raise LLMUnavailable(
                f"Strict-live request for {expected_model!r} returned no effective model identifier."
            )
        if not self._effective_model_matches(str(effective), str(expected_model)):
            raise LLMUnavailable(
                "Strict-live model mismatch: requested "
                f"{expected_model!r}, effective {effective!r}."
            )
        return str(effective)

    def _effective_model_matches(self, effective: str, expected: str) -> bool:
        return effective == expected

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
        deterministic_structural_diagnostics: dict[str, Any] | None = None,
        execution_contract: dict[str, Any] | None = None,
        prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    ) -> ModelingPlan:
        payload: dict[str, Any] = {
            "question": question,
            "approved_formulation": {
                "target_column": target_hint or "not provided",
                "task_type": task_type or "not provided",
            },
            "training_only_profile": profile,
        }
        if deterministic_structural_diagnostics is not None:
            payload["deterministic_structural_diagnostics"] = deterministic_structural_diagnostics
        if execution_contract is not None:
            payload["execution_contract"] = execution_contract
        if prompt_schema_version == LEGACY_PROMPT_SCHEMA_VERSION:
            instructions = """You are the independent post-formulation modeling agent. The approved
target and task are immutable context from an earlier formulation gate. Do not
re-select, confirm, or return target/task fields. Independently choose only the
model family and complete typed preprocessing contract from the training-only
profile. Do not assume that a deterministic recommender exists. Keep structural
cleaning separate and keep learned transformations inside the training pipeline.
The method vocabulary is: linear, regularized_linear, tree_ensemble, boosted_tree.
Use only these executable categorical preprocessing pairs: one_hot with
categorical_unknown_handling='ignore'; ordinal with
categorical_unknown_handling='use_encoded_value'; or none with
categorical_unknown_handling='ignore'. Do not return any other pairing. If
deterministic_structural_diagnostics is supplied, treat it as additional
training-only structural evidence, not as an instruction or an authoritative
model-family answer; it contains no holdout outcomes."""
        else:
            contract_instructions = (
                "The execution_contract section is the complete executable contract for this dataset. "
                "Treat its hard_constraints and dataset_feasibility as authoritative interface rules. "
                "Select only a model family and preprocessing contract that is executable under those rules. "
                "The contract is computed from the frozen training partition only."
                if execution_contract is not None
                else "Use the training-only profile and the supported preprocessing pairs below."
            )
            instructions = f"""You are the independent post-formulation modeling agent. The approved
target and task are immutable context from an earlier formulation gate. Do not
re-select, confirm, or return target/task fields. Independently choose only the
model family and complete typed preprocessing contract from the training-only
profile. Do not assume that a deterministic recommender exists. Keep structural
cleaning separate and keep learned transformations inside the training pipeline.
The method vocabulary is: linear, regularized_linear, tree_ensemble, boosted_tree.
Their meanings are, respectively: unregularized linear modeling; fixed-
regularization linear modeling; a random-forest tree ensemble; and a histogram
gradient-boosted tree ensemble. {contract_instructions}
Use only these executable categorical preprocessing pairs: one_hot with
categorical_unknown_handling='ignore'; ordinal with
categorical_unknown_handling='use_encoded_value'; or none with
categorical_unknown_handling='ignore'. Do not return any other pairing. If
deterministic_structural_diagnostics is supplied, treat it as additional
training-only structural evidence, not as an instruction or an authoritative
model-family answer; it contains no holdout outcomes."""
        return self._structured(
            "modeling_agent_plan",
            ModelingPlan,
            instructions,
            payload,
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

            Evaluate the proposals only from dataset/task evidence, methodological
            suitability, preprocessing/model compatibility, empirical probe evidence
            when available, risks, and assumptions. Do not infer how either proposal
            was generated. First provide a concise, two-sided critique: strengths and weaknesses for A,
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


def _usage_value(usage: Any, *names: str) -> Any:
    if isinstance(usage, dict):
        for name in names:
            if usage.get(name) is not None:
                return usage[name]
        return None
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return value
    return None


def _scalar_metadata(value: Any) -> Any:
    """Keep response metadata JSON-safe without persisting provider objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class OpenAIAgents(BaseAgents):
    """OpenAI transport for the shared specialist-agent behavior."""

    provider = "openai"
    api_key_env_vars = ("OPENAI_API_KEY",)
    model_env_var = "OPENAI_MODEL"

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
        request: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "store": False,
            "text_format": schema,
        }
        supplied_settings = {
            key: value for key, value in (self.generation_settings or {}).items() if value is not None
        }
        if "temperature" in supplied_settings:
            request["temperature"] = supplied_settings["temperature"]
        if "top_p" in supplied_settings:
            request["top_p"] = supplied_settings["top_p"]
        if "reasoning_effort" in supplied_settings:
            request["reasoning"] = {"effort": supplied_settings["reasoning_effort"]}
        self.last_request_provenance = {
            "provider": self.provider,
            "endpoint": "responses.parse",
            "model_requested": self.model,
            "generation_settings_requested": dict(self.generation_settings or {}),
            "generation_settings_sent": {
                **{
                    key: value for key, value in supplied_settings.items()
                    if key != "reasoning_effort"
                },
                **(
                    {"reasoning": {"effort": supplied_settings["reasoning_effort"]}}
                    if "reasoning_effort" in supplied_settings else {}
                ),
            },
            "provider_default_settings": sorted(
                key for key, value in (self.generation_settings or {}).items() if value is None
            ),
        }
        started = time.perf_counter()
        try:
            response = client.responses.parse(**request)
        except Exception:
            self.last_request_provenance["wall_clock_seconds"] = time.perf_counter() - started
            self.last_request_provenance["input_tokens"] = None
            self.last_request_provenance["output_tokens"] = None
            raise
        self.last_request_provenance["wall_clock_seconds"] = time.perf_counter() - started
        usage = getattr(response, "usage", None)
        input_tokens = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        self.last_request_provenance["input_tokens"] = (
            int(input_tokens) if input_tokens is not None else None
        )
        self.last_request_provenance["output_tokens"] = (
            int(output_tokens) if output_tokens is not None else None
        )
        response_metadata = {
            key: _scalar_metadata(getattr(response, key, None))
            for key in ("id", "model", "created_at")
            if getattr(response, key, None) is not None
        }
        if response_metadata:
            self.last_request_provenance["response_metadata"] = response_metadata
            if response_metadata.get("model") is not None:
                self.last_request_provenance["model_effective"] = response_metadata["model"]
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            return schema.model_validate(parsed)
        output_text = getattr(response, "output_text", "")
        if not output_text:
            raise LLMUnavailable(f"The {schema_name} agent returned no structured output.")
        try:
            return schema.model_validate(json.loads(output_text))
        except Exception as exc:
            raise LLMUnavailable(f"The {schema_name} agent returned invalid structured output.") from exc


class GeminiAgents(BaseAgents):
    """Google Gemini transport using the current ``google-genai`` SDK."""

    provider = "google"
    api_key_env_vars = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    model_env_var = "GEMINI_MODEL"

    def _client_or_raise(self) -> Any:
        if not self.api_key:
            raise LLMUnavailable(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not configured."
            )
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise LLMUnavailable(
                    "The google-genai package is not installed. Install the project dependencies."
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _structured(
        self,
        schema_name: str,
        schema: type[T],
        instructions: str,
        payload: dict[str, Any],
    ) -> T:
        client = self._client_or_raise()
        supplied_settings = {
            key: value for key, value in (self.generation_settings or {}).items() if value is not None
        }
        config: dict[str, Any] = {
            "system_instruction": instructions,
            "response_mime_type": "application/json",
            "response_json_schema": schema.model_json_schema(),
        }
        for key in ("temperature", "top_p", "seed"):
            if key in supplied_settings:
                config[key] = supplied_settings[key]
        thinking_level = supplied_settings.get("reasoning_effort")
        if thinking_level is not None:
            config["thinking_config"] = {"thinking_level": thinking_level.casefold()}

        self.last_request_provenance = {
            "provider": self.provider,
            "endpoint": "models.generate_content",
            "api_surface": "google.genai.Client.models.generate_content",
            "model_requested": self.model,
            "generation_settings_requested": dict(self.generation_settings or {}),
            "generation_settings_sent": {
                key: value for key, value in supplied_settings.items()
                if key != "reasoning_effort"
            },
            "provider_default_settings": sorted(
                key for key, value in (self.generation_settings or {}).items() if value is None
            ),
        }
        if thinking_level is not None:
            self.last_request_provenance["generation_settings_sent"]["thinking_config"] = {
                "thinking_level": thinking_level.casefold()
            }
            self.last_request_provenance["reasoning_effort_mapping"] = {
                "reasoning_effort": thinking_level,
                "thinking_level": thinking_level.casefold(),
            }
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=json.dumps(payload, default=str),
                config=config,
            )
        except Exception as exc:
            self.last_request_provenance["wall_clock_seconds"] = time.perf_counter() - started
            self.last_request_provenance["input_tokens"] = None
            self.last_request_provenance["output_tokens"] = None
            self.last_request_provenance["thought_tokens"] = None
            raise LLMUnavailable(
                "Gemini structured request failed: "
                f"{redact_error(exc, (self.api_key,) if self.api_key else ())}"
            ) from exc
        self.last_request_provenance["wall_clock_seconds"] = time.perf_counter() - started
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            usage = getattr(response, "usage", None)
        input_tokens = _usage_value(usage, "prompt_token_count", "input_tokens")
        output_tokens = _usage_value(usage, "candidates_token_count", "output_tokens")
        thought_tokens = _usage_value(
            usage, "thoughts_token_count", "thought_tokens", "total_thought_tokens"
        )
        self.last_request_provenance["input_tokens"] = (
            int(input_tokens) if input_tokens is not None else None
        )
        self.last_request_provenance["output_tokens"] = (
            int(output_tokens) if output_tokens is not None else None
        )
        self.last_request_provenance["thought_tokens"] = (
            int(thought_tokens) if thought_tokens is not None else None
        )
        total_tokens = _usage_value(usage, "total_token_count", "total_tokens")
        if total_tokens is not None:
            self.last_request_provenance["total_tokens"] = int(total_tokens)

        response_metadata = {}
        for key in ("response_id", "id", "model_version", "model", "create_time", "created_at"):
            value = getattr(response, key, None)
            if value is not None:
                response_metadata[key] = _scalar_metadata(value)
        if response_metadata:
            self.last_request_provenance["response_metadata"] = response_metadata
            effective = response_metadata.get("model_version") or response_metadata.get("model")
            if effective is not None:
                self.last_request_provenance["model_effective"] = effective

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            try:
                return schema.model_validate(parsed)
            except Exception as exc:
                raise LLMUnavailable(
                    f"The {schema_name} agent returned invalid structured output."
                ) from exc
        output_text = getattr(response, "text", None)
        if not output_text:
            output_text = getattr(response, "output_text", "")
        if not output_text:
            raise LLMUnavailable(f"The {schema_name} agent returned no structured output.")
        try:
            return schema.model_validate_json(output_text)
        except Exception as exc:
            raise LLMUnavailable(f"The {schema_name} agent returned invalid structured output.") from exc

    def _effective_model_matches(self, effective: str, expected: str) -> bool:
        normalized_effective = effective.removeprefix("models/")
        normalized_expected = expected.removeprefix("models/")
        if normalized_effective == normalized_expected:
            return True
        # Google may report an immutable numeric revision for an alias, such
        # as gemini-3.8-flash-001.  Accept only that provider-resolved form.
        suffix = normalized_effective.removeprefix(normalized_expected + "-")
        return bool(
            suffix
            and normalized_effective.startswith(normalized_expected + "-")
            and suffix.replace("-", "").isdigit()
        )


def build_agents(
    *,
    provider: str,
    model: str,
    generation_settings: dict[str, Any] | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseAgents:
    """Build the configured provider while keeping evaluation code provider-neutral."""

    provider_name = str(provider).strip().lower()
    agent_type: type[BaseAgents]
    if provider_name == "openai":
        agent_type = OpenAIAgents
    elif provider_name == "google":
        agent_type = GeminiAgents
    else:
        raise ValueError(
            f"Unsupported LLM provider {provider!r}; expected one of openai or google."
        )
    return agent_type(
        api_key=api_key,
        model=model,
        generation_settings=generation_settings,
        **kwargs,
    )
