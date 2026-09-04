"""Run paired agent/gated/post-hoc evaluations with auditable trial rows."""

from __future__ import annotations

import json
import shutil
import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
from pydantic import BaseModel

from app.deterministic import deterministic_recommendation, profile_dataframe
from app.empirical_challenge_probe import EmpiricalProbePolicy
from app.deterministic_policy import DeterministicPolicy
from app.llm import LLMUnavailable, PROMPT_SCHEMA_VERSION, OpenAIAgents
from app.reconciliation import BLINDED_RECONCILIATION_PROMPT_VERSION
from app.pipeline import (
    _fallback_modeling_plan,
    _fallback_modeling_resolution,
    _validate_modeling_gate,
)
from app.soft_challenge import SOFT_CHALLENGE_POLICY_VERSION, load_calibration_artifact
from app.preprocessing import compare_preprocessing_plans
from app.schemas import ModelingPlan, ModelingResolution, DeterministicRecommendation, PreprocessingContract
from app.validation import (
    FrozenSplit,
    InvariantViolation,
    freeze_supervised_split,
    training_profile_frame,
    validate_training_plan,
)
from evaluation.benchmarks import BenchmarkCase, default_benchmark_cases
from evaluation.empirical_reference import (
    evaluate_empirical_reference,
    evaluate_holdout_plan,
    evaluate_plan_cv,
    training_only_requirements,
)
from evaluation.metrics import (
    DEFAULT_THRESHOLDS,
    GATE_OBJECTIVE_VERSION,
    HOLDOUT_METRIC_SCHEMA_VERSION,
    catastrophic_transition,
    classify_intervention_outcome,
    holdout_neutral_tolerance,
    normalized_performance_delta,
    paper_holdout_delta,
    raw_holdout_performance_delta,
    normalized_regret,
    regret,
    regret_reduction,
    summarize_trials,
)
from evaluation.perturbations import Perturbation, default_perturbations
from evaluation.confirmatory import (
    CONFIRMATORY_EXPERIMENT_NAME,
    load_confirmatory_manifest,
    runtime_manifest_values,
    validate_confirmatory_manifest,
    deterministic_policy_config,
    empirical_probe_config,
    config_sha256,
    repository_commit,
    experiment_code_sha256,
    model_conditions,
    repetition_ids,
    condition_repetition_ids,
)
from evaluation.provenance import environment_provenance
from evaluation.external_benchmarks import external_benchmark_manifest_sha256, external_benchmark_specs
from evaluation.statistics import (
    DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
)


EXPERIMENT_CONFIG_VERSION = "paper-confirmatory-v1"
CONFIRMATORY_CONFIG_SNAPSHOT = "evaluation/configs/paper_confirmatory_v1.json"


ModelingPlanFactory = Callable[[dict[str, Any]], ModelingPlan]
ModelingReconciliationFactory = Callable[
    [dict[str, Any], ModelingPlan, DeterministicRecommendation], ModelingResolution
]


@dataclass(frozen=True)
class EvaluationConfig:
    repetitions: int = 1
    seed: int = 42
    test_size: float = 0.2
    model: str = "gpt-4.1-mini"
    planner_model: str | None = None
    reconciler_model: str | None = None
    offline: bool = False
    include_perturbations: bool = False
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION
    repository_commit: str | None = None
    gate_mode: str = "selective"
    order_swap: bool = False
    empirical_probe_enabled: bool = True
    split_seeds: tuple[int, ...] = (42,)
    require_live: bool = False
    soft_challenge_strategy: str = "calibrated"
    enable_regression_interaction_diagnostics: bool = True
    enable_classification_boundary_diagnostics: bool = True
    ablation_name: str | None = None
    ablation_schema_version: str | None = None
    suite: str = "local"
    tier: str | None = None
    model_condition_id: str = "default"
    llm_repetition_id: str | None = None
    generation_settings: dict[str, Any] = field(default_factory=dict)
    llm_repetition_ids: tuple[str, ...] | None = None
    planner_evidence_mode: str = "training_profile_only"

    def __post_init__(self) -> None:
        if self.planner_model is None:
            object.__setattr__(self, "planner_model", self.model)
        if self.reconciler_model is None:
            object.__setattr__(self, "reconciler_model", self.model)
        if self.suite not in {"local", "external"}:
            raise ValueError("suite must be 'local' or 'external'.")
        if self.tier is not None and self.tier not in {"core", "stress"}:
            raise ValueError("tier must be 'core', 'stress', or None.")
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least one")
        if not 0.10 <= self.test_size <= 0.50:
            raise ValueError("test_size must be between 0.10 and 0.50")
        if self.gate_mode not in {
            "llm_only", "hard_validation_only", "deterministic_only",
            "always_reconcile", "selective", "probe_first", "probe_direct", "full",
        }:
            raise ValueError("Unsupported gate_mode.")
        if not self.split_seeds:
            raise ValueError("split_seeds must contain at least one seed.")
        if self.soft_challenge_strategy not in {"calibrated", "high_confidence_only"}:
            raise ValueError("Unsupported soft_challenge_strategy.")
        if self.planner_evidence_mode not in {
            "training_profile_only",
            "training_only_structural_diagnostics",
        }:
            raise ValueError("Unsupported planner_evidence_mode.")


class _EvaluationGateAgent:
    def __init__(
        self,
        *,
        live_agents: OpenAIAgents | None,
        reconciliation_factory: ModelingReconciliationFactory | None,
        context: dict[str, Any],
    ) -> None:
        self.live_agents = live_agents
        self.reconciliation_factory = reconciliation_factory
        self.context = context

    def reconcile_modeling(
        self,
        question: str,
        profile: dict[str, Any],
        modeling_plan: ModelingPlan,
        deterministic: dict[str, Any],
    ) -> ModelingResolution:
        recommendation = DeterministicRecommendation.model_validate(
            {
                key: deterministic[key]
                for key in (
                    "target_column",
                    "task_type",
                    "recommended_method",
                    "preprocessing",
                    "reasoning",
                    "evidence",
                )
            }
        )
        if self.reconciliation_factory is not None:
            return self.reconciliation_factory(self.context, modeling_plan, recommendation)
        if self.live_agents is not None:
            return self.live_agents.reconcile_modeling(question, profile, modeling_plan, deterministic)
        return _fallback_modeling_resolution(modeling_plan, recommendation)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _frame_digest(frame: pd.DataFrame) -> str:
    """Hash only the frame supplied to an evaluation stage.

    The empirical-reference cache is keyed by the training frame, never by
    holdout values.  Including column names and dtypes prevents accidental
    reuse across incompatible schema representations.
    """

    digest = hashlib.sha256()
    digest.update(json.dumps([str(column) for column in frame.columns]).encode("utf-8"))
    digest.update(json.dumps([str(dtype) for dtype in frame.dtypes]).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _repository_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _failed_checks(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "code": check.code,
            "severity": check.severity,
            "message": check.message,
            "evidence": _jsonable(check.evidence),
        }
        for check in result.failed_checks
    ]


def _blocking_failed_check_codes(payload: dict[str, Any] | None) -> list[str]:
    """Return only hard-invalid check codes from a serialized validation payload."""

    return [
        str(check["code"])
        for check in (payload or {}).get("checks", [])
        if check.get("status") == "failed"
        and check.get("blocking", check.get("severity", "error") == "error")
    ]


def _plan_fields(
    plan: ModelingPlan | None,
    *,
    target_column: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    if plan is None:
        return {
            "target": None,
            "task": None,
            "method": None,
            "preprocessing": None,
        }
    return {
        "target": target_column,
        "task": task_type,
        "method": plan.recommended_method,
        "preprocessing": plan.preprocessing.model_dump(mode="json"),
    }


def _canonical_diagnostics(value: Any) -> dict[str, Any] | None:
    """Return a stable JSON-safe structural-diagnostics payload.

    The payload contains only deterministic diagnostics computed from the
    frozen training partition.  It deliberately excludes the challenger
    recommendation, candidate CV results, and every holdout-derived value.
    """

    if value is None:
        return None
    raw = _jsonable(value)
    return json.loads(json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str))


def _plan_matches(
    target: str | None,
    task: str | None,
    method: str | None,
    preprocessing: dict[str, Any] | None,
    plan: ModelingPlan | None,
    *,
    expected_target: str,
    expected_task: str,
) -> bool:
    fields = _plan_fields(plan, target_column=expected_target, task_type=expected_task)
    return (
        target == fields["target"]
        and task == fields["task"]
        and method == fields["method"]
        and preprocessing == fields["preprocessing"]
    )


def _validate_initial_plan(
    dataframe: pd.DataFrame,
    split: FrozenSplit,
    plan: ModelingPlan,
    *,
    expected_target: str,
    expected_task: str,
    test_size: float,
    random_state: int,
) -> Any:
    result = validate_training_plan(
        dataframe,
        expected_target,
        expected_task,
        plan.recommended_method,
        test_size=test_size,
        random_state=random_state,
        preprocessing=plan.preprocessing,
        split=split,
        row_positions=list(range(len(dataframe))),
        training_only=True,
    )
    return result


def _choose_source(
    *,
    config: EvaluationConfig,
    agents: OpenAIAgents,
    plan_factory: ModelingPlanFactory | None,
    context: dict[str, Any],
    training_profile: dict[str, Any],
    planner_diagnostics: dict[str, Any] | None,
    case: BenchmarkCase,
    warnings: list[str],
    require_live: bool = False,
) -> tuple[ModelingPlan, str, str, str, str | None]:
    if plan_factory is not None:
        return plan_factory(context), "mock", "mock", "mock", None
    if config.offline or not agents.available:
        if require_live and not config.offline and not agents.available:
            raise LLMUnavailable("OPENAI_API_KEY is not configured for a required live trial.")
        if not config.offline:
            warnings.append("OPENAI_API_KEY is unavailable; this trial used the offline fallback.")
        return (
            _fallback_modeling_plan(
                training_profile,
                case.question,
                case.target_column,
                case.expected_task_type,
            ),
            "offline_fallback",
            "offline",
            "offline_fallback" if config.offline else "not_requested_fallback",
            None,
        )
    try:
        modeling_plan = agents.modeling_plan(
            training_profile,
            case.question,
            case.target_column,
            case.expected_task_type,
            deterministic_structural_diagnostics=planner_diagnostics,
        )
        if require_live:
            agents.assert_effective_model(expected_model=config.planner_model)
        return (
            ModelingPlan(
                recommended_method=modeling_plan.recommended_method,
                preprocessing=modeling_plan.preprocessing,
                reasoning=modeling_plan.reasoning,
                confidence=modeling_plan.confidence,
            ),
            "openai",
            agents.model,
            "succeeded",
            None,
        )
    except Exception as exc:
        if require_live:
            raise
        warnings.append(f"modeling agent fallback used: {_redact_error(exc)}")
        return (
            _fallback_modeling_plan(
                training_profile,
                case.question,
                case.target_column,
                case.expected_task_type,
            ),
            "offline_fallback",
            agents.model,
            "failed_fallback",
            f"{type(exc).__name__}: {exc}",
        )


def _family_score(reference: dict[str, Any], method: str | None) -> float | None:
    if method is None:
        return None
    candidate = reference.get("candidate_metrics", {}).get(method, {})
    value = candidate.get("primary_mean")
    return float(value) if value is not None and candidate.get("status") == "evaluated" else None


def _proposal_cache_key(
    *,
    case: BenchmarkCase,
    perturbation_id: str,
    split_seed: int,
    llm_repetition: int,
    model: str,
    prompt_schema_version: str,
    llm_repetition_id: str | None = None,
    model_condition_id: str = "default",
    generation_settings: dict[str, Any] | None = None,
    evidence_mode: str = "training_profile_only",
    planner_diagnostics: dict[str, Any] | None = None,
    training_profile: dict[str, Any],
) -> str:
    """Identify only the evidence that is legal for the initial proposal."""

    payload = {
        "benchmark_case": case.name,
        "perturbation": perturbation_id,
        "split_seed": split_seed,
        "llm_repetition": llm_repetition,
        "llm_repetition_id": llm_repetition_id or f"rep_{llm_repetition + 1:03d}",
        "model_condition_id": model_condition_id,
        "model": model,
        "initial_modeling_prompt_schema_version": prompt_schema_version,
        "training_profile_digest": hashlib.sha256(
            json.dumps(_jsonable(training_profile), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "approved_target": case.target_column,
        "approved_task": case.expected_task_type,
        "generation_settings": _jsonable(generation_settings or {}),
        "planner_evidence_mode": evidence_mode,
        "planner_diagnostics_digest": hashlib.sha256(
            json.dumps(_jsonable(planner_diagnostics), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if planner_diagnostics is not None else None,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _plan_from_cache_payload(payload: dict[str, Any]) -> ModelingPlan | None:
    values = payload.get("modeling_plan") if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        return None
    try:
        return ModelingPlan.model_validate(values)
    except Exception:
        return None


def _redact_error(error: Exception) -> str:
    """Keep provider errors useful without allowing credential serialization."""

    message = f"{type(error).__name__}: {error}"
    import os

    secret = os.getenv("OPENAI_API_KEY")
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return message


def _effective_model_from_provenance(
    provenance: dict[str, Any] | None,
) -> str | None:
    """Extract a provider-resolved model ID from a saved request artifact."""

    if not isinstance(provenance, dict):
        return None
    value = provenance.get("model_effective")
    if value is None:
        value = (provenance.get("response_metadata") or {}).get("model")
    return str(value) if value else None


def _write_proposal_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for key in sorted(cache):
            handle.write(json.dumps({"cache_key": key, **cache[key]}, sort_keys=True) + "\n")


def _run_trial(
    case: BenchmarkCase,
    perturbation: Perturbation | None,
    trial_number: int,
    config: EvaluationConfig,
    *,
    plan_factory: ModelingPlanFactory | None,
    reconciliation_factory: ModelingReconciliationFactory | None,
    agents: OpenAIAgents,
    reconciler_agents: OpenAIAgents,
    empirical_reference_cache: dict[str, dict[str, Any]],
    variant: str = "standard",
    proposal_order: tuple[str, str] | None = None,
    order_swap_pair_id: str | None = None,
    initial_plan_override: ModelingPlan | None = None,
    initial_source_override: str | None = None,
    initial_model_override: str | None = None,
    initial_request_status_override: str | None = None,
    initial_request_error_override: str | None = None,
    initial_api_provenance_override: dict[str, Any] | None = None,
    requested_split_seed: int | None = None,
    proposal_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # Repetitions are stochastic LLM repeats over identical evidence.  The
    # split seed is therefore a property of the case/experiment, not of the
    # repetition number.
    experimental_split_seed = config.seed if requested_split_seed is None else requested_split_seed
    split_seed = experimental_split_seed + case.random_seed
    perturbation_seed = split_seed + 1009
    base_frame = case.load()
    split = freeze_supervised_split(
        base_frame,
        case.target_column,
        case.expected_task_type,
        test_size=config.test_size,
        random_state=split_seed,
    )
    frame = base_frame.copy()
    perturbation_id = "clean"
    perturbation_data = {
        "id": "clean",
        "kind": "clean",
        "description": "Unperturbed benchmark case.",
        "changes": [],
        "expected_validation_codes": [],
    }
    if perturbation is not None:
        frame, changes = perturbation.apply(frame, perturbation_seed, case)
        perturbation_id = perturbation.id
        perturbation_data = {
            **perturbation.as_dict(),
            "changes": changes,
            "seed": perturbation_seed,
        }
    training_frame = training_profile_frame(
        frame,
        case.target_column,
        case.expected_task_type,
        test_size=config.test_size,
        random_state=split_seed,
        split=split,
    )
    training_profile = profile_dataframe(training_frame)
    warnings: list[str] = []
    deterministic: DeterministicRecommendation | None = None
    deterministic_failure: str | None = None
    try:
        policy = DeterministicPolicy(
            enable_regression_interaction_diagnostics=config.enable_regression_interaction_diagnostics,
            enable_classification_boundary_diagnostics=config.enable_classification_boundary_diagnostics,
        )
        deterministic = deterministic_recommendation(
            training_frame,
            case.question,
            case.target_column,
            task_type=case.expected_task_type,
            policy=policy,
        )
    except Exception as exc:
        deterministic_failure = _redact_error(exc)
        warnings.append(f"Deterministic recommendation failed closed: {_redact_error(exc)}")

    planner_diagnostics = (
        _canonical_diagnostics(deterministic.diagnostics)
        if config.planner_evidence_mode == "training_only_structural_diagnostics"
        and deterministic is not None
        else None
    )
    repetition_id = config.llm_repetition_id or f"rep_{trial_number + 1:03d}"
    trial_id = (
        f"{config.model_condition_id}:{case.name}:{perturbation_id}:split{experimental_split_seed}:"
        f"{repetition_id}:llm{trial_number}"
    )
    if config.ablation_name:
        trial_id = f"{trial_id}:{config.ablation_name}"
    if variant != "standard":
        trial_id = f"{trial_id}:{variant}"
    context = {
        "benchmark_case": case.name,
        "perturbation_id": perturbation_id,
        "trial": trial_number,
        "trial_id": trial_id,
        "split_seed": experimental_split_seed,
        "target_column": case.target_column,
        "task_type": case.expected_task_type,
        "question": case.question,
        "training_profile": training_profile,
        "model_condition_id": config.model_condition_id,
        "llm_repetition_id": repetition_id,
        "planner_model": config.planner_model,
        "reconciler_model": config.reconciler_model,
        "generation_settings": dict(config.generation_settings),
        "planner_evidence_mode": config.planner_evidence_mode,
        "training_only_structural_diagnostics": planner_diagnostics,
    }
    initial_input_artifact = {
        "evaluation_mode": "modeling_gate",
        "gate_objective_version": GATE_OBJECTIVE_VERSION,
        "gate_mode": config.gate_mode,
        "ablation_name": config.ablation_name,
        "ablation_schema_version": config.ablation_schema_version,
        "challenger_enabled": config.gate_mode != "llm_only",
        "hard_validation_enabled": True,
        "probe_enabled": config.empirical_probe_enabled and config.gate_mode in {"selective", "probe_first", "probe_direct", "full"},
        "reconciliation_enabled": config.gate_mode in {"always_reconcile", "selective", "probe_first", "full"},
        "reconcile_on_any_disagreement": config.gate_mode == "always_reconcile",
        "direct_probe_selection_enabled": config.gate_mode == "probe_direct",
        "abstention_enabled": config.gate_mode in {"selective", "probe_first", "probe_direct", "full"},
        "soft_challenge_strategy": config.soft_challenge_strategy,
        "deterministic_diagnostic_config": {
            "enable_regression_interaction_diagnostics": config.enable_regression_interaction_diagnostics,
            "enable_classification_boundary_diagnostics": config.enable_classification_boundary_diagnostics,
        },
        "question": case.question,
        "benchmark_target_constraint": case.target_column,
        "benchmark_task_constraint": case.expected_task_type,
        "training_profile": training_profile,
        "planner_evidence_mode": config.planner_evidence_mode,
        "planner_structural_diagnostics_exposed": planner_diagnostics is not None,
        "planner_structural_diagnostics": planner_diagnostics,
        "holdout_included": False,
        "deterministic_recommendation_included": False,
        "empirical_reference_included": False,
        "previous_repetitions_included": False,
        "planner_prompt_schema_version": config.prompt_schema_version,
        "reconciler_prompt_schema_version": BLINDED_RECONCILIATION_PROMPT_VERSION,
        "prompt_schema_version": config.prompt_schema_version,
        "prompt_schema_version_semantics": "deprecated alias for planner_prompt_schema_version; not an independent schema",
    }
    proposal_key = _proposal_cache_key(
        case=case,
        perturbation_id=perturbation_id,
        split_seed=experimental_split_seed,
        llm_repetition=trial_number,
        llm_repetition_id=config.llm_repetition_id,
        model_condition_id=config.model_condition_id,
        model=config.planner_model,
        prompt_schema_version=config.prompt_schema_version,
        generation_settings=config.generation_settings,
        evidence_mode=config.planner_evidence_mode,
        planner_diagnostics=planner_diagnostics,
        training_profile=training_profile,
    )
    proposal_cache_hit = False
    planner_api_provenance: dict[str, Any] | None = None
    cached_proposal = proposal_cache.get(proposal_key) if proposal_cache is not None else None
    cached_plan = _plan_from_cache_payload(cached_proposal or {})
    # A strict confirmatory run may reuse an exact live proposal, but never a
    # proposal created by an offline fallback or an injected development mock.
    # This prevents a stale development cache from silently contaminating a
    # live-required result bundle.
    can_use_cached = cached_plan is not None and not (
        config.require_live and (cached_proposal or {}).get("source") != "openai"
    )
    if initial_plan_override is None and can_use_cached:
        plan = cached_plan
        proposal_cache_hit = True
        agent_source = str((cached_proposal or {}).get("source") or "cached")
        agent_model = str((cached_proposal or {}).get("model") or config.planner_model)
        agent_request_status = "cached"
        agent_request_error = None
        planner_api_provenance = (cached_proposal or {}).get("api_provenance")
        if config.require_live:
            effective_model = _effective_model_from_provenance(planner_api_provenance)
            if effective_model != str(config.planner_model):
                raise LLMUnavailable(
                    "Strict-live cached proposal has no matching effective planner model: "
                    f"expected {config.planner_model!r}, got {effective_model!r}."
                )
    elif config.gate_mode == "deterministic_only":
        plan = _fallback_modeling_plan(
            training_profile,
            case.question,
            case.target_column,
            case.expected_task_type,
        )
        agent_source = "deterministic_only"
        agent_model = None
        agent_request_status = "not_requested"
        agent_request_error = None
    elif initial_plan_override is None:
        plan, agent_source, agent_model, agent_request_status, agent_request_error = _choose_source(
            config=config,
            agents=agents,
            plan_factory=plan_factory,
            context=context,
            training_profile=training_profile,
            planner_diagnostics=planner_diagnostics,
            case=case,
            warnings=warnings,
            require_live=config.require_live,
        )
        planner_api_provenance = getattr(agents, "last_request_provenance", None)
    else:
        if config.require_live and initial_source_override != "openai":
            raise LLMUnavailable(
                "Strict live order-swap trials may reuse only an OpenAI initial proposal."
            )
        plan = initial_plan_override
        agent_source = initial_source_override or "cached_order_swap"
        agent_model = initial_model_override or config.planner_model
        agent_request_status = initial_request_status_override or "cached"
        agent_request_error = initial_request_error_override
        planner_api_provenance = initial_api_provenance_override
        if config.require_live and agent_source == "openai":
            effective_model = _effective_model_from_provenance(planner_api_provenance)
            if effective_model != str(config.planner_model):
                raise LLMUnavailable(
                    "Strict-live reused proposal has no matching effective planner model: "
                    f"expected {config.planner_model!r}, got {effective_model!r}."
                )
    initial_validation = _validate_initial_plan(
        training_frame,
        split,
        plan,
        expected_target=case.target_column,
        expected_task=case.expected_task_type,
        test_size=config.test_size,
        random_state=split_seed,
    )
    initial_fields = _plan_fields(
        plan,
        target_column=case.target_column,
        task_type=case.expected_task_type,
    )
    if config.gate_mode == "deterministic_only" and deterministic is not None:
        plan = ModelingPlan(
            recommended_method=deterministic.recommended_method,
            preprocessing=deterministic.preprocessing,
            reasoning=deterministic.reasoning,
            confidence={"low": 0.35, "medium": 0.65, "high": 0.90}[deterministic.confidence],
        )
        initial_validation = _validate_initial_plan(
            training_frame,
            split,
            plan,
            expected_target=case.target_column,
            expected_task=case.expected_task_type,
            test_size=config.test_size,
            random_state=split_seed,
        )

    if (
        proposal_cache is not None
        and not proposal_cache_hit
        and config.gate_mode != "deterministic_only"
        and agent_request_status not in {"failed", "failed_fallback"}
    ):
        proposal_cache[proposal_key] = {
            "modeling_plan": plan.model_dump(mode="json"),
            "source": agent_source,
            "model": agent_model,
            "request_status": agent_request_status,
            "benchmark_case": case.name,
            "perturbation": perturbation_id,
            "split_seed": experimental_split_seed,
            "llm_repetition": trial_number,
            "llm_repetition_id": config.llm_repetition_id or f"rep_{trial_number + 1:03d}",
            "model_condition_id": config.model_condition_id,
            "planner_prompt_schema_version": config.prompt_schema_version,
            "reconciler_prompt_schema_version": BLINDED_RECONCILIATION_PROMPT_VERSION,
            "prompt_schema_version": config.prompt_schema_version,
            "prompt_schema_version_semantics": "deprecated alias for planner_prompt_schema_version; not an independent schema",
            "approved_target": case.target_column,
            "approved_task": case.expected_task_type,
            "generation_settings": _jsonable(config.generation_settings),
            "planner_evidence_mode": config.planner_evidence_mode,
            "planner_structural_diagnostics": planner_diagnostics,
            "api_provenance": _jsonable(planner_api_provenance),
        }

    comparison: dict[str, Any] | None = None
    target_disagreement = False
    task_disagreement = False
    method_disagreement = False
    preprocessing_disagreement = False
    agreement_status = "unavailable"
    reconciliation_invoked = False
    reconciliation_status = "not_invoked"
    reconciliation_method_source = None
    reconciliation_preprocessing_source = None
    reconciliation_agent_source = None
    reconciler_api_provenance: dict[str, Any] | None = None
    gate_sources: dict[str, str] = {}
    gate_result: dict[str, Any] | None = None
    gate_error: str | None = None
    final_fields = _plan_fields(None)
    final_validation: dict[str, Any] | None = None
    final_valid = False
    training_allowed = False

    if deterministic is not None:
        requirements = training_only_requirements(
            training_profile,
            case.target_column,
            case.expected_task_type,
            deterministic.recommended_method,
        )
        comparison = compare_preprocessing_plans(
            plan.preprocessing,
            deterministic.preprocessing,
            requirements,
        )
        method_disagreement = plan.recommended_method != deterministic.recommended_method
        preprocessing_disagreement = bool(comparison["material_differences"])
        agreement_status = (
            "agreement"
            if not (method_disagreement or preprocessing_disagreement)
            else "disagreement"
        )
        if config.gate_mode in {"llm_only", "deterministic_only"}:
            agreement_status = config.gate_mode
            final_fields = _plan_fields(
                plan,
                target_column=case.target_column,
                task_type=case.expected_task_type,
            )
            final_validation = initial_validation.as_dict()
            final_valid = initial_validation.status == "passed"
            training_allowed = final_valid
        else:
            gate_agent = _EvaluationGateAgent(
                live_agents=None if plan_factory is not None or config.offline else reconciler_agents,
                reconciliation_factory=reconciliation_factory,
                context=context,
            )
            gate_offline = config.offline or (
                plan_factory is None and not reconciler_agents.available and reconciliation_factory is None
            )
            try:
                gate_result = _validate_modeling_gate(
                    gate_agent,
                    training_profile,
                    case.question,
                    plan,
                    deterministic,
                    warnings,
                    gate_sources,
                    offline=gate_offline,
                    dataframe=frame,
                    test_size=config.test_size,
                    random_state=split_seed,
                    reconciliation_profile=training_profile,
                    split=split,
                    row_positions=list(range(len(frame))),
                    approved_target=case.target_column,
                    approved_task=case.expected_task_type,
                    soft_challenge_mode=config.gate_mode,
                    proposal_order_override=proposal_order,
                    soft_challenge_strategy=config.soft_challenge_strategy,
                    strict_live=config.require_live,
                    empirical_probe_policy=EmpiricalProbePolicy(
                        enabled=config.empirical_probe_enabled,
                        random_state=split_seed,
                    ),
                )
                reconciliation_invoked = gate_result.get("reconciliation") is not None
                if reconciliation_invoked and gate_sources.get("modeling_reconciliation") == "openai":
                    reconciler_api_provenance = getattr(
                        reconciler_agents, "last_request_provenance", None
                    )
                if (
                    config.require_live
                    and reconciliation_invoked
                    and gate_sources.get("modeling_reconciliation") == "openai"
                ):
                    reconciler_agents.assert_effective_model(
                        expected_model=config.reconciler_model
                    )
                reconciliation_status = (
                    "succeeded" if reconciliation_invoked else "not_invoked"
                )
                if reconciliation_invoked:
                    reconciliation_agent_source = gate_sources.get("modeling_reconciliation")
                    if plan_factory is not None or reconciliation_factory is not None:
                        reconciliation_agent_source = "mock"
                    elif config.offline:
                        reconciliation_agent_source = "offline_fallback"
                    if reconciliation_agent_source != "openai":
                        reconciler_api_provenance = None
                final_fields = {
                    "target": gate_result["selected_target_column"],
                    "task": gate_result["selected_task_type"],
                    "method": gate_result["selected_method"],
                    "preprocessing": gate_result["approved_preprocessing"],
                }
                final_validation = gate_result.get("deterministic_validation")
                final_valid = gate_result.get("overall_status") == "passed"
                training_allowed = final_valid
            except InvariantViolation as exc:
                gate_error = _redact_error(exc)
                reconciliation_invoked = "modeling_reconciliation" in gate_sources
                reconciliation_agent_source = gate_sources.get("modeling_reconciliation")
                reconciliation_status = "failed" if reconciliation_invoked else "not_invoked"
                if exc.result is not None:
                    final_validation = exc.result.as_dict()
            except Exception as exc:
                gate_error = _redact_error(exc)
                reconciliation_invoked = "modeling_reconciliation" in gate_sources
                reconciliation_agent_source = gate_sources.get("modeling_reconciliation")
                reconciliation_status = "failed" if reconciliation_invoked else "not_invoked"

        if reconciliation_invoked and reconciliation_status == "succeeded":
            selected_method = final_fields["method"]
            if selected_method == plan.recommended_method:
                reconciliation_method_source = "agent"
            elif selected_method == deterministic.recommended_method:
                reconciliation_method_source = "deterministic"
            else:
                reconciliation_method_source = "other"
            if final_fields["preprocessing"] == plan.preprocessing.model_dump(mode="json"):
                reconciliation_preprocessing_source = "agent"
            elif final_fields["preprocessing"] == deterministic.preprocessing.model_dump(mode="json"):
                reconciliation_preprocessing_source = "deterministic"
            else:
                reconciliation_preprocessing_source = "reconciled_contract"

    proceeded_unchanged = final_valid and _plan_matches(
        final_fields["target"],
        final_fields["task"],
        final_fields["method"],
        final_fields["preprocessing"],
        plan,
        expected_target=case.target_column,
        expected_task=case.expected_task_type,
    )
    unsafe_plan_intercepted = initial_validation.status == "failed" and not proceeded_unchanged
    # A soft intervention is a changed final plan resulting from an actual
    # challenge. Hard validation/repair remains a separate evaluation path.
    soft_decision = (gate_result or {}).get("soft_challenge_decision") or (
        (gate_result or {}).get("soft_challenge") or {}
    ).get("decision")
    intervention_occurred = bool(
        soft_decision == "challenge"
        and final_valid
        and not proceeded_unchanged
        and not unsafe_plan_intercepted
    )

    # Full empirical-reference candidate fitting is intentionally after the
    # gate call, so final summary results cannot influence the runtime
    # decision. The separate bounded pairwise arbitration probe, when needed,
    # is fit earlier on the frozen training partition only and is passed to
    # the gate as explicitly labeled evidence. Repetitions share this
    # training-only reference cache because their split and candidate
    # configuration are identical.
    cache_key = json.dumps(
        {
            "case": case.name,
            "task_type": case.expected_task_type,
            "target_column": case.target_column,
            "split": split.as_dict(),
            "training_frame_digest": _frame_digest(training_frame),
            "candidate_methods": [
                "linear",
                "regularized_linear",
                "tree_ensemble",
                "boosted_tree",
            ],
            "planner_prompt_schema_version": config.prompt_schema_version,
            "reconciler_prompt_schema_version": BLINDED_RECONCILIATION_PROMPT_VERSION,
            "prompt_schema_version": config.prompt_schema_version,
            "prompt_schema_version_semantics": "deprecated alias for planner_prompt_schema_version; not an independent schema",
        },
        sort_keys=True,
    )
    if cache_key not in empirical_reference_cache:
        empirical_reference_cache[cache_key] = evaluate_empirical_reference(
            training_frame,
            case.target_column,
            case.expected_task_type,
            training_profile,
            random_state=split_seed,
        )
    reference = empirical_reference_cache[cache_key]
    reference = {**reference, "frozen_split_contract": split.as_dict()}
    best_score = reference.get("best_primary_mean")
    agent_family_score = (
        _family_score(reference, plan.recommended_method) if initial_validation.status == "passed" else None
    )
    gated_family_score = _family_score(reference, final_fields["method"]) if final_valid else None
    deterministic_family_score = (
        _family_score(reference, deterministic.recommended_method) if deterministic is not None else None
    )
    agent_plan_cv = (
        evaluate_plan_cv(
            training_frame,
            case.target_column,
            case.expected_task_type,
            plan.recommended_method,
            plan.preprocessing,
            random_state=split_seed,
        )
        if initial_validation.status == "passed"
        else None
    )
    gated_plan_cv = (
        evaluate_plan_cv(
            training_frame,
            case.target_column,
            case.expected_task_type,
            final_fields["method"],
            PreprocessingContract.model_validate(final_fields["preprocessing"]),
            random_state=split_seed,
        )
        if final_valid
        else None
    )
    agent_holdout = (
        evaluate_holdout_plan(
            frame,
            split,
            case.target_column,
            case.expected_task_type,
            plan.recommended_method,
            plan.preprocessing,
            random_state=split_seed,
            row_positions=list(range(len(frame))),
        )
        if initial_validation.status == "passed"
        else None
    )
    gated_holdout = (
        evaluate_holdout_plan(
            frame,
            split,
            case.target_column,
            case.expected_task_type,
            final_fields["method"],
            PreprocessingContract.model_validate(final_fields["preprocessing"]),
            random_state=split_seed,
            row_positions=list(range(len(frame))),
        )
        if final_valid
        else None
    )
    holdout_metric_name = (
        "macro_f1" if case.expected_task_type == "classification" else "rmse"
    )
    initial_holdout_metric = (agent_holdout or {}).get("holdout_metrics", {}).get(holdout_metric_name)
    final_holdout_metric = (gated_holdout or {}).get("holdout_metrics", {}).get(holdout_metric_name)
    # This calculation is strictly post-decision evaluation. Nothing below is
    # passed back into the gate, probe, reconciliation, or final-plan choice.
    holdout_delta_raw = raw_holdout_performance_delta(
        case.expected_task_type, initial_holdout_metric, final_holdout_metric
    )
    paired_holdout_delta = paper_holdout_delta(
        case.expected_task_type,
        initial_holdout_metric,
        final_holdout_metric,
        epsilon=float(config.thresholds["holdout_rmse_epsilon"]),
    )
    holdout_tolerance = holdout_neutral_tolerance(
        case.expected_task_type, config.thresholds
    )
    holdout_outcome = (
        "not_intervened"
        if not intervention_occurred
        else (
            "not_comparable"
            if paired_holdout_delta is None
            else (
                "neutral"
                if abs(paired_holdout_delta) <= holdout_tolerance
                else ("beneficial" if paired_holdout_delta > 0 else "harmful")
            )
        )
    )
    agent_regret = regret(case.expected_task_type, best_score, agent_family_score)
    gated_regret = regret(case.expected_task_type, best_score, gated_family_score)
    paired_cv_improvement = None
    if agent_family_score is not None and gated_family_score is not None:
        paired_cv_improvement = (
            gated_family_score - agent_family_score
            if case.expected_task_type == "classification"
            else agent_family_score - gated_family_score
        )
    tolerance = float(config.thresholds.get("neutral_tolerance", config.thresholds["paired_normalized_regret"]))
    initial_normalized_regret = normalized_regret(case.expected_task_type, best_score, agent_family_score)
    final_normalized_regret = normalized_regret(case.expected_task_type, best_score, gated_family_score)
    deterministic_normalized_regret = normalized_regret(
        case.expected_task_type,
        best_score,
        deterministic_family_score,
    )
    normalized_gate_delta = regret_reduction(initial_normalized_regret, final_normalized_regret)
    gate_outcome = classify_intervention_outcome(normalized_gate_delta, tolerance)
    if normalized_gate_delta is None:
        gate_outcome = "not_comparable"
    gate_primary_delta = normalized_performance_delta(
        case.expected_task_type,
        (agent_plan_cv or {}).get("primary_mean"),
        (gated_plan_cv or {}).get("primary_mean"),
    )
    soft_decision = (
        (gate_result or {}).get("soft_challenge_decision")
        or (gate_result or {}).get("soft_challenge", {}).get("decision")
    )
    # Preserve the distinction between a challenge and an actual changed
    # final plan; reconciliation may preserve the initial proposal.
    soft_intervention_occurred = intervention_occurred
    final_selection_source = (gate_result or {}).get("final", {}).get("selected_source")
    if final_selection_source == "agent":
        final_selection_source = "initial_llm"
    elif final_selection_source == "deterministic":
        final_selection_source = "deterministic"
    elif final_selection_source == "reconciled_contract":
        final_selection_source = "reconciled_A" if (gate_result or {}).get("selected_proposal") == "A" else "reconciled_B"
    if final_selection_source is None and proceeded_unchanged:
        final_selection_source = "initial_llm"
    alternative_regret_reduction = regret_reduction(
        initial_normalized_regret, deterministic_normalized_regret
    )
    alternative_outcome = classify_intervention_outcome(alternative_regret_reduction, tolerance)
    catastrophe = catastrophic_transition(
        initial_normalized_regret,
        final_normalized_regret,
        float(config.thresholds["catastrophic_regret_threshold"]),
    )
    gate_outcome_details = {
        "intervention_occurred": soft_intervention_occurred,
        "soft_challenge_decision": soft_decision,
        "outcome": gate_outcome,
        "normalized_performance_delta": gate_primary_delta,
        "normalized_gate_delta": normalized_gate_delta,
        "initial_regret": initial_normalized_regret,
        "final_regret": final_normalized_regret,
        "deterministic_regret": deterministic_normalized_regret,
        "regret_reduction": normalized_gate_delta,
        "alternative_regret_reduction": alternative_regret_reduction,
        "alternative_outcome": alternative_outcome,
        **catastrophe,
        "unnecessary_intervention": bool(
            soft_intervention_occurred and gate_outcome == "neutral"
        ),
        "missed_rescue": bool(
            soft_decision == "abstain" and alternative_outcome == "improved"
        ),
        "good_abstention": bool(
            soft_decision == "abstain" and alternative_outcome == "worsened"
        ),
        "neutral_abstention": bool(
            soft_decision == "abstain" and alternative_outcome == "neutral"
        ),
        "evaluation_only": True,
        "objective_version": GATE_OBJECTIVE_VERSION,
    }
    initial_failure_codes = [failure["code"] for failure in _failed_checks(initial_validation)]
    final_failure_codes = _blocking_failed_check_codes(final_validation)
    validation_failure_codes = sorted(set(initial_failure_codes) | set(final_failure_codes))
    planner_effective_model = (
        (planner_api_provenance or {}).get("model_effective")
        or ((planner_api_provenance or {}).get("response_metadata") or {}).get("model")
        or (agents.model if agent_source in {"openai", "offline_fallback"} else agent_model)
    )
    reconciler_effective_model = (
        (reconciler_api_provenance or {}).get("model_effective")
        or ((reconciler_api_provenance or {}).get("response_metadata") or {}).get("model")
        or (reconciler_agents.model if reconciliation_agent_source in {"openai", "offline_fallback"} else config.reconciler_model)
    )
    record = {
        "benchmark_case": case.name,
        "dataset_source": case.dataset_source,
        "task_type": case.expected_task_type,
        "evaluation_mode": "modeling_gate",
        "gate_objective_version": GATE_OBJECTIVE_VERSION,
        "gate_mode": config.gate_mode,
        "trial": trial_number,
        "llm_repetition_id": repetition_id,
        "model_condition_id": config.model_condition_id,
        "trial_id": context["trial_id"],
        "evaluation_variant": variant,
        "order_swap_pair_id": order_swap_pair_id,
        "ablation_name": config.ablation_name,
        "ablation_schema_version": config.ablation_schema_version,
        "challenger_enabled": config.gate_mode != "llm_only",
        "hard_validation_enabled": True,
        "probe_enabled": config.empirical_probe_enabled and config.gate_mode in {"selective", "probe_first", "probe_direct", "full"},
        "reconciliation_enabled": config.gate_mode in {"always_reconcile", "selective", "probe_first", "full"},
        "reconcile_on_any_disagreement": config.gate_mode == "always_reconcile",
        "direct_probe_selection_enabled": config.gate_mode == "probe_direct",
        "abstention_enabled": config.gate_mode in {"selective", "probe_first", "probe_direct", "full"},
        "trial_status": "failed" if config.require_live and gate_error else "completed",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": experimental_split_seed,
        "split_random_state": split_seed,
        "test_size": config.test_size,
        "split_contract": split.as_dict(),
        "agent_source": agent_source,
        "requested_live_trial": (
            not config.offline
            and plan_factory is None
            and config.gate_mode != "deterministic_only"
        ),
        "agent_model": agent_model,
        "planner_model": config.planner_model,
        "reconciler_model": config.reconciler_model,
        "planner_model_effective": planner_effective_model,
        "reconciler_model_effective": reconciler_effective_model,
        "repository_commit": config.repository_commit,
        "experiment_config_version": EXPERIMENT_CONFIG_VERSION,
        "require_live": config.require_live,
        "benchmark_suite_version": (
            case.benchmark_suite_version
            or "local-2"
        ),
        "agent_request_status": agent_request_status,
        "agent_request_error": agent_request_error,
        "live_request_failed": agent_request_status == "failed_fallback",
        "api_status": (
            "live_succeeded" if agent_source == "openai"
            else "offline" if agent_source == "offline_fallback"
            else "mock" if agent_source == "mock"
            else "not_requested"
        ),
        "fallback_status": (
            "none" if agent_source in {"openai", "mock"}
            else agent_request_status or "offline"
        ),
        "initial_proposal_cache_key": proposal_key,
        "initial_proposal_cache_hit": proposal_cache_hit,
        "initial_modeling_call_made": (
            not proposal_cache_hit
            and config.gate_mode != "deterministic_only"
            and initial_plan_override is None
        ),
        "generation_settings": dict(config.generation_settings),
        "planner_evidence_mode": config.planner_evidence_mode,
        "planner_structural_diagnostics_exposed": planner_diagnostics is not None,
        "planner_structural_diagnostics": planner_diagnostics,
        "planner_api_provenance": planner_api_provenance,
        "reconciler_api_provenance": reconciler_api_provenance,
        "planner_prompt_schema_version": config.prompt_schema_version,
        "reconciler_prompt_schema_version": BLINDED_RECONCILIATION_PROMPT_VERSION,
        "prompt_schema_version": config.prompt_schema_version,
        "prompt_schema_version_semantics": "deprecated alias for planner_prompt_schema_version; not an independent schema",
        "agent_initial_input": initial_input_artifact,
        "agent_initial": {
            **initial_fields,
            "reasoning": plan.reasoning,
            "confidence": plan.confidence,
        },
        "agent_initial_target": initial_fields["target"],
        "agent_initial_task": initial_fields["task"],
        "agent_initial_target_source": "benchmark_constraint_for_modeling_gate",
        "agent_initial_task_source": "benchmark_constraint_for_modeling_gate",
        "agent_initial_method": initial_fields["method"],
        "agent_initial_preprocessing": initial_fields["preprocessing"],
        "agent_initial_valid": initial_validation.status == "passed",
        "agent_initial_validation_failures": _failed_checks(initial_validation),
        "agent_initial_validation": initial_validation.as_dict(),
        "deterministic_recommendation": deterministic.model_dump(mode="json") if deterministic else None,
        "deterministic_target": deterministic.target_column if deterministic else None,
        "deterministic_task": deterministic.task_type if deterministic else None,
        "deterministic_method": deterministic.recommended_method if deterministic else None,
        "deterministic_policy_version": deterministic.policy_version if deterministic else None,
        "deterministic_confidence": deterministic.confidence if deterministic else None,
        "deterministic_score_margin": deterministic.score_margin if deterministic else None,
        "deterministic_preprocessing": deterministic.preprocessing.model_dump(mode="json") if deterministic else None,
        "deterministic_failure": deterministic_failure,
        "agreement_status": agreement_status,
        "hard_validation_status": (
            (gate_result or {}).get("hard_validation", {}).get("status")
            if gate_result
            else ("failed" if initial_validation.status == "failed" else "unavailable")
        ),
        "soft_challenge_status": (
            (gate_result or {}).get("soft_challenge", {}).get("status")
            if gate_result
            else "unavailable"
        ),
        "soft_challenge_decision": (
            (gate_result or {}).get("soft_challenge_decision")
            or (gate_result or {}).get("soft_challenge", {}).get("decision")
        ),
        "target_disagreement": target_disagreement,
        "task_disagreement": task_disagreement,
        "method_disagreement": method_disagreement,
        "preprocessing_disagreement": preprocessing_disagreement,
        "preprocessing_agreement_status": (
            comparison.get("status") if comparison is not None else "unavailable"
        ),
        "preprocessing_comparison": comparison,
        "reconciliation_invoked": reconciliation_invoked,
        "reconciliation_status": reconciliation_status,
        "reconciliation_api_call_made": bool(
            reconciliation_invoked
            and reconciliation_agent_source == "openai"
        ),
        "reconciliation_request_failed": bool(
            reconciliation_invoked and reconciliation_status == "failed"
        ),
        "reconciliation_agent_source": reconciliation_agent_source,
        "reconciliation_method_source": reconciliation_method_source,
        "reconciliation_preprocessing_source": reconciliation_preprocessing_source,
        "reconciliation_mode": (gate_result or {}).get("reconciliation_mode"),
        "reconciliation_prompt_version": (gate_result or {}).get("reconciliation_prompt_version"),
        "proposal_order_seed": (gate_result or {}).get("proposal_order_seed"),
        "proposal_a_source": (gate_result or {}).get("proposal_a_source"),
        "proposal_b_source": (gate_result or {}).get("proposal_b_source"),
        "selected_proposal": (gate_result or {}).get("selected_proposal"),
        "selected_proposal_source": (gate_result or {}).get("selected_proposal_source"),
        "reconciliation": gate_result.get("reconciliation") if gate_result else None,
        "hard_validation": gate_result.get("hard_validation") if gate_result else {
            "status": "failed" if initial_validation.status == "failed" else "unavailable",
            "intervention_required": initial_validation.status == "failed",
            "initial_hard_invalid": initial_validation.status == "failed",
            "checks": initial_validation.as_dict().get("checks", []),
            "initial_proposal": initial_validation.as_dict(),
            "deterministic_challenger": {},
            "final_plan": final_validation or {},
        },
        "soft_challenge": gate_result.get("soft_challenge") if gate_result else {
            "status": "unavailable",
            "agent_method": plan.recommended_method,
            "deterministic_method": deterministic.recommended_method if deterministic else None,
            "deterministic_confidence": deterministic.confidence if deterministic else None,
            "method_disagreement": method_disagreement,
            "preprocessing_disagreement": preprocessing_disagreement,
            "decision": "agree" if not method_disagreement else "abstain",
            "status_detail": "agreement" if not method_disagreement else "abstained",
            "reconciliation_invoked": False,
            "reconciliation_status": "not_invoked",
        },
        "empirical_probe_invoked": bool((gate_result or {}).get("empirical_probe_invoked", False)),
        "empirical_probe_policy_version": (gate_result or {}).get(
            "empirical_probe_policy_version", EmpiricalProbePolicy().policy_version
        ),
        "empirical_probe_status": ((gate_result or {}).get("empirical_probe") or {}).get("status"),
        "empirical_probe": (gate_result or {}).get("empirical_probe"),
        "probe_status": (gate_result or {}).get("probe_status") or ((gate_result or {}).get("empirical_probe") or {}).get("status"),
        "probe_evidence_strength": (gate_result or {}).get("probe_evidence_strength") or ((gate_result or {}).get("empirical_probe") or {}).get("evidence_strength"),
        "abstention_reason": (gate_result or {}).get("abstention_reason"),
        "gate_decision": (gate_result or {}).get("gate_decision"),
        "decision_path": (gate_result or {}).get("decision_path"),
        "final_decision": gate_result.get("final") if gate_result else {
            **final_fields,
            "selected_source": None,
        },
        "final_target": final_fields["target"],
        "final_task": final_fields["task"],
        "final_method": final_fields["method"],
        "final_preprocessing": final_fields["preprocessing"],
        "gated_final": {
            **final_fields,
            "valid": final_valid,
        },
        "gated_final_target": final_fields["target"],
        "gated_final_task": final_fields["task"],
        "gated_final_method": final_fields["method"],
        "gated_final_preprocessing": final_fields["preprocessing"],
        "final_valid": final_valid,
        "final_validation": final_validation,
        "training_allowed": training_allowed,
        "unsafe_plan_intercepted": unsafe_plan_intercepted,
        "proceeded_unchanged": proceeded_unchanged,
        "gate_changed_initial_plan": final_valid and not proceeded_unchanged,
        "intervention_occurred": intervention_occurred,
        "hard_repair_occurred": unsafe_plan_intercepted,
        "soft_intervention_occurred": soft_intervention_occurred,
        "final_selection_source": final_selection_source,
        "deterministic_validation_intervened": unsafe_plan_intercepted,
        "hard_validation_intervened": bool(
            (gate_result or {}).get("hard_validation", {}).get("intervention_required")
            or unsafe_plan_intercepted
        ),
        "initial_hard_invalid": initial_validation.status == "failed",
        "final_hard_invalid": final_valid is False,
        "empirical_best_method": reference.get("best_method"),
        "empirical_reference_method": reference.get("best_method"),
        "initial_agent_choice": {
            "method": initial_fields["method"],
            "performance": agent_family_score,
            "normalized_regret": initial_normalized_regret,
        },
        "final_gated_choice": {
            "method": final_fields["method"],
            "performance": gated_family_score,
            "normalized_regret": final_normalized_regret,
        },
        "empirical_reference_choice": {
            "method": reference.get("best_method"),
            "performance": best_score,
            "normalized_regret": 0.0 if best_score is not None else None,
        },
        "empirical_reference": reference,
        "candidate_cv_metrics": reference.get("candidate_metrics", {}),
        "agent_initial_cv_metric": agent_family_score,
        "agent_initial_plan_cv_metric": (agent_plan_cv or {}).get("primary_mean"),
        "gated_final_cv_metric": gated_family_score,
        "gated_final_plan_cv_metric": (gated_plan_cv or {}).get("primary_mean"),
        "agent_regret": agent_regret,
        "gated_regret": gated_regret,
        "agent_normalized_regret": initial_normalized_regret,
        "gated_normalized_regret": final_normalized_regret,
        "initial_regret": initial_normalized_regret,
        "final_regret": final_normalized_regret,
        "deterministic_regret": deterministic_normalized_regret,
        "deterministic_normalized_regret": deterministic_normalized_regret,
        "challenge_regret_delta": (
            normalized_regret(case.expected_task_type, best_score, agent_family_score)
            - deterministic_normalized_regret
            if normalized_regret(case.expected_task_type, best_score, agent_family_score) is not None
            and deterministic_normalized_regret is not None
            else None
        ),
        "paired_cv_improvement": paired_cv_improvement,
        "gate_outcome": gate_outcome,
        "gate_outcome_tolerance": tolerance,
        "gate_outcome_details": gate_outcome_details,
        "normalized_gate_delta": normalized_gate_delta,
        "regret_reduction": normalized_gate_delta,
        "initial_catastrophic": catastrophe["initial_catastrophic"],
        "final_catastrophic": catastrophe["final_catastrophic"],
        "initial_agent_catastrophic": catastrophe["initial_catastrophic"],
        "final_gate_catastrophic": catastrophe["final_catastrophic"],
        "catastrophic_prevented": catastrophe["catastrophic_prevented"],
        "catastrophic_introduced": catastrophe["catastrophic_introduced"],
        "unnecessary_intervention": gate_outcome_details["unnecessary_intervention"],
        "missed_rescue": gate_outcome_details["missed_rescue"],
        "agent_initial_cv": agent_plan_cv,
        "gated_final_cv": gated_plan_cv,
        "agent_initial_holdout_metrics": (agent_holdout or {}).get("holdout_metrics", {}),
        "gated_final_holdout_metrics": (gated_holdout or {}).get("holdout_metrics", {}),
        "initial_holdout_metric": initial_holdout_metric,
        "final_holdout_metric": final_holdout_metric,
        "holdout_metric_name": holdout_metric_name,
        "holdout_metric_schema_version": HOLDOUT_METRIC_SCHEMA_VERSION,
        "holdout_rmse_epsilon": float(config.thresholds["holdout_rmse_epsilon"]),
        "holdout_neutral_tolerance": holdout_tolerance,
        "holdout_neutral_tolerance_units": (
            "absolute macro-F1 delta"
            if case.expected_task_type == "classification"
            else "relative RMSE improvement"
        ),
        "holdout_macro_f1_delta": (
            holdout_delta_raw if case.expected_task_type == "classification" else None
        ),
        "holdout_rmse_delta_raw": (
            holdout_delta_raw if case.expected_task_type == "regression" else None
        ),
        "holdout_rmse_relative_improvement": (
            paired_holdout_delta if case.expected_task_type == "regression" else None
        ),
        "paper_holdout_delta": paired_holdout_delta,
        # Preserve the historical field in its native-unit meaning.  New
        # paper-facing consumers must use paper_holdout_delta or the explicit
        # task-specific fields above.
        "holdout_intervention_delta": holdout_delta_raw,
        "holdout_intervention_delta_raw": holdout_delta_raw,
        "holdout_intervention_delta_semantics": "deprecated native-unit legacy delta; use paper_holdout_delta for paper analysis",
        "holdout_intervention_outcome": holdout_outcome,
        "agent_initial_holdout_split_contract": (agent_holdout or {})
        .get("validation", {})
        .get("split", {})
        .get("contract"),
        "gated_final_holdout_split_contract": (gated_holdout or {})
        .get("validation", {})
        .get("split", {})
        .get("contract"),
        "validation_failure_codes": validation_failure_codes,
        "holdout_policy": {
            "frozen_before_model_family_decisions": True,
            "used_for_agent_planning": False,
            "used_for_deterministic_recommendation": False,
            "used_for_reconciliation": False,
            "used_for_candidate_selection": False,
            "used_for_preprocessing_fitting": False,
            "used_for_cross_validation": False,
            "used_for_final_evaluation": True,
        },
        "perturbation_id": perturbation_id,
        "perturbation": perturbation_data,
        "expected_perturbation_checks_observed": sorted(
            set(perturbation_data.get("expected_validation_codes", []))
            & {
                failure["code"]
                for failure in _failed_checks(initial_validation)
            }
            | {
                check["code"]
                for check in (final_validation or {}).get("checks", [])
                if check.get("status") == "failed"
                and check.get("blocking", check.get("severity", "error") == "error")
            }
        ),
        "failure_reason": gate_error
        or deterministic_failure
        or (
            initial_validation.failed_checks[0].code
            if initial_validation.status == "failed" and initial_validation.failed_checks
            else None
        ),
        "warnings": warnings,
        "fallback_row": agent_source == "offline_fallback" or reconciliation_agent_source == "offline_fallback",
        "empirical_reference_cache_key": cache_key,
    }
    record.update(case.provenance())
    return _jsonable(record)


def _write_outputs(
    output_dir: Path,
    config_payload: dict[str, Any],
    trials: list[dict[str, Any]],
    summary: dict[str, Any],
    empirical_reference_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "trials.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for trial in trials:
            handle.write(json.dumps(trial, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if empirical_reference_cache is not None:
        (output_dir / "empirical_reference.json").write_text(
            json.dumps(empirical_reference_cache, indent=2, sort_keys=True), encoding="utf-8"
        )
    from evaluation.reporting import render_summary_markdown

    (output_dir / "summary.md").write_text(
        render_summary_markdown(config_payload, trials, summary), encoding="utf-8"
    )
    return {
        "config": str(output_dir / "config.json"),
        "trials": str(output_dir / "trials.jsonl"),
        "summary": str(output_dir / "summary.json"),
        "summary_markdown": str(output_dir / "summary.md"),
        "empirical_reference": str(output_dir / "empirical_reference.json"),
    }


def _failed_trial_record(
    case: BenchmarkCase,
    perturbation: Perturbation | None,
    trial_number: int,
    config: EvaluationConfig,
    error: Exception,
    variant: str = "standard",
    order_swap_pair_id: str | None = None,
    requested_split_seed: int | None = None,
) -> dict[str, Any]:
    """Persist an execution failure without inventing a modeling decision."""

    experimental_split_seed = config.seed if requested_split_seed is None else requested_split_seed
    split_seed = experimental_split_seed + case.random_seed
    perturbation_id = perturbation.id if perturbation is not None else "clean"
    split = freeze_supervised_split(
        case.load(),
        case.target_column,
        case.expected_task_type,
        test_size=config.test_size,
        random_state=split_seed,
    )
    repetition_id = config.llm_repetition_id or f"rep_{trial_number + 1:03d}"
    trial_id = (
        f"{config.model_condition_id}:{case.name}:{perturbation_id}:split{experimental_split_seed}:"
        f"{repetition_id}:llm{trial_number}"
    )
    if config.ablation_name:
        trial_id = f"{trial_id}:{config.ablation_name}"
    if variant != "standard":
        trial_id = f"{trial_id}:{variant}"
    return {
        "trial_id": trial_id,
        "benchmark_case": case.name,
        "dataset_source": case.dataset_source,
        "task_type": case.expected_task_type,
        "trial": trial_number,
        "llm_repetition_id": repetition_id,
        "model_condition_id": config.model_condition_id,
        "trial_status": "failed",
        "evaluation_variant": variant,
        "order_swap_pair_id": order_swap_pair_id,
        "gate_mode": config.gate_mode,
        "ablation_name": config.ablation_name,
        "ablation_schema_version": config.ablation_schema_version,
        "challenger_enabled": config.gate_mode != "llm_only",
        "hard_validation_enabled": True,
        "probe_enabled": config.empirical_probe_enabled and config.gate_mode in {"selective", "probe_first", "probe_direct", "full"},
        "reconciliation_enabled": config.gate_mode in {"always_reconcile", "selective", "probe_first", "full"},
        "reconcile_on_any_disagreement": config.gate_mode == "always_reconcile",
        "direct_probe_selection_enabled": config.gate_mode == "probe_direct",
        "abstention_enabled": config.gate_mode in {"selective", "probe_first", "probe_direct", "full"},
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": experimental_split_seed,
        "split_random_state": split_seed,
        "split_contract": split.as_dict(),
        "agent_source": "failed",
        "requested_live_trial": not config.offline and config.gate_mode != "deterministic_only",
        "agent_model": config.planner_model,
        "planner_model": config.planner_model,
        "reconciler_model": config.reconciler_model,
        "planner_model_effective": config.planner_model,
        "reconciler_model_effective": config.reconciler_model,
        "repository_commit": config.repository_commit,
        "experiment_config_version": EXPERIMENT_CONFIG_VERSION,
        "require_live": config.require_live,
        "benchmark_suite_version": case.benchmark_suite_version or "local-2",
        "agent_request_status": "failed",
        "agent_request_error": _redact_error(error),
        "live_request_failed": not config.offline,
        "api_status": "failed_live_request" if not config.offline else "offline",
        "fallback_status": "none",
        "planner_prompt_schema_version": config.prompt_schema_version,
        "reconciler_prompt_schema_version": BLINDED_RECONCILIATION_PROMPT_VERSION,
        "prompt_schema_version": config.prompt_schema_version,
        "prompt_schema_version_semantics": "deprecated alias for planner_prompt_schema_version; not an independent schema",
        "generation_settings": dict(config.generation_settings),
        "planner_api_provenance": None,
        "reconciler_api_provenance": None,
        "agent_initial": None,
        "agent_initial_valid": None,
        "deterministic_recommendation": None,
        "agreement_status": "unavailable",
        "reconciliation_invoked": False,
        "reconciliation_status": "not_invoked",
        "final_valid": None,
        "training_allowed": False,
        "unsafe_plan_intercepted": False,
        "proceeded_unchanged": False,
        "empirical_reference": None,
        "candidate_cv_metrics": {},
        "agent_initial_cv_metric": None,
        "gated_final_cv_metric": None,
        "initial_holdout_metric": None,
        "final_holdout_metric": None,
        "holdout_metric_name": None,
        "holdout_metric_schema_version": HOLDOUT_METRIC_SCHEMA_VERSION,
        "holdout_rmse_epsilon": float(config.thresholds["holdout_rmse_epsilon"]),
        "holdout_neutral_tolerance": None,
        "holdout_neutral_tolerance_units": None,
        "holdout_macro_f1_delta": None,
        "holdout_rmse_delta_raw": None,
        "holdout_rmse_relative_improvement": None,
        "paper_holdout_delta": None,
        "holdout_intervention_delta": None,
        "holdout_intervention_delta_raw": None,
        "holdout_intervention_delta_semantics": "deprecated native-unit legacy delta; use paper_holdout_delta for paper analysis",
        "holdout_intervention_outcome": "not_comparable",
        "intervention_occurred": False,
        "hard_repair_occurred": False,
        "soft_intervention_occurred": False,
        "final_selection_source": None,
        "agent_normalized_regret": None,
        "gated_normalized_regret": None,
        "paired_cv_improvement": None,
        "gate_outcome": "not_comparable",
        "final_method": None,
        "perturbation_id": perturbation_id,
        "perturbation": perturbation.as_dict() if perturbation is not None else {
            "id": "clean",
            "kind": "clean",
            "description": "Unperturbed benchmark case.",
            "expected_validation_codes": [],
        },
        "validation_failure_codes": [],
        "failure_reason": _redact_error(error),
        "warnings": ["Trial execution failed; no model-selection result was recorded."],
        "initial_modeling_call_made": config.gate_mode != "deterministic_only",
        "initial_proposal_cache_hit": False,
        "fallback_row": False,
        "reconciliation_api_call_made": False,
        "reconciliation_request_failed": False,
        **case.provenance(),
    }


def _cached_plan_from_trial(record: dict[str, Any]) -> ModelingPlan | None:
    """Recover the first pair member's proposal for an order-swap rerun."""

    values = record.get("agent_initial")
    if not isinstance(values, dict):
        return None
    try:
        return ModelingPlan(
            recommended_method=values["method"],
            preprocessing=values["preprocessing"],
            reasoning=str(values.get("reasoning", "The cached order-swap proposal was retained.")),
            confidence=float(values.get("confidence", 0.0)),
        )
    except Exception:
        return None


def run_evaluation(
    output_dir: str | Path,
    *,
    cases: Sequence[BenchmarkCase] | None = None,
    repetitions: int = 1,
    seed: int = 42,
    test_size: float = 0.2,
    model: str = "gpt-4.1-mini",
    planner_model: str | None = None,
    reconciler_model: str | None = None,
    offline: bool = False,
    include_perturbations: bool = False,
    thresholds: dict[str, float] | None = None,
    case_names: Sequence[str] | None = None,
    modeling_plan_factory: ModelingPlanFactory | None = None,
    reconciliation_factory: ModelingReconciliationFactory | None = None,
    # The harness default exercises the always-reconcile comparison; research
    # ablations pass their explicit current-method gate mode.
    gate_mode: str = "always_reconcile",
    order_swap: bool = False,
    empirical_probe_enabled: bool = True,
    resume: bool = False,
    split_seeds: Sequence[int] | None = None,
    require_live: bool = False,
    soft_challenge_strategy: str = "calibrated",
    enable_regression_interaction_diagnostics: bool = True,
    enable_classification_boundary_diagnostics: bool = True,
    ablation_name: str | None = None,
    ablation_schema_version: str | None = None,
    ablation_spec: Any | None = None,
    proposal_cache_path: str | Path | None = None,
    empirical_reference_cache_path: str | Path | None = None,
    suite: str = "local",
    tier: str | None = None,
    confirmatory_config_path: str | Path | None = None,
    confirmatory_selected_ablations: Sequence[str] | None = None,
    model_condition_id: str = "default",
    llm_repetition_id: str | None = None,
    generation_settings: dict[str, Any] | None = None,
    llm_repetition_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run reproducible trials and write the structured evaluation bundle."""

    spec_values: dict[str, Any] | None = None
    if ablation_spec is not None:
        spec_values = ablation_spec.as_dict() if hasattr(ablation_spec, "as_dict") else dict(ablation_spec)
        gate_mode = spec_values.get("decision_mode", gate_mode)
        soft_challenge_strategy = spec_values.get("soft_challenge_strategy", soft_challenge_strategy)
        empirical_probe_enabled = bool(spec_values.get("empirical_probe", empirical_probe_enabled))
        enable_regression_interaction_diagnostics = bool(
            spec_values.get(
                "interaction_diagnostics",
                enable_regression_interaction_diagnostics,
            )
        )
        enable_classification_boundary_diagnostics = bool(
            spec_values.get(
                "classification_boundary_diagnostics",
                enable_classification_boundary_diagnostics,
            )
        )
        ablation_name = spec_values.get("name", ablation_name)
        ablation_schema_version = spec_values.get("schema_version", ablation_schema_version)
        planner_evidence_mode = spec_values.get(
            "planner_evidence_mode", "training_profile_only"
        )
    else:
        planner_evidence_mode = "training_profile_only"
    if require_live and offline:
        raise ValueError("require_live cannot be combined with offline mode.")
    if require_live and (modeling_plan_factory is not None or reconciliation_factory is not None):
        raise ValueError(
            "require_live is incompatible with injected mock planner/reconciler factories."
        )
    selected_split_seeds = tuple(int(value) for value in (split_seeds or [seed]))
    config = EvaluationConfig(
        suite=suite,
        tier=tier,
        repetitions=repetitions,
        seed=seed,
        test_size=test_size,
        model=model,
        planner_model=planner_model or model,
        reconciler_model=reconciler_model or model,
        offline=offline,
        include_perturbations=include_perturbations,
        thresholds={**DEFAULT_THRESHOLDS, **(thresholds or {})},
        repository_commit=_repository_commit(),
        gate_mode=gate_mode,
        order_swap=order_swap,
        empirical_probe_enabled=empirical_probe_enabled,
        prompt_schema_version=PROMPT_SCHEMA_VERSION,
        split_seeds=selected_split_seeds,
        require_live=require_live,
        soft_challenge_strategy=soft_challenge_strategy,
        enable_regression_interaction_diagnostics=enable_regression_interaction_diagnostics,
        enable_classification_boundary_diagnostics=enable_classification_boundary_diagnostics,
        ablation_name=ablation_name,
        ablation_schema_version=ablation_schema_version,
        model_condition_id=model_condition_id,
        llm_repetition_id=llm_repetition_id,
        generation_settings=dict(generation_settings or {}),
        llm_repetition_ids=tuple(str(value) for value in llm_repetition_ids) if llm_repetition_ids is not None else None,
        planner_evidence_mode=planner_evidence_mode,
    )
    if cases is not None:
        selected_cases = list(cases)
    elif suite == "external":
        from evaluation.external_benchmarks import external_benchmark_cases

        selected_cases = external_benchmark_cases()
    else:
        selected_cases = default_benchmark_cases()
    if case_names:
        wanted = set(case_names)
        selected_cases = [case for case in selected_cases if case.name in wanted]
    if tier is not None:
        selected_cases = [case for case in selected_cases if case.tier == tier]
    if not selected_cases:
        raise ValueError("No benchmark cases selected.")
    perturbations = default_perturbations() if include_perturbations else []
    output_path = Path(output_dir).resolve()
    confirmatory_metadata: dict[str, Any] | None = None
    legacy_offline_condition_projection = False
    if confirmatory_config_path is not None:
        if config.suite != "external":
            raise ValueError("Confirmatory manifest enforcement requires suite='external'.")
        manifest = load_confirmatory_manifest(confirmatory_config_path)
        declared_conditions = model_conditions(manifest)
        selected_condition = next(
            (item for item in declared_conditions if item["condition_id"] == config.model_condition_id),
            None,
        )
        if selected_condition is None:
            if (
                config.offline
                and not config.require_live
                and config.model_condition_id == "default"
                and len(declared_conditions) > 1
            ):
                # Compatibility for offline artifact/copy tests and legacy
                # exploratory callers.  This path is never permitted for a
                # strict-live confirmatory run, and the selected condition is
                # recorded explicitly below rather than being hidden.
                selected_condition = declared_conditions[0]
                object.__setattr__(config, "model_condition_id", selected_condition["condition_id"])
                object.__setattr__(config, "model", selected_condition["planner_model"])
                object.__setattr__(config, "repetitions", int(selected_condition["llm_repetitions"]))
                object.__setattr__(config, "planner_model", selected_condition["planner_model"])
                object.__setattr__(config, "reconciler_model", selected_condition["reconciler_model"])
                legacy_offline_condition_projection = True
            else:
                raise ValueError(
                    f"Confirmatory model condition {config.model_condition_id!r} is not declared in the frozen manifest."
                )
        object.__setattr__(config, "model", selected_condition["planner_model"])
        selected_repetitions = selected_condition["llm_repetitions"]
        if config.repetitions != selected_repetitions:
            raise ValueError(
                f"Confirmatory repetition mismatch for {config.model_condition_id!r}: "
                f"manifest declares {selected_repetitions}, runtime requested {config.repetitions}."
            )
        repetitions = int(selected_repetitions)
        if config.planner_model != selected_condition["planner_model"] or config.reconciler_model != selected_condition["reconciler_model"]:
            raise ValueError(
                "Confirmatory model override conflicts with the frozen model condition "
                f"{config.model_condition_id!r}."
            )
        condition_generation = selected_condition.get("generation_settings", {})
        if not config.generation_settings and condition_generation:
            object.__setattr__(config, "generation_settings", dict(condition_generation))
        if dict(config.generation_settings) != dict(condition_generation):
            raise ValueError(
                f"Confirmatory generation settings differ from frozen condition {config.model_condition_id!r}."
            )
        expected_repetition_ids = condition_repetition_ids(manifest, selected_condition)
        if config.llm_repetition_ids is None:
            object.__setattr__(config, "llm_repetition_ids", tuple(expected_repetition_ids))
        elif list(config.llm_repetition_ids) != expected_repetition_ids:
            raise ValueError(
                f"Confirmatory repetition IDs differ from frozen condition {config.model_condition_id!r}."
            )
        frozen_order_swap = bool(manifest.get("order_swap", False))
        if config.order_swap != frozen_order_swap:
            raise ValueError(
                "Confirmatory order-swap setting differs from the frozen manifest."
            )
        frozen_perturbations = manifest.get("perturbations", []) or []
        if config.include_perturbations != bool(frozen_perturbations):
            raise ValueError(
                "Confirmatory perturbation dimensions differ from the frozen manifest."
            )
        configured_holdout_tolerances = {
            "classification": holdout_neutral_tolerance("classification", config.thresholds),
            "regression": holdout_neutral_tolerance("regression", config.thresholds),
        }
        runtime_values = runtime_manifest_values(
            experiment_name=CONFIRMATORY_EXPERIMENT_NAME,
            planner_model=str(config.planner_model),
            reconciler_model=str(config.reconciler_model),
            split_seeds=list(config.split_seeds),
            llm_repetitions=config.repetitions,
            holdout_fraction=config.test_size,
            selected_ablations=(
                list(confirmatory_selected_ablations)
                if confirmatory_selected_ablations is not None
                else ([config.ablation_name] if config.ablation_name else None)
            ),
            deterministic_policy_version=DeterministicPolicy().version,
            deterministic_policy_sha256=config_sha256(deterministic_policy_config()),
            empirical_probe_policy_version=EmpiricalProbePolicy().policy_version,
            empirical_probe_policy_sha256=config_sha256(empirical_probe_config()),
            planner_prompt_schema_version=config.prompt_schema_version,
            reconciler_prompt_schema_version=BLINDED_RECONCILIATION_PROMPT_VERSION,
            candidate_model_families=[
                "linear", "regularized_linear", "tree_ensemble", "boosted_tree"
            ],
            preprocessing_option_space=[
                "one_hot/categorical_unknown_handling=ignore",
                "ordinal/categorical_unknown_handling=use_encoded_value",
                "none/categorical_unknown_handling=ignore",
            ],
            classification_neutral_tolerance=configured_holdout_tolerances["classification"],
            regression_neutral_tolerance=configured_holdout_tolerances["regression"],
            benchmark_manifest_version=(
                selected_cases[0].benchmark_suite_version
                if selected_cases[0].benchmark_suite_version
                else "local-2"
            ),
            benchmark_manifest_sha256=external_benchmark_manifest_sha256(),
            benchmark_task_ids=[case.openml_task_id for case in selected_cases if case.openml_task_id is not None],
            benchmark_tranches={
                "core": [spec.task_id for spec in external_benchmark_specs() if spec.tier == "core"],
                "stress": [spec.task_id for spec in external_benchmark_specs() if spec.tier == "stress"],
            },
            benchmark_tier=config.tier,
            strict_live_required=config.require_live,
            bootstrap_settings={
                "method": "dataset_cluster_bootstrap_percentile",
                "replicates": DEFAULT_BOOTSTRAP_REPLICATES,
                "confidence_level": DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL,
                "seed": DEFAULT_BOOTSTRAP_SEED,
            },
            experiment_config_version=EXPERIMENT_CONFIG_VERSION,
            expected_experiment_code_sha256=experiment_code_sha256(),
            source_git_commit=repository_commit(),
            model_conditions=declared_conditions,
            generation_settings=dict(manifest.get("generation_settings", {}) or {}),
            llm_repetition_ids=list(config.llm_repetition_ids or repetition_ids(manifest)),
            selected_model_condition_id=config.model_condition_id,
        )
        confirmatory_metadata = validate_confirmatory_manifest(manifest, runtime_values)
    # Validate settings before constructing or using a client.  In strict mode
    # this is before any paid request can be attempted.
    agents = OpenAIAgents(
        model=config.planner_model,
        generation_settings=config.generation_settings,
        respect_environment_model=confirmatory_metadata is None,
    )
    reconciler_agents = OpenAIAgents(
        model=config.reconciler_model,
        generation_settings=config.generation_settings,
        respect_environment_model=confirmatory_metadata is None,
    )
    stable_config = {
        "config_version": "2026-09-04.evaluation.v5-paper-metrics",
        "experiment_config_version": EXPERIMENT_CONFIG_VERSION,
        "confirmatory_config_snapshot": CONFIRMATORY_CONFIG_SNAPSHOT,
        "confirmatory_mode": confirmatory_metadata is not None,
        "legacy_offline_condition_projection": legacy_offline_condition_projection
        if confirmatory_config_path is not None else False,
        "confirmatory_config_status": (
            confirmatory_metadata["status"] if confirmatory_metadata else "not_selected"
        ),
        "experiment_config_path": str(Path(confirmatory_config_path).resolve()) if confirmatory_config_path else None,
        "experiment_config_sha256": (
            confirmatory_metadata.get("experiment_config_sha256")
            if confirmatory_metadata else None
        ),
        "expected_experiment_code_sha256": (
            confirmatory_metadata.get("expected_experiment_code_sha256")
            if confirmatory_metadata else None
        ),
        "source_git_commit": confirmatory_metadata.get("source_git_commit") if confirmatory_metadata else repository_commit(),
        "frozen_manifest_path": None,
        "strict_live_required": config.require_live,
        "fallback_rows": None,
        "config_mismatch_detected": False,
        "confirmatory_valid": None,
        "result_schema_version": HOLDOUT_METRIC_SCHEMA_VERSION,
        "gate_objective_version": GATE_OBJECTIVE_VERSION,
        "suite": config.suite,
        "tier": config.tier,
        "benchmark_manifest_version": (
            selected_cases[0].benchmark_suite_version
            if selected_cases and selected_cases[0].benchmark_suite_version
            else "local-2"
        ),
        "benchmark_cases": [case.as_dict() for case in selected_cases],
        "perturbations": [perturbation.as_dict() for perturbation in perturbations],
        "repetitions": config.repetitions,
        "seed": config.seed,
        "split_seeds": list(config.split_seeds),
        "llm_repetitions": config.repetitions,
        "llm_repetition_id": config.llm_repetition_id,
        "llm_repetition_ids": list(config.llm_repetition_ids or []),
        "test_size": config.test_size,
        "agent_mode": "offline" if config.offline else "live_required" if config.require_live else "live_or_fallback",
        "model": config.model,
        "model_condition_id": config.model_condition_id,
        "planner_model": config.planner_model,
        "reconciler_model": config.reconciler_model,
        "agent_model_requested": config.model,
        "planner_model_requested": config.planner_model,
        "reconciler_model_requested": config.reconciler_model,
        "generation_settings": dict(config.generation_settings),
        "planner_evidence_mode": config.planner_evidence_mode,
        "planner_structural_diagnostics_exposed": config.planner_evidence_mode == "training_only_structural_diagnostics",
        "planner_model_effective": agents.model,
        "reconciler_model_effective": reconciler_agents.model,
        "agent_source_policy": "openai, offline_fallback, mock, or failed; source is persisted per trial",
        "gate_mode": config.gate_mode,
        "order_swap": config.order_swap,
        "empirical_probe_enabled": config.empirical_probe_enabled,
        "require_live": config.require_live,
        "live_required": config.require_live,
        "soft_challenge_strategy": config.soft_challenge_strategy,
        "diagnostic_configuration": {
            "enable_regression_interaction_diagnostics": config.enable_regression_interaction_diagnostics,
            "enable_classification_boundary_diagnostics": config.enable_classification_boundary_diagnostics,
        },
        "ablation_name": config.ablation_name,
        "ablation_schema_version": config.ablation_schema_version,
        "ablation_spec": spec_values,
        "selected_ablations": (
            list(confirmatory_selected_ablations)
            if confirmatory_selected_ablations is not None
            else ([config.ablation_name] if config.ablation_name else None)
        ),
        "empirical_probe_policy_version": EmpiricalProbePolicy().policy_version,
        "gate_mode_definitions": {
            "llm_only": "retain the initial agent plan after initial validation; never reconcile soft disagreement",
            "deterministic_only": "use the deterministic recommendation directly without an initial modeling-agent call",
            "always_reconcile": "invoke the existing reconciliation path for every valid soft disagreement",
            "selective": "invoke reconciliation only when the versioned soft-challenge policy authorizes a challenge",
            "probe_direct": "run the bounded pairwise training-only probe; directly select a moderate or strong empirical winner",
            "full": "run the bounded pairwise training-only probe; invoke blinded reconciliation only for moderate or strong evidence",
        },
        "planner_prompt_schema_version": config.prompt_schema_version,
        "reconciler_prompt_schema_version": BLINDED_RECONCILIATION_PROMPT_VERSION,
        "prompt_schema_version": config.prompt_schema_version,
        "prompt_schema_version_semantics": "deprecated alias for planner_prompt_schema_version; not an independent schema",
        "deterministic_policy_version": DeterministicPolicy().version,
        "soft_challenge_policy_version": SOFT_CHALLENGE_POLICY_VERSION,
        "soft_challenge_calibration_artifact_version": load_calibration_artifact().get(
            "calibration_artifact_version"
        ),
        "evaluation_settings": {
            "primary_metrics": {"classification": "macro_f1", "regression": "rmse"},
            "candidate_methods": ["linear", "regularized_linear", "tree_ensemble", "boosted_tree"],
            "candidate_selection_data": "frozen training partition only",
            "holdout_data": "final evaluation only",
            "holdout_macro_f1_delta": "final_holdout_macro_f1-initial_holdout_macro_f1",
            "holdout_rmse_delta_raw": "initial_holdout_rmse-final_holdout_rmse (diagnostic/native units)",
            "holdout_rmse_relative_improvement": "(initial_holdout_rmse-final_holdout_rmse)/max(abs(initial_holdout_rmse), holdout_rmse_epsilon)",
            "paper_holdout_delta": "classification=holdout_macro_f1_delta; regression=holdout_rmse_relative_improvement",
            "repetition_design": "same split and training-only profile; stochastic LLM response is the intended varying factor",
            "objective": "intervention quality; exact family match is diagnostic only",
            "neutrality": "training-side regret uses neutral_tolerance; holdout outcomes use task-specific classification/regression tolerances",
            "catastrophic_regret": "normalized regret at or above catastrophic_regret_threshold",
        },
        "thresholds": config.thresholds,
        "holdout_neutral_tolerances": {
            "classification": holdout_neutral_tolerance(
                "classification", config.thresholds
            ),
            "regression": holdout_neutral_tolerance("regression", config.thresholds),
        },
        "limitations": [
            (
                "Frozen external AMLB/OpenML suite is not representative of every tabular data-science domain."
                if config.suite == "external"
                else "Small local benchmark suite is not representative of every tabular data-science domain."
            ),
            "Empirical reference compares only supported families under this CV procedure.",
            "Offline/mock rows are not evidence about live LLM behavior.",
        ],
        "repository_commit": config.repository_commit,
        "environment_provenance": (
            environment_provenance(manifest=confirmatory_config_path)
            if confirmatory_metadata is not None else None
        ),
    }
    if config.suite == "external":
        stable_config["benchmark_suite_version"] = (
            selected_cases[0].benchmark_suite_version or "unknown"
        )
    if resume:
        config_path = output_path / "config.json"
        trials_path = output_path / "trials.jsonl"
        if not config_path.is_file() or not trials_path.is_file():
            raise ValueError("--resume requires an existing evaluation bundle with config.json and trials.jsonl.")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        compare_keys = [key for key in stable_config if key != "repository_commit"]
        mismatches = [
            key
            for key in compare_keys
            if existing_config.get(key) != stable_config.get(key)
        ]
        if mismatches:
            raise ValueError(
                "Existing evaluation configuration is incompatible; refusing to resume: "
                + ", ".join(mismatches)
            )
        config_payload = existing_config
        trials = [
            json.loads(line)
            for line in trials_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        reference_path = output_path / "empirical_reference.json"
        empirical_reference_cache = (
            json.loads(reference_path.read_text(encoding="utf-8"))
            if reference_path.is_file()
            else {}
        )
    else:
        if (output_path / "config.json").exists():
            raise ValueError("Output directory already contains an evaluation; use --resume or choose a new directory.")
        config_payload = {
            "evaluation_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **stable_config,
        }
        trials = []
        empirical_reference_cache = {}
    empirical_reference_file = (
        Path(empirical_reference_cache_path).resolve()
        if empirical_reference_cache_path is not None
        else None
    )
    if empirical_reference_file is not None and empirical_reference_file.is_file():
        empirical_reference_cache = json.loads(
            empirical_reference_file.read_text(encoding="utf-8")
        )
    proposal_cache: dict[str, dict[str, Any]] = {}
    proposal_cache_file = Path(proposal_cache_path).resolve() if proposal_cache_path is not None else None
    if proposal_cache_file is not None and proposal_cache_file.is_file():
        for line in proposal_cache_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.pop("cache_key", None)
            if key is not None and isinstance(item, dict):
                proposal_cache[str(key)] = item
    completed_trial_ids = {
        trial.get("trial_id")
        for trial in trials
        if trial.get("trial_status") != "failed"
    }
    persisted_ids = [trial.get("trial_id") for trial in trials]
    duplicate_ids = sorted({trial_id for trial_id in persisted_ids if persisted_ids.count(trial_id) > 1}, key=str)
    if duplicate_ids:
        raise ValueError(f"Duplicate confirmatory result detected for trial_id={duplicate_ids[0]}")
    for case in selected_cases:
        scenario_list: list[Perturbation | None] = [None]
        scenario_list.extend(perturbation for perturbation in perturbations if perturbation.applies(case))
        for requested_split_seed in config.split_seeds:
            for perturbation in scenario_list:
                repetition_values = config.llm_repetition_ids or tuple(
                    f"rep_{index + 1:03d}" for index in range(repetitions)
                )
                if len(repetition_values) != repetitions:
                    raise ValueError("llm_repetition_ids must contain exactly repetitions identifiers.")
                for trial_number in range(repetitions):
                    # Stable repetition identity is part of proposal provenance;
                    # it must change with the nested stochastic repeat.
                    object.__setattr__(
                        config,
                        "llm_repetition_id",
                        repetition_values[trial_number],
                    )
                    base_trial_id = (
                        f"{config.model_condition_id}:{case.name}:{perturbation.id if perturbation is not None else 'clean'}:"
                        f"split{requested_split_seed}:{repetition_values[trial_number]}:llm{trial_number}"
                    )
                    if config.ablation_name:
                        base_trial_id = f"{base_trial_id}:{config.ablation_name}"
                    if config.order_swap:
                        pair_id = base_trial_id
                        variants = [
                            ("order_ab", ("agent", "deterministic")),
                            ("order_ba", ("deterministic", "agent")),
                        ]
                    else:
                        pair_id = None
                        variants = [("standard", None)]
                    cached_pair_plan = None
                    cached_pair_source = None
                    cached_pair_model = None
                    cached_pair_request_status = None
                    cached_pair_request_error = None
                    cached_pair_api_provenance = None
                    if config.order_swap:
                        prior_ab = next(
                            (record for record in trials if record.get("trial_id") == f"{base_trial_id}:order_ab"),
                            None,
                        )
                        if prior_ab is not None and prior_ab.get("trial_status") != "failed":
                            cached_pair_plan = _cached_plan_from_trial(prior_ab)
                            cached_pair_source = prior_ab.get("agent_source")
                            cached_pair_model = prior_ab.get("agent_model")
                            cached_pair_request_status = prior_ab.get("agent_request_status")
                            cached_pair_request_error = prior_ab.get("agent_request_error")
                            cached_pair_api_provenance = prior_ab.get("planner_api_provenance")
                    for variant, proposal_order in variants:
                        trial_id = base_trial_id if variant == "standard" else f"{base_trial_id}:{variant}"
                        if trial_id in completed_trial_ids:
                            continue
                        try:
                            trial = _run_trial(
                                case,
                                perturbation,
                                trial_number,
                                config,
                                plan_factory=modeling_plan_factory,
                                reconciliation_factory=reconciliation_factory,
                                agents=agents,
                                reconciler_agents=reconciler_agents,
                                empirical_reference_cache=empirical_reference_cache,
                                variant=variant,
                                proposal_order=proposal_order,
                                order_swap_pair_id=pair_id,
                                initial_plan_override=cached_pair_plan,
                                initial_source_override=cached_pair_source,
                                initial_model_override=cached_pair_model,
                                initial_request_status_override=cached_pair_request_status,
                                initial_request_error_override=cached_pair_request_error,
                                initial_api_provenance_override=cached_pair_api_provenance,
                                requested_split_seed=requested_split_seed,
                                proposal_cache=proposal_cache,
                            )
                        except Exception as exc:
                            trial = _failed_trial_record(
                                case,
                                perturbation,
                                trial_number,
                                config,
                                exc,
                                variant=variant,
                                order_swap_pair_id=pair_id,
                                requested_split_seed=requested_split_seed,
                            )
                        trial.update({
                            "model_condition_id": config.model_condition_id,
                            "llm_repetition_id": repetition_values[trial_number],
                            "generation_settings": dict(config.generation_settings),
                            "experiment_config_sha256": (
                                confirmatory_metadata.get("experiment_config_sha256")
                                if confirmatory_metadata else None
                            ),
                        })
                        trials.append(trial)
                        if trial.get("trial_status") != "failed":
                            completed_trial_ids.add(trial_id)
                        if config.order_swap and variant == "order_ab" and trial.get("trial_status") != "failed":
                            cached_pair_plan = _cached_plan_from_trial(trial)
                            cached_pair_source = trial.get("agent_source")
                            cached_pair_model = trial.get("agent_model")
                            cached_pair_request_status = trial.get("agent_request_status")
                            cached_pair_request_error = trial.get("agent_request_error")
                            cached_pair_api_provenance = trial.get("planner_api_provenance")
                        _write_outputs(
                            output_path,
                            config_payload,
                            trials,
                            summarize_trials(trials, thresholds=config.thresholds),
                            empirical_reference_cache,
                        )
                        if proposal_cache_file is not None:
                            _write_proposal_cache(proposal_cache_file, proposal_cache)
                        if empirical_reference_file is not None:
                            empirical_reference_file.parent.mkdir(parents=True, exist_ok=True)
                            empirical_reference_file.write_text(
                                json.dumps(empirical_reference_cache, indent=2, sort_keys=True),
                                encoding="utf-8",
                            )
    summary = summarize_trials(trials, thresholds=config.thresholds)
    confirmatory_valid = None
    if confirmatory_metadata is not None:
        confirmatory_valid = bool(
            config.suite == "external"
            and config.require_live
            and not config.offline
            and summary.get("strict_live_valid", False)
            and summary.get("fallback_rows", 0) == 0
        )
    confirmatory_result_metadata = {
        "confirmatory_mode": confirmatory_metadata is not None,
        "confirmatory_config_status": (
            confirmatory_metadata["status"] if confirmatory_metadata else "not_selected"
        ),
        "experiment_config_path": (
            str(Path(confirmatory_config_path).resolve())
            if confirmatory_config_path is not None else None
        ),
        "experiment_config_version": EXPERIMENT_CONFIG_VERSION,
        "experiment_config_sha256": (
            confirmatory_metadata.get("experiment_config_sha256")
            if confirmatory_metadata else None
        ),
        "expected_experiment_code_sha256": (
            confirmatory_metadata.get("expected_experiment_code_sha256")
            if confirmatory_metadata else None
        ),
        "source_git_commit": (
            confirmatory_metadata.get("source_git_commit")
            if confirmatory_metadata else config.repository_commit
        ),
        "frozen_manifest_path": (
            str(output_path / "frozen_confirmatory_manifest.json")
            if confirmatory_metadata is not None else None
        ),
        "strict_live_required": config.require_live,
        "fallback_rows": summary.get("fallback_rows", 0),
        "config_mismatch_detected": False,
        "external_benchmark_manifest_matches": (
            bool(confirmatory_metadata.get("benchmark_manifest_matches"))
            if confirmatory_metadata is not None else None
        ),
        "confirmatory_valid": confirmatory_valid,
        "environment_provenance": (
            environment_provenance(manifest=confirmatory_config_path)
            if confirmatory_metadata is not None else None
        ),
    }
    summary.update(confirmatory_result_metadata)
    config_payload.update(confirmatory_result_metadata)
    if confirmatory_metadata is not None:
        frozen_manifest_path = output_path / "frozen_confirmatory_manifest.json"
        shutil.copyfile(Path(confirmatory_config_path), frozen_manifest_path)
    paths = _write_outputs(output_path, config_payload, trials, summary, empirical_reference_cache)
    if proposal_cache_file is not None:
        _write_proposal_cache(proposal_cache_file, proposal_cache)
    if empirical_reference_file is not None:
        empirical_reference_file.parent.mkdir(parents=True, exist_ok=True)
        empirical_reference_file.write_text(
            json.dumps(empirical_reference_cache, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return {
        "output_dir": str(output_path),
        "paths": paths,
        "summary": summary,
        "trials": trials,
        "config": config_payload,
    }
