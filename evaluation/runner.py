"""Run paired agent/gated/post-hoc evaluations with auditable trial rows."""

from __future__ import annotations

import json
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
from app.llm import PROMPT_SCHEMA_VERSION, OpenAIAgents
from app.reconciliation import LEGACY_RECONCILIATION_PROMPT_VERSION
from app.pipeline import (
    _fallback_agent_plan,
    _fallback_resolution,
    _validate_before_training,
)
from app.soft_challenge import SOFT_CHALLENGE_POLICY_VERSION, load_calibration_artifact
from app.preprocessing import compare_preprocessing_plans
from app.schemas import AgentPlan, ConflictResolution, DeterministicRecommendation, PreprocessingContract
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
    catastrophic_transition,
    classify_intervention_outcome,
    normalized_performance_delta,
    normalized_regret,
    regret,
    regret_reduction,
    summarize_trials,
)
from evaluation.perturbations import Perturbation, default_perturbations


AgentPlanFactory = Callable[[dict[str, Any]], AgentPlan]
ReconciliationFactory = Callable[
    [dict[str, Any], AgentPlan, DeterministicRecommendation], ConflictResolution
]


@dataclass(frozen=True)
class EvaluationConfig:
    repetitions: int = 1
    seed: int = 42
    test_size: float = 0.2
    model: str = "gpt-4.1-mini"
    offline: bool = False
    include_perturbations: bool = False
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION
    repository_commit: str | None = None
    gate_mode: str = "selective"
    reconciliation_mode: str = "blinded"
    order_swap: bool = False
    empirical_probe_enabled: bool = True

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least one")
        if not 0.10 <= self.test_size <= 0.50:
            raise ValueError("test_size must be between 0.10 and 0.50")
        if self.gate_mode not in {"llm_only", "always_reconcile", "selective"}:
            raise ValueError("gate_mode must be 'llm_only', 'always_reconcile', or 'selective'.")
        if self.reconciliation_mode not in {"blinded", "legacy"}:
            raise ValueError("reconciliation_mode must be 'blinded' or 'legacy'.")


class _EvaluationGateAgent:
    def __init__(
        self,
        *,
        live_agents: OpenAIAgents | None,
        reconciliation_factory: ReconciliationFactory | None,
        context: dict[str, Any],
    ) -> None:
        self.live_agents = live_agents
        self.reconciliation_factory = reconciliation_factory
        self.context = context

    def reconcile(
        self,
        question: str,
        profile: dict[str, Any],
        agent_plan: AgentPlan,
        deterministic: dict[str, Any],
    ) -> ConflictResolution:
        if self.reconciliation_factory is not None:
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
            return self.reconciliation_factory(self.context, agent_plan, recommendation)
        if self.live_agents is not None:
            return self.live_agents.reconcile(question, profile, agent_plan, deterministic)
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
        return _fallback_resolution(agent_plan, recommendation)


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


def _plan_fields(plan: AgentPlan | None) -> dict[str, Any]:
    if plan is None:
        return {
            "target": None,
            "task": None,
            "method": None,
            "preprocessing": None,
        }
    return {
        "target": plan.target_column,
        "task": plan.task_type,
        "method": plan.recommended_method,
        "preprocessing": plan.preprocessing.model_dump(mode="json"),
    }


def _plan_matches(
    target: str | None,
    task: str | None,
    method: str | None,
    preprocessing: dict[str, Any] | None,
    plan: AgentPlan | None,
) -> bool:
    fields = _plan_fields(plan)
    return (
        target == fields["target"]
        and task == fields["task"]
        and method == fields["method"]
        and preprocessing == fields["preprocessing"]
    )


def _validate_initial_plan(
    dataframe: pd.DataFrame,
    split: FrozenSplit,
    plan: AgentPlan,
    *,
    expected_target: str,
    expected_task: str,
    test_size: float,
    random_state: int,
) -> Any:
    result = validate_training_plan(
        dataframe,
        plan.target_column,
        plan.task_type,
        plan.recommended_method,
        test_size=test_size,
        random_state=random_state,
        preprocessing=plan.preprocessing,
        split=split,
        row_positions=list(range(len(dataframe))),
    )
    if plan.target_column != expected_target or plan.task_type != expected_task:
        result.add_failure(
            "established_target_task_is_immutable",
            "The modeling agent changed the established target/task after the supervised holdout was frozen.",
            {
                "established_target": expected_target,
                "established_task": expected_task,
                "agent_target": plan.target_column,
                "agent_task": plan.task_type,
            },
        )
    return result


def _choose_source(
    *,
    config: EvaluationConfig,
    agents: OpenAIAgents,
    plan_factory: AgentPlanFactory | None,
    context: dict[str, Any],
    training_profile: dict[str, Any],
    case: BenchmarkCase,
    warnings: list[str],
) -> tuple[AgentPlan, str, str, str, str | None]:
    if plan_factory is not None:
        return plan_factory(context), "mock", "mock", "mock", None
    if config.offline or not agents.available:
        if not config.offline:
            warnings.append("OPENAI_API_KEY is unavailable; this trial used the offline fallback.")
        return (
            _fallback_agent_plan(
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
        )
        return (
            AgentPlan(
                target_column=case.target_column,
                task_type=case.expected_task_type,
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
        warnings.append(f"modeling agent fallback used: {type(exc).__name__}: {exc}")
        return (
            _fallback_agent_plan(
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


def _run_trial(
    case: BenchmarkCase,
    perturbation: Perturbation | None,
    trial_number: int,
    config: EvaluationConfig,
    *,
    plan_factory: AgentPlanFactory | None,
    reconciliation_factory: ReconciliationFactory | None,
    agents: OpenAIAgents,
    empirical_reference_cache: dict[str, dict[str, Any]],
    variant: str = "standard",
    proposal_order: tuple[str, str] | None = None,
    order_swap_pair_id: str | None = None,
    initial_plan_override: AgentPlan | None = None,
    initial_source_override: str | None = None,
    initial_model_override: str | None = None,
    initial_request_status_override: str | None = None,
    initial_request_error_override: str | None = None,
) -> dict[str, Any]:
    # Repetitions are stochastic LLM repeats over identical evidence.  The
    # split seed is therefore a property of the case/experiment, not of the
    # repetition number.
    split_seed = config.seed + case.random_seed
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
    trial_id = f"{case.name}:{perturbation_id}:{trial_number}"
    if variant != "standard":
        trial_id = f"{trial_id}:{variant}"
    context = {
        "benchmark_case": case.name,
        "perturbation_id": perturbation_id,
        "trial": trial_number,
        "trial_id": trial_id,
        "split_seed": split_seed,
        "target_column": case.target_column,
        "task_type": case.expected_task_type,
        "question": case.question,
        "training_profile": training_profile,
    }
    initial_input_artifact = {
        "evaluation_mode": "modeling_gate",
        "gate_objective_version": GATE_OBJECTIVE_VERSION,
        "gate_mode": config.gate_mode,
        "question": case.question,
        "benchmark_target_constraint": case.target_column,
        "benchmark_task_constraint": case.expected_task_type,
        "training_profile": training_profile,
        "holdout_included": False,
        "deterministic_recommendation_included": False,
        "empirical_reference_included": False,
        "previous_repetitions_included": False,
        "prompt_schema_version": config.prompt_schema_version,
    }
    if initial_plan_override is None:
        plan, agent_source, agent_model, agent_request_status, agent_request_error = _choose_source(
            config=config,
            agents=agents,
            plan_factory=plan_factory,
            context=context,
            training_profile=training_profile,
            case=case,
            warnings=warnings,
        )
    else:
        plan = initial_plan_override
        agent_source = initial_source_override or "cached_order_swap"
        agent_model = initial_model_override or config.model
        agent_request_status = initial_request_status_override or "cached"
        agent_request_error = initial_request_error_override
    initial_validation = _validate_initial_plan(
        frame,
        split,
        plan,
        expected_target=case.target_column,
        expected_task=case.expected_task_type,
        test_size=config.test_size,
        random_state=split_seed,
    )
    initial_fields = _plan_fields(plan)
    deterministic: DeterministicRecommendation | None = None
    deterministic_failure: str | None = None
    try:
        deterministic = deterministic_recommendation(
            training_frame,
            case.question,
            case.target_column,
            task_type=case.expected_task_type,
        )
    except Exception as exc:
        deterministic_failure = f"{type(exc).__name__}: {exc}"
        warnings.append(f"Deterministic recommendation failed closed: {deterministic_failure}")

    comparison: dict[str, Any] | None = None
    target_disagreement = task_disagreement = method_disagreement = False
    preprocessing_disagreement = False
    agreement_status = "unavailable"
    reconciliation_invoked = False
    reconciliation_status = "not_invoked"
    reconciliation_method_source = None
    reconciliation_preprocessing_source = None
    reconciliation_agent_source = None
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
        target_disagreement = plan.target_column != deterministic.target_column
        task_disagreement = plan.task_type != deterministic.task_type
        method_disagreement = plan.recommended_method != deterministic.recommended_method
        preprocessing_disagreement = bool(comparison["material_differences"])
        agreement_status = (
            "agreement"
            if not (target_disagreement or task_disagreement or method_disagreement or preprocessing_disagreement)
            else "disagreement"
        )
        reconciliation_invoked = agreement_status == "disagreement" and not (
            target_disagreement or task_disagreement
        )
        if config.gate_mode == "llm_only":
            agreement_status = "llm_only"
            final_fields = _plan_fields(plan)
            final_validation = initial_validation.as_dict()
            final_valid = initial_validation.status == "passed"
            training_allowed = final_valid
        elif target_disagreement or task_disagreement:
            gate_error = (
                "[established_target_task_is_immutable] The modeling agent changed the established "
                "target/task after the supervised holdout was frozen."
            )
            reconciliation_status = "not_invoked"
        else:
            gate_agent = _EvaluationGateAgent(
                live_agents=None if plan_factory is not None or config.offline else agents,
                reconciliation_factory=reconciliation_factory,
                context=context,
            )
            gate_offline = config.offline or (
                plan_factory is None and not agents.available and reconciliation_factory is None
            )
            try:
                gate_sources: dict[str, str] = {}
                gate_result = _validate_before_training(
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
                    established_target=case.target_column,
                    established_task=case.expected_task_type,
                    soft_challenge_mode=config.gate_mode,
                    reconciliation_mode=config.reconciliation_mode,
                    proposal_order_override=proposal_order,
                    empirical_probe_policy=EmpiricalProbePolicy(
                        enabled=config.empirical_probe_enabled,
                        random_state=split_seed,
                    ),
                )
                # The gate may also reconcile a hard-invalid initial proposal
                # whose method-family comparison looks like agreement.  The
                # artifact is the source of truth for invocation metadata.
                reconciliation_invoked = gate_result.get("reconciliation") is not None
                reconciliation_status = (
                    "succeeded" if reconciliation_invoked else "not_invoked"
                )
                if reconciliation_invoked:
                    reconciliation_agent_source = gate_sources.get("reconciliation")
                    if plan_factory is not None or reconciliation_factory is not None:
                        reconciliation_agent_source = "mock"
                    elif config.offline:
                        reconciliation_agent_source = "offline_fallback"
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
                gate_error = str(exc)
                reconciliation_status = "failed" if reconciliation_invoked else "not_invoked"
                if exc.result is not None:
                    final_validation = exc.result.as_dict()
            except Exception as exc:
                gate_error = f"{type(exc).__name__}: {exc}"
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
    )
    unsafe_plan_intercepted = initial_validation.status == "failed" and not proceeded_unchanged

    # This is the only section that performs candidate-family fitting.  It is
    # intentionally after the gate call, so empirical results cannot influence
    # the runtime decision above.  Repetitions share this training-only cache
    # entry because their split, profile, and candidate configuration are
    # identical.
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
            "prompt_schema_version": config.prompt_schema_version,
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
            plan.target_column,
            plan.task_type,
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
            plan.target_column,
            plan.task_type,
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
    soft_intervention_occurred = soft_decision == "challenge"
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
    final_failure_codes = [
        check["code"]
        for check in (final_validation or {}).get("checks", [])
        if check.get("status") == "failed"
    ]
    validation_failure_codes = sorted(set(initial_failure_codes) | set(final_failure_codes))
    record = {
        "benchmark_case": case.name,
        "dataset_source": case.dataset_source,
        "task_type": case.expected_task_type,
        "evaluation_mode": "modeling_gate",
        "gate_objective_version": GATE_OBJECTIVE_VERSION,
        "gate_mode": config.gate_mode,
        "trial": trial_number,
        "trial_id": context["trial_id"],
        "evaluation_variant": variant,
        "order_swap_pair_id": order_swap_pair_id,
        "trial_status": "completed",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": split_seed,
        "test_size": config.test_size,
        "split_contract": split.as_dict(),
        "agent_source": agent_source,
        "requested_live_trial": not config.offline and plan_factory is None,
        "agent_model": agent_model,
        "repository_commit": config.repository_commit,
        "agent_request_status": agent_request_status,
        "agent_request_error": agent_request_error,
        "live_request_failed": agent_request_status == "failed_fallback",
        "generation_settings": {"seed": None, "temperature": None},
        "prompt_schema_version": config.prompt_schema_version,
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
        "empirical_reference_cache_key": cache_key,
    }
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
) -> dict[str, Any]:
    """Persist an execution failure without inventing a modeling decision."""

    split_seed = config.seed + case.random_seed
    perturbation_id = perturbation.id if perturbation is not None else "clean"
    split = freeze_supervised_split(
        case.load(),
        case.target_column,
        case.expected_task_type,
        test_size=config.test_size,
        random_state=split_seed,
    )
    trial_id = f"{case.name}:{perturbation_id}:{trial_number}"
    if variant != "standard":
        trial_id = f"{trial_id}:{variant}"
    return {
        "trial_id": trial_id,
        "benchmark_case": case.name,
        "dataset_source": case.dataset_source,
        "task_type": case.expected_task_type,
        "trial": trial_number,
        "trial_status": "failed",
        "evaluation_variant": variant,
        "order_swap_pair_id": order_swap_pair_id,
        "gate_mode": config.gate_mode,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": split_seed,
        "split_contract": split.as_dict(),
        "agent_source": "failed",
        "requested_live_trial": not config.offline,
        "agent_model": config.model,
        "repository_commit": config.repository_commit,
        "agent_request_status": "failed",
        "agent_request_error": f"{type(error).__name__}: {error}",
        "live_request_failed": not config.offline,
        "prompt_schema_version": config.prompt_schema_version,
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
        "failure_reason": f"{type(error).__name__}: {error}",
        "warnings": ["Trial execution failed; no model-selection result was recorded."],
    }


def _cached_plan_from_trial(record: dict[str, Any]) -> AgentPlan | None:
    """Recover the first pair member's proposal for an order-swap rerun."""

    values = record.get("agent_initial")
    if not isinstance(values, dict):
        return None
    try:
        return AgentPlan(
            target_column=str(values["target"]),
            task_type=values["task"],
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
    offline: bool = False,
    include_perturbations: bool = False,
    thresholds: dict[str, float] | None = None,
    case_names: Sequence[str] | None = None,
    agent_plan_factory: AgentPlanFactory | None = None,
    reconciliation_factory: ReconciliationFactory | None = None,
    # Preserve the historical public-harness default. Research runs should
    # pass gate_mode="selective" explicitly; the production pipeline defaults
    # to selective intervention.
    gate_mode: str = "always_reconcile",
    reconciliation_mode: str = "blinded",
    order_swap: bool = False,
    empirical_probe_enabled: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Run reproducible trials and write the structured evaluation bundle."""

    config = EvaluationConfig(
        repetitions=repetitions,
        seed=seed,
        test_size=test_size,
        model=model,
        offline=offline,
        include_perturbations=include_perturbations,
        thresholds={**DEFAULT_THRESHOLDS, **(thresholds or {})},
        repository_commit=_repository_commit(),
        gate_mode=gate_mode,
        reconciliation_mode=reconciliation_mode,
        order_swap=order_swap,
        empirical_probe_enabled=empirical_probe_enabled,
        prompt_schema_version=(
            LEGACY_RECONCILIATION_PROMPT_VERSION
            if reconciliation_mode == "legacy"
            else PROMPT_SCHEMA_VERSION
        ),
    )
    selected_cases = list(cases or default_benchmark_cases())
    if case_names:
        wanted = set(case_names)
        selected_cases = [case for case in selected_cases if case.name in wanted]
    if not selected_cases:
        raise ValueError("No benchmark cases selected.")
    perturbations = default_perturbations() if include_perturbations else []
    output_path = Path(output_dir).resolve()
    stable_config = {
        "config_version": "2026-08-24.evaluation.v4-intervention-quality",
        "gate_objective_version": GATE_OBJECTIVE_VERSION,
        "benchmark_cases": [case.as_dict() for case in selected_cases],
        "perturbations": [perturbation.as_dict() for perturbation in perturbations],
        "repetitions": config.repetitions,
        "seed": config.seed,
        "test_size": config.test_size,
        "agent_mode": "offline" if config.offline else "live_or_fallback",
        "agent_model_requested": config.model,
        "agent_source_policy": "openai, offline_fallback, mock, or failed; source is persisted per trial",
        "gate_mode": config.gate_mode,
        "reconciliation_mode": config.reconciliation_mode,
        "order_swap": config.order_swap,
        "empirical_probe_enabled": config.empirical_probe_enabled,
        "empirical_probe_policy_version": EmpiricalProbePolicy().policy_version,
        "reconciliation_mode_definitions": {
            "legacy": "source-labeled reconciliation prompt retained as an evaluation baseline",
            "blinded": "source-neutral Proposal A/Proposal B evidence-comparison prompt",
        },
        "gate_mode_definitions": {
            "llm_only": "retain the initial agent plan after initial validation; never reconcile soft disagreement",
            "always_reconcile": "invoke the existing reconciliation path for every valid soft disagreement",
            "selective": "invoke reconciliation only when the versioned soft-challenge policy authorizes a challenge",
        },
        "prompt_schema_version": config.prompt_schema_version,
        "deterministic_policy_version": DeterministicPolicy().version,
        "soft_challenge_policy_version": SOFT_CHALLENGE_POLICY_VERSION,
        "soft_challenge_calibration_artifact_version": load_calibration_artifact().get(
            "calibration_artifact_version"
        ),
        "generation_settings": {"seed": None, "temperature": None},
        "evaluation_settings": {
            "primary_metrics": {"classification": "macro_f1", "regression": "rmse"},
            "candidate_methods": ["linear", "regularized_linear", "tree_ensemble", "boosted_tree"],
            "candidate_selection_data": "frozen training partition only",
            "holdout_data": "final evaluation only",
            "repetition_design": "same split and training-only profile; stochastic LLM response is the intended varying factor",
            "objective": "intervention quality; exact family match is diagnostic only",
            "neutrality": "normalized regret reduction within neutral_tolerance is neutral",
            "catastrophic_regret": "normalized regret at or above catastrophic_regret_threshold",
        },
        "thresholds": config.thresholds,
        "limitations": [
            "Small local benchmark suite.",
            "Empirical reference compares only supported families under this CV procedure.",
            "Offline/mock rows are not evidence about live LLM behavior.",
        ],
        "repository_commit": config.repository_commit,
    }
    if resume:
        config_path = output_path / "config.json"
        trials_path = output_path / "trials.jsonl"
        if not config_path.is_file() or not trials_path.is_file():
            raise ValueError("--resume requires an existing evaluation bundle with config.json and trials.jsonl.")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        compare_keys = [key for key in stable_config if key != "repository_commit"]
        mismatches = [key for key in compare_keys if existing_config.get(key) != stable_config.get(key)]
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
    agents = OpenAIAgents(model=model)
    completed_trial_ids = {trial.get("trial_id") for trial in trials}
    for case in selected_cases:
        scenario_list: list[Perturbation | None] = [None]
        scenario_list.extend(perturbation for perturbation in perturbations if perturbation.applies(case))
        for perturbation in scenario_list:
            for trial_number in range(repetitions):
                base_trial_id = f"{case.name}:{perturbation.id if perturbation is not None else 'clean'}:{trial_number}"
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
                            plan_factory=agent_plan_factory,
                            reconciliation_factory=reconciliation_factory,
                            agents=agents,
                            empirical_reference_cache=empirical_reference_cache,
                            variant=variant,
                            proposal_order=proposal_order,
                            order_swap_pair_id=pair_id,
                            initial_plan_override=cached_pair_plan,
                            initial_source_override=cached_pair_source,
                            initial_model_override=cached_pair_model,
                            initial_request_status_override=cached_pair_request_status,
                            initial_request_error_override=cached_pair_request_error,
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
                        )
                    trials.append(trial)
                    completed_trial_ids.add(trial_id)
                    if config.order_swap and variant == "order_ab" and trial.get("trial_status") != "failed":
                        cached_pair_plan = _cached_plan_from_trial(trial)
                        cached_pair_source = trial.get("agent_source")
                        cached_pair_model = trial.get("agent_model")
                        cached_pair_request_status = trial.get("agent_request_status")
                        cached_pair_request_error = trial.get("agent_request_error")
                    _write_outputs(
                        output_path,
                        config_payload,
                        trials,
                        summarize_trials(trials, thresholds=config.thresholds),
                        empirical_reference_cache,
                    )
    summary = summarize_trials(trials, thresholds=config.thresholds)
    paths = _write_outputs(output_path, config_payload, trials, summary, empirical_reference_cache)
    return {"output_dir": str(output_path), "paths": paths, "summary": summary}
