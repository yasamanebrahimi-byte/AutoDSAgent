"""The complete agent-vs-deterministic workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.deterministic import (
    apply_cleaning,
    deterministic_recommendation,
    establish_target_task,
    eda_summary,
    make_plots,
    profile_dataframe,
)
from app.llm import OpenAIAgents
from app.modeling import fit_selected_model
from app.preprocessing import compare_preprocessing_plans, requirements_from_records
from app.reporting import render_code, render_report, write_json
from app.schemas import (
    AgentPlan,
    CleaningPlan,
    ConflictResolution,
    DeterministicRecommendation,
    PreprocessingContract,
    ReportDraft,
)
from app.validation import (
    DeterministicRecommendationUnavailable,
    FrozenSplit,
    InvariantViolation,
    ValidationResult,
    freeze_supervised_split,
    prepare_validated_frame,
    training_profile_frame,
    validated_row_positions,
    validate_training_plan,
)


def run_analysis(
    dataset_path: str | Path,
    question: str,
    target_column: str | None = None,
    output_dir: str | Path = "runs",
    api_key: str | None = None,
    model: str = "gpt-4.1-mini",
    offline: bool = False,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    dataframe = pd.read_csv(dataset_path)
    if dataframe.empty or len(dataframe.columns) < 2:
        raise ValueError("The input must be a non-empty tabular file with at least two columns.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir).resolve() / run_id
    for child in ["data", "plots", "model"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    profile = profile_dataframe(dataframe)
    write_json(run_dir / "profile.json", profile)

    agents = OpenAIAgents(api_key=api_key, model=model)
    warnings: list[str] = []
    agent_sources: dict[str, str] = {}
    agent_plan: AgentPlan | None = None
    deterministic: DeterministicRecommendation | None = None
    validation: dict[str, Any] | None = None
    planning_frame: pd.DataFrame | None = None
    planning_profile: dict[str, Any] | None = None
    split: FrozenSplit | None = None
    established_target: str | None = None
    established_task: str | None = None
    decision_payload: dict[str, Any] = {
        "agent_plan": None,
        "deterministic_recommendation": None,
        "validation": None,
        "warnings": warnings,
        "agent_sources": agent_sources,
        "gate_completed_before_training": False,
        "validation_gate_status": "not_completed",
        "model_training_occurred": False,
        "holdout_policy": {
            "frozen_before_modeling_recommendations": True,
            "planning_data": "training_partition_only",
            "holdout_used_for": "final_evaluation_only",
        },
        "cleaning_policy": {
            "decision_evidence": "training_partition_only",
            "structural_actions": "applied_with_original_row_positions_preserved",
            "learned_preprocessing": "fit_inside_training_pipeline_only",
        },
    }
    try:
        # Target/task establishment is the one deliberately pre-split stage.
        # No model family or preprocessing recommendation is made here.
        try:
            established_target, established_task = establish_target_task(
                dataframe,
                question,
                target_column,
            )
        except Exception as exc:
            raise InvariantViolation(
                f"[target_task_establishment_failed] Could not establish a valid target/task before freezing the holdout: {exc}"
            ) from exc
        split = freeze_supervised_split(
            dataframe,
            established_target,
            established_task,
            test_size=test_size,
            random_state=random_state,
        )
        planning_frame = training_profile_frame(
            dataframe,
            established_target,
            established_task,
            test_size=test_size,
            random_state=random_state,
            split=split,
        )
        planning_profile = profile_dataframe(planning_frame)
        write_json(run_dir / "planning_profile.json", planning_profile)
        decision_payload["target_establishment"] = {
            "target_column": established_target,
            "task_type": established_task,
            "target_source": "user_supplied" if target_column else "deterministic_schema_and_question",
            "completed_before_holdout_freeze": True,
        }
        decision_payload["split_contract"] = split.as_dict()
        agent_plan = _call_or_fallback(
            "modeling",
            lambda: agents.modeling_plan(
                planning_profile,
                question,
                established_target,
                established_task,
            ),
            lambda: _fallback_agent_plan(
                planning_profile,
                question,
                established_target,
                established_task,
            ),
            warnings,
            agent_sources,
            offline=offline,
        )
        _ensure_established_target_task(agent_plan, established_target, established_task)
        deterministic = _deterministic_recommendation_or_fail(
            planning_frame,
            question,
            established_target,
            warnings,
            task_type=established_task,
        )
        validation = _validate_before_training(
            agents,
            planning_profile,
            question,
            agent_plan,
            deterministic,
            warnings,
            agent_sources,
            offline=offline,
            dataframe=dataframe,
            test_size=test_size,
            random_state=random_state,
            reconciliation_profile=planning_profile,
            split=split,
            row_positions=list(range(len(dataframe))),
            established_target=established_target,
            established_task=established_task,
        )
        decision_payload.update(
            {
                "agent_plan": agent_plan.model_dump(mode="json"),
                "deterministic_recommendation": deterministic.model_dump(mode="json"),
                "validation": validation,
            }
        )
        write_json(run_dir / "decision.json", decision_payload)

        selected_target = validation["selected_target_column"]
        cleaning_plan = _call_or_fallback(
            "cleaning",
            lambda: agents.cleaning(planning_profile, selected_target),
            lambda: _fallback_cleaning_plan(planning_profile),
            warnings,
            agent_sources,
            offline=offline,
        )
        row_position_column = "__autods_row_position__"
        if row_position_column in dataframe.columns:
            raise InvariantViolation(
                f"The reserved row-position column '{row_position_column}' is already present in the dataset."
            )
        cleaning_input = dataframe.copy()
        cleaning_input[row_position_column] = range(len(cleaning_input))
        cleaned, cleaning_log = apply_cleaning(
            cleaning_input,
            selected_target,
            list(cleaning_plan.actions),
            row_position_column=row_position_column,
        )
        cleaned_row_positions = cleaned.pop(row_position_column).to_numpy(dtype=int)
        cleaning_log["original_shape"] = [int(dataframe.shape[0]), int(dataframe.shape[1])]
        cleaning_log["cleaned_shape"] = [int(cleaned.shape[0]), int(cleaned.shape[1])]
        post_cleaning_result = validate_training_plan(
            cleaned,
            selected_target,
            validation["selected_task_type"],
            validation["selected_method"],
            test_size=test_size,
            random_state=random_state,
            preprocessing=validation["approved_preprocessing"],
            split=split,
            row_positions=cleaned_row_positions,
        )
        post_cleaning_result.raise_if_failed()
        validation["pre_cleaning_deterministic_validation"] = validation[
            "deterministic_validation"
        ]
        validation["deterministic_validation"] = post_cleaning_result.as_dict()
        validation["validated_after_cleaning"] = True
        cleaned_row_positions = validated_row_positions(
            cleaned,
            post_cleaning_result,
            cleaned_row_positions,
        )
        cleaned = prepare_validated_frame(cleaned, post_cleaning_result)
        cleaning_log["deterministic_target_rows_removed"] = post_cleaning_result.target_rows_removed
        cleaning_log["removed_rows"] += post_cleaning_result.target_rows_removed
        cleaning_log["cleaned_shape"] = [int(cleaned.shape[0]), int(cleaned.shape[1])]
        cleaned_path = run_dir / "data" / "cleaned.csv"
        cleaned.to_csv(cleaned_path, index=False)
        write_json(
            run_dir / "cleaning.json",
            {"plan": cleaning_plan.model_dump(mode="json"), "log": cleaning_log},
        )
        decision_payload["validation"] = validation
        write_json(run_dir / "decision.json", decision_payload)

        computed_eda = eda_summary(cleaned, selected_target)
        plot_paths = make_plots(cleaned, selected_target, run_dir / "plots")
        findings = _call_or_fallback(
            "eda",
            lambda: agents.eda(question, computed_eda),
            lambda: _fallback_findings(computed_eda),
            warnings,
            agent_sources,
            offline=offline,
        )
        write_json(
            run_dir / "eda.json",
            {"computed": computed_eda, "agent_findings": findings, "plots": plot_paths},
        )

        decision_payload["gate_completed_before_training"] = True
        decision_payload["validation_gate_status"] = "completed"
        decision_payload["warnings"] = warnings
        decision_payload["agent_sources"] = agent_sources
        write_json(run_dir / "decision.json", decision_payload)
        modeling = fit_selected_model(
            cleaned,
            target_column=selected_target,
            task_type=validation["selected_task_type"],
            method=validation["selected_method"],
            preprocessing=validation["approved_preprocessing"],
            output_dir=run_dir / "model",
            test_size=test_size,
            random_state=random_state,
            split=split,
            row_positions=cleaned_row_positions,
        )
        write_json(run_dir / "modeling.json", modeling)
        decision_payload["model_training_occurred"] = True
        write_json(run_dir / "decision.json", decision_payload)

        report_context = {
            "profile": profile,
            "planning_profile": planning_profile,
            "validation": validation,
            "cleaning": cleaning_log,
            "eda": computed_eda,
            "findings": findings,
            "modeling": modeling,
        }
        draft = _call_or_fallback(
            "report",
            lambda: agents.report(question, report_context),
            lambda: _fallback_report(modeling, findings),
            warnings,
            agent_sources,
            offline=offline,
        )
        artifact_names = [
            "profile.json",
            "planning_profile.json",
            "decision.json",
            "cleaning.json",
            "data/cleaned.csv",
            "eda.json",
            "modeling.json",
            "model/selected_model.joblib",
            "report.md",
            "reproduce_analysis.py",
        ]
        report = render_report(
            question,
            profile,
            agent_plan,
            deterministic,
            validation,
            cleaning_plan,
            cleaning_log,
            computed_eda,
            findings,
            modeling,
            draft,
            artifact_names,
        )
        (run_dir / "report.md").write_text(report, encoding="utf-8")
        code = render_code(
            str(dataset_path),
            selected_target,
            question,
            validation["selected_method"],
            validation["selected_task_type"],
            random_state,
            test_size,
        )
        (run_dir / "reproduce_analysis.py").write_text(code, encoding="utf-8")
        decision_payload["warnings"] = warnings
        decision_payload["agent_sources"] = agent_sources
        write_json(run_dir / "decision.json", decision_payload)
        manifest = {
            "run_id": run_id,
            "dataset": str(dataset_path),
            "question": question,
            "api_model": agents.model,
            "api_used": any(source == "openai" for source in agent_sources.values()),
            "agent_sources": agent_sources,
            "warnings": warnings,
            "validation_status": validation["status"],
            "selected_method": validation["selected_method"],
            "artifacts": artifact_names,
        }
        write_json(run_dir / "run.json", manifest)
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest": manifest,
            "modeling": modeling,
            "validation": validation,
        }
    except InvariantViolation as exc:
        if exc.result is not None:
            failed_validation = _failed_validation_payload(exc.result, validation)
            decision_payload["validation"] = failed_validation
            decision_payload["validation_gate_status"] = "failed"
        decision_payload["warnings"] = warnings
        decision_payload["agent_sources"] = agent_sources
        decision_payload["gate_completed_before_training"] = False
        decision_payload["model_training_occurred"] = False
        decision_payload["failure"] = {
            "code": getattr(exc, "code", "validation_failed"),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "check_codes": [check.code for check in exc.result.failed_checks]
            if exc.result is not None
            else [],
        }
        if isinstance(exc, DeterministicRecommendationUnavailable):
            decision_payload["failure"].update(
                {
                    "original_error_type": exc.original_error_type,
                    "original_error_message": exc.original_error_message,
                }
            )
        if agent_plan is not None:
            decision_payload["agent_plan"] = agent_plan.model_dump(mode="json")
        if deterministic is not None:
            decision_payload["deterministic_recommendation"] = deterministic.model_dump(mode="json")
        write_json(run_dir / "decision.json", decision_payload)
        write_json(
            run_dir / "run.json",
            {
                "run_id": run_id,
                "dataset": str(dataset_path),
                "validation_status": "failed",
                "failure": decision_payload["failure"],
                "api_used": any(source == "openai" for source in agent_sources.values()),
                "gate_completed_before_training": False,
                "model_training_occurred": False,
            },
        )
        model_path = run_dir / "model" / "selected_model.joblib"
        if model_path.exists():
            model_path.unlink()
        raise


def _validate_before_training(
    agents: OpenAIAgents,
    profile: dict[str, Any],
    question: str,
    agent_plan: AgentPlan,
    deterministic: DeterministicRecommendation,
    warnings: list[str],
    agent_sources: dict[str, str],
    offline: bool,
    dataframe: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    reconciliation_profile: dict[str, Any] | None = None,
    split: FrozenSplit | None = None,
    row_positions: list[int] | None = None,
    established_target: str | None = None,
    established_task: str | None = None,
) -> dict[str, Any]:
    requirements = None
    if dataframe is not None:
        records = [
            record
            for record in (reconciliation_profile or profile).get("column_details", [])
            if str(record.get("name")) not in {
                agent_plan.target_column,
                deterministic.target_column,
            }
        ]
        requirements = requirements_from_records(
            records,
            deterministic.task_type,
            deterministic.recommended_method,
        )
    preprocessing_comparison = compare_preprocessing_plans(
        agent_plan.preprocessing,
        deterministic.preprocessing,
        requirements,
    )
    core_same = (
        agent_plan.target_column == deterministic.target_column
        and agent_plan.task_type == deterministic.task_type
        and agent_plan.recommended_method == deterministic.recommended_method
    )
    same = core_same and preprocessing_comparison["status"] == "agreement"
    resolution: ConflictResolution | None = None
    if same:
        selected_target = deterministic.target_column
        selected_task = deterministic.task_type
        selected_method = deterministic.recommended_method
        selected_preprocessing = deterministic.preprocessing
        justification = "The independent agent and deterministic recommender agreed on target, task type, method, and material preprocessing behavior before training."
        checks: Any = [
            "target_match",
            "task_match",
            "method_match",
            "preprocessing_materially_matches",
        ]
        status = "agreement"
    else:
        resolution = _call_or_fallback(
            "reconciliation",
            lambda: agents.reconcile(
                question,
                reconciliation_profile or profile,
                agent_plan,
                {
                    **deterministic.model_dump(mode="json"),
                    "preprocessing_comparison": preprocessing_comparison,
                    "preprocessing_requirements": requirements.as_dict() if requirements else None,
                },
            ),
            lambda: _fallback_resolution(agent_plan, deterministic),
            warnings,
            agent_sources,
            offline=offline,
        )
        selected_method = resolution.selected_method
        selected_target = resolution.selected_target_column
        selected_task = resolution.selected_task_type
        selected_preprocessing = resolution.selected_preprocessing
        justification = resolution.justification
        checks = resolution.checks
        status = "disagreement_resolved"

    if established_target is not None and (
        selected_target != established_target or selected_task != established_task
    ):
        raise InvariantViolation(
            "[established_target_task_is_immutable] Reconciliation cannot change the target/task after the supervised holdout is frozen."
        )

    if dataframe is not None:
        deterministic_validation = validate_training_plan(
            dataframe,
            selected_target,
            selected_task,
            selected_method,
            test_size=test_size,
            random_state=random_state,
            preprocessing=selected_preprocessing,
            split=split,
            row_positions=row_positions,
        )
        if requirements is not None:
            deterministic_validation.add_check(
                "preprocessing_requirements_recorded",
                True,
                requirements.as_dict(),
                "Preprocessing requirements were derived deterministically from the observed feature schema and selected model family.",
            )
        deterministic_validation.add_check(
            "reconciliation_method_is_proposed",
            same or selected_method in {
                agent_plan.recommended_method,
                deterministic.recommended_method,
            },
            {
                "selected_method": selected_method,
                "proposed_methods": sorted(
                    {agent_plan.recommended_method, deterministic.recommended_method}
                ),
            },
            "Reconciliation may select only one of the two proposed methods; do not invent or silently substitute a method.",
        )
        if not same and selected_method not in {
            agent_plan.recommended_method,
            deterministic.recommended_method,
        }:
            deterministic_validation.checks.insert(
                0,
                deterministic_validation.checks.pop(),
            )
        if not same:
            deterministic_validation.add_check(
                "reconciliation_target_is_proposed",
                selected_target in {agent_plan.target_column, deterministic.target_column},
                {
                    "selected_target": selected_target,
                    "proposed_targets": sorted({agent_plan.target_column, deterministic.target_column}),
                },
                "Reconciliation must select one of the proposed targets so the final target decision remains inspectable.",
            )
            deterministic_validation.add_check(
                "reconciliation_task_is_proposed",
                selected_task in {agent_plan.task_type, deterministic.task_type},
                {
                    "selected_task": selected_task,
                    "proposed_tasks": sorted({agent_plan.task_type, deterministic.task_type}),
                },
                "Reconciliation must select one of the proposed task types; the selected target/task pair is then validated together.",
            )
        if not same:
            deterministic_validation.add_check(
                "reconciliation_preprocessing_is_complete",
                isinstance(selected_preprocessing, PreprocessingContract),
                {
                    "selected_preprocessing": selected_preprocessing.model_dump(mode="json")
                    if isinstance(selected_preprocessing, PreprocessingContract)
                    else str(selected_preprocessing),
                },
                "Reconciliation must return one complete schema-bound preprocessing contract; unsupported or invented transformations are rejected.",
            )
            if preprocessing_comparison["material_differences"]:
                deterministic_validation.add_check(
                    "reconciliation_justification_discusses_preprocessing",
                    _justification_discusses_preprocessing(justification),
                    {
                        "material_differences": preprocessing_comparison["material_differences"],
                        "justification": justification,
                    },
                    "When preprocessing materially disagrees, reconciliation must explicitly discuss the affected preprocessing behavior and evidence.",
                )
        deterministic_validation.raise_if_failed()
        checks = deterministic_validation.as_dict()["checks"]
    return {
        "status": status,
        "overall_status": "passed",
        "selected_target_column": selected_target,
        "selected_task_type": selected_task,
        "selected_method": selected_method,
        "justification": justification,
        "checks": checks,
        "confidence": 1.0 if same else resolution.confidence,
        "agent_preprocessing": agent_plan.preprocessing.model_dump(mode="json"),
        "deterministic_preprocessing": deterministic.preprocessing.model_dump(mode="json"),
        "approved_preprocessing": selected_preprocessing.model_dump(mode="json"),
        "preprocessing_comparison": preprocessing_comparison,
        "reconciliation": resolution.model_dump(mode="json") if resolution is not None else None,
        "preprocessing_requirements": requirements.as_dict() if requirements else None,
        "deterministic_validation": deterministic_validation.as_dict()
        if dataframe is not None
        else None,
        "agent_method": agent_plan.recommended_method,
        "deterministic_method": deterministic.recommended_method,
    }


def _justification_discusses_preprocessing(justification: str) -> bool:
    text = justification.casefold()
    terms = (
        "preprocess",
        "imput",
        "scal",
        "encod",
        "categor",
        "missing",
        "infin",
        "identifier",
        "feature exclusion",
    )
    return any(term in text for term in terms)


def _deterministic_recommendation_or_fail(
    dataframe: pd.DataFrame,
    question: str,
    target_hint: str | None,
    warnings: list[str],
    task_type: str | None = None,
) -> DeterministicRecommendation:
    try:
        return deterministic_recommendation(dataframe, question, target_hint, task_type=task_type)
    except Exception as exc:
        warnings.append(
            f"Deterministic recommendation failed closed: {type(exc).__name__}: {exc}"
        )
        raise DeterministicRecommendationUnavailable(exc) from exc


def _call_or_fallback(
    name: str,
    call: Callable[[], Any],
    fallback: Callable[[], Any],
    warnings: list[str],
    agent_sources: dict[str, str],
    offline: bool,
) -> Any:
    if offline:
        agent_sources[name] = "offline_fallback"
        return fallback()
    try:
        value = call()
    except Exception as exc:
        warnings.append(f"{name} agent fallback used: {exc}")
        agent_sources[name] = "offline_fallback"
        return fallback()
    agent_sources[name] = "openai"
    return value


def _fallback_agent_plan(
    profile: dict[str, Any],
    question: str,
    target_hint: str | None,
    task_type_hint: str | None = None,
) -> AgentPlan:
    columns = [record["name"] for record in profile["column_details"]]
    target = target_hint or next(
        (column for column in columns if column.lower() in question.lower()),
        columns[-1],
    )
    target_record = next(
        (record for record in profile["column_details"] if record["name"] == target),
        {"semantic_type": "unknown"},
    )
    task = task_type_hint or (
        "classification"
        if target_record["semantic_type"] in {"categorical", "boolean", "text", "unknown"}
        else "regression"
    )
    features = [record for record in profile["column_details"] if record["name"] != target]
    has_categories = any(
        record["semantic_type"] in {"categorical", "boolean"} for record in features
    )
    method = "tree_ensemble" if has_categories else "regularized_linear"
    from app.preprocessing import requirements_from_records

    preprocessing = requirements_from_records(
        features,
        task,
        method,
    ).expected_contract
    return AgentPlan(
        target_column=target,
        task_type=task,
        recommended_method=method,
        preprocessing=preprocessing,
        reasoning="Offline fallback uses an independent schema heuristic so the workflow remains runnable without an API key.",
        confidence=0.4,
    )


def _ensure_established_target_task(
    plan: AgentPlan,
    target_column: str,
    task_type: str,
) -> None:
    if plan.target_column != target_column or plan.task_type != task_type:
        raise InvariantViolation(
            "[established_target_task_is_immutable] The modeling agent changed the target/task after the supervised holdout was frozen."
        )


def _failed_validation_payload(
    result: ValidationResult,
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep a stable gate-shaped record even when validation rejects a run."""

    payload = dict(existing or {})
    payload.update(
        {
            "status": "rejected",
            "overall_status": "failed",
            "selected_target_column": result.target_column,
            "selected_task_type": result.task_type,
            "selected_method": result.method,
            "deterministic_validation": result.as_dict(),
            "checks": result.as_dict()["checks"],
            "justification": "Deterministic validation rejected the approved plan before model fitting.",
            "confidence": 0.0,
        }
    )
    return payload


def _fallback_resolution(
    agent_plan: AgentPlan,
    deterministic: DeterministicRecommendation,
):
    from app.schemas import ConflictResolution

    return ConflictResolution(
        selected_target_column=deterministic.target_column,
        selected_task_type=deterministic.task_type,
        selected_method=deterministic.recommended_method,
        selected_preprocessing=deterministic.preprocessing,
        checks=[
            "deterministic_target_exists",
            "deterministic_task_is_feasible",
            "deterministic_method_is_allow_listed",
        ],
        justification="The offline validation fallback selected the deterministic target, task, method, and preprocessing contract because they are tied directly to the observed schema and do not invent a new executable transformation.",
        confidence=0.5,
    )


def _fallback_cleaning_plan(profile: dict[str, Any]) -> CleaningPlan:
    actions = [
        "trim_strings",
        "drop_exact_duplicates",
        "drop_all_null_columns",
        "drop_constant_features",
        "drop_rows_missing_target",
    ]
    if any(
        record["semantic_type"] == "numeric_like"
        for record in profile["column_details"]
    ):
        actions.append("coerce_numeric_strings")
    return CleaningPlan(
        actions=actions,
        reasoning="The offline cleaning fallback applies only structural operations and leaves learned imputation inside the model pipeline.",
    )


def _fallback_findings(summary: dict[str, Any]) -> list[str]:
    findings = [
        f"The cleaned analysis frame contains {summary['rows']} rows and {summary['columns']} columns."
    ]
    if summary["missing_by_column"]:
        findings.append(
            "Missing values remain visible to the modeling pipeline, where imputation is learned from training folds only."
        )
    else:
        findings.append("No missing values were present in the cleaned frame.")
    if summary.get("strongest_numeric_relationships"):
        top = summary["strongest_numeric_relationships"][0]
        findings.append(
            f"The strongest absolute numeric relationship observed was {top['feature_a']} with {top['feature_b']} (absolute r={top['abs_correlation']:.3f})."
        )
    return findings


def _fallback_report(modeling: dict[str, Any], findings: list[str]) -> ReportDraft:
    primary = next(iter(modeling["holdout_metrics"].items()))
    return ReportDraft(
        executive_summary=(
            f"The approved {modeling['selected_model']} model completed cross-validation "
            f"and one untouched holdout evaluation. The first reported holdout metric "
            f"was {primary[0]}={primary[1]:.4f}."
        ),
        key_findings=findings[:5],
        modeling_interpretation=(
            "Cross-validation metrics are used for model selection, while the holdout "
            "metrics are reported once after the approved method was selected."
        ),
        limitations=[
            "This is a compact baseline workflow, not a production deployment or causal analysis.",
            "Performance estimates depend on the supplied data and split seed.",
        ],
        next_steps=[
            "Review feature definitions and leakage risks with a domain expert.",
            "Compare additional models and calibration strategies on a representative evaluation set.",
        ],
    )
