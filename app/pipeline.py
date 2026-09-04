"""The complete agent-vs-deterministic workflow."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.deterministic import (
    deterministic_formulation,
    deterministic_recommendation,
    eda_summary,
    fit_cleaning_spec,
    make_plots,
    profile_dataframe,
    transform_cleaning,
)
from app.llm import OpenAIAgents
from app.modeling import fit_selected_model
from app.empirical_challenge_probe import (
    EmpiricalProbePolicy,
    run_pairwise_model_probe,
)
from app.preprocessing import compare_preprocessing_plans, requirements_from_records
from app.reporting import render_code, render_report, write_json
from app.reconciliation import (
    BLINDED_RECONCILIATION_MODE,
    BLINDED_RECONCILIATION_PROMPT_VERSION,
    build_blinded_reconciliation,
    infer_selected_proposal,
)
from app.schemas import (
    CleaningPlan,
    DeterministicFormulation,
    DeterministicRecommendation,
    FormulationComparison,
    FormulationPlan,
    FormulationResolution,
    HardValidationArtifact,
    ModelingGateArtifact,
    ModelingPlan,
    ModelingResolution,
    PreprocessingContract,
    ReportDraft,
    SoftChallengeArtifact,
)
from app.soft_challenge import SoftChallengePolicy, decide_soft_challenge
from app.validation import (
    DeterministicRecommendationUnavailable,
    FrozenSplit,
    InvariantViolation,
    ValidationResult,
    freeze_supervised_split,
    prepare_validated_frame,
    training_partition_frame,
    training_profile_frame,
    validated_row_positions,
    validate_formulation,
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
    empirical_probe_policy: EmpiricalProbePolicy | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    # Preserve source values as object/string representations until the
    # training-fitted cleaning specification decides which columns are
    # numeric-like.  Inferring a dtype from the complete CSV can otherwise let
    # holdout-only strings change the training profile before cleaning fits.
    dataframe = pd.read_csv(dataset_path, dtype=object)
    if dataframe.empty or len(dataframe.columns) < 2:
        raise ValueError("The input must be a non-empty tabular file with at least two columns.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir).resolve() / run_id
    for child in ["data", "plots", "model"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    agents = OpenAIAgents(api_key=api_key, model=model)
    warnings: list[str] = []
    agent_sources: dict[str, str] = {}
    formulation_agent: FormulationPlan | None = None
    deterministic_formulation_result: DeterministicFormulation | None = None
    formulation_resolution: FormulationResolution | None = None
    formulation_validation: dict[str, Any] | None = None
    modeling_plan: ModelingPlan | None = None
    deterministic: DeterministicRecommendation | None = None
    validation: dict[str, Any] | None = None
    planning_frame: pd.DataFrame | None = None
    planning_profile: dict[str, Any] | None = None
    split: FrozenSplit | None = None
    established_target: str | None = None
    established_task: str | None = None
    decision_payload: dict[str, Any] = {
        "formulation": {
            "user_target_constraint": None,
            "agent_initial": None,
            "deterministic": None,
            "comparison": None,
            "reconciliation": None,
            "final": None,
            "status": "not_completed",
        },
        "modeling_gate": {
            "agent_initial": None,
            "deterministic": None,
            "comparison": None,
            "reconciliation": None,
            "final": None,
            "status": "not_completed",
        },
        "modeling_plan": None,
        "deterministic_recommendation": None,
        "validation": None,
        "warnings": warnings,
        "agent_sources": agent_sources,
        "gate_completed_before_training": False,
        "formulation_gate_status": "not_completed",
        "split_frozen_after_formulation_gate": False,
        "modeling_gate_status": "not_completed",
        "validation_gate_status": "not_completed",
        "model_training_occurred": False,
        "holdout_policy": {
            "frozen_before_modeling_recommendations": True,
            "planning_data": "training_partition_only",
            "holdout_used_for": "final_evaluation_only",
        },
        "cleaning_policy": {
            "decision_evidence": "training_partition_only",
            "structural_actions": "fitted_on_training_and_transformed_per_partition_with_original_row_positions_preserved",
            "duplicate_policy": "within_partition_only_keep_first",
            "holdout_used_for_cleaning_decisions": False,
            "learned_preprocessing": "fit_inside_training_pipeline_only",
        },
    }
    try:
        # Formulation is deliberately complete before a split exists.  Both
        # initial paths receive only raw compact schema evidence and an
        # explicit user target constraint, never each other's proposal.
        formulation_profile = profile_dataframe(dataframe)
        write_json(run_dir / "formulation_profile.json", formulation_profile)
        user_target_constraint = (
            {
                "target_column": target_column,
                "target_source": "user_supplied",
                "target_is_mutable": False,
            }
            if target_column
            else {
                "target_column": None,
                "target_source": "inferred",
                "target_is_mutable": True,
            }
        )
        decision_payload["formulation"]["user_target_constraint"] = user_target_constraint
        formulation_agent = _call_or_fallback(
            "formulation",
            lambda: agents.formulate_problem(
                formulation_profile,
                question,
                user_target_constraint if target_column else None,
            ),
            lambda: _fallback_formulation_plan(
                dataframe,
                formulation_profile,
                question,
                target_column,
            ),
            warnings,
            agent_sources,
            offline=offline,
        )
        if target_column and formulation_agent.target_column != target_column:
            # The external proposal is retained in the warning, but the
            # explicit user constraint is enforced as an immutable invariant.
            warnings.append(
                "Formulation agent proposed a different target; explicit user target constraint was enforced."
            )
            formulation_agent = formulation_agent.model_copy(
                update={
                    "target_column": target_column,
                    "reasoning": (
                        formulation_agent.reasoning
                        + " Explicit user target constraint enforced by deterministic guardrail."
                    )[:1200],
                }
            )
        deterministic_formulation_result = deterministic_formulation(
            dataframe,
            question,
            target_column,
        )
        decision_payload["formulation"]["agent_initial"] = formulation_agent.model_dump(mode="json")
        decision_payload["formulation"]["deterministic"] = deterministic_formulation_result.model_dump(mode="json")
        if deterministic_formulation_result.status != "proposed":
            failed = validate_formulation(
                dataframe,
                deterministic_formulation_result.target_column or "",
                deterministic_formulation_result.task_type or "unsupported",
                user_target=target_column,
                test_size=test_size,
                random_state=random_state,
            )
            failed.add_failure(
                "deterministic_formulation_is_defensible",
                deterministic_formulation_result.reasoning,
                {"evidence": deterministic_formulation_result.evidence},
            )
            raise InvariantViolation.from_result(failed)

        formulation_comparison = _compare_formulations(
            formulation_agent,
            deterministic_formulation_result,
            target_column,
        )
        decision_payload["formulation"]["comparison"] = formulation_comparison.model_dump(mode="json")
        if formulation_comparison.overall_agreement:
            established_target = target_column or formulation_agent.target_column
            established_task = formulation_agent.task_type
            formulation_status = "agreement"
            formulation_justification = (
                "The independent formulation agent and deterministic formulation engine agreed "
                "on the target and supported task before split construction."
            )
        else:
            formulation_resolution = _call_or_fallback(
                "formulation_reconciliation",
                lambda: agents.reconcile_formulation(
                    question,
                    formulation_profile,
                    user_target_constraint if target_column else None,
                    formulation_agent,
                    deterministic_formulation_result.model_dump(mode="json"),
                ),
                lambda: _fallback_formulation_resolution(
                    formulation_agent,
                    deterministic_formulation_result,
                    target_column,
                ),
                warnings,
                agent_sources,
                offline=offline,
            )
            established_target = formulation_resolution.selected_target_column
            established_task = formulation_resolution.selected_task_type
            formulation_status = "disagreement_resolved"
            formulation_justification = formulation_resolution.justification
            _validate_formulation_resolution(
                formulation_resolution,
                formulation_agent,
                deterministic_formulation_result,
                target_column,
            )
        formulation_validation_result = validate_formulation(
            dataframe,
            established_target,
            established_task,
            user_target=target_column,
            test_size=test_size,
            random_state=random_state,
        )
        formulation_validation_result.raise_if_failed()
        formulation_validation = formulation_validation_result.as_dict()
        decision_payload["formulation"].update(
            {
                "reconciliation": formulation_resolution.model_dump(mode="json")
                if formulation_resolution is not None
                else None,
                "final": {
                    "target_column": established_target,
                    "task_type": established_task,
                    "target_source": "user_supplied" if target_column else "inferred",
                    "target_is_mutable": target_column is None,
                    "justification": formulation_justification,
                },
                "validation": formulation_validation,
                "status": formulation_status,
            }
        )
        decision_payload["formulation_gate_status"] = "completed"
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
        decision_payload["split_frozen_after_formulation_gate"] = True
        decision_payload["split_contract"] = split.as_dict()
        modeling_plan = _call_or_fallback(
            "modeling",
            lambda: agents.modeling_plan(
                planning_profile,
                question,
                established_target,
                established_task,
            ),
            lambda: _fallback_modeling_plan(
                planning_profile,
                question,
                established_target,
                established_task,
            ),
            warnings,
            agent_sources,
            offline=offline,
        )
        deterministic = _deterministic_recommendation_or_fail(
            planning_frame,
            question,
            established_target,
            warnings,
            task_type=established_task,
        )
        validation = _validate_modeling_gate(
            agents,
            planning_profile,
            question,
            modeling_plan,
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
            approved_target=established_target,
            approved_task=established_task,
            empirical_probe_policy=empirical_probe_policy,
        )
        validation["formulation"] = decision_payload["formulation"]
        decision_payload["modeling_gate_status"] = "completed"
        decision_payload.update(
            {
                "modeling_plan": modeling_plan.model_dump(mode="json"),
                "modeling_gate": validation.get("modeling_gate", {}),
                "deterministic_recommendation": deterministic.model_dump(mode="json"),
                "deterministic_policy_version": deterministic.policy_version,
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
        training_cleaning_input = cleaning_input.iloc[list(split.train_row_positions)].copy()
        cleaning_specification = fit_cleaning_spec(
            training_cleaning_input,
            selected_target,
            list(cleaning_plan.actions),
            row_position_column=row_position_column,
        )
        valid_positions = set(split.valid_row_positions)
        partition_positions = {
            "training": list(split.train_row_positions),
            "holdout": list(split.holdout_row_positions),
            "unassigned": [
                int(position)
                for position in range(len(cleaning_input))
                if int(position) not in valid_positions
            ],
        }
        transformed_partitions: list[pd.DataFrame] = []
        transformed_by_partition: dict[str, pd.DataFrame] = {}
        partition_logs: dict[str, dict[str, Any]] = {}
        for partition_name, positions in partition_positions.items():
            partition_input = cleaning_input.iloc[positions].copy()
            transformed, partition_log = transform_cleaning(
                partition_input,
                cleaning_specification,
                partition=partition_name,
            )
            transformed_partitions.append(transformed)
            transformed_by_partition[partition_name] = transformed
            partition_logs[partition_name] = partition_log
        cleaned = (
            pd.concat(transformed_partitions, axis=0, ignore_index=True)
            .sort_values(row_position_column, kind="stable")
            .reset_index(drop=True)
        )
        removed_columns = list(
            dict.fromkeys(
                cleaning_specification.all_null_columns
                + cleaning_specification.constant_columns
            )
        )
        cleaning_log = {
            "original_shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
            "cleaned_shape": [int(cleaned.shape[0]), int(cleaned.shape[1])],
            "requested_actions": list(cleaning_plan.actions),
            "applied_actions": list(cleaning_plan.actions),
            "removed_rows": int(sum(log["removed_rows"] for log in partition_logs.values())),
            "removed_columns": removed_columns,
            "partition_logs": partition_logs,
            "decision_scope": "training_partition_only",
            "holdout_used_for_cleaning_decisions": False,
            "duplicate_policy": "within_partition_only_keep_first",
        }
        cleaned_row_positions = cleaned.pop(row_position_column).to_numpy(dtype=int)
        training_evidence_frame = transformed_by_partition["training"].drop(
            columns=[row_position_column]
        )
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
            evidence_dataframe=training_evidence_frame,
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
            {
                "decision_scope": "training_partition_only",
                "holdout_used_for_cleaning_decisions": False,
                "requested_actions": list(cleaning_plan.actions),
                "plan": cleaning_plan.model_dump(mode="json"),
                "fitted_specification": cleaning_specification.model_dump(mode="json"),
                "training_only_evidence": cleaning_specification.training_only_evidence,
                "applied_transformations": partition_logs,
                "rows_removed": {
                    "training": partition_logs["training"]["removed_rows"],
                    "holdout": partition_logs["holdout"]["removed_rows"],
                    "unassigned": partition_logs["unassigned"]["removed_rows"],
                    "total_before_validation": cleaning_log["removed_rows"],
                },
                "columns_removed": removed_columns,
                "log": cleaning_log,
            },
        )
        decision_payload["validation"] = validation
        write_json(run_dir / "decision.json", decision_payload)

        eda_frame = training_partition_frame(cleaned, split, cleaned_row_positions)
        computed_eda = eda_summary(eda_frame, selected_target)
        plot_paths = make_plots(eda_frame, selected_target, run_dir / "plots")
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
            {
                "data_scope": "training_partition_only",
                "training_rows": int(len(eda_frame)),
                "holdout_rows_included": 0,
                "train_positions_digest": split.as_dict()["train_positions_digest"],
                "computed": computed_eda,
                "agent_findings": findings,
                "plots": plot_paths,
            },
        )

        decision_payload["gate_completed_before_training"] = True
        decision_payload["validation_gate_status"] = "completed"
        decision_payload["modeling_gate_status"] = "completed"
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
            evidence_dataframe=training_evidence_frame,
        )
        write_json(run_dir / "modeling.json", modeling)
        decision_payload["model_training_occurred"] = True
        write_json(run_dir / "decision.json", decision_payload)
        profile = profile_dataframe(dataframe)
        write_json(run_dir / "profile.json", profile)

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
            "formulation_profile.json",
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
            modeling_plan,
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
            "formulation_gate_status": decision_payload["formulation_gate_status"],
            "split_frozen_after_formulation_gate": decision_payload["split_frozen_after_formulation_gate"],
            "modeling_gate_status": decision_payload["modeling_gate_status"],
            "selected_method": validation["selected_method"],
            "deterministic_policy_version": deterministic.policy_version,
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
            failed_validation = _failed_validation_payload(
                exc.result,
                validation,
                getattr(exc, "metadata", None),
            )
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
        if modeling_plan is not None:
            decision_payload["modeling_plan"] = modeling_plan.model_dump(mode="json")
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
                "deterministic_policy_version": deterministic.policy_version if deterministic else None,
                "gate_completed_before_training": False,
                "formulation_gate_status": decision_payload["formulation_gate_status"],
                "split_frozen_after_formulation_gate": decision_payload["split_frozen_after_formulation_gate"],
                "modeling_gate_status": decision_payload["modeling_gate_status"],
                "model_training_occurred": False,
            },
        )
        model_path = run_dir / "model" / "selected_model.joblib"
        if model_path.exists():
            model_path.unlink()
        raise


def _validate_modeling_gate(
    agents: OpenAIAgents,
    profile: dict[str, Any],
    question: str,
    modeling_plan: ModelingPlan,
    deterministic: DeterministicRecommendation,
    warnings: list[str],
    agent_sources: dict[str, str],
    offline: bool,
    dataframe: pd.DataFrame,
    test_size: float,
    random_state: int,
    reconciliation_profile: dict[str, Any],
    split: FrozenSplit,
    row_positions: list[int],
    approved_target: str,
    approved_task: str,
    soft_challenge_policy: SoftChallengePolicy | None = None,
    soft_challenge_mode: str = "selective",
    empirical_probe_policy: EmpiricalProbePolicy | None = None,
    soft_challenge_strategy: str = "calibrated",
    strict_live: bool = False,
    proposal_order_override: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the initial proposal before treating the deterministic output as a challenger.

    The gate has three intentionally separate stages:

    * hard validation of the agent proposal and deterministic challenger;
    * soft comparison of model family and material preprocessing behavior;
    * hard validation of the selected final plan.

    The returned artifact records the independent proposals, safety checks,
    selective decision, and final approved plan in separate sections.
    """
    records = [
        record
        for record in reconciliation_profile.get("column_details", [])
        if str(record.get("name")) != approved_target
    ]
    requirements = requirements_from_records(records, approved_task, deterministic.recommended_method)

    def probe_training_frame() -> pd.DataFrame:
        # Production reaches this gate before structural cleaning, so the
        # frozen split positions directly identify the training rows.  The
        # alternate mapping keeps the helper safe for callers that pass a
        # cleaned frame with original source positions.
        if row_positions == list(range(len(dataframe))):
            return training_profile_frame(
                dataframe,
                approved_target,
                approved_task,
                test_size=test_size,
                random_state=random_state,
                split=split,
            )
        return training_partition_frame(dataframe, split, row_positions)

    def hard_validate(method: str, preprocessing: PreprocessingContract) -> ValidationResult:
        result = validate_training_plan(
            dataframe,
            approved_target,
            approved_task,
            method,
            test_size=test_size,
            random_state=random_state,
            preprocessing=preprocessing,
            split=split,
            row_positions=row_positions,
        )
        result.add_check(
            "approved_formulation_is_immutable",
            split.target_column == approved_target and split.task_type == approved_task,
            {
                "approved_target": approved_target,
                "approved_task": approved_task,
                "split_target": split.target_column,
                "split_task": split.task_type,
            },
            "Modeling validation must use the immutable formulation approved before split construction.",
        )
        return result

    agent_hard_validation = hard_validate(
        modeling_plan.recommended_method,
        modeling_plan.preprocessing,
    )
    challenger_hard_validation = hard_validate(
        deterministic.recommended_method,
        deterministic.preprocessing,
    )
    preprocessing_comparison = compare_preprocessing_plans(
        modeling_plan.preprocessing,
        deterministic.preprocessing,
        requirements,
    )
    method_disagreement = modeling_plan.recommended_method != deterministic.recommended_method
    preprocessing_disagreement = bool(preprocessing_comparison["material_differences"])
    same = not method_disagreement and not preprocessing_disagreement
    soft_status = "agreement" if same else "disagreement"
    if challenger_hard_validation.status != "passed" and agent_hard_validation.status == "passed":
        soft_status = "invalid"

    if soft_challenge_mode not in {"selective", "always_reconcile", "probe_first"}:
        raise ValueError(f"Unsupported soft_challenge_mode: {soft_challenge_mode!r}")

    # The calibrated soft policy remains available as audit metadata, but it
    # no longer gates the production probe.  A valid family disagreement first
    # receives bounded pairwise training-only evidence.
    hard_reconciliation_required = (
        agent_hard_validation.status != "passed"
        or challenger_hard_validation.status != "passed"
    )

    resolution: ModelingResolution | None = None
    blinded_reconciliation = None
    selected_proposal = None
    selected_proposal_source = None
    empirical_probe: dict[str, Any] | None = None
    empirical_probe_invoked = False
    probe_status = "not_invoked"
    probe_evidence_strength = "not_invoked"
    abstention_reason: str | None = None

    if not hard_reconciliation_required and method_disagreement:
        configured_probe_policy = empirical_probe_policy or EmpiricalProbePolicy()
        if configured_probe_policy.enabled:
            if configured_probe_policy.random_state is None:
                configured_probe_policy = replace(configured_probe_policy, random_state=random_state)
            blinded_reconciliation = build_blinded_reconciliation(
                reconciliation_profile,
                modeling_plan,
                deterministic,
                target_column=approved_target,
                task_type=approved_task,
                preprocessing_comparison=preprocessing_comparison,
                preprocessing_requirements=requirements.as_dict(),
                hard_validation={
                    "agent": _reconciliation_hard_summary(agent_hard_validation),
                    "deterministic": _reconciliation_hard_summary(challenger_hard_validation),
                },
                order_seed=random_state,
                proposal_order=proposal_order_override,
            )
            source_by_label = {
                "A": blinded_reconciliation.proposal_a_source,
                "B": blinded_reconciliation.proposal_b_source,
            }
            proposal_by_source = {
                "agent": modeling_plan,
                "deterministic": deterministic,
            }
            empirical_probe_invoked = True
            try:
                empirical_probe = run_pairwise_model_probe(
                    probe_training_frame(),
                    approved_target,
                    approved_task,
                    proposal_by_source[source_by_label["A"]],
                    proposal_by_source[source_by_label["B"]],
                    policy=configured_probe_policy,
                    random_state=configured_probe_policy.random_state,
                    validation_by_method={
                        modeling_plan.recommended_method: agent_hard_validation,
                        deterministic.recommended_method: challenger_hard_validation,
                    },
                )
            except Exception as exc:  # advisory evidence fails closed to abstention
                empirical_probe = {
                    "status": "failed",
                    "policy_version": configured_probe_policy.policy_version,
                    "task_type": approved_task,
                    "metric": "macro_f1" if approved_task == "classification" else "rmse",
                    "higher_is_better": approved_task == "classification",
                    "cv_folds": 0,
                    "training_rows": None,
                    "data_used": "frozen_training_partition_only",
                    "holdout_used": False,
                    "fit_count": 0,
                    "winner": "tie",
                    "difference": None,
                    "relative_advantage": 0.0,
                    "normalized_advantage": 0.0,
                    "evidence_strength": "tie",
                    "reason": "probe_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            probe_status = str(empirical_probe.get("status", "unavailable"))
            if probe_status not in {"completed", "unavailable", "failed"}:
                probe_status = "unavailable"
            probe_evidence_strength = str(empirical_probe.get("evidence_strength", "tie"))
            if probe_evidence_strength not in {"tie", "weak", "moderate", "strong"}:
                probe_evidence_strength = "tie"
            if probe_evidence_strength not in {"moderate", "strong"}:
                abstention_reason = (
                    "probe_evidence_unavailable"
                    if probe_status in {"unavailable", "failed"}
                    else f"probe_evidence_{probe_evidence_strength}"
                )
        else:
            probe_status = "unavailable"
            probe_evidence_strength = "tie"
            abstention_reason = "probe_disabled"

    # The calibrated decision is computed only after the probe so
    # it cannot act as a pre-gate. It is retained for audit/evaluation metadata
    # and is overridden below by the probe-first production policy.
    soft_decision = decide_soft_challenge(
        agent_method=modeling_plan.recommended_method,
        deterministic_method=deterministic.recommended_method,
        deterministic_confidence=deterministic.confidence,
        score_margin=deterministic.score_margin,
        diagnostics=deterministic.diagnostics,
        task_type=approved_task,
        training_row_count=(
            deterministic.diagnostics.training_row_count
            if deterministic.diagnostics is not None
            else None
        ),
        policy=soft_challenge_policy,
        strategy=soft_challenge_strategy,
    )
    soft_reconciliation_required = method_disagreement and (
        soft_challenge_mode == "always_reconcile"
        or probe_evidence_strength in {"moderate", "strong"}
    )
    if hard_reconciliation_required:
        soft_decision = replace(
            soft_decision,
            decision="challenge",
            decision_reason="hard_validation_correction",
        )
    elif method_disagreement and soft_reconciliation_required:
        soft_decision = replace(
            soft_decision,
            decision="challenge",
            decision_reason=(
                "always_reconcile_baseline"
                if soft_challenge_mode == "always_reconcile"
                else f"probe_{probe_evidence_strength}_evidence"
            ),
        )
    elif method_disagreement:
        soft_decision = replace(
            soft_decision,
            decision="abstain",
            decision_reason=abstention_reason or "probe_evidence_not_sufficient_for_reconciliation",
        )
    if (
        not hard_reconciliation_required
        and not soft_reconciliation_required
        and not method_disagreement
        and agent_hard_validation.status == "passed"
    ):
        # Preserve the initial proposal as a legitimate final option.  The
        # deterministic output agrees materially, but it is not authoritative
        # merely because it was generated by deterministic code.
        selected_method = modeling_plan.recommended_method
        selected_preprocessing = modeling_plan.preprocessing
        justification = (
            "The independent modeling agent and deterministic challenger agreed on "
            "model family and material preprocessing behavior after both proposals passed "
            "hard safety and executability validation."
        )
        status = "agreement"
    else:
        if not hard_reconciliation_required and not soft_reconciliation_required:
            selected_method = modeling_plan.recommended_method
            selected_preprocessing = modeling_plan.preprocessing
            justification = (
                "The deterministic challenger recorded a model-family disagreement but abstained "
                "because its heuristic confidence and calibration evidence did not justify intervention."
            )
            status = "disagreement_abstained"
            resolution = None
        else:
            blinded_reconciliation = build_blinded_reconciliation(
                reconciliation_profile,
                modeling_plan,
                deterministic,
                target_column=approved_target,
                task_type=approved_task,
                preprocessing_comparison=preprocessing_comparison,
                preprocessing_requirements=requirements.as_dict(),
                hard_validation={
                    "agent": _reconciliation_hard_summary(agent_hard_validation),
                    "deterministic": _reconciliation_hard_summary(challenger_hard_validation),
                },
                order_seed=random_state,
                proposal_order=proposal_order_override,
            )
            if (not empirical_probe_invoked) and method_disagreement and soft_decision.decision == "challenge" and not hard_reconciliation_required and (
                empirical_probe_policy is None or empirical_probe_policy.enabled
            ) and (
                agent_hard_validation.status == "passed" and challenger_hard_validation.status == "passed"
            ):
                # The first blinded payload establishes the randomized A/B
                # mapping.  The probe is then translated into that same
                # vocabulary before the prompt payload is rebuilt.
                empirical_probe_invoked = True
                configured_probe_policy = empirical_probe_policy or EmpiricalProbePolicy()
                if configured_probe_policy.random_state is None:
                    configured_probe_policy = replace(configured_probe_policy, random_state=random_state)
                source_by_label = {
                    "A": blinded_reconciliation.proposal_a_source,
                    "B": blinded_reconciliation.proposal_b_source,
                }
                proposal_by_source = {
                    "agent": modeling_plan,
                    "deterministic": deterministic,
                }
                empirical_probe = run_pairwise_model_probe(
                    probe_training_frame(),
                    approved_target,
                    approved_task,
                    proposal_by_source[source_by_label["A"]],
                    proposal_by_source[source_by_label["B"]],
                    policy=configured_probe_policy,
                    random_state=configured_probe_policy.random_state,
                    validation_by_method={
                        modeling_plan.recommended_method: agent_hard_validation,
                        deterministic.recommended_method: challenger_hard_validation,
                    },
                )
                blinded_reconciliation = build_blinded_reconciliation(
                    reconciliation_profile,
                    modeling_plan,
                    deterministic,
                    target_column=approved_target,
                    task_type=approved_task,
                    preprocessing_comparison=preprocessing_comparison,
                    preprocessing_requirements=requirements.as_dict(),
                    hard_validation={
                        "agent": _reconciliation_hard_summary(agent_hard_validation),
                        "deterministic": _reconciliation_hard_summary(challenger_hard_validation),
                    },
                    order_seed=random_state,
                    proposal_order=(
                        blinded_reconciliation.proposal_a_source,
                        blinded_reconciliation.proposal_b_source,
                    ),
                    empirical_probe=empirical_probe,
                )
            if empirical_probe_invoked and empirical_probe is not None:
                blinded_reconciliation = build_blinded_reconciliation(
                    reconciliation_profile,
                    modeling_plan,
                    deterministic,
                    target_column=approved_target,
                    task_type=approved_task,
                    preprocessing_comparison=preprocessing_comparison,
                    preprocessing_requirements=requirements.as_dict(),
                    hard_validation={
                        "agent": _reconciliation_hard_summary(agent_hard_validation),
                        "deterministic": _reconciliation_hard_summary(challenger_hard_validation),
                    },
                    order_seed=random_state,
                    proposal_order=(
                        blinded_reconciliation.proposal_a_source,
                        blinded_reconciliation.proposal_b_source,
                    ),
                    empirical_probe=empirical_probe,
                )
            resolution = _call_or_fallback(
                "modeling_reconciliation",
                lambda: agents.reconcile_modeling(
                    question,
                    reconciliation_profile,
                    modeling_plan,
                    {
                        **deterministic.model_dump(mode="json"),
                        "_reconciliation_order_seed": blinded_reconciliation.proposal_order_seed,
                        "_reconciliation_proposal_order": [
                            blinded_reconciliation.proposal_a_source,
                            blinded_reconciliation.proposal_b_source,
                        ],
                        "_blinded_reconciliation_payload": blinded_reconciliation.payload,
                        "preprocessing_comparison": preprocessing_comparison,
                        "preprocessing_requirements": requirements.as_dict(),
                        "hard_validation": {
                            "agent_proposal": _reconciliation_hard_summary(agent_hard_validation),
                            "deterministic_challenger": _reconciliation_hard_summary(challenger_hard_validation),
                        },
                        "soft_challenge": {
                            **soft_decision.as_dict(),
                            "status": soft_status,
                            "method_disagreement": method_disagreement,
                            "preprocessing_disagreement": preprocessing_disagreement,
                            "reconciliation_justified": soft_reconciliation_required,
                        },
                        "empirical_probe": empirical_probe,
                    },
                ),
                lambda: _fallback_modeling_resolution(modeling_plan, deterministic),
                warnings,
                agent_sources,
                offline=offline,
                strict_live=strict_live,
            )
            if blinded_reconciliation is not None:
                try:
                    selected_proposal, selected_proposal_source = infer_selected_proposal(
                        resolution,
                        blinded_reconciliation,
                    )
                except ValueError as exc:
                    raise InvariantViolation(
                        f"[reconciliation_selected_proposal_mismatch] {exc}"
                    ) from exc
                if selected_proposal is not None and resolution.selected_proposal is None:
                    resolution = resolution.model_copy(update={"selected_proposal": selected_proposal})
            selected_method = resolution.selected_method
            selected_preprocessing = resolution.selected_preprocessing
            justification = resolution.justification
            status = "disagreement_resolved"
            if selected_method not in {modeling_plan.recommended_method, deterministic.recommended_method}:
                raise InvariantViolation(
                    "[reconciliation_method_is_proposed] Modeling reconciliation invented an unsupported method."
                )

    deterministic_validation = hard_validate(selected_method, selected_preprocessing)
    deterministic_validation.add_check(
        "modeling_reconciliation_method_is_proposed",
        same or selected_method in {modeling_plan.recommended_method, deterministic.recommended_method},
        {
            "selected_method": selected_method,
            "proposed_methods": sorted({modeling_plan.recommended_method, deterministic.recommended_method}),
        },
        "Modeling reconciliation may select only one of the two proposed methods.",
    )
    deterministic_validation.add_check(
        "preprocessing_requirements_recorded",
        True,
        requirements.as_dict(),
        "Preprocessing requirements were derived from the training-only profile after formulation.",
    )
    if deterministic_validation.status == "failed":
        raise InvariantViolation.from_result(
            deterministic_validation,
            {
                "hard_validation": {
                    "status": "failed",
                    "intervention_required": agent_hard_validation.status != "passed",
                    "initial_hard_invalid": agent_hard_validation.status != "passed",
                    "checks": deterministic_validation.as_dict()["checks"],
                    "initial_proposal": agent_hard_validation.as_dict(),
                    "deterministic_challenger": challenger_hard_validation.as_dict(),
                    "final_plan": deterministic_validation.as_dict(),
                },
                "soft_challenge": {
                    "status": soft_status,
                    "agent_method": modeling_plan.recommended_method,
                    "deterministic_method": deterministic.recommended_method,
                    "deterministic_confidence": deterministic.confidence,
                    "method_disagreement": method_disagreement,
                    "preprocessing_disagreement": preprocessing_disagreement,
                    "reconciliation_invoked": resolution is not None,
                    "reconciliation_status": "succeeded" if resolution is not None else "not_invoked",
                },
                "final": {
                    "target_column": approved_target,
                    "task_type": approved_task,
                    "recommended_method": selected_method,
                    "preprocessing": selected_preprocessing.model_dump(mode="json"),
                    "selected_source": "reconciled_contract",
                },
                "hard_validation_intervened": agent_hard_validation.status != "passed",
            },
        )
    hard_artifact = HardValidationArtifact(
        status=deterministic_validation.status,
        intervention_required=agent_hard_validation.status != "passed",
        initial_hard_invalid=agent_hard_validation.status != "passed",
        checks=deterministic_validation.as_dict()["checks"],
        initial_proposal=agent_hard_validation.as_dict(),
        deterministic_challenger=challenger_hard_validation.as_dict(),
        final_plan=deterministic_validation.as_dict(),
    )
    reconciliation_method_source = None
    reconciliation_preprocessing_source = None
    if resolution is not None:
        reconciliation_method_source = (
            selected_proposal_source
            if selected_proposal_source in {"agent", "deterministic"}
            else "agent"
            if selected_method == modeling_plan.recommended_method
            else "deterministic"
            if selected_method == deterministic.recommended_method
            else "other"
        )
        selected_preprocessing_dict = selected_preprocessing.model_dump(mode="json")
        if selected_preprocessing_dict == modeling_plan.preprocessing.model_dump(mode="json"):
            reconciliation_preprocessing_source = "agent"
        elif selected_preprocessing_dict == deterministic.preprocessing.model_dump(mode="json"):
            reconciliation_preprocessing_source = "deterministic"
        else:
            reconciliation_preprocessing_source = "reconciled_contract"
    final_source = (
        "agent"
        if selected_method == modeling_plan.recommended_method
        and selected_preprocessing.model_dump(mode="json") == modeling_plan.preprocessing.model_dump(mode="json")
        else "deterministic"
        if selected_method == deterministic.recommended_method
        and selected_preprocessing.model_dump(mode="json") == deterministic.preprocessing.model_dump(mode="json")
        else "reconciled_contract"
    )
    if empirical_probe_invoked and empirical_probe is not None and probe_status == "not_invoked":
        probe_status = str(empirical_probe.get("status", "unavailable"))
        if probe_status not in {"completed", "unavailable", "failed"}:
            probe_status = "unavailable"
        probe_evidence_strength = str(empirical_probe.get("evidence_strength", "tie"))
        if probe_evidence_strength not in {"tie", "weak", "moderate", "strong"}:
            probe_evidence_strength = "tie"

    decision_path = (
        "hard_validation_correction"
        if hard_reconciliation_required
        else "agreement"
        if not method_disagreement
        else "probe_triggered_blinded_reconciliation"
        if resolution is not None
        else "probe_abstention"
    )
    soft_artifact = SoftChallengeArtifact(
        status=soft_status,
        agent_method=modeling_plan.recommended_method,
        deterministic_method=deterministic.recommended_method,
        deterministic_confidence=deterministic.confidence,
        method_disagreement=method_disagreement,
        preprocessing_disagreement=preprocessing_disagreement,
        reconciliation_invoked=resolution is not None,
        reconciliation_status="succeeded" if resolution is not None else "not_invoked",
        reconciliation_method_source=reconciliation_method_source,
        reconciliation_preprocessing_source=reconciliation_preprocessing_source,
        reconciliation_mode=(
            BLINDED_RECONCILIATION_MODE if blinded_reconciliation is not None else None
        ),
        reconciliation_prompt_version=(
            BLINDED_RECONCILIATION_PROMPT_VERSION if blinded_reconciliation is not None else None
        ),
        proposal_order_seed=(
            blinded_reconciliation.proposal_order_seed if blinded_reconciliation is not None else None
        ),
        proposal_a_source=(
            blinded_reconciliation.proposal_a_source if blinded_reconciliation is not None else None
        ),
        proposal_b_source=(
            blinded_reconciliation.proposal_b_source if blinded_reconciliation is not None else None
        ),
        selected_proposal=selected_proposal,
        selected_proposal_source=selected_proposal_source,
        decision=soft_decision.decision,
        status_detail=soft_decision.status if soft_status != "invalid" else "invalid",
        decision_reason=soft_decision.decision_reason,
        policy_version=soft_decision.policy_version,
        calibration_artifact_version=soft_decision.calibration_artifact_version,
        calibration_regime=soft_decision.calibration_regime,
        calibration_support=soft_decision.calibration_support,
        empirical_reliability=soft_decision.empirical_reliability,
        challenge_win_rate=soft_decision.challenge_win_rate,
        challenge_loss_rate=soft_decision.challenge_loss_rate,
        mean_regret_delta=soft_decision.mean_regret_delta,
        catastrophic_regret_prevention_rate=soft_decision.catastrophic_regret_prevention_rate,
        catastrophic_regret_support=soft_decision.catastrophic_regret_support,
        score_margin=soft_decision.score_margin,
        training_row_count=soft_decision.training_row_count,
        empirical_probe_invoked=empirical_probe_invoked,
        empirical_probe_policy_version=(
            empirical_probe_policy.policy_version
            if empirical_probe_policy is not None
            else EmpiricalProbePolicy().policy_version
        ),
        empirical_probe=empirical_probe,
        probe_status=probe_status,
        probe_evidence_strength=probe_evidence_strength,
        abstention_reason=abstention_reason,
        decision_path=decision_path,
    )
    final_artifact = {
        "target_column": approved_target,
        "task_type": approved_task,
        "recommended_method": selected_method,
        "preprocessing": selected_preprocessing.model_dump(mode="json"),
        "selected_source": final_source,
        "selected_proposal": selected_proposal,
        "selected_proposal_source": selected_proposal_source,
    }
    gate_artifact = ModelingGateArtifact(
        hard_validation=hard_artifact,
        soft_challenge=soft_artifact,
        final=final_artifact,
    )
    validation = {
        "status": status,
        "overall_status": "passed",
        "selected_target_column": approved_target,
        "selected_task_type": approved_task,
        "selected_method": selected_method,
        "justification": justification,
        "checks": deterministic_validation.as_dict()["checks"],
        "confidence": 1.0 if same else resolution.confidence if resolution is not None else modeling_plan.confidence,
        "agent_preprocessing": modeling_plan.preprocessing.model_dump(mode="json"),
        "deterministic_preprocessing": deterministic.preprocessing.model_dump(mode="json"),
        "approved_preprocessing": selected_preprocessing.model_dump(mode="json"),
        "preprocessing_comparison": preprocessing_comparison,
        "reconciliation": resolution.model_dump(mode="json") if resolution else None,
        "reconciliation_mode": BLINDED_RECONCILIATION_MODE if blinded_reconciliation else None,
        "reconciliation_prompt_version": (
            BLINDED_RECONCILIATION_PROMPT_VERSION if blinded_reconciliation else None
        ),
        "proposal_order_seed": (
            blinded_reconciliation.proposal_order_seed if blinded_reconciliation else None
        ),
        "proposal_a_source": blinded_reconciliation.proposal_a_source if blinded_reconciliation else None,
        "proposal_b_source": blinded_reconciliation.proposal_b_source if blinded_reconciliation else None,
        "selected_proposal": selected_proposal,
        "selected_proposal_source": selected_proposal_source,
        "preprocessing_requirements": requirements.as_dict(),
        "deterministic_validation": deterministic_validation.as_dict(),
        "initial_hard_validation": agent_hard_validation.as_dict(),
        "deterministic_challenger_hard_validation": challenger_hard_validation.as_dict(),
        "hard_validation": hard_artifact.model_dump(mode="json"),
        "soft_challenge": soft_artifact.model_dump(mode="json"),
        "soft_challenge_decision": soft_decision.decision,
        "empirical_probe_invoked": empirical_probe_invoked,
        "empirical_probe_policy_version": (
            empirical_probe_policy.policy_version
            if empirical_probe_policy is not None
            else EmpiricalProbePolicy().policy_version
        ),
        "empirical_probe": empirical_probe,
        "probe_status": probe_status,
        "probe_evidence_strength": probe_evidence_strength,
        "abstention_reason": abstention_reason,
        "decision_path": decision_path,
        "final": final_artifact,
        "hard_validation_intervened": agent_hard_validation.status != "passed",
        "agent_method": modeling_plan.recommended_method,
        "deterministic_method": deterministic.recommended_method,
        "deterministic_policy_version": deterministic.policy_version,
        "deterministic_recommendation_evidence": deterministic.model_dump(mode="json"),
    }
    validation["modeling_gate"] = {
        "agent_initial": modeling_plan.model_dump(mode="json"),
        "deterministic": deterministic.model_dump(mode="json"),
        "comparison": {
            "method_agreement": not method_disagreement,
            "preprocessing_agreement": not preprocessing_disagreement,
            "overall_agreement": same,
            "status": soft_status,
        },
        "reconciliation": resolution.model_dump(mode="json") if resolution else None,
        "hard_validation": hard_artifact.model_dump(mode="json"),
        "soft_challenge": soft_artifact.model_dump(mode="json"),
        "soft_challenge_decision": soft_decision.decision,
        "empirical_probe_invoked": empirical_probe_invoked,
        "empirical_probe_policy_version": (
            empirical_probe_policy.policy_version
            if empirical_probe_policy is not None
            else EmpiricalProbePolicy().policy_version
        ),
        "empirical_probe": empirical_probe,
        "probe_status": probe_status,
        "probe_evidence_strength": probe_evidence_strength,
        "abstention_reason": abstention_reason,
        "decision_path": decision_path,
        "final": final_artifact,
        "status": status,
        "artifact": gate_artifact.model_dump(mode="json"),
    }
    return validation


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


def _reconciliation_hard_summary(result: ValidationResult | None) -> dict[str, Any]:
    """Expose hard outcomes to reconciliation without exposing holdout data."""

    if result is None:
        return {"status": "unavailable", "failed_check_codes": [], "passed_check_codes": []}
    return {
        "status": result.status,
        "failed_check_codes": [check.code for check in result.failed_checks],
        "passed_check_codes": [check.code for check in result.checks if check.passed],
    }


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
    strict_live: bool = False,
) -> Any:
    if offline:
        agent_sources[name] = "offline_fallback"
        return fallback()
    try:
        value = call()
    except Exception as exc:
        if strict_live:
            agent_sources[name] = "failed"
            raise
        warnings.append(f"{name} agent fallback used: {exc}")
        agent_sources[name] = "offline_fallback"
        return fallback()
    agent_sources[name] = "openai"
    return value


def _fallback_formulation_plan(
    dataframe: pd.DataFrame,
    profile: dict[str, Any],
    question: str,
    target_hint: str | None,
) -> FormulationPlan:
    del profile
    result = deterministic_formulation(dataframe, question, target_hint)
    if result.status != "proposed" or result.target_column is None or result.task_type is None:
        failed = validate_formulation(
            dataframe,
            result.target_column or target_hint or "",
            result.task_type or "unsupported",
            user_target=target_hint,
        )
        failed.add_failure(
            "formulation_agent_fallback_failed_closed",
            result.reasoning,
            {"evidence": result.evidence},
        )
        raise InvariantViolation.from_result(failed)
    return FormulationPlan(
        target_column=result.target_column,
        task_type=result.task_type,
        reasoning=(
            "Offline fallback uses the deterministic schema formulation as a local heuristic; "
            "this is not evidence of independent LLM reasoning. "
            + result.reasoning
        )[:1200],
        confidence=0.4,
    )


def _compare_formulations(
    agent: FormulationPlan,
    deterministic: DeterministicFormulation,
    user_target: str | None,
) -> FormulationComparison:
    target_agreement = (
        user_target is not None
        or (deterministic.target_column is not None and agent.target_column == deterministic.target_column)
    )
    task_agreement = deterministic.task_type is not None and agent.task_type == deterministic.task_type
    differences: list[str] = []
    if not target_agreement:
        differences.append("target_disagreement")
    if not task_agreement:
        differences.append("task_disagreement")
    overall = target_agreement and task_agreement
    return FormulationComparison(
        target_agreement=target_agreement,
        task_agreement=task_agreement,
        overall_agreement=overall,
        status="agreement" if overall else "disagreement",
        differences=differences,
    )


def _fallback_formulation_resolution(
    agent: FormulationPlan,
    deterministic: DeterministicFormulation,
    user_target: str | None,
) -> FormulationResolution:
    target = user_target or deterministic.target_column or agent.target_column
    task = deterministic.task_type or agent.task_type
    return FormulationResolution(
        selected_target_column=target,
        selected_task_type=task,
        checks=["deterministic_schema_evidence_checked", "user_target_constraint_checked"],
        justification=(
            "Offline formulation reconciliation selected the deterministic schema-based proposal "
            "and preserved any explicit user target constraint; no LLM reconciliation evidence is claimed."
        ),
        confidence=0.4,
    )


def _validate_formulation_resolution(
    resolution: FormulationResolution,
    agent: FormulationPlan,
    deterministic: DeterministicFormulation,
    user_target: str | None,
) -> None:
    proposed_targets = {agent.target_column}
    if deterministic.target_column:
        proposed_targets.add(deterministic.target_column)
    proposed_tasks = {agent.task_type}
    if deterministic.task_type:
        proposed_tasks.add(deterministic.task_type)
    if user_target is not None and resolution.selected_target_column != user_target:
        raise InvariantViolation(
            "[user_target_constraint_violated] Formulation reconciliation cannot override an explicit user target."
        )
    if user_target is None and resolution.selected_target_column not in proposed_targets:
        raise InvariantViolation(
            "[reconciliation_target_is_proposed] Formulation reconciliation selected a target not proposed by either initial path."
        )
    if resolution.selected_task_type not in proposed_tasks:
        raise InvariantViolation(
            "[reconciliation_task_is_proposed] Formulation reconciliation selected a task not proposed by either initial path."
        )


def _fallback_modeling_plan(
    profile: dict[str, Any],
    question: str,
    target_hint: str,
    task_type_hint: str,
) -> ModelingPlan:
    del question
    features = [record for record in profile["column_details"] if record["name"] != target_hint]
    has_categories = any(
        record["semantic_type"] in {"categorical", "boolean"} for record in features
    )
    method = "tree_ensemble" if has_categories else "regularized_linear"
    preprocessing = requirements_from_records(features, task_type_hint, method).expected_contract
    return ModelingPlan(
        recommended_method=method,
        preprocessing=preprocessing,
        reasoning="Offline fallback selects a supported model family from the training-only schema profile; target and task remain immutable formulation context.",
        confidence=0.4,
    )


def _fallback_modeling_resolution(
    modeling_plan: ModelingPlan,
    deterministic: DeterministicRecommendation,
) -> ModelingResolution:
    return ModelingResolution(
        selected_method=deterministic.recommended_method,
        selected_preprocessing=deterministic.preprocessing,
        checks=["deterministic_model_evidence_checked", "preprocessing_contract_checked"],
        justification="Offline modeling reconciliation selected the deterministic model recommendation; no LLM modeling reconciliation evidence is claimed.",
        confidence=0.5,
    )


def _failed_validation_payload(
    result: ValidationResult,
    existing: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
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
    if metadata:
        payload.update(metadata)
    return payload


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
