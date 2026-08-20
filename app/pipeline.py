"""The complete agent-vs-deterministic workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.deterministic import (
    apply_cleaning,
    deterministic_recommendation,
    eda_summary,
    make_plots,
    profile_dataframe,
)
from app.llm import OpenAIAgents
from app.modeling import fit_selected_model
from app.reporting import render_code, render_report, write_json
from app.schemas import AgentPlan, CleaningPlan, DeterministicRecommendation, ReportDraft


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

    agent_plan = _call_or_fallback(
        "planning",
        lambda: agents.planning(profile, question, target_column),
        lambda: _fallback_agent_plan(profile, question, target_column),
        warnings,
        agent_sources,
        offline=offline,
    )
    deterministic = deterministic_recommendation(dataframe, question, target_column)
    validation = _validate_before_training(
        agents,
        profile,
        question,
        agent_plan,
        deterministic,
        warnings,
        agent_sources,
        offline=offline,
    )
    write_json(
        run_dir / "decision.json",
        {
            "agent_plan": agent_plan.model_dump(mode="json"),
            "deterministic_recommendation": deterministic.model_dump(mode="json"),
            "validation": validation,
            "warnings": warnings,
            "agent_sources": agent_sources,
        },
    )
    selected_target = validation["selected_target_column"]
    if selected_target not in dataframe.columns:
        raise ValueError("The validation agent selected a target that is not in the input data.")

    cleaning_plan = _call_or_fallback(
        "cleaning",
        lambda: agents.cleaning(profile, selected_target),
        lambda: _fallback_cleaning_plan(profile),
        warnings,
        agent_sources,
        offline=offline,
    )
    cleaned, cleaning_log = apply_cleaning(
        dataframe,
        selected_target,
        list(cleaning_plan.actions),
    )
    cleaned_path = run_dir / "data" / "cleaned.csv"
    cleaned.to_csv(cleaned_path, index=False)
    write_json(
        run_dir / "cleaning.json",
        {"plan": cleaning_plan.model_dump(mode="json"), "log": cleaning_log},
    )

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

    modeling = fit_selected_model(
        cleaned,
        target_column=selected_target,
        task_type=validation["selected_task_type"],
        method=validation["selected_method"],
        output_dir=run_dir / "model",
        test_size=test_size,
        random_state=random_state,
    )
    write_json(run_dir / "modeling.json", modeling)

    report_context = {
        "profile": profile,
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
    )
    (run_dir / "reproduce_analysis.py").write_text(code, encoding="utf-8")
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


def _validate_before_training(
    agents: OpenAIAgents,
    profile: dict[str, Any],
    question: str,
    agent_plan: AgentPlan,
    deterministic: DeterministicRecommendation,
    warnings: list[str],
    agent_sources: dict[str, str],
    offline: bool,
) -> dict[str, Any]:
    same = (
        agent_plan.target_column == deterministic.target_column
        and agent_plan.task_type == deterministic.task_type
        and agent_plan.recommended_method == deterministic.recommended_method
    )
    if same:
        return {
            "status": "agreement",
            "selected_target_column": deterministic.target_column,
            "selected_task_type": deterministic.task_type,
            "selected_method": deterministic.recommended_method,
            "justification": "The independent agent and deterministic recommender agreed on target, task type, and method before training.",
            "checks": ["target_match", "task_match", "method_match"],
            "confidence": 1.0,
        }

    resolution = _call_or_fallback(
        "reconciliation",
        lambda: agents.reconcile(
            question,
            profile,
            agent_plan,
            deterministic.model_dump(mode="json"),
        ),
        lambda: _fallback_resolution(agent_plan, deterministic),
        warnings,
        agent_sources,
        offline=offline,
    )
    proposed_methods = {
        agent_plan.recommended_method,
        deterministic.recommended_method,
    }
    selected_method = (
        resolution.selected_method
        if resolution.selected_method in proposed_methods
        else deterministic.recommended_method
    )
    selected_target = (
        resolution.selected_target_column
        if resolution.selected_target_column
        in {agent_plan.target_column, deterministic.target_column}
        else deterministic.target_column
    )
    selected_task = (
        resolution.selected_task_type
        if resolution.selected_task_type in {agent_plan.task_type, deterministic.task_type}
        else deterministic.task_type
    )
    return {
        "status": "disagreement_resolved",
        "selected_target_column": selected_target,
        "selected_task_type": selected_task,
        "selected_method": selected_method,
        "justification": resolution.justification,
        "checks": resolution.checks,
        "confidence": resolution.confidence,
        "agent_method": agent_plan.recommended_method,
        "deterministic_method": deterministic.recommended_method,
    }


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
) -> AgentPlan:
    columns = [record["name"] for record in profile["column_details"]]
    target = target_hint or next(
        (column for column in columns if column.lower() in question.lower()),
        columns[-1],
    )
    target_record = next(record for record in profile["column_details"] if record["name"] == target)
    task = (
        "classification"
        if target_record["semantic_type"] in {"categorical", "boolean", "text"}
        else "regression"
    )
    features = [record for record in profile["column_details"] if record["name"] != target]
    has_categories = any(
        record["semantic_type"] in {"categorical", "boolean"} for record in features
    )
    method = "tree_ensemble" if has_categories else "regularized_linear"
    return AgentPlan(
        target_column=target,
        task_type=task,
        recommended_method=method,
        preprocessing=["training_only_imputation", "schema_aware_encoding"],
        reasoning="Offline fallback uses an independent schema heuristic so the workflow remains runnable without an API key.",
        confidence=0.4,
    )


def _fallback_resolution(
    agent_plan: AgentPlan,
    deterministic: DeterministicRecommendation,
):
    from app.schemas import ConflictResolution

    return ConflictResolution(
        selected_target_column=deterministic.target_column,
        selected_task_type=deterministic.task_type,
        selected_method=deterministic.recommended_method,
        checks=[
            "deterministic_target_exists",
            "deterministic_task_is_feasible",
            "deterministic_method_is_allow_listed",
        ],
        justification="The offline validation fallback selected the deterministic recommendation because it is tied directly to the observed schema and does not invent a new method.",
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
