"""Deterministic Markdown rendering for evaluation results."""

from __future__ import annotations

from typing import Any


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def render_summary_markdown(
    config: dict[str, Any], trials: list[dict[str, Any]], summary: dict[str, Any]
) -> str:
    """Render quantitative conclusions directly from saved structured rows."""

    by_task = summary.get("by_task", {})
    rows = [
        "# AutoDSAgent Validation Architecture Evaluation",
        "",
        "Offline/mock results are not evidence of live LLM performance.",
        "",
        "## Evaluation setup",
        "",
        f"- Trials: **{summary.get('trial_count', 0)}** ({summary.get('clean_trial_count', 0)} clean, {summary.get('perturbation_trial_count', 0)} perturbation).",
        f"- Repetitions per case/scenario: **{config.get('repetitions', 'n/a')}**.",
        f"- Agent mode: **{config.get('agent_mode', 'n/a')}**; requested model: `{config.get('agent_model_requested', 'n/a')}`.",
        "- The final holdout was frozen before model-family decisions and was reserved for final evaluation.",
        "",
        "## Benchmark composition",
        "",
        "| Case | Task | Rows | Source |",
        "|---|---|---:|---|",
    ]
    for case in config.get("benchmark_cases", []):
        rows.append(
            f"| {case['name']} | {case['expected_task_type']} | {case.get('rows', 'n/a')} | {case['dataset_source']} |"
        )
    rows.extend(
        [
            "",
            "## Agent vs deterministic agreement",
            "",
            f"- Agreement rate: **{_percent(summary.get('agreement_rate'))}**; disagreement rate: **{_percent(summary.get('disagreement_rate'))}**.",
            "",
            "| Dimension | Disagreements |",
            "|---|---:|",
        ]
    )
    for dimension in ("target", "task", "method", "preprocessing"):
        count = sum(bool(trial.get(f"{dimension}_disagreement")) for trial in trials)
        rows.append(f"| {dimension} | {count} |")
    rows.extend(
        [
            "",
            "## Safety / validity results",
            "",
            f"- Final valid/invalid trials: **{summary.get('valid_trial_count', 0)} / {summary.get('invalid_trial_count', 0)}**.",
            f"- Initial agent validity rate: **{_percent(summary.get('agent_initial_validity_rate'))}**.",
            f"- Initial invalid proposals: **{summary.get('agent_initial_invalid_count', 0)}**.",
            f"- Unsafe proposal interception: **{summary.get('unsafe_plan_interception_count', 0)}** / invalid initial proposals (**{_percent(summary.get('unsafe_plan_interception_rate'))}**).",
            f"- Intentionally unsafe perturbation interception: **{summary.get('validation_interception_count', 0)}** (**{_percent(summary.get('validation_interception_rate'))}**).",
            f"- Final invalid rate: **{_percent(summary.get('final_invalid_rate'))}**.",
            f"- Validation failures by code: `{summary.get('failure_counts_by_validation_code', {})}`.",
            "",
            "## Reconciliation results",
            "",
            f"- Reconciliation invocation rate: **{_percent(summary.get('reconciliation_invocation_rate'))}**.",
            f"- Reconciliation success rate: **{_percent(summary.get('reconciliation_success_rate'))}**.",
            "",
            "| Selection source | Successful reconciliations |",
            "|---|---:|",
        ]
    )
    for source in ("agent", "deterministic", "other"):
        count = sum(
            trial.get("reconciliation_method_source") == source
            and trial.get("reconciliation_status") == "succeeded"
            for trial in trials
        )
        rows.append(f"| {source} | {count} |")
    rows.extend(
        [
            "",
            "## Empirical model-family comparison",
            "",
            f"- Empirical-reference match rate: agent initial **{_percent(summary.get('agent_empirical_reference_match_rate'))}**; gated final **{_percent(summary.get('gated_empirical_reference_match_rate'))}**.",
            "- The empirical reference is a post-hoc training-only benchmark over the supported candidate set, not a universal optimum.",
            "",
            "| Case | Trial | Best family | Candidate ranking |",
            "|---|---:|---|---|",
        ]
    )
    for trial in trials:
        reference = trial.get("empirical_reference", {})
        rows.append(
            f"| {trial.get('benchmark_case')} / {trial.get('perturbation_id')} | {trial.get('trial')} | {reference.get('best_method', 'n/a')} | {', '.join(reference.get('ranking', [])) or 'none'} |"
        )
    rows.extend(
        [
            "",
            "## Regret analysis",
            "",
            f"- Mean normalized regret: agent **{_number(summary.get('agent_normalized_regret_mean'))}**, gated **{_number(summary.get('gated_normalized_regret_mean'))}**.",
            f"- Median normalized regret: agent **{_number(summary.get('agent_normalized_regret_median'))}**, gated **{_number(summary.get('gated_normalized_regret_median'))}**.",
            f"- Paired comparison: gated better **{summary.get('gating_outcome_counts', {}).get('gated_better_count', 0)}**, agent better **{summary.get('gating_outcome_counts', {}).get('agent_better_count', 0)}**, tie **{summary.get('gating_outcome_counts', {}).get('tie_count', 0)}** (eligible **{summary.get('gating_outcome_counts', {}).get('eligible_count', 0)}**).",
            "- Classification regret is `best_macro_f1 - selected_macro_f1`; regression regret is `selected_rmse - best_rmse`. Regression aggregate regret is normalized by the best RMSE for that trial.",
            "",
            "| Task | Trials | Agent mean normalized regret | Gated mean normalized regret | Improved / worse / unchanged |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for task in ("classification", "regression"):
        item = by_task.get(task, {})
        outcomes = item.get("gating_outcome_counts", {})
        rows.append(
            f"| {task} | {item.get('trial_count', 0)} | {_number(item.get('agent_normalized_regret_mean'))} | {_number(item.get('gated_normalized_regret_mean'))} | {outcomes.get('improved', 0)} / {outcomes.get('worsened', 0)} / {outcomes.get('unchanged', 0)} |"
        )
    rows.extend(
        [
            "",
            "## Perturbation results",
            "",
            "| Scenario | Kind | Trials | Expected failed checks observed |",
            "|---|---|---:|---:|",
        ]
    )
    perturbation_ids = sorted({trial.get("perturbation_id") for trial in trials if trial.get("perturbation_id") != "clean"})
    for perturbation_id in perturbation_ids:
        subset = [trial for trial in trials if trial.get("perturbation_id") == perturbation_id]
        observed = sum(bool(trial.get("expected_perturbation_checks_observed")) for trial in subset)
        kind = subset[0].get("perturbation", {}).get("kind", "n/a") if subset else "n/a"
        rows.append(f"| {perturbation_id} | {kind} | {len(subset)} | {observed} |")
    rows.extend(
        [
            "",
            "## Potentially unnecessary intervention",
            "",
            f"- Count: **{summary.get('potentially_unnecessary_intervention_count', 0)}**; denominator is valid initial plans that materially disagreed on model family (**{_percent(summary.get('potentially_unnecessary_intervention_rate'))}**).",
            "- This is an exploratory heuristic, not a correctness theorem: the configured approximation thresholds are shown in `config.json`.",
            "",
            "## Limitations",
            "",
            "- The benchmark suite is small and may not represent real project domains.",
            "- Only the currently supported tabular model families and one CV procedure are compared.",
            "- The empirical reference is not ground truth or a universal optimum.",
            "- LLM behavior is stochastic; meaningful live conclusions require repeated live-agent trials.",
            "- Model-family choice is only one component of data-science quality.",
            "- Semantic/domain leakage and feature availability cannot be fully validated automatically.",
            "- Offline fallback and mock rows must be filtered out before making claims about actual LLM behavior.",
            "",
        ]
    )
    return "\n".join(rows)
