"""Deterministic Markdown rendering for evaluation results."""

from __future__ import annotations

from typing import Any


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _agreement(records: list[dict[str, Any]], field: str) -> float | None:
    eligible = [record for record in records if record.get(field) is not None]
    if not eligible:
        return None
    return sum(record[field] == "agreement" for record in eligible) / len(eligible)


def _dimension_agreement(records: list[dict[str, Any]], disagreement_field: str) -> float | None:
    eligible = [record for record in records if record.get(disagreement_field) is not None]
    if not eligible:
        return None
    return sum(not bool(record[disagreement_field]) for record in eligible) / len(eligible)


def render_summary_markdown(
    config: dict[str, Any], trials: list[dict[str, Any]], summary: dict[str, Any]
) -> str:
    """Render only from saved rows and summary values; no LLM prose is used."""

    openai_trials = [record for record in trials if record.get("agent_source") == "openai"]
    openai_only = summary.get("openai_only", {})
    openai_pair = summary.get("openai_only_paired_stats", {})
    outcomes = summary.get("gating_outcome_counts", {})
    openai_outcomes = {
        "improved": openai_pair.get("improved_count", 0),
        "worsened": openai_pair.get("worsened_count", 0),
        "tie": openai_pair.get("tie_count", 0),
    }
    rows = [
        "# AutoDS Validation Architecture Evaluation",
        "",
        "This report is generated deterministically from `config.json`, `trials.jsonl`, and the computed summary. Offline fallback and mock rows are not evidence of live LLM performance.",
        "",
        "## Experiment Configuration",
        "",
        f"- Repetitions per benchmark/scenario: **{config.get('repetitions', 'n/a')}**.",
        f"- Base seed: **{config.get('seed', 'n/a')}**; holdout fraction: **{config.get('test_size', 'n/a')}**.",
        f"- Requested model: `{config.get('agent_model_requested', 'n/a')}`; prompt/schema version: `{config.get('prompt_schema_version', 'n/a')}`.",
        f"- Repository commit: `{config.get('repository_commit') or 'unavailable'}`.",
        "- Each repetition keeps the case, frozen train/holdout membership, and training-only profile fixed; the intended varying factor is the stochastic LLM response.",
        "- These rows are `modeling_gate` evaluations: benchmark target/task values are fixed context, while `agent_initial` represents only the post-split model-family and preprocessing proposal. `gated_final` is the approved plan after comparison, optional reconciliation, and deterministic validation. Formulation accuracy requires a separate formulation-gate evaluation mode.",
        "- `empirical_reference` is an evaluation-only ranking of the four supported families using training-only CV; it is not an oracle and never enters runtime decisions.",
        "",
        "## Trial Coverage",
        "",
        "| Trial category | Count |",
        "|---|---:|",
        f"| Requested live trials | {summary.get('requested_live_trials', 0)} |",
        f"| Successful OpenAI trials | {summary.get('successful_openai_trials', 0)} |",
        f"| Offline fallback trials | {summary.get('offline_fallback_trials', 0)} |",
        f"| Failed trials | {summary.get('failed_trials', 0)} |",
        f"| Mock trials | {summary.get('mock_trials', 0)} |",
        f"| Completed trials | {summary.get('completed_trial_count', 0)} |",
        "",
        "Claims about LLM behavior below use `agent_source == \"openai\"` only.",
        "",
        "## LLM Decision Stability",
        "",
    ]
    stability = summary.get("stability_by_dataset", {})
    if stability:
        rows.extend([
            "| Dataset | OpenAI trials | Unique initial methods | Modal method | Modal frequency | Pairwise consistency |",
            "|---|---:|---:|---|---:|---:|",
        ])
        for dataset, item in stability.items():
            rows.append(
                f"| {dataset} | {item.get('trial_count', 0)} | {item.get('unique_initial_methods_selected', 0)} | {item.get('modal_method', 'n/a')} | {_percent(item.get('modal_method_rate'))} | {_percent(item.get('pairwise_consistency'))} |"
            )
    else:
        rows.append("No successful OpenAI trials were recorded, so live decision stability is not estimable.")
    rows.extend([
        "",
        "## Agent vs Deterministic Soft Challenge",
        "",
        f"- All operational trials: soft agreement **{_percent(summary.get('agreement_rate'))}**, soft disagreement **{_percent(summary.get('disagreement_rate'))}**.",
        f"- Model-family disagreement rate: **{_percent(summary.get('model_family_disagreement_rate'))}**; preprocessing disagreement rate: **{_percent(summary.get('preprocessing_disagreement_rate'))}**.",
        f"- OpenAI-only method agreement: **{_percent(_dimension_agreement(openai_trials, 'method_disagreement'))}**; preprocessing agreement: **{_percent(_dimension_agreement(openai_trials, 'preprocessing_disagreement'))}**.",
        "",
        "| Method distribution | Initial agent | Gated final |",
        "|---|---|---|",
        f"| All trials | {summary.get('initial_method_distribution', {})} | {summary.get('final_method_distribution', {})} |",
        f"| OpenAI only | {openai_only.get('initial_method_distribution', {})} | {openai_only.get('final_method_distribution', {})} |",
        "",
        "## Empirical Reference Comparison",
        "",
        f"- All operational trials: initial reference match **{_percent(summary.get('agent_empirical_reference_match_rate'))}**; gated reference match **{_percent(summary.get('gated_empirical_reference_match_rate'))}**.",
        f"- OpenAI only: initial reference match **{_percent(summary.get('openai_only_match_rates', {}).get('initial_reference_match_rate'))}**; gated reference match **{_percent(summary.get('openai_only_match_rates', {}).get('gated_reference_match_rate'))}**.",
        "- The empirical reference represents the best-performing candidate among the four supported model families under the configured training-only cross-validation procedure. It is not a universal optimum or ground truth.",
        "",
        "## Effect of the Validation Gate",
        "",
        f"- OpenAI-only gate outcomes: **{openai_outcomes['improved']} improved**, **{openai_outcomes['worsened']} worsened**, **{openai_outcomes['tie']} tied**.",
        f"- OpenAI-only potentially unnecessary interventions: **{openai_only.get('potentially_unnecessary_intervention_count', 0)}**.",
        f"- Operational outcomes: improved **{outcomes.get('improved', outcomes.get('gated_better_count', 0))}**, worsened **{outcomes.get('worsened', outcomes.get('agent_better_count', 0))}**, tie **{outcomes.get('tie', outcomes.get('tie_count', 0))}**.",
        "- Improved/worsened/tie is defined from paired training-only CV regret using the configured tolerance; holdout results do not define this label.",
        "",
        "## Soft-Challenge Reconciliation Outcomes",
        "",
        f"- Total disagreements: **{summary.get('total_disagreements', summary.get('soft_challenge_count', 0))}**; challenges: **{summary.get('challenges', 0)}**; abstentions: **{summary.get('abstentions', 0)}**.",
        f"- Challenge rate: **{_percent(summary.get('challenge_rate'))}**; abstention rate: **{_percent(summary.get('abstention_rate'))}**.",
        f"- Soft-challenge reconciliation invocation rate: **{_percent(summary.get('soft_challenge_reconciliation_invocation_rate'))}**.",
        f"- Reconciliation invocation rate: **{_percent(summary.get('reconciliation_invocation_rate'))}**; success rate: **{_percent(summary.get('reconciliation_success_rate'))}**.",
        f"- Sided with agent: **{_percent(summary.get('reconciliation_sided_with_agent_rate'))}**; sided with deterministic challenger: **{_percent(summary.get('reconciliation_sided_with_deterministic_rate'))}**.",
        f"- Proposal A selected: **{_percent(summary.get('reconciliation_a_selected_rate'))}**; Proposal B selected: **{_percent(summary.get('reconciliation_b_selected_rate'))}**; A/B selection imbalance: **{_percent(summary.get('reconciliation_a_b_selection_imbalance'))}**.",
        f"- Order-swap consistency: **{_percent(summary.get('order_swap_consistency_rate'))}**; order-flip rate: **{_percent(summary.get('order_swap_flip_rate'))}** over **{summary.get('order_swap_pair_count', 0)}** paired cases.",
        f"- Reconciliation modes observed: **{sorted({record.get('reconciliation_mode') for record in trials if record.get('reconciliation_mode')}) or ['none']}**.",
        f"- Soft-challenge outcomes: **{summary.get('soft_challenge_improved_count', 0)} improved**, **{summary.get('soft_challenge_worsened_count', 0)} worsened**, **{summary.get('soft_challenge_neutral_count', 0)} neutral**.",
        f"- Challenge outcomes: **{summary.get('challenge_improved_count', 0)} improved**, **{summary.get('challenge_worsened_count', 0)} worsened**, **{summary.get('challenge_neutral_count', 0)} neutral**; intervention precision: **{_percent(summary.get('intervention_precision'))}**.",
        f"- Abstentions where agent was better: **{summary.get('abstained_agent_better_count', 0)}**; where deterministic was better: **{summary.get('abstained_deterministic_better_count', 0)}**.",
        f"- Mean challenge regret improvement: **{_number(summary.get('mean_performance_delta_conditional_on_challenge'))}**; unnecessary interventions: **{summary.get('unnecessary_intervention_count', 0)}** (**{_percent(summary.get('unnecessary_intervention_rate'))}**).",
        f"- Catastrophic-regret rate: **{_percent(summary.get('catastrophic_regret_rate'))}**; catastrophic cases prevented by challenge: **{summary.get('catastrophic_regret_prevented_by_challenge_count', 0)}** (**{_percent(summary.get('catastrophic_regret_prevented_by_challenge_rate'))}**).",
        "- A soft disagreement is competing advisory evidence, not an invalid plan. Every challenge row retains the initial plan, deterministic plan, preprocessing comparison, reconciliation response, selected source, and final hard-validation result.",
        "",
        "## Predictive Performance",
        "",
        f"- OpenAI-only mean paired CV improvement: **{_number(openai_pair.get('mean_paired_improvement'))}**; median: **{_number(openai_pair.get('median_paired_improvement'))}**; standard deviation: **{_number(openai_pair.get('std_paired_improvement'))}**.",
        f"- OpenAI-only mean paired holdout improvement: **{_number(openai_pair.get('mean_paired_holdout_improvement'))}** (descriptive only; not used to define gate outcomes).",
        "- Classification improvement is `gated_macro_f1 - initial_macro_f1`; regression improvement is `initial_rmse - gated_rmse`, so positive always means gating helped.",
        "- Untouched holdout metrics are retained per trial as a descriptive external check after decisions and the empirical ranking are frozen.",
        "",
        "## Dataset-Level Results",
        "",
        "| Dataset | Trials | Initial match | Gated match | Improved / worsened / tie | Mean paired CV improvement |",
        "|---|---:|---:|---:|---|---:|",
    ])
    for dataset, item in summary.get("by_dataset", {}).items():
        live_item = item.get("openai_only", {})
        item_outcomes = live_item.get("gating_outcome_counts", {})
        rows.append(
            f"| {dataset} | {item.get('openai_trial_count', 0)} | {_percent(live_item.get('agent_empirical_reference_match_rate'))} | {_percent(live_item.get('gated_empirical_reference_match_rate'))} | {item_outcomes.get('improved', 0)} / {item_outcomes.get('worsened', 0)} / {item_outcomes.get('tie', 0)} | {_number(live_item.get('paired_cv_improvement_mean'))} |"
        )
    rows.extend([
        "",
        "## Validation / Safety Interceptions",
        "",
        f"- Initial hard-invalid proposals: **{summary.get('initial_hard_invalid_proposal_count', summary.get('agent_initial_invalid_count', 0))}**; hard-validation interventions: **{summary.get('hard_validation_intervention_count', summary.get('unsafe_plan_interception_count', 0))}**; interception rate: **{_percent(summary.get('hard_validation_interception_rate'))}**.",
        f"- Final hard-invalid trials: **{summary.get('final_hard_invalid_count', summary.get('final_invalid_count', 0))}**; validation failure codes: `{summary.get('failure_counts_by_validation_code', {})}`.",
        "- Hard validation is authoritative for safety and executability. Model-family disagreement is reported above as a soft challenge and is not counted as an invalid plan by itself.",
        f"- Intentionally unsafe perturbations intercepted: **{summary.get('validation_interception_count', 0)}** / **{summary.get('perturbation_trial_count', 0)}** perturbation trials where applicable.",
        "",
        "## Limitations",
        "",
        "- The benchmark suite is small and local; it is not representative of every tabular data-science domain.",
        "- The empirical reference is not a universal optimum or ground truth; it ranks only the supported families under one CV design.",
        "- Method-family match is not equivalent to predictive or deployment quality, and a one-split study cannot establish generalization.",
        "- Offline fallback and mock rows must not be used to make claims about live LLM behavior.",
        "- Semantic leakage, feature availability, and domain-specific safety still require expert review.",
        "",
    ])
    return "\n".join(rows)
