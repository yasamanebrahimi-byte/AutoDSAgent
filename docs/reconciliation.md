# Modeling reconciliation

The default probe-triggered modeling reconciler is
`2026-09-04.blinded-canonical-proposals.v1-empirical-probe` (`blinded_evidence_comparison`). It
is invoked only after the existing formulation gate, frozen split, and hard
validation. For a valid model-family disagreement, the bounded pairwise probe
runs first on the frozen training partition. Only moderate or strong evidence
authorizes reconsideration; unavailable, tied, or weak evidence preserves the
initial LLM plan. The initial modeling proposal and deterministic recommendation
remain independent. The deterministic recommendation is an advisory hypothesis,
and its compatibility scores are heuristics rather than predicted accuracy or
probabilities.

## Prompt contract

The live reconciler receives the following instruction text, followed by an
`INPUT JSON` payload:

```text
You are comparing two independently generated modeling proposals.
They are deliberately presented as Proposal A and Proposal B; do not infer,
mention, or favor their origins. Target and task are immutable approved context.
Choose exactly one of Proposal A or Proposal B, return its model family and a
complete supported preprocessing contract, and never invent Proposal C.

Evaluate A and B only from dataset/task evidence, methodological suitability,
preprocessing/model compatibility, empirical probe evidence when available,
risks, and assumptions. Do not infer how either proposal was generated. First provide a concise, two-sided critique: strengths and weaknesses for A,
strengths and weaknesses for B, including the strongest case against each.
Then list the decisive observed evidence and select A or B. The output must be
methodological justification, not hidden chain-of-thought. If evidence is close,
still select one proposal and prefer the one whose assumptions are less fragile
and whose complexity is more proportional to the observed evidence; do not use a
universal simplicity or complexity rule.

Distinguish observed dataset evidence from each proposal's interpretation. The
compatibility diagnostics in the input are heuristic structural evidence only.
They are not probabilities, cross-validation results, empirical performance,
expected accuracy/RMSE, or proof that either proposal is better. If present,
`LIMITED TRAINING-ONLY EMPIRICAL COMPARISON` is a small directional comparison of
only Proposal A and Proposal B using training-side folds. It is not final holdout
performance or a guarantee of future generalization; fold variability matters and
preprocessing was fitted inside each fold. Weigh strong, consistent direct evidence
more heavily than heuristic point scores, but do not treat its winner as an
automatic final decision. Do not use holdout values, empirical-reference rankings,
or calibration reliability. Hard validation outcomes describe safety
constraints, not comparative predictive quality. Re-check preprocessing, leakage,
immutable context, supported methods, and the complete contract before returning
the selected plan.
```

The JSON payload contains shared `dataset_evidence`, `proposal_a`, `proposal_b`,
shared preprocessing requirements, neutral preprocessing differences, and
source-neutral hard-validation results. Each proposal has the same fields:
`model_family`, `task_type`, `preprocessing`, `preprocessing_plan`,
`feature_handling`, `modeling_strategy`, `rationale`, `supporting_evidence`,
`contradicting_evidence`, `assumptions`, `risks`, `constraints`, and
`expected_failure_modes`. Both candidates are rendered by the same deterministic
canonical formatter; raw source rationale and recommendation metadata are not
copied. A challenged payload may also contain an
`empirical_probe` block with symmetric A/B mean and fold scores, standard
deviations, fold wins, metric direction, relative advantage, and evidence
strength. A failed or unavailable probe is recorded as such and is not treated as
evidence for either proposal.

Raw compatibility scores, ranked methods, score margins, deterministic
confidence, soft-challenge calibration reliability, holdout results,
candidate-model CV results, empirical-reference results, and source labels are
not included in the default prompt. They remain in runtime artifacts and
evaluation logs for audit, policy calibration, and reporting.

The structured response adds `selected_proposal` (`A` or `B`), concise strengths
and weaknesses for both proposals, `decisive_evidence`, `selection_confidence`,
the selected method, the complete preprocessing contract, and a justification.
The pipeline still maps the selected contract back to the underlying source and
re-runs hard validation; invalid or unsupported reconciled plans fail closed.

## Ordering and artifacts

Proposal order is reproducible. The pipeline hashes the trial/split seed,
approved target/task, and a stable aggregate training-profile representation,
then uses a local seeded shuffle of `agent` and `deterministic`. Global random
state is never used. The resolved ordering is retained privately in artifacts:

```json
{
  "reconciliation_mode": "blinded_evidence_comparison",
  "reconciliation_prompt_version": "2026-09-04.blinded-canonical-proposals.v1-empirical-probe",
  "proposal_order_seed": 123,
  "proposal_a_source": "agent",
  "proposal_b_source": "deterministic",
  "selected_proposal": "A",
  "selected_proposal_source": "agent"
}
```

These fields are for evaluation and reporting and are not sent to the live
reconciler. Internally, the architecture is: independently generate both
proposals; canonicalize both; construct shared evidence; randomly assign the
canonical candidates to A/B; reconcile; then remap the selected label to the
private source record. Provenance is withheld not only by removing source labels,
but also by canonical formatting that removes source-specific stylistic cues.
The reconciler can see the approved target/task, aggregate training-only dataset
evidence, both canonical plans and their preprocessing contracts, neutralized
preprocessing differences, hard safety outcomes, and optional training-only
pairwise probe results. It cannot see source labels, raw rationales, method
scores/ranks, source identifiers, calibration metadata, holdout results, or
other source-specific recommendation fields.

## Ablation semantics

| Ablation | Initial LLM | Hard validation | Deterministic challenger | Empirical probe | Abstention | LLM reconciliation | Direct probe selection |
|---|---|---|---|---|---|---|---|
| `llm_only` | yes | minimum execution guard | no soft use | no | no | no | no |
| `hard_validation_only` | yes | yes; repairs invalid plans only | advisory only | no | no | no | no |
| `deterministic_only` | no final LLM choice | yes | final choice | no | no | no | no |
| `always_reconcile` | yes | yes | yes | no | no | every valid disagreement | no |
| `probe_direct` | yes | yes | yes | yes | weak/tied/unavailable | no | moderate/strong winner |
| `full` | yes | yes | yes | yes | weak/tied/unavailable | moderate/strong disagreement | no |

The primary sequence answers distinct causal questions: the quality of the raw
planner; the safety contribution of invariants; the deterministic alternative;
the cost of intervening on every disagreement; whether probe evidence alone is
enough; and whether blinded reconciliation improves on that direct choice.
`selective_calibrated` and `probe_first` are retained only as legacy aliases or
diagnostic modes. Calibration metadata is recorded for audit but is not the
production reconciliation gate.

## Evaluation modes and metrics

The evaluation harness supports the current modeling gate with:

```text
python -m evaluation.run --gate-mode selective ...
python -m evaluation.run --gate-mode selective --order-swap ...
```

`--order-swap` runs paired trials with the same split and evidence in A/B and
B/A order. The summary reports reconciliation invocation, agent versus
deterministic selection, Proposal A/B selection rates and imbalance, and paired
`order_swap_consistency_rate` / `order_swap_flip_rate`. The existing predictive,
intervention, safety, and catastrophic-regret metrics remain unchanged.

The `probe_direct` preset reuses the same cached initial LLM proposal and the
same training-only probe configuration as `full`, but directly selects a
moderate/strong probe winner and never invokes the LLM reconciler. The `full`
preset is the production policy: weak, tied, or unavailable evidence abstains;
moderate or strong evidence invokes blinded reconciliation.
