# Modeling reconciliation

The default post-challenge modeling reconciler is
`2026-08-24.blinded-evidence-comparison.v1` (`blinded_evidence_comparison`). It
is invoked only after the existing formulation gate, frozen split, hard
validation, and soft-challenge policy. The initial modeling proposal and the
deterministic recommendation remain independent; this change only affects the
comparison after a challenge is authorized.

## Prompt contract

The live reconciler receives the following instruction text, followed by an
`INPUT JSON` payload:

```text
You are comparing two independently generated modeling proposals.
They are deliberately presented as Proposal A and Proposal B; do not infer,
mention, or favor their origins. Target and task are immutable approved context.
Choose exactly one of Proposal A or Proposal B, return its model family and a
complete supported preprocessing contract, and never invent Proposal C.

First provide a concise, two-sided critique: strengths and weaknesses for A,
strengths and weaknesses for B, including the strongest case against each.
Then list the decisive observed evidence and select A or B. The output must be
methodological justification, not hidden chain-of-thought. If evidence is close,
still select one proposal and prefer the one whose assumptions are less fragile
and whose complexity is more proportional to the observed evidence; do not use a
universal simplicity or complexity rule.

Distinguish observed dataset evidence from each proposal's interpretation. The
compatibility diagnostics in the input are heuristic structural evidence only.
They are not probabilities, cross-validation results, empirical performance,
expected accuracy/RMSE, or proof that either proposal is better. Do not use holdout
values, candidate-model CV results, empirical-reference rankings, or historical
challenge reliability. Hard validation outcomes describe safety constraints, not
comparative predictive quality. Re-check preprocessing, leakage, immutable context,
supported methods, and the complete contract before returning the selected plan.
```

The JSON payload contains shared `dataset_evidence`, `proposal_a`, `proposal_b`,
shared preprocessing requirements, neutral preprocessing differences, and
source-neutral hard-validation results. Each proposal has the same fields:
`model_family`, `preprocessing`, `rationale`, `supporting_evidence`,
`assumptions`, and `risks`.

Raw compatibility scores, ranked methods, score margins, deterministic
confidence, calibration reliability, holdout results, candidate-model CV
results, empirical-reference results, and source labels are not included in the
default prompt. They remain in runtime artifacts and evaluation logs for audit,
policy calibration, and reporting. A legacy prompt mode is retained for
evaluation baselines only.

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
  "reconciliation_prompt_version": "2026-08-24.blinded-evidence-comparison.v1",
  "proposal_order_seed": 123,
  "proposal_a_source": "agent",
  "proposal_b_source": "deterministic",
  "selected_proposal": "A",
  "selected_proposal_source": "agent"
}
```

These fields are for evaluation and reporting and are not sent to the live
reconciler.

## Evaluation modes and metrics

The evaluation harness supports equivalent runs with:

```text
python -m evaluation.run --reconciliation-mode legacy ...
python -m evaluation.run --reconciliation-mode blinded ...
python -m evaluation.run --reconciliation-mode blinded --order-swap ...
```

`--order-swap` runs paired trials with the same split and evidence in A/B and
B/A order. The summary reports reconciliation invocation, agent versus
deterministic selection, Proposal A/B selection rates and imbalance, and paired
`order_swap_consistency_rate` / `order_swap_flip_rate`. The existing predictive,
intervention, safety, and catastrophic-regret metrics remain unchanged.

