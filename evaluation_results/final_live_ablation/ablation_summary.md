# Paired Modeling-Gate Ablation Study

- Ablation schema: `modeling-gate-ablation-v1`
- Split seeds: `[42, 123, 2027]`
- LLM repetitions per split: `5`
- Strict live: `True`

## Central Comparison

| Ablation | Improved | Worsened | Neutral | Precision | Harm rate | Median regret reduction | Catastrophic net | Initial calls | Reconciliation calls | Probe invocations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_only | 0 | 0 | 0 | None | None | 0.0 | 0 | 270 | 0 | 0 |
| legacy_gate | 3 | 13 | 80 | 0.1875 | 0.13131313131313133 | 0.0 | -2 | 0 | 101 | 0 |
| selective_calibrated | 0 | 0 | 0 | None | None | 0.0 | 0 | 0 | 5 | 0 |
| full | 3 | 0 | 16 | 1.0 | 0.0 | 0.0 | 0 | 0 | 24 | 96 |

## Paired Comparisons

- `full` vs `llm_only`: first better `3`, second better `0`, tied `262`, mean first advantage `0.0011205347757560842`.
- `selective_calibrated` vs `llm_only`: first better `0`, second better `0`, tied `265`, mean first advantage `0.0`.

## Live-Trial Integrity

- `llm_only`: requested `270`, initial failures `0`, reconciliation failures `0`, fallback rows `0`.
- `legacy_gate`: requested `270`, initial failures `0`, reconciliation failures `0`, fallback rows `0`.
- `selective_calibrated`: requested `270`, initial failures `0`, reconciliation failures `0`, fallback rows `0`.
- `full`: requested `270`, initial failures `0`, reconciliation failures `0`, fallback rows `0`.

Initial proposals are keyed by case, perturbation, split seed, LLM repetition, model, prompt schema, training-profile digest, target, and task. They are generated once and reused across compatible ablations; reconciliation outputs are never shared across prompt variants.

Split-seed variation is represented by `split_seed`; stochastic LLM variation is represented independently by `trial`/LLM repetition. Every paired comparison uses the same unit key.
