# Paired Modeling-Gate Ablation Study

- Ablation schema: `modeling-gate-ablation-v1`
- Split seeds: `[42, 123]`
- LLM repetitions per split: `2`
- Strict live: `False`

## Central Comparison

| Ablation | Improved | Worsened | Neutral | Precision | Harm rate | Median regret reduction | Catastrophic net | Initial calls | Reconciliation calls | Probe invocations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_only | 0 | 0 | 0 | None | None | 0.0 | 0 | 0 | 0 | 0 |
| deterministic_only | 0 | 0 | 0 | None | None | 0.0 | 0 | 0 | 0 | 0 |
| selective_calibrated | 0 | 0 | 0 | None | None | 0.0 | 0 | 0 | 0 | 0 |
| interaction_boundary_aware | 0 | 0 | 0 | None | None | 0.0 | 0 | 0 | 0 | 0 |
| empirical_probe | 0 | 0 | 0 | None | None | 0.0 | 0 | 0 | 0 | 0 |

## Paired Comparisons

- `interaction_boundary_aware` vs `selective_calibrated`: first better `0`, second better `0`, tied `4`, mean first advantage `0.0`.
- `empirical_probe` vs `interaction_boundary_aware`: first better `0`, second better `0`, tied `4`, mean first advantage `0.0`.
- `selective_calibrated` vs `llm_only`: first better `0`, second better `0`, tied `4`, mean first advantage `0.0`.

## Live-Trial Integrity

- `llm_only`: requested `0`, initial failures `0`, reconciliation failures `0`, fallback rows `4`.
- `deterministic_only`: requested `0`, initial failures `0`, reconciliation failures `0`, fallback rows `0`.
- `selective_calibrated`: requested `0`, initial failures `0`, reconciliation failures `0`, fallback rows `4`.
- `interaction_boundary_aware`: requested `0`, initial failures `0`, reconciliation failures `0`, fallback rows `4`.
- `empirical_probe`: requested `0`, initial failures `0`, reconciliation failures `0`, fallback rows `4`.

Initial proposals are keyed by case, perturbation, split seed, LLM repetition, model, prompt schema, training-profile digest, target, and task. They are generated once and reused across compatible ablations; reconciliation outputs are never shared across prompt variants.

Split-seed variation is represented by `split_seed`; stochastic LLM variation is represented independently by `trial`/LLM repetition. Every paired comparison uses the same unit key.
