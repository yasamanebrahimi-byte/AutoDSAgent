# Deterministic policy v4: development justification

This is a development-only justification artifact for the frozen deterministic policy version `4`. It is included to make the policy reviewable before the confirmatory run. It is not a production tuning record, and it does not authorize changing the runtime policy from version `4`.

Machine-readable companion: [`evaluation/configs/deterministic_policy_v4_development.json`](../evaluation/configs/deterministic_policy_v4_development.json).

## What the policy is intended to do

The policy maps observable dataset structure to bounded, auditable compatibility points for four model families: `linear`, `regularized_linear`, `tree_ensemble`, and `boosted_tree`. It is a transparent model-family recommender, not a predictor of the empirical winner and not a probability model.

The policy is intended to respond to these predeclared failure modes:

- underpowered linear representations when nonlinear signal, pairwise structure, or a complex classification boundary is visible;
- unstable or over-flexible methods when sample-to-feature ratio, effective dimensionality, missingness, outliers, class imbalance, or class support make the problem fragile;
- poor handling of mixed or categorical structure and high-cardinality representations;
- overly confident recommendations when structural evidence is weak, conflicting, or resource-constrained;
- interaction evidence being inferred from marginal associations alone.

The policy therefore combines coarse structural observations with explicit score contributions. Each contribution records the observed factor, direction, and bounded point value. A score is a ranking aid, not a calibrated probability.

## What is fixed in v4

The full parameter object and compatibility-point table are captured by the production hash `9a450828f27dcce86b005f0c21d0271ca08a0b19e6dd8a802e047ec39effcc90`. The important threshold bands are:

| Signal | Low/moderate/high or key boundary |
|---|---|
| sample-to-feature ratio | `3 / 8 / 20` |
| effective features | `20 / 100 / 300` |
| missingness fraction | `0.10 / 0.25`; widespread missing-feature fraction `0.40` |
| regression nonlinearity | `0.15 / 0.35` |
| classification boundary complexity | `0.10 / 0.20` |
| interaction strength | `0.18 / 0.38`; strong pair `0.30`; report pair `0.10` |
| association evidence | regression `0.20`; classification `0.20` |
| outlier and class-imbalance signals | outlier fraction `0.05`; minority fraction `0.10`; imbalance ratio `9.0` |
| target shape | skewness `2.0`; high target-outlier fraction `0.10` |
| categorical structure | elevated cardinality `40`; high-cardinality fraction `0.50` |

The structural-complexity weights are explicit: mixed types `0.15`, categorical structure `0.15`, nonlinear fraction `0.20`, nonlinearity strength `0.25`, heterogeneity `0.15`, weak marginal evidence `0.10`, boundary evidence `0.35`, and direct interaction evidence `0.20`. These are additive heuristic weights; they are not claimed to be independent probabilities.

Engineering limits are also part of the policy contract: boundary diagnostics use at most `16` numeric features and `5,000` rows, interaction diagnostics at most `12` features, `48` pairs, and `5,000` rows, and report at most `5` interaction pairs. Classification boundary probes use three folds, random state `1729`, and neighbor `k=7`.

## Why these choices are defensible

The choices separate three kinds of justification:

1. Domain reasoning: linear and regularized families are sensible when signal is simple, stable, and well-supported; tree families are plausible when nonlinear, interaction, mixed-type, or boundary structure is visible; stronger flexibility should be penalized under sparse or fragile data.
2. Engineering constraints: diagnostics are bounded, deterministic, and auditable; compatibility points are named rather than learned; the runtime does not fit an empirical winner selector.
3. Development evidence: the policy is checked on a small, explicit development panel and nearby threshold perturbations. This evidence can reveal brittle bands and failure cases, but it cannot establish global optimality.

No claim is made that the point table or thresholds are globally optimal. The confirmatory benchmark remains a held-out evaluation of the frozen policy and is not a source of policy tuning.

## Information asymmetry and leakage boundary

The main runtime recommendation uses deterministic structural diagnostics. Some diagnostics are simple counts, ratios, missingness, categorical summaries, correlations, and bounded structural probes. A bounded training-only fitted diagnostic may be used where the policy explicitly needs it, such as a classification boundary probe or regression interaction probe. “Deterministic” means fixed code, fixed seed, fixed limits, and no learned empirical-winner selection; it does not mean “no fitted statistic ever appears.”

The initial LLM sees the canonical training profile. The `llm_with_diagnostics` secondary ablation additionally sees a compact canonical digest of deterministic structural diagnostics computed from the training partition only. It never receives holdout rows, holdout labels, holdout metrics, or empirical-reference outcomes. The normal `llm_only` condition does not receive this extra digest. The empirical pairwise probe and final model/holdout results are separate evidence layers and are not folded into the initial LLM prompt.

The model gate may inspect training labels to establish the declared split and to validate the training-only plan. It does not use holdout labels for planning, validation, prompting, reconciliation, or intervention decisions. Holdout outcomes are computed only after the recommendation path is complete.

## Development panel and sensitivity check

The policy-development panel contains 12 local/synthetic cases: `breast_cancer`, `diabetes`, `synthetic_regression`, `synthetic_linear_regression`, `synthetic_nonlinear_regression`, `synthetic_high_dim_regression`, `synthetic_binary_linear`, `synthetic_binary_nonlinear`, `synthetic_imbalanced_classification`, `synthetic_multiclass`, `synthetic_missingness`, and `synthetic_outlier_regression`. It uses split seeds `42`, `123`, and `2027`, producing `288` development records across the declared candidate set.

The sensitivity grid multiplies continuous structural thresholds by approximately `-20%`, `-10%`, `+10%`, and `+20%` in the sense of `0.80`, `0.90`, `1.10`, and `1.20` multipliers. It excludes categorical logic, compatibility points, integer resource limits, and bounded correlation thresholds. The production object is not overwritten or selected by this run.

| Candidate | Mean normalized regret | Harmful intervention rate | Catastrophic introduced | Utility |
|---|---:|---:|---:|---:|
| `current` | `0.2925` | `0.3000` | `4` | `-22.75` |
| `global_thresholds_minus_10pct` | `0.3816` | `0.3333` | `4` | `-15.00` |
| `global_thresholds_plus_10pct` | `0.0410` | `0.1667` | `1` | `-10.50` |
| `global_thresholds_minus_20pct` | `0.3816` | `0.3333` | `4` | `-15.00` |
| `global_thresholds_plus_20pct` | `0.0493` | `0.1667` | `1` | `-14.75` |

The local panel’s diagnostic ranking favored `global_thresholds_plus_10pct`, with an observed mean-regret improvement of `0.2514` versus `current`. That result is recorded as sensitivity evidence only. It does not promote the candidate, rewrite the manifest, or change the production policy hash. The broader named candidates and their full metrics are in the JSON companion.

The metric set intentionally includes harmful intervention rate, catastrophic-regret introductions, intervention precision, challenge recall, transparent utility, regret, and exact reference match as a secondary diagnostic. A candidate cannot be justified by exact family match alone, and the local panel is too small to support a general performance claim.

## Why the LLM and challenger can still be wrong

The LLM can misread a compact structural profile, over-weight a salient but weak signal, propose an invalid preprocessing plan, or recommend a flexible family that is poorly supported by sample size. Its reasoning is therefore constrained by a schema, deterministic hard validation, provenance checks, and the frozen model/generation settings.

The empirical challenger can also be wrong: its training-only cross-validation estimate has finite sample noise, its candidate family set is limited, and its preprocessing/model search can mismatch the final pipeline. Disagreement is evidence for review, not automatic proof that either side is correct. Soft challenge calibration and the abstention/reconciliation rules preserve ties and weak evidence rather than converting them into forced interventions.

## Review conclusion

Policy v4 is justified as an interpretable, bounded, predeclared heuristic with explicit domain rationale, explicit engineering limits, and development-only stress tests. The artifact documents where the heuristic may fail and how those failures are measured. It does not claim that v4 is globally optimal, and no confirmatory or external holdout outcome is used to tune it.
