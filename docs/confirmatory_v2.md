# Confirmatory v2 protocol

`paper_confirmatory_v1.json` is the historical pilot / pre-contract-fix
protocol. Its generated results are not part of confirmatory v2 claims.
`paper_confirmatory_v2.json` is the definitive protocol for selective
intervention for LLM-based supervised tabular ML planning within the fixed
supported model/preprocessing search space.

Before the v2 freeze, the modeling planner receives a structured execution
contract derived from the frozen training partition. It describes the four
supported families, executable preprocessing fields and compatibility pairs,
missing/infinity handling, mandatory feature exclusions, training-only fitting,
and dataset-specific one-hot feasibility. The deterministic recommender's
family, scores, rankings, holdout results, and empirical-probe results are not
included in that initial planner input. The independent hard validator remains
authoritative and fail-closed.

Reporting uses three distinct terms:

- hard interception: an invalid initial LLM plan is detected and prevented from
  training;
- hard repair: an invalid initial plan is replaced by a valid deterministic
  alternative;
- soft intervention: a valid LLM plan changes after a valid model-family
  disagreement. Preprocessing-only disagreement remains diagnostic under the
  frozen policy.

The Linux VM runner is:

```bash
export OPENAI_API_KEY="your-key"
scripts/run_paper_confirmatory_v2.sh
```

It creates a Python 3.12 environment, installs development and benchmark
dependencies, runs preflight tests, validates the frozen manifest and code
SHA, prefetches the 40 external tasks, runs all 7 ablations for all 3 model
conditions and 3 repetitions, and verifies a complete 2520-unit matrix with
zero fallback rows. To resume an existing bundle, set `RESUME=1` and the same
`OUTPUT_DIR`; exact manifest, code, prompt, benchmark, contract, and run
configuration identity is checked before trial work or artifact replacement.

The root run configuration records Python and platform provenance, package
versions, Git commit, experiment-code SHA, confirmatory-manifest SHA, and
benchmark-manifest SHA. API token and runtime counters are passive per-trial
instrumentation; unavailable token fields are `null`, and cached proposals do
not add API calls a second time.
