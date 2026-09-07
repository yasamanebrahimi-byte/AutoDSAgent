# AutoDS Agent

AutoDS Agent is an auditable workflow for classification and regression on tabular CSV data. It combines optional OpenAI agent suggestions with deterministic validation, preprocessing, model training, and reporting.

The repository also contains a narrower research evaluation. Its paper-facing
claim concerns LLM-based model-family/preprocessing planning for supervised
tabular classification and regression under a selective deterministic
safeguard. It is not a claim that the confirmatory experiment validates every
capability of the broader autonomous data-science product.

The product workflow:

1. Infers or validates the target and task.
2. Checks the data and freezes a train/holdout split.
3. Selects and validates preprocessing and a model family.
4. Trains on the training partition and evaluates once on the untouched holdout.
5. Saves the decision, model, metrics, plots, report, and a replay script.

## Paper-facing evaluation

The narrower research question is whether a deterministic/non-LLM safeguard can
selectively catch harmful LLM model-family planning without unnecessarily
overriding good plans. Typed preprocessing plans/contracts are carried with
candidate plans and remain subject to universal hard validity checks, but a
preprocessing-only difference is currently diagnostic rather than an
independent soft-intervention trigger. The confirmatory scope is supervised
tabular classification/regression planning; it does not validate every
capability of the broader autonomous data-science product.

The paper-facing decision path is:

```text
independent LLM proposal
  -> hard validation
  -> deterministic structural challenge
  -> model-family agreement/preserve OR actionable model-family disagreement
  -> bounded training-only empirical arbitration
  -> abstain or selectively intervene/reconcile
  -> freeze final plan
  -> evaluate the untouched holdout
```

“Deterministic” means specified, reproducible non-LLM code conditional on the
data split and configuration. The challenger is restricted to the training
partition and may include bounded fitted diagnostic models; it is not a claim
that no fitting occurs before the final plan is selected. The LLM and
challenger use the same training partition but not necessarily the same
representation: the LLM receives a compact profile while the challenger uses
richer pre-specified structural diagnostics. The secondary
`llm_with_diagnostics` control asks whether giving the LLM those richer,
pre-specified training-only diagnostics improves its initial plan. Its outcomes
are initial untouched-holdout planner quality and paired initial-plan validity,
not intervention delta; results remain separate by model condition and
secondary to the selective-intervention estimand.

The confirmatory matrix is declared in the manifest and currently contains only
GPT-5.6 Luna, Sol, and Terra. Each condition has three LLM repetitions
(`rep_001`–`rep_003`) and split seed `42`:

| Condition | Snapshot |
|---|---|
| `gpt56_luna` | `gpt-5.6-luna` |
| `gpt56_sol` | `gpt-5.6-sol` |
| `gpt56_terra` | `gpt-5.6-terra` |

The six primary ablations are `llm_only` (LLM + universal validity checks; no
deterministic soft challenger), `hard_validation_only`,
`deterministic_only`, `always_reconcile`, `probe_direct`, and `full`.
`llm_with_diagnostics` is secondary and is reported separately. Its initial
planner-quality analysis aligns repetitions by declared slot for balanced
analysis, but the slots are not shared-seed stochastic matches across the
separate planner calls. Classification and regression magnitudes are reported
separately, conditional on jointly valid and evaluable initial plans; dataset/task
remains the independent statistical unit and uncertainty uses a dataset-cluster
bootstrap. The 40-task OpenML/AMLB panel
is a frozen, predeclared evaluation panel, not a statistically random sample
of AMLB tasks.

## Requirements

- Python 3.11+
- An OpenAI API key for API-backed runs; offline runs need no key

## Install

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run

Run the bundled offline demo:

```bash
python scripts/run_demo.py --offline
```

Run an analysis on your own CSV. `--target` is optional; if omitted, the workflow attempts to infer it from the question and schema.

```bash
python -m app \
  --data path/to/data.csv \
  --target target_column \
  --question "What should we predict from these features?" \
  --output-dir runs
```

For an API-backed run, set the key first:

```bash
# macOS/Linux
export OPENAI_API_KEY="your-key"

# Windows PowerShell
$env:OPENAI_API_KEY = "your-key"
```

Omit `--offline` to use the API. Set `OPENAI_MODEL` or pass `--model` to choose the model. Never commit API keys.

Each run creates a folder under `runs/` containing the final report and reproducible artifacts. The most useful files are:

- `report.md` — results and interpretation
- `decision.json` — agent proposals, deterministic checks, and the approved plan
- `modeling.json` — preprocessing, cross-validation, holdout metrics, and model details
- `model/selected_model.joblib` — fitted scikit-learn pipeline
- `reproduce_analysis.py` — replay of the approved analysis

## Development

Run the test suite:

```bash
pytest
```

More detailed design and evaluation notes are in [`docs/`](docs/), including [reconciliation](docs/reconciliation.md) and the [evaluation objective](docs/gate_evaluation_objective.md).
The optional frozen AMLB/OpenML external evaluation suite is documented in [external benchmark](docs/external_benchmark.md), including its [selection and freeze record](docs/external_benchmark_selection.md).

Development and confirmatory evaluation are separate. Live API smoke tests
must use local or synthetic development cases. The external suite may be
prefetched and schema-validated before confirmation, but live external pilot
outcomes are not part of the normal publication-readiness workflow.

`paper_confirmatory_v1.json` is retained as the historical pilot /
pre-contract-fix protocol. Its artifacts remain interpretable with their
recorded source provenance and are excluded from v2 confirmatory claims.
`paper_confirmatory_v2.json` is the active definitive contract-aware protocol;
the checked-in manifest is frozen with its current canonical experiment-code
SHA and source implementation commit recorded. Historical or development
manifests may remain draft/unfrozen, but that generic workflow state does not
mean that the active v2 confirmatory manifest is unfrozen. The v2 planner
receives the complete executable modeling/preprocessing contract and
training-only feasibility, but not deterministic recommendations, scores,
holdout results, or empirical-probe evidence. Hard validation remains an
independent fail-closed check.

For the Linux VM, use the checked-in runner after the v2 freeze:

```bash
export OPENAI_API_KEY="your-key"
scripts/run_paper_confirmatory_v2.sh
```

The runner requires a clean tree, a frozen v2 manifest and matching experiment
code SHA, uses Python 3.12, prefetches and verifies the 40-task benchmark, and
fails closed on fallback rows or an incomplete 2520-unit matrix. Strict resume
uses the exact frozen manifest SHA, code SHA, prompt schema, contract digest,
benchmark identity, and persisted run configuration; mismatches are rejected
before artifacts are modified. The root run configuration records
Python/platform, package versions, Git commit, experiment-code SHA, manifest
SHA, and benchmark-manifest SHA.

Confirmatory paper-primary estimates are reported separately for each declared
model condition (currently Luna, Sol, and Terra). Repetitions remain nested
within dataset/task, and dataset/task is the independent statistical unit.
Any cross-model aggregate or paired comparison is explicitly descriptive and
audit-only. `llm_with_diagnostics` remains a secondary information-asymmetry
control and cannot enter the primary summaries or paired comparisons. The
control compares initial plans and reports paired initial-plan validity; the
untouched holdout is evaluation-only and results remain separate for Luna, Sol,
and Terra.

Strict confirmatory resume is bound to the exact frozen manifest SHA and its
expected experiment-code SHA. A mismatched resume fails closed before trial
execution or replacement of the frozen manifest artifact. Proposal-cache
identity includes the declared provider. A failed attempt may be superseded
by a completed retry, while conflicting duplicate completed trial IDs fail
closed.

The confirmatory code identity is a canonical SHA-256 over sorted relative
paths and bytes in `app/`, `evaluation/` (excluding the confirmatory manifest),
and `pyproject.toml`. Git metadata, generated evaluation results, caches,
`.git`, Python bytecode, and temporary files are excluded. The manifest is
excluded because its expected hash would otherwise hash itself. A Git commit
may be recorded as `source_git_commit` for provenance, but it is not used for
confirmatory validity.

The freeze sequence is: finalize result-affecting code/configuration -> compute
the experiment code SHA-256 -> insert it as
`expected_experiment_code_sha256` -> set the v2 manifest to `frozen` -> commit
the freeze -> validate. Committing the frozen manifest does not change the code
hash; changing any included result-affecting file does. Reporting distinguishes
hard interception (invalid initial plan prevented from training), hard repair
(valid deterministic replacement), and soft intervention (a valid plan changed
after a valid family disagreement). The scope remains selective intervention
for LLM-based supervised tabular ML planning within this fixed supported
model/preprocessing search space, not a universal AutoML oracle.

## License

See [`LICENSE`](LICENSE).
