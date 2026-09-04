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
selectively catch harmful LLM model-family or preprocessing plans without
unnecessarily overriding good plans. The confirmatory scope is supervised
tabular classification/regression planning; it does not validate every
capability of the broader autonomous data-science product.

The paper-facing decision path is:

```text
independent LLM proposal
  -> hard validation
  -> deterministic structural challenge
  -> agreement/preserve OR disagreement
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
`llm_with_diagnostics` ablation tests whether that information asymmetry alone
explains any observed difference.

The confirmatory matrix has three same-model planner/reconciler conditions,
three LLM repetitions per condition (`rep_001`–`rep_003`), and split seed `42`:

| Condition | Snapshot |
|---|---|
| `gpt5_mini_2025_08_07` | `gpt-5-mini-2025-08-07` |
| `gpt54_mini_2026_03_17` | `gpt-5.4-mini-2026-03-17` |
| `gpt54_2026_03_05` | `gpt-5.4-2026-03-05` |

The six primary ablations are `llm_only`, `hard_validation_only`,
`deterministic_only`, `always_reconcile`, `probe_direct`, and `full`.
`llm_with_diagnostics` is secondary and is reported separately. Dataset/task
is the independent statistical unit; repetitions are nested observations and
are retained by the dataset-cluster bootstrap. The 40-task OpenML/AMLB panel
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
outcomes are not part of the normal publication-readiness workflow. The checked
in configuration snapshot is a draft template (`status: "draft"`); intentionally
set it to `"frozen"` only after reviewing the predeclared models, seeds, repetitions,
and other experiment settings. Then pass it explicitly with
`--confirmatory-config` to enable runtime validation and manifest hashing:
[`evaluation/configs/paper_confirmatory_v1.json`](evaluation/configs/paper_confirmatory_v1.json)
before launching a strict-live external run.

The confirmatory code identity is a canonical SHA-256 over sorted relative
paths and bytes in `app/`, `evaluation/` (excluding the confirmatory manifest),
and `pyproject.toml`. Git metadata, generated evaluation results, caches,
`.git`, Python bytecode, and temporary files are excluded. The manifest is
excluded because its expected hash would otherwise hash itself. A Git commit
may be recorded as `source_git_commit` for provenance, but it is not used for
confirmatory validity.

The freeze sequence is: finalize result-affecting code/configuration -> compute
the experiment code SHA-256 -> insert it as
`expected_experiment_code_sha256` -> set the manifest to `frozen` -> commit the
freeze -> validate. Committing the frozen manifest does not change the code
hash; changing any included result-affecting file does.

## License

See [`LICENSE`](LICENSE).
