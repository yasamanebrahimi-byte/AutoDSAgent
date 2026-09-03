# AutoDS Agent

AutoDS Agent is an auditable workflow for classification and regression on tabular CSV data. It combines optional OpenAI agent suggestions with deterministic validation, preprocessing, model training, and reporting.

The workflow:

1. Infers or validates the target and task.
2. Checks the data and freezes a train/holdout split.
3. Selects and validates preprocessing and a model family.
4. Trains on the training partition and evaluates once on the untouched holdout.
5. Saves the decision, model, metrics, plots, report, and a replay script.

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
The optional frozen AMLB/OpenML external evaluation suite is documented in [external benchmark](docs/external_benchmark.md).

## License

See [`LICENSE`](LICENSE).
