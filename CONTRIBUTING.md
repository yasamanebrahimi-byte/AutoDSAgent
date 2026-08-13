# Contributing

AutoDS Agent is a portfolio project, so contributions should preserve clarity, reproducibility, and a small dependency footprint.

## Local Setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before Opening A Pull Request

Run:

```bash
python scripts/smoke_test.py
python scripts/validate_project.py
pytest
```

## Contribution Guidelines

- Keep raw uploaded data preservation intact.
- Do not add paid API requirements.
- Do not commit secrets, `.env`, or large generated run artifacts.
- Prefer deterministic services for computation.
- Add tests when changing workflow, service, model, report, or artifact behavior.
- Keep documentation honest about current capabilities and limitations.
