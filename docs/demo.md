# Demo walkthrough

The bundled demos exercise the current workflow on the Breast Cancer Wisconsin
classification dataset and the diabetes progression regression dataset.

Run both demos offline with:

```bash
python scripts/run_demo.py --offline
```

Each run performs formulation, freezes the supervised split, creates an
independent modeling proposal, computes the deterministic recommendation,
applies hard feasibility checks, selectively probes and reconciles valid
disagreements when warranted, then performs cleaning, EDA, model fitting, and
reporting. Run artifacts are written under `runs/<run_id>/`, which is ignored
by Git.

The bundled CSV files are software fixtures, not a source of benchmark
performance claims. Inspect the generated artifacts for the details of an
individual run.
