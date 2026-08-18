# Demo Outputs

This folder documents what successful demo runs produce. It intentionally does not store large generated run artifacts. Run artifacts are created under `runs/<run_id>/` and are ignored by Git except for `.gitkeep`.

Run a demo with:

```bash
python scripts/run_full_demo.py --dataset classification
python scripts/run_full_demo.py --dataset regression
```

Each command prints:

- Run ID.
- Dataset.
- Target column.
- Workflow status.
- Selected model, best candidate, and primary metric, when modeling succeeds.
- Final report path.
- Key artifact paths.

See:

- [Classification Demo Summary](classification_demo_summary.md)
- [Regression Demo Summary](regression_demo_summary.md)
