# Demo Outputs

This folder documents the artifacts produced by the current offline demos. It
does not store generated run folders; those are created under
`runs/<run_id>/` and ignored by Git except for `.gitkeep`.

Run both bundled datasets with:

```bash
python scripts/run_demo.py --offline
```

Each run writes a profile, the validation decision, cleaning and EDA evidence,
modeling metrics, plots, a fitted model, a Markdown report, and a reproduction
script. See the [classification demo summary](classification_demo_summary.md)
and [regression demo summary](regression_demo_summary.md) for the expected
artifact layout.
