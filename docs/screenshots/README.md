# Verified Demo Gallery

These figures are genuine artifacts from a completed local run of the 569-row
[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)
benchmark with target `diagnosis`. The workflow profiled the data, generated
EDA, compared models with training-only cross-validation, evaluated exactly one
selected model on the held-out partition, and persisted a completed report.

The images are presentation artifacts, not a clinical validation. Exact model,
split, cross-validation, and holdout details are recorded in the accompanying
workflow artifacts and can be regenerated with:

- Train/holdout rows: 455 / 114
- Cross-validation: 5-fold stratified CV on the training partition
- Selected model: logistic regression
- CV macro F1: 0.9716 ± 0.0162
- Holdout macro F1: 0.9619
- Holdout accuracy: 0.9649

```bash
python scripts/run_full_demo.py --dataset classification
```

Files:

- `target-distribution.png`
- `correlation-heatmap.png`
- `model-comparison.png`
