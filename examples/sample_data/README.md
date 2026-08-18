# Public Benchmark Datasets

The product demo uses two established tabular benchmarks distributed with
scikit-learn. They are large enough for meaningful train/holdout evaluation,
remain small enough for a local walkthrough, and can be regenerated without a
network request:

```bash
python scripts/export_benchmark_datasets.py
```

`benchmark_manifest.json` records the exact row count, target, source, loader,
license/attribution, citation, and SHA-256 digest for each generated CSV.

## Classification: Breast Cancer Wisconsin (Diagnostic)

- File: `breast_cancer_wisconsin.csv`
- Shape: 569 rows, 30 features, and one target
- Target: `diagnosis` (`malignant` or `benign`)
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)
- DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Loader: `sklearn.datasets.load_breast_cancer`

Attribution: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
*Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository.

## Regression: Diabetes Disease Progression

- File: `diabetes_progression.csv`
- Shape: 442 rows, 10 features, and one target
- Target: `disease_progression`
- Source and field definitions: [scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- Loader: `sklearn.datasets.load_diabetes(as_frame=True, scaled=False)`

Reference: Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004).
“Least Angle Regression.” *The Annals of Statistics*, 32(2), 407–499.

These health-related datasets are included only to demonstrate the software
pipeline. AutoDS Agent is not a medical device, and the outputs must not be used
for diagnosis, treatment, or other clinical decisions.

## Fast Test Fixtures

The former 25-row synthetic churn and housing CSVs now live under
`tests/fixtures/sample_data/`. They intentionally remain tiny and are used only
where the test suite needs a fast smoke fixture; they are not product benchmarks
or evidence of model quality.
