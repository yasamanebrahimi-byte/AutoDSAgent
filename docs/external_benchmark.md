# External AMLB/OpenML benchmark

External Benchmark v1 is an independent, frozen evaluation suite for testing
the AutoDS validation architecture on public tabular tasks. It is benchmark
infrastructure only: the external results must not be used for policy
calibration, threshold changes, prompt changes, or model-family changes.

The suite version is `1.0.0` and contains exactly 40 immutable OpenML task IDs:
22 classification tasks from AMLB classification suite `271` and 18 regression
tasks from AMLB regression suite `269`. OpenML task ID, rather than dataset
name, is the canonical source identifier. The suite should remain frozen before
the first live research evaluation.

The original AMLB citation is:

> Pieter Gijsbers et al., “AMLB: an AutoML Benchmark,” *Journal of Machine
> Learning Research* 25 (2024).

## Frozen manifest

### Classification (AMLB suite 271)

| OpenML task ID | Name | Rows | Features | Classes | Tier |
|---:|---|---:|---:|---:|---|
| 359983 | adult | 48842 | 15 | 2 | core |
| 359979 | Amazon_employee_access | 32769 | 10 | 2 | core |
| 168868 | APSFailure | 76000 | 171 | 2 | stress |
| 146818 | Australian | 690 | 15 | 2 | core |
| 359982 | bank-marketing | 45211 | 17 | 2 | core |
| 359967 | Bioresponse | 3751 | 1777 | 2 | stress |
| 359955 | blood-transfusion-service-center | 748 | 5 | 2 | core |
| 359960 | car | 1728 | 7 | 4 | stress |
| 359968 | churn | 5000 | 21 | 2 | core |
| 359992 | Click_prediction_small | 39948 | 12 | 2 | core |
| 168757 | credit-g | 1000 | 21 | 2 | core |
| 359964 | dna | 3186 | 181 | 3 | core |
| 359954 | eucalyptus | 736 | 20 | 5 | core |
| 359970 | GesturePhaseSegmentationProcessed | 9873 | 33 | 5 | core |
| 359966 | Internet-Advertisements | 3279 | 1559 | 2 | stress |
| 359962 | kc1 | 2109 | 22 | 2 | core |
| 190137 | ozone-level-8hr | 2534 | 73 | 2 | core |
| 359971 | PhishingWebsites | 11055 | 31 | 2 | core |
| 168350 | phoneme | 5404 | 6 | 2 | core |
| 359956 | qsar-biodeg | 1055 | 42 | 2 | core |
| 168784 | steel-plates-fault | 1941 | 28 | 7 | core |
| 359974 | wine-quality-white | 4898 | 12 | 7 | stress |

### Regression (AMLB suite 269)

| OpenML task ID | Name | Rows | Features | Tier |
|---:|---|---:|---:|---|
| 359944 | abalone | 4177 | 9 | core |
| 359938 | Brazilian_houses | 10692 | 13 | core |
| 359942 | colleges | 7063 | 45 | core |
| 233211 | diamonds | 53940 | 10 | stress |
| 359936 | elevators | 16599 | 19 | core |
| 359952 | house_16H | 22784 | 17 | core |
| 359951 | house_prices_nominal | 1460 | 80 | stress |
| 359949 | house_sales | 21613 | 22 | core |
| 233215 | Mercedes_Benz_Greener_Manufacturing | 4209 | 377 | stress |
| 360945 | MIP-2016-regression | 1090 | 145 | core |
| 167210 | Moneyball | 1232 | 15 | core |
| 359941 | OnlineNewsPopularity | 39644 | 60 | stress |
| 359930 | quake | 2178 | 4 | core |
| 359931 | sensory | 576 | 12 | core |
| 359932 | socmob | 1156 | 6 | core |
| 359933 | space_ga | 3107 | 7 | core |
| 359934 | tecator | 240 | 125 | stress |
| 359935 | wine_quality | 6497 | 12 | core |

## Loading and reproducibility

OpenML is an optional benchmark dependency. Data are downloaded on demand by
the official `openml` Python package and cached by OpenML; downloaded data are
not committed to the repository. Set `AUTODS_OPENML_CACHE` to choose a stable
local cache directory, for example `C:\\data\\autods-openml-cache` on Windows.

The loader preserves raw feature values and dtypes. It does not impute, encode,
scale, remove outliers, select features, subsample rows, or drop columns/rows.
The original OpenML target is exposed to AutoDS as the collision-resistant
`__target__` column after a shape and target validation. A mismatch fails
loudly; no task is substituted.

External Benchmark v1 deliberately uses AutoDS’s existing deterministic
`freeze_supervised_split` train/holdout policy and split seeds. It does not use
AMLB’s original predefined cross-validation folds, so these results must not be
presented as directly comparable to AMLB leaderboard numbers. The OpenML task
IDs are retained so standardized folds can be added in a later, explicitly
versioned suite.

## Commands

Install the optional dependency:

```bash
python -m pip install -e ".[benchmark]"
```

Prefetch and validate every task, producing the non-performance reproducibility
manifest at `evaluation_results/external_dataset_manifest.json`:

```bash
python scripts/prefetch_external_benchmarks.py
```

Run one external case offline (offline refers to the LLM/evaluation mode; the
dataset still needs to be cached or downloaded):

```bash
python -m evaluation.run --suite external --case adult --offline --output evaluation_results/external_adult_smoke
```

Run the complete external suite offline:

```bash
python -m evaluation.run --suite external --offline --output evaluation_results/external_offline
```

Run the primary paired ablation set offline:

```bash
python -m evaluation.ablation --suite external --offline --output evaluation_results/external_ablation_offline --ablation llm_only --ablation blinded_always_reconcile --ablation selective_calibrated --ablation full
```

Run only one tier:

```bash
python -m evaluation.run --suite external --tier stress --offline --output evaluation_results/external_stress_offline
```

After the suite is frozen and live research approval is in place, run the full
suite live with strict failure reporting:

```bash
python -m evaluation.ablation --suite external --live --require-live --output evaluation_results/external_ablation_live --ablation llm_only --ablation blinded_always_reconcile --ablation selective_calibrated --ablation full
```

The historical `legacy_gate` preset remains available as a secondary ablation
by passing `--ablation legacy_gate` explicitly.

The default `python -m evaluation.run` command remains the local suite.
