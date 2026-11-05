# Milestone 1 clustering guide

This contribution runs one provisional untuned K-Means baseline. Read
`reports/clustering_m1.md` for feature decisions, actual results and limitations.

## Setup

Use the shared Python 3.13.6 environment for all three tasks. Create an
isolated environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Place the unchanged course H2 at `data/H2.csv`. Its SHA-256 must match the
frozen split metadata: `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.
Keep the existing source ordering, manifest, config and feature policy.

## Run and check

```powershell
.\.venv\Scripts\python.exe -m src.clustering.check_preprocessing
.\.venv\Scripts\python.exe main.py baseline --task clustering
.\.venv\Scripts\python.exe -m pytest tests/test_clustering.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Alternative configuration:
`python main.py --config experiment.yaml baseline --task clustering`.
The result records that CLI argument and effective configuration. Regeneration
replaces only clustering outputs and its ignored local artifacts.

The training representation uses the shared numeric imputation/log/scaling
and categorical encoding without an additional cap. K-Means fits all training
rows, k=3, n_init=10, seed 42. Internal metrics use prepared features before
PCA; silhouette uses fixed sampling. PCA is only a training-fitted 2D view.

Outputs are under `outputs/clustering/`: metrics, original-unit profiles,
category frequencies, silhouette row IDs, preprocessing checks and two PNGs.
Fitted objects are local under ignored `artifacts/clustering/`; exclude them,
raw data, environments and scratch files from submissions.

## Explain the result

K-Means alternates nearest-center assignments and center updates. k is the
number of groups; n_init is the number of initializations tried. IDs are
arbitrary, not target predictions. Higher silhouette and lower Davies-Bouldin
generally indicate clearer geometry, but do not validate a business story.
The small repeated-guest cluster shows sensitivity to feature weighting.
PCA omits most variance, and k=3 is not optimized. Snapshot timing remains
provisional. Reserved rows are not used for fitting, selection or evaluation;
the loader first reads the source CSV to check identity and select training.
