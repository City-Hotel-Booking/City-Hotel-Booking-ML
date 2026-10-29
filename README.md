# City Hotel Booking ML

CSE382 university machine learning project using the City Hotel H2 dataset.

The project covers ADR regression, cancellation classification and booking
segmentation through clustering.

- Milestone 1: 5 November 2026.
- Milestone 2: 11 December 2026.

## Current status

Implemented:

- H2 loading with stable source row IDs and checksum tracking.
- Data validation, EDA and unusual-record review.
- A frozen grouped and stratified train/validation/test split.
- Task-specific starting feature lists and a preliminary leakage audit.
- Unfitted preprocessing for linear, distance-based and tree models.
- Shared task-data access with configuration propagation.
- Common experiment-result serialization.
- Data, preparation, preprocessing and result-writer tests.

The regression, classification and clustering baseline runners remain assigned
to their model owners. No baseline model results are included yet.
The integrated Milestone 1 report remains in progress.

## Team

| Member | Name | Responsibility |
| --- | --- | --- |
| 1 | Zeina | Data validation and EDA |
| 2 | Michael | Shared preparation and leakage policy |
| 3 | Paula | Regression |
| 4 | Youssef | Classification |
| 5 | Aya | Clustering |
| 6 | Makarious | Integration and report assembly |

See `CONTRIBUTING.md` for file ownership, deliverables and the review workflow.

## Local setup

Run commands from the repository root.

Create an isolated Python environment:

```powershell
python -m venv .venv
```

Install runtime and test dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Check the command interface:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Using the environment's executable directly avoids requiring activation.

The Member 6 environment uses Python 3.13.6. The dependency lists are currently
unpinned. Earlier source verification used the environment recorded in
`outputs/verification.json`; that record describes its original run.

## Dataset

Use City Hotel H2 only. The unchanged CSV is included at:

```text
data/H2.csv
```

Expected dimensions: 79,330 rows and 31 source columns.

Expected SHA-256:

```text
6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4
```

Verify the local file:

```powershell
Get-FileHash -LiteralPath ".\data\H2.csv" -Algorithm SHA256
```

Do not edit or resave the source CSV to perform cleaning.
The frozen split refers to this exact source file and row ordering.

The loader preserves original tokens separately from a normalized view.
It trims surrounding whitespace, parses numeric fields and retains stable
zero-based source row IDs. It does not remove or impute rows.

Agent and Company NULL values represent not applicable and remain categorical.
Their identifiers must not be treated as numeric measurements.
Unknown Children values must not be assumed to mean zero.

See `data/README.md` for source references and identity details.

## Project structure

```text
main.py                    Shared command interface
config.yaml                Starting experiment configuration
src/data/                  Loading, validation, EDA, splitting and task data
src/features/              Feature availability and leakage policy
src/preprocessing/         Unfitted transformers and preparation checks
src/evaluation/            Common result serialization
src/regression/            Regression experiments
src/classification/        Classification experiments
src/clustering/            Clustering experiments
src/tuning/                Milestone 2 tuning extension
src/inference/             Milestone 2 inference extension
data/H2.csv                Unchanged H2 source
outputs/data/              Validation, review and frozen split evidence
outputs/eda/               Descriptive tables and figures
outputs/regression/        Regression results
outputs/classification/    Classification results
outputs/clustering/        Clustering results
reports/                   Decisions, task notes and milestone report
tests/                     Behavioral and integration checks
scripts/                   Output verification utilities
artifacts/                 Ignored local fitted pipelines and models
app/                       Milestone 2 application extension
```

## Commands

The commands below are shown with the project environment's executable.

### Data validation

```powershell
.\.venv\Scripts\python.exe main.py validate
```

Writes validation evidence and the source schema under `outputs/data/`.

### Exploratory analysis

```powershell
.\.venv\Scripts\python.exe main.py analyze
```

Generates 15 EDA tables, seven figures, unusual-record evidence and reports.

### Shared preparation

```powershell
.\.venv\Scripts\python.exe main.py prepare
```

Routes through `src.data.prepare` to create or verify the shared split.
An incompatible existing split is rejected rather than silently replaced.

### Preparation checks

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.checks
```

Checks source identity, frozen assignments, duplicate-group containment,
feature exclusions, training-only fitting and transformed data.

### Independent output verification

```powershell
.\.venv\Scripts\python.exe scripts/verify_outputs.py
```

Reconciles saved data evidence against the CSV and runs the unittest-compatible checks. Run pytest separately for the complete test suite.

Validation, analysis and verification commands write their respective outputs.
Inspect output differences before committing regenerated evidence.

### Baseline experiments

The owners will implement these existing command routes:

```powershell
.\.venv\Scripts\python.exe main.py baseline --task regression
.\.venv\Scripts\python.exe main.py baseline --task classification
.\.venv\Scripts\python.exe main.py baseline --task clustering
```

Until implemented, each baseline command reports an unfinished-task message.

### Alternative configuration

Place `--config` before the command:

```powershell
.\.venv\Scripts\python.exe main.py --config experiment.yaml baseline --task regression
```

Paths inside configuration files are relative to the project root.

## Starting experiment policy

The starting prediction point is initial booking entry into the hotel PMS.
H2 is a later snapshot, so feature timing assumptions require explicit
discussion in experiment reports.

Starting input counts:

- Regression: 16.
- Classification: 16.
- Clustering: 11.

The source of the exact lists is `src/features/policy.py`.
Feature-policy version: `m1-draft-1`.

Shared split version: `m1-group-stratified-v1`, seed 42.

| Subset | Rows |
| --- | ---: |
| Train | 55,532 |
| Validation | 11,899 |
| Reserved test | 11,899 |

Exact duplicate groups remain together. The split is random and does not
measure future-year drift.

Start regression with all finite ADR targets, including canceled and zero-rate
records. Apply no automatic exclusions or clipping. An owner may define a
different scientifically justified population and document affected counts.

Start clustering with the existing 11 inputs and frozen training rows.
Do not use cancellation or final-status outcomes to select clusters.

Owners may make justified task-specific changes to algorithms, features,
population, preprocessing and prediction point. Record the change, reason,
evidence, exact inputs and impact in the task report and result settings.

Preserve source identity and frozen split assignments. Shared changes affecting
other tasks are reviewed through a pull request.

See `reports/decisions.md` for the full policy.

## Model-owner interface

Use the configuration and project root received by the task runner:

```python
from sklearn.pipeline import Pipeline

from src.data.prepare import get_task_data
from src.preprocessing.builders import build_preprocessor


def run(config, project_root):
    X_train, y_train = get_task_data(
        "classification", project_root, "train", config=config
    )
    X_validation, y_validation = get_task_data(
        "classification", project_root, "validation", config=config
    )

    # The owner supplies a selected, configured estimator.
    classifier = ...

    pipeline = Pipeline([
        (
            "preprocess",
            build_preprocessor(
                "classification", config, model_family="linear"
            ),
        ),
        ("model", classifier),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_validation)
```

This is an interface illustration; replace the estimator placeholder with the
owner's implementation.

`get_task_data` returns inputs and a supervised target. For clustering,
the target is `None`. Returned indexes preserve source row IDs.

Preprocessing choices:

- `model_family="linear"` or `"distance"`: numeric median imputation,
  the configured LeadTime log1p transform and standard scaling.
- `model_family="tree"`: numeric median imputation without scaling or log1p.
- Clustering: a separate distance preprocessor with dense output.
- Categorical inputs: missing marker and one-hot encoding with unknown
  categories ignored.
- Agent, Company and Country remain excluded from starting inputs.
  Rare-category encoding is available for justified future policies.

All builders return fresh unfitted objects. Fit supervised preprocessing inside
the model Pipeline and inside each training cross-validation fold.

## Results and deliverables

Use `save_result(result, path)` from `src/evaluation/results.py`.

Each owner supplies:

- A working task runner and relevant tests.
- `metrics.json` under their task's output folder.
- Actual figures or tables.
- A completed task report with interpretation and limitations.
- Exact settings, input population, features, command and runtime.
- An accurate contribution statement.

Use `None` with an explanation for unavailable metrics.
Convert NumPy values to ordinary Python values before JSON serialization.
NaN and infinity are rejected.

Detailed requirements are in `CONTRIBUTING.md`.

## Leakage and evaluation

- Reserve final supervised test evaluation for Milestone 2.
- Fit learned transformations and models on training folds only.
- Use group-aware folds if introducing cross-validation.
- Exclude future information at the stated prediction point.
- Apply task-specific exclusions within existing splits.
- Do not use supervised outcomes to select clustering parameters.
- Treat full-source EDA thresholds as descriptions, not fitted treatment rules.
- Explain comparisons involving different populations or prediction points.

See `reports/leakage_audit_m1.md` for the preliminary audit.

## Reports

- `reports/eda_notes.md`: source understanding and EDA.
- `reports/data_review.md`: unusual records and duplicate evidence.
- `reports/member1_integration.md`: source-format reconciliation.
- `reports/decisions.md`: starting assumptions and experiment rules.
- `reports/leakage_audit_m1.md`: preliminary leakage audit.
- `reports/handoff_member2.md`: preparation handoff.
- `reports/regression_m1.md`: Paula's regression report.
- `reports/classification_m1.md`: Youssef's classification report.
- `reports/clustering_m1.md`: Aya's clustering report.
- `reports/milestone1_report.md`: integrated progress report.

## Milestone 2

Extend the project with at least three algorithms per task, baseline and
tuned comparisons, an ensemble/boosting supervised model, PCA and LDA
experiments, final supervised test evaluation, clustering interpretation and
visualization, saved inference pipelines and a usable application.

The application must load saved pipelines for inference.

## Repository and submission

Repository:

https://github.com/City-Hotel-Booking/City-Hotel-Booking-ML

Submitted commit SHA: pending milestone submission.

Commit source, configuration, tests, the unchanged H2 CSV, relevant result
tables and figures, and reports.

Exclude environments, caches, secrets, processed datasets, fitted pipelines
and models, scratch outputs and the two local reference documents.

Record the tested environment, reproduction commands and submitted revision
when preparing the milestone package.