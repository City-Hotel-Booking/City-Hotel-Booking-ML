# City Hotel Booking Demand

CSE382 university machine learning project for a team of six. Use H2.csv (City Hotel) only.

## Current status

This repository contains the agreed folder layout, command routing, a provisional feature policy, a working supervised split, an unfitted shared preprocessing builder, and report templates. Member 1 validation and EDA are implemented, including source checks, plots and unusual-record review. Model and result-writing functions still need their owners. There are no model experiment results yet.

Milestone 1 is due **5 November 2026**. Milestone 2 is due **11 December 2026**, according to the supplied project Description.

## Local environment

The Member 1 integration is checked with Python 3.12.14. The shared dependency list remains unpinned; outputs/verification.json records the installed versions used for its verification. Model owners must record their own tested environment when experiments are added.

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --help
```

For development tests, install `requirements-dev.txt` instead. Editors can differ; everyone uses the same source code and configuration. Use the environment's Python directly if PowerShell activation is unavailable.

## Data

Place your own copy of H2 at data/H2.csv; see data/README.md. The raw CSV and the two root reference documents are excluded from Git. The local Word coordination plan is shared separately; the team ownership summary is also in CONTRIBUTING.md.

## Commands and ownership

The following command routes exist. Help/configuration/routing, validation, analysis, a provisional feature policy, and prepare are implemented. Baseline commands still exit with an unfinished-task message until their owners complete them.

| Command | Owner | Expected output after implementation |
| --- | --- | --- |
| `python main.py validate` | Member 1 | `outputs/data/validation.json`, `schema.csv` |
| `python main.py analyze` | Member 1 | `outputs/eda/`, `reports/eda_notes.md` |
| `python main.py prepare` | Member 2 | `outputs/data/split_manifest.csv`, `split_metadata.json` |
| `python main.py baseline --task regression` | Member 3 | `outputs/regression/`, `reports/regression_m1.md` |
| `python main.py baseline --task classification` | Member 4 | `outputs/classification/`, `reports/classification_m1.md` |
| `python main.py baseline --task clustering` | Member 5 | `outputs/clustering/`, `reports/clustering_m1.md` |

Use `python main.py --config config.yaml <command>` to select a configuration explicitly. Paths in the configuration are relative to the project root, not to an individual laptop.

## Member 1 data and EDA handoff

Both `validate` and `analyze` expose `run(config, project_root)` and use configured repository-relative paths. The shared `load_h2(path)` now returns a normalized DataFrame with an explicit zero-based `row_id` and all 31 source columns. Its checksum is in `frame.attrs["data_checksum_sha256"]`. Validation/EDA use `read_dataset(path)` internally when original string tokens are also needed. No row is removed or imputed by the loader.

`python main.py analyze` generates 15 tables in `outputs/eda/tables/`, seven plots in `outputs/eda/figures/`, validation/schema/review evidence in `outputs/data/`, and two reports:

- [EDA notes](reports/eda_notes.md)
- [Unusual records and duplicate review](reports/data_review.md)
- [Integration and source-identity notes](reports/member1_integration.md)
- [Manual checklist and observations](reports/zeina_checklist.md)

Run `python scripts/verify_outputs.py` to independently reconcile generated evidence and run the combined test suite. The script uses the shared configuration by default. Analysis regenerates its reports but preserves the manual checklist and Michael's existing split files. Full-source descriptions are not test-performance evaluation and must not become all-data fitted model-selection rules.

The source file should match the split metadata checksum. The supplied spreadsheet export has equivalent row values but different byte formatting; this integration uses the course mirror whose exact checksum matches the existing split. See the integration notes. Agent/Company NULL is preserved as not applicable in EDA. These fields remain excluded by the provisional modeling policy; any future activation needs agreement with Member 2 on categorical handling.

## Member 2 preprocessing handoff

`get_features(task, config)` returns the provisional input columns and
`build_preprocessor(task, config)` returns a fresh **unfitted** transformer.
Members 3 and 4 should select their own train/validation row IDs from
`outputs/data/split_manifest.csv`, then put the transformer inside their
model Pipeline before fitting. For example:

```python
from sklearn.pipeline import Pipeline
from src.features.policy import get_features
from src.preprocessing.builders import build_preprocessor

features = get_features("classification", config)
model = Pipeline([
    ("preprocess", build_preprocessor("classification", config)),
    ("estimator", classifier),
])
model.fit(train_rows[features], train_rows["IsCanceled"])
predictions = model.predict(validation_rows[features])
```

Create a separate unfitted transformer for clustering. The current feature
lists exclude disputed inputs; agree on any policy change and bump its version
before adding them. `NULL`, blank, and literal `NA` are missing values inside
the transformer; `NA` appears in four source `Children` entries. Numeric
statistics and one-hot categories are learned only during `fit`.

## Milestone 1 sequence

1. Member 1 checks H2 and provides evidence for cleaning decisions.
2. Member 2 coordinates the prediction point, task inputs, cleaning rules, and split with Members 1 and 3-5, then implements shared preparation.
3. Member 6 supplies the shared result writer before Members 3-5 finish their outputs.
4. Members 3-5 run one untuned experiment each using the common preparation.
5. Member 6 connects the commands, reruns the workflow, and assembles the report with checked contributions from every member.

Provisional booking-time and split settings are recorded in config.yaml and reports/decisions.md. The team has not yet approved the prediction point, ADR population, row-deletion rules, baseline models, or final feature lists.

## Shared working rules

- Keep original H2 labels. Preserve stable source row IDs if rows are filtered.
- Both supervised tasks use the same split assignments, subject to documented task-specific exclusions.
- Keep final test rows untouched during M1 model development.
- Fit supervised imputation, encoding, scaling, and later PCA/LDA inside training folds.
- Clustering uses its own justified inputs and population. Do not use cancellation/final-status labels to fit clusters or choose parameters. Do not transfer a transformer fitted on all clustering rows into a supervised experiment.
- Save each task's outputs in its own folder. Use the result fields documented in `src/evaluation/results.py`.
- Include the exact run command, interpretation, and actual contribution. A notebook can supplement the Python modules, not replace them.

## Later work

M2 extends the same modules with at least three models/algorithms per task, baseline and tuned/optimized results for every selection, an ensemble/boosting supervised model, PCA for at least two supervised models, an original/PCA/LDA classifier comparison, final test evaluation, and a usable CLI or GUI loading saved inference pipelines. `src/tuning/`, `src/inference/`, `app/`, and `artifacts/` are reserved for this extension.

## Submission record

GitHub repository URL: https://github.com/City-Hotel-Booking/City-Hotel-Booking-ML

Submitted commit SHA: not available yet.

Member 6 records the submitted revision and repository link for each release and packages source, configuration, dependencies, report, and relevant result tables/figures. Exclude trained/fitted artifacts, local environments, caches, credentials, and scratch files. Keep a separate release record so the submitted SHA does not imply a self-referential commit.

See `CONTRIBUTING.md` for file ownership and the future collaboration workflow.
