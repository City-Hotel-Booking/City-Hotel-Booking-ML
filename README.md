# City Hotel Booking Demand

CSE382 university machine learning project for a team of six. Use H2.csv (City Hotel) only.

## Current status

This repository contains the agreed folder layout, command routing, a provisional feature policy, a working supervised split, and report templates. EDA, preprocessing, model, and result-writing functions still need their owners. There are no model experiment results yet.

Milestone 1 is due **5 November 2026**. Milestone 2 is due **11 December 2026**, according to the supplied project Description.

## Local environment

Use Python 3. The integration owner will record a specific tested Python version and tested dependency versions after checking the first working implementation. The dependency list is currently unpinned.

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

The following command routes exist. Help/configuration/routing, a provisional feature policy, and prepare are implemented. Validate, analyze, and baseline still exit with an unfinished-task message until their owners complete them.

| Command | Owner | Expected output after implementation |
| --- | --- | --- |
| `python main.py validate` | Member 1 | `outputs/data/validation.json`, `schema.csv` |
| `python main.py analyze` | Member 1 | `outputs/eda/`, `reports/eda_notes.md` |
| `python main.py prepare` | Member 2 | `outputs/data/split_manifest.csv`, `split_metadata.json` |
| `python main.py baseline --task regression` | Member 3 | `outputs/regression/`, `reports/regression_m1.md` |
| `python main.py baseline --task classification` | Member 4 | `outputs/classification/`, `reports/classification_m1.md` |
| `python main.py baseline --task clustering` | Member 5 | `outputs/clustering/`, `reports/clustering_m1.md` |

Use `python main.py --config config.yaml <command>` to select a configuration explicitly. Paths in the configuration are relative to the project root, not to an individual laptop.

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

