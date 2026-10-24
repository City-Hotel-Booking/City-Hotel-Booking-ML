# Team contributions

Member labels are work packages, not assigned names. The team will map its six members to these labels.

| Member | Primary files and folders |
| --- | --- |
| 1 Data inspection and EDA | `src/data/load.py`, `validate.py`, `eda.py`; `outputs/data/validation.json`, `schema.csv`; `outputs/eda/`; `reports/eda_notes.md` |
| 2 Shared preparation | `src/data/split.py`, `src/features/policy.py`, `src/preprocessing/builders.py`; `config.yaml` scientific settings; split manifest/metadata; `reports/decisions.md`, `leakage_audit_m1.md` |
| 3 Regression | `src/regression/`, `outputs/regression/`, `reports/regression_m1.md` |
| 4 Classification | `src/classification/`, `outputs/classification/`, `reports/classification_m1.md` |
| 5 Clustering | `src/clustering/`, `outputs/clustering/`, `reports/clustering_m1.md` |
| 6 Integration and report | `main.py`, `src/settings.py`, `src/evaluation/results.py`, root setup/docs; `reports/milestone1_report.md`; release packaging |

Each owner adds relevant tests under `tests/`. All members contribute evidence and review their sections in the final report.

## Shared handoff

Every task runner exposes `run(config, project_root)`. It uses repository-relative paths from the shared configuration. Members 3 and 4 use the common loader, split manifest, feature policy, and unfitted preprocessing builder. Member 5 uses a separate justified clustering input list.

Members 3-5 each provide `metrics.json` in their own output folder, actual figures/tables, a short interpretation, and the exact reproduction command. Use the shared result fields in `src/evaluation/results.py`; do not invent scores. Add specific details such as classifier threshold/positive class or clustering sample/population as appropriate.

Member 2 changes scientific settings after coordinating with affected owners. Member 6 coordinates dependency and command changes so simultaneous edits do not overwrite one another.

## GitHub collaboration

Create each task branch from the latest `main`. Open a pull request that explains the purpose, changed files, run command, and output paths. The repository owner reviews the work before it is merged; additional reviewers can be requested when needed. Use a merge commit so the task commits and their authors remain visible in the project history.

Do not upload raw datasets, trained models, fitted transformers, environments, caches, credentials, or the two local root reference documents. Upload relevant result JSON/CSV files and report figures. Keep the same folder paths when contributing; do not submit separate copies of the whole project.

