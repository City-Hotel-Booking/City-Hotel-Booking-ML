# Generated evidence and experiment results

Keep reproducible evidence in the task-specific folders below.

| Folder | Owners | Contents |
| --- | --- | --- |
| `data/` | Zeina and Michael | Validation, schema, unusual-record review, frozen assignments, split metadata and preparation checks |
| `eda/` | Zeina | Descriptive tables, figures and their manifest |
| `regression/` | Paula | Regression metrics and error figures |
| `classification/` | Youssef | Classification metrics and confusion matrix |
| `clustering/` | Aya | Clustering metrics, profiles and visualization |

## Source evidence

Files under `data/` include:

- `validation.json`
- `schema.csv`
- `review.json`
- `split_manifest.csv`
- `split_metadata.json`
- `preprocessing_checks.json`

The frozen split contains source row IDs and subset assignments only.
Preserve its source identity and assignments.

`eda/manifest.json` lists the descriptive tables and figures.
These describe the source; they are not model-performance results.

`verification.json` records the environment and findings of its saved
verification run. Regenerate it through the verification script when needed;
do not manually change its test counts or dependency versions.

## Model results

Each model owner saves `metrics.json` in their task folder using
`src/evaluation/results.py`.

Record:

- Task, model and fixed parameters.
- Source checksum and split/feature-policy versions.
- Evaluation subset and random seed.
- Actual metrics and training time.
- Exact reproduction command.
- Repository-relative figure paths.
- Exact features, preprocessing and population.
- Task-specific thresholds, exclusions or sampling.

Use None for an unavailable metric and explain it in `metric_notes`.
Do not substitute a fabricated zero.

For M1, supervised metrics use validation or training cross-validation.
Final supervised test results remain reserved for Milestone 2.
Clustering evaluation uses the documented development population and
unsupervised criteria.

## Figures and tables

Regression includes an error figure.
Classification includes a labeled confusion matrix.
Clustering includes `cluster_profiles.csv` and a two-dimensional figure.

Use descriptive filenames, labeled axes and captions where appropriate.
The task report must explain the results and any limitations.

## Local artifacts and scratch files

Fitted models and preprocessing pipelines belong in the ignored
`artifacts/` directory.

Use ignored `outputs/tmp/` and `outputs/logs/` for scratch output.
Do not store raw or processed dataset copies in result folders.

Relevant result JSON/CSV files and report figures may be committed.
Review regenerated output differences before staging them.