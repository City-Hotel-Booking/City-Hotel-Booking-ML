# Results

These folders will contain actual reproducible outputs; .gitkeep files only preserve the empty folders in Git.

| Folder | Owner | Expected files |
| --- | --- | --- |
| `data/` | Members 1 and 2 | `validation.json`, `schema.csv`, `split_manifest.csv`, `split_metadata.json` |
| `eda/` | Member 1 | Descriptive tables and named analysis figures |
| `regression/` | Member 3 | `metrics.json` and regression error figure |
| `classification/` | Member 4 | `metrics.json` and labeled confusion matrix figure |
| `clustering/` | Member 5 | `metrics.json`, `cluster_profiles.csv`, and cluster visualization |

Use one consistent result format from `src/evaluation/results.py`. Include input checksum, feature/split versions where relevant, parameters, seed, training time, evaluation subset, command, and figure paths. Missing metrics are null with an explanation, not zero.

Final tables/figures are eligible for Git. Keep fitted models in the ignored `artifacts/` folder, not here. Use ignored `outputs/tmp/` or `outputs/logs/` for scratch output. No supervised test results belong in the M1 model-comparison results.

