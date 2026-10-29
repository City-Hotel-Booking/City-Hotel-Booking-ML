# Team contributions and workflow

## Ownership

| Member | Name | GitHub account | Primary responsibility |
| --- | --- | --- | --- |
| 1 | Zeina | zeihany0375 | Data loading, validation, EDA and data-quality evidence |
| 2 | Michael | MichaelHany48 | Shared splitting, feature policy, preprocessing and leakage audit |
| 3 | Paula | PaulaAdelDev | ADR regression |
| 4 | Youssef | youssefklopp-del | Cancellation classification |
| 5 | Aya | ayaoraby63 | Clustering |
| 6 | Makarious | makarioussafa | Integration, common result handling, documentation and report assembly |

Each member implements, tests and explains their own contribution.
The integrated report records actual work and supporting files or commits.

## Shared experiment rules

Use the starting assumptions in `reports/decisions.md`.

- Preserve the source file, source ordering and zero-based row IDs.
- Use the frozen split assignments and matching source checksum.
- Fit learned preprocessing and models on training data only.
- Keep the final supervised test set reserved for Milestone 2.
- Use group-aware folds if introducing cross-validation.
- Exclude information unavailable at the stated prediction point.
- Use unsupervised criteria to select clustering parameters.

Owners may independently make justified task-specific changes to algorithms,
features, populations, preprocessing and prediction points. Record each
change, its reason, supporting evidence and effect on the experiment.

Keep alternatives within the task's module or configuration. Shared changes
affecting other tasks are reviewed through a pull request.

## Shared data and preprocessing interfaces

Every task runner exposes:

```python
def run(config: dict, project_root: Path) -> None:
    ...
```

Use the configuration passed by the command router:

```python
from src.data.prepare import get_task_data

X_train, y_train = get_task_data(
    task, project_root, "train", config=config
)
```

For supervised validation, request `"validation"` using the same interface.
Clustering returns `(X, None)`.

Create a fresh transformer using:

```python
from src.preprocessing.builders import build_preprocessor

preprocessor = build_preprocessor(
    task, config, model_family="linear"
)
```

Choose `"linear"` or `"distance"` for scaled preprocessing and `"tree"`
for unscaled tree preprocessing. Clustering requires `"distance"`.

For supervised experiments, place preprocessing and the estimator inside
one scikit-learn Pipeline. Fit the pipeline on training rows only.

Approved task-specific exclusions occur within the existing splits.
Preserve row IDs and record exclusion reasons and affected counts.

## Model deliverables

### Paula: regression

Implement:

- `src/regression/baseline.py`
- `tests/test_regression.py`
- `reports/regression_m1.md`

Save under `outputs/regression/`:

- `metrics.json`
- An error figure, such as residuals or actual versus predicted ADR.

Report validation MAE, RMSE or MSE, R2 and training time.
Define the ADR population and explain any exclusions or target transformations.

Command:

```powershell
python main.py baseline --task regression
```

### Youssef: classification

Implement:

- `src/classification/baseline.py`
- `tests/test_classification.py`
- `reports/classification_m1.md`

Save under `outputs/classification/`:

- `metrics.json`
- A labeled confusion-matrix figure.

Report validation accuracy, precision, recall, F1 and ROC-AUC or PR-AUC
where appropriate. Record training time, positive class and threshold.
The starting positive class is `IsCanceled=1`.

Command:

```powershell
python main.py baseline --task classification
```

### Aya: clustering

Implement:

- `src/clustering/baseline.py`
- `tests/test_clustering.py`
- `reports/clustering_m1.md`

Save under `outputs/clustering/`:

- `metrics.json`
- `cluster_profiles.csv`
- A two-dimensional cluster visualization.

State the segmentation purpose, population, features, preprocessing,
algorithm and parameter choices. Report cluster sizes, silhouette and
Davies-Bouldin scores where defined, plus training time.

Document sampling and seed for expensive evaluations. Explain unavailable
scores rather than replacing them with zero. Do not use outcome labels
to choose features, algorithms or clustering parameters.

Command:

```powershell
python main.py baseline --task clustering
```

## Result records

Use `save_result(result, path)` from `src/evaluation/results.py`.

Required fields:

- `task`
- `model`
- `parameters`
- `data_checksum`
- `feature_policy_version`
- `split_version`
- `evaluation_subset`
- `seed`
- `metrics`
- `train_seconds`
- `command`
- `figure_paths`

Also record the exact features, model family, preprocessing settings,
population definition and relevant row counts. Include an experiment
identifier and document departures from the shared starting policy.

Classification adds positive class and threshold.
Clustering adds sampling and population details. Record the frozen split
version when clustering uses its training assignments.

Use ordinary Python values in result dictionaries. Convert NumPy scalars
and arrays before saving. NaN and infinity are rejected.
Unavailable metrics use `None`, with explanations in `metric_notes`.

Figure paths are repository-relative. Keep fitted models and pipelines
under the ignored `artifacts/` directory.

## Task reports

Each task report includes:

1. Purpose, prediction or analysis point and population.
2. Selected untuned baseline and fixed settings.
3. Inputs, preprocessing, source checksum and versions.
4. Actual metrics and links to saved figures or tables.
5. Interpretation and limitations.
6. Departures from the starting policy and supporting reasons.
7. Exact command and actual member contribution.
8. Planned Milestone 2 extensions.

Do not present planned work as completed experiments.

## Integration responsibilities

Makarious maintains command routing, configuration loading, the common
result writer, root documentation and `reports/milestone1_report.md`.

Zeina maintains source evidence and EDA.
Michael maintains shared preparation and leakage documentation.
Model owners verify their sections of the integrated report.

## GitHub workflow

1. Start a task branch from the latest merged `main`.
2. Make meaningful commits using your own account.
3. Run relevant tests and inspect the diff.
4. Stage specific files.
5. Push the branch and open a pull request into `main`.
6. Describe the change, verification commands, outputs and limitations.
7. Resolve review feedback before merging.
8. Use Create a merge commit to preserve original commits and authorship.

Makarious reviews Michael's pull requests.
Michael reviews Makarious's pull requests.
Either may review other members' pull requests.
A member must not approve their own pull request.

## Tracked and local files

Keep the unchanged `data/H2.csv` locally; it is ignored by Git and must match
the checksum in the frozen split metadata.

Commit source code, configuration, tests, relevant results and report figures.
Do not commit raw data, environments, caches, secrets, processed datasets or fitted
models and pipelines. Keep the two local reference documents excluded.
