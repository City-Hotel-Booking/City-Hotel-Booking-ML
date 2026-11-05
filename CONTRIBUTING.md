# Team Contributions and Workflow

> Ownership, experiment integrity and review rules for the City Hotel project.

## Contributions and ownership

| Member | GitHub account | Main responsibility and supporting files |
| --- | --- | --- |
| Zeina | `zeihany0375` | Loading, validation and EDA: `src/data/`, source evidence and data reports |
| Michael | `MichaelHany48` | Shared split, feature policy, preprocessing and leakage audit |
| Paula | `PaulaAdelDev` | `src/regression/`, regression tests, results and reports |
| Youssef | `youssefklopp-del` | `src/classification/`, classification tests, results and report |
| Aya | `ayaoraby63` | `src/clustering/`, clustering checks, tests, profiles and reports |
| Makarious | `makarioussafa` | Shared command routing, configuration integration, result writer, documentation and submission verification |

Michael maintains `src/data/split.py` and `src/data/prepare.py` within the
shared data package. Preserve each member's implementation and contribution
history. Explain focused corrections rather than replacing completed work.

The integrated M1 report is maintained in
`Milestone_1_Report/milestone1_report.md`, with task owners checking their sections.

## Experiment rules

- Preserve source bytes, source row IDs and frozen assignments.
- Use the active feature policy and record prediction point and population.
- Fit learned preprocessing inside the training pipeline and each training fold.
- Use group-aware folds when introducing cross-validation.
- Reserve final supervised test evaluation for M2.
- Use unsupervised criteria for clustering selection.
- Document exclusions within existing splits, policy departures and their reasons.
- Keep provisional assumptions distinct from recorded scientific approval.

See [decisions](reports/decisions.md) and
[leakage audit](reports/leakage_audit_m1.md) for the current protocol.

## Shared interfaces

Each task runner exposes `run(config, project_root)` and uses the supplied
configuration. Request data through:

```python
X_train, y_train = get_task_data(
    task, project_root, "train", config=config
)
```

The function is provided by `src.data.prepare`; validation uses the same
interface with `"validation"`. Clustering returns `(X, None)`.

Create a fresh transformer with `build_preprocessor` from
`src.preprocessing.builders`. Use `model_family="linear"` or `"distance"`
for scaled inputs and `"tree"` for unscaled inputs. Clustering uses the
separate dense distance representation.

## Results and task reports

Save results through `save_result` in `src/evaluation/results.py`.
Required fields are:

```text
task, model, parameters, data_checksum, feature_policy_version,
split_version, evaluation_subset, seed, metrics, train_seconds,
command, figure_paths
```

Also record actual inputs, preprocessing, population, row counts and
departures from the shared policy. Classification records positive class,
threshold and confusion matrix; clustering records profiles, metric sampling
and cluster sizes. Use ordinary JSON-compatible values. Unavailable metrics
use `None` with an explanation; NaN and infinity are rejected.

| Task | Result location | Main evidence |
| --- | --- | --- |
| Regression | `outputs/regression/` | MAE, RMSE, R² and residual figure |
| Classification | `outputs/classification/` | Accuracy, precision, recall, F1, ranking metrics and confusion matrix |
| Clustering | `outputs/clustering/` | Internal metrics, cluster profiles and 2D visualization |

Task reports explain settings, measured results, limitations, exact commands
and future work. Planned experiments must not be presented as completed.

## GitHub review workflow

1. Start a task branch from the latest merged `main`.
2. Make focused changes using your own account.
3. Run relevant checks, inspect the diff and stage specific files.
4. Push the task branch and open a pull request.
5. Resolve review feedback and obtain another member's approval.
6. Use **Create a merge commit** to preserve original commits.

Michael reviews Makarious's pull requests; Makarious reviews Michael's.
Either may review model-owner work. A pull-request author does not approve
their own request. Reproducing another member's experiment is integration
work and does not transfer implementation ownership.

## Verification and submission

Install the shared environment using `requirements-dev.txt`.
Use pytest for the full suite and `scripts/verify_outputs.py` for independent
data reconciliation and unittest-compatible checks. Run commands from the
repository root; [README](README.md) contains setup and execution instructions.

Commit code, configuration, dependencies, tests, reports and relevant result
tables/figures. Keep raw H2, environments, caches, temporary output, secrets
and fitted artifacts local. Preserve `artifacts/README.md`.

For each milestone, package the submitted repository version as a ZIP,
including its applicable report. The submission README must contain the
repository URL and submitted commit SHA. Final inference artifacts are
excluded from the ZIP and must be available locally for the M2 discussion.
