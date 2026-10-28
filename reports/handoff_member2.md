# Member 2 handoff for Milestone 1

Owner: Michael. Updated 29 October 2026. Scientific choices remain provisional
until the named owners respond in `reports/decisions.md`.

## What is ready

- Source: City Hotel H2 only, 79,330 rows and 31 official source columns;
  SHA-256 `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.
- Frozen label-free manifest: `outputs/data/split_manifest.csv`, with 55,532
  train, 11,899 validation and 11,899 reserved test row IDs. Split version:
  `m1-group-stratified-v1`. Exact normalized groups do not cross splits.
- Current feature policy: `m1-draft-1`, 16 ADR inputs, 16 cancellation inputs,
  and 11 clustering inputs. See `src/features/policy.py` for the exact lists
  and every 31-column decision. These lists are provisional.
- `src/data/prepare.py` exposes `load_manifest()`, `get_row_ids(split)`,
  `get_task_data(task, project_root, split)`, and `run(config, project_root)`.
  `get_task_data` returns `(X, y)` for supervised tasks and `(X, None)` for
  clustering. Row IDs are the index of X and y.
- `src/preprocessing/builders.py` exposes `build_preprocessor(task, config,
  model_family="distance")`. It returns a **new unfitted** ColumnTransformer.
  Use `model_family="tree"` for unscaled tree inputs, `"linear"` or the
  default `"distance"` for scaled inputs. Clustering requires the scaled
  distance choice and returns dense data.
- `python -m src.data.prepare` and `python main.py prepare` both create or
  verify the frozen split. Member 6 should route the latter to
  `src.data.prepare` for one shared public entry point.
- `python -m src.preprocessing.checks` writes
  `outputs/data/preprocessing_checks.json`; all nine checks passed on the
  confirmed H2 file. The test suite reported 31 passed and two subtests
  passed. No model evaluation was performed on final test rows.

## Still awaiting a team decision

Prediction time and mutable fields, ADR target population, use of high-cardinality
IDs, duplicate interpretation, and clustering purpose remain open. The tracker,
owners, proposed dates and provisional defaults are in `reports/decisions.md`.
In particular, Member 1 found 1,208 zero-ADR and 331 zero-night rows across
the full source; these sets overlap. The **training-only** counts are 785 zero
ADR, 224 zero nights and 23,172 canceled rows. Do not add overlapping counts
or filter rows without a recorded population decision. Member 1's evidence is
in `outputs/data/validation.json`, `outputs/data/review.json`,
`outputs/data/schema.csv`, and `reports/data_review.md`.

## Rules for every baseline

1. Select rows through the frozen row IDs. Any approved task-specific
   exclusion happens **inside each split**, without reshuffling the manifest.
2. Fit the entire `Pipeline` on training rows. Fit preprocessing again inside
   each training cross-validation fold. Validation rows can be transformed
   and scored; final test rows remain untouched until the Milestone 2 final
   evaluation. Never tune on test results.
3. Use only `get_features(task, config)` inputs. IsCanceled is never an input;
   ReservationStatus and ReservationStatusDate are outcome fields; ADR is
   excluded from all active input lists. The positive class for cancellation
   is `IsCanceled=1`.
4. Clustering chooses features, k and other parameters using only unsupervised
   criteria such as silhouette and Davies-Bouldin. Do not use cancellation or
   status labels to choose clusters. They may describe fixed clusters later.
5. Save result records with at least `task`, `model`, `parameters`,
   `data_checksum`, `split_version`, `feature_policy_version`,
   `evaluation_subset`, `seed`, `metrics`, `train_seconds`, and `command`.
   Include `model_family`, preprocessing settings, and train/validation row
   counts. Classification also records positive class and threshold;
   clustering records population, sampling, and unsupervised scores. Member 6
   owns the common result writer in `src/evaluation/results.py`.

## Copy-paste starting points

Run these from the repository root. They illustrate **untuned baselines**;
the model owners choose and document their actual algorithms and metrics.
None of the snippets loads test rows.

### Member 3: ADR regression

```python
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from src.data.prepare import get_task_data
from src.preprocessing.builders import build_preprocessor
from src.settings import load_config

root = Path.cwd()
config = load_config(root / "config.yaml")
X_train, y_train = get_task_data("regression", root, "train")
X_val, y_val = get_task_data("regression", root, "validation")
model = Pipeline([
    ("preprocess", build_preprocessor("regression", config, model_family="linear")),
    ("regressor", Ridge()),
])
model.fit(X_train, y_train.astype(float))
print("validation_MAE", mean_absolute_error(y_val.astype(float), model.predict(X_val)))
```

Record the agreed ADR population and any excluded row IDs within each split.
Report MAE, RMSE and R2 for the baseline; save settings and runtime.

### Member 4: cancellation classification

```python
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from src.data.prepare import get_task_data
from src.preprocessing.builders import build_preprocessor
from src.settings import load_config

root = Path.cwd()
config = load_config(root / "config.yaml")
X_train, y_train = get_task_data("classification", root, "train")
X_val, y_val = get_task_data("classification", root, "validation")
model = Pipeline([
    ("preprocess", build_preprocessor("classification", config, model_family="linear")),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train.astype(int))
predicted = model.predict(X_val)
print("validation_F1", f1_score(y_val.astype(int), predicted, pos_label=1))
```

Record threshold 0.5 if using the default. Also report precision, recall,
accuracy, confusion matrix and AUC when supported. Do not include final
ReservationStatus or ReservationStatusDate in X.

### Member 5: booking-time clustering

```python
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from src.data.prepare import get_task_data
from src.preprocessing.builders import build_preprocessor
from src.settings import load_config

root = Path.cwd()
config = load_config(root / "config.yaml")
X_train, y = get_task_data("clustering", root, "train")
assert y is None
preprocess = build_preprocessor("clustering", config)
Z_train = preprocess.fit_transform(X_train)
labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(Z_train)
print("cluster_sizes", dict(zip(*np.unique(labels, return_counts=True))))
```

The `3` is an example baseline setting, not an approved k. Compare candidates
using unsupervised criteria on a documented training sample; a full silhouette
calculation on all 55,532 rows can be expensive. Fit PCA for the 2D plot on
training rows only. Never use cancellation/status labels in selection.

## Copy-paste messages to teammates

**Zeina, Member 1:**
> Thanks for the H2 validation and EDA. Your zero-based row_id, 79,330 rows,
> 31 official labels and source checksum match our frozen split. I kept your
> raw-token loader for grouping. Please confirm the 4 unknown Children values,
> Agent/Company NULL as not applicable, and whether the original booking-time
> values of guest counts and stay nights can be established. The NA and NULL
> grouping difference is documented; we have not changed the manifest.

**Paula, Member 3:**
> The frozen train/validation row IDs and 16 provisional ADR inputs are ready
> through `get_task_data("regression", root, split)`. The training set has 785
> zero-ADR and 224 zero-night rows; Member 1's full-source review also flags
> the ADR maximum of 5,400. Please agree the ADR population with Zeina and me
> before accepting your baseline. Keep exclusions inside each existing split.

**Youssef, Member 4:**
> The provisional prediction point is first booking entry. The positive class
> is `IsCanceled=1`. Use the 16 approved inputs from `get_task_data` and an
> unfitted preprocessor inside your model Pipeline. Final status and status
> date are excluded, as are recorded ADR and later booking-change fields.
> Please record the classification threshold and class-aware validation metrics.

**Aya, Member 5:**
> The current clustering policy has 11 booking-time inputs. Use the separate
> scaled clustering preprocessor on training rows only. IsCanceled, ADR,
> ReservationStatus and ReservationStatusDate are not active cluster inputs.
> Please confirm the segment purpose and population with me, then choose k
> using unsupervised scores and record any sample and seed.

**Makarious, Member 6:**
> Member 2 now provides `src.data.prepare.run(config, project_root)` and
> `python -m src.data.prepare`. Please route `python main.py prepare` to
> `src.data.prepare` when updating main.py. Example: set
> `RUNNERS["prepare"] = "src.data.prepare"`. I have not edited main.py or your
> config sections. Result records need the data checksum, split version and
> feature policy version, plus task/model/parameters/metrics/runtime/command.
> The current versions are `m1-group-stratified-v1` and `m1-draft-1`.

These are prepared text for Michael to send; this file does not claim the
messages have been delivered or decisions approved.

## Review checklist for each later baseline

- [ ] Uses the same frozen train and validation row IDs and source checksum.
- [ ] Uses only the task's currently approved inputs and records policy version.
- [ ] Fits preprocessing, feature selection and model on training folds only.
- [ ] Does not load, transform for scoring, or tune on final test rows in M1.
- [ ] Documents task-specific row exclusions without moving other rows.
- [ ] Reports positive class/threshold for classification or unsupervised
      selection criteria for clustering.
- [ ] Reported values match saved result files, commands, seed and runtime.
- [ ] Missing or unavailable metrics are explained, never filled with zero.
