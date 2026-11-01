# Clustering Milestone 1 handoff

Owner: Aya, Member 5.

The clustering contribution runs locally through the existing command interface.
The untuned K-Means baseline, metrics, original-unit profiles, two figures and
preliminary leakage audit are ready for owner and team review. Scientific
approval and GitHub integration remain pending. Nothing has been committed,
pushed or sent to teammates; all deliverables are local.

## Progress report contribution

The clustering task used all 55,532 frozen training bookings and the existing
11 provisional booking-time inputs. A fresh clustering preprocessor applied
training-median imputation, log1p on LeadTime, numeric standardization and
categorical one-hot encoding, producing 45 numeric columns. Cancellation,
ADR and final-status fields were excluded from clustering and PCA inputs.

An untuned K-Means experiment used three clusters, k-means++ initialization,
10 initializations and seed 42. It produced 39,945, 1,328 and 14,259 bookings
in clusters 0, 1 and 2 respectively. Silhouette was 0.244930 on a reproducible
uniform sample of 3,000 training rows, and Davies-Bouldin was 1.834343 on all
training rows. Model fitting took 1.039 seconds in the recorded environment.
No optimal cluster count or held-out predictive performance is claimed.

The largest group had a median lead time of 131 days; the third group had a
median of 11 days and a higher mean child count. The smallest group contained
all repeated-guest training bookings, indicating sensitivity to the scaled
repeat-guest indicator. The first two PCA components explained 32.02% of
variance. The visualization is a partial view of the 45-dimensional solution.

All 15 preprocessing checks passed. The repository suite passed 76 tests and
2 subtests. Reloading the saved pipeline reproduced all training assignments.
The final supervised test set was not evaluated. Snapshot feature timing,
distance weighting and the inherited outcome-stratified split remain explicit
limitations. Further algorithms, parameter optimization and stability analysis
are planned for Milestone 2.

## Files for integration

| File | Purpose | Integration |
| --- | --- | --- |
| `src/clustering/baseline.py` | K-Means runner and supporting evaluation/profile functions | Existing `run(config, project_root)` route |
| `src/clustering/check_preprocessing.py` | Reproducible training-only representation diagnostic | Run as a module |
| `tests/test_clustering.py` | Undefined metrics, deterministic sampling and profile checks | Existing pytest suite |
| `outputs/clustering/metrics.json` | Parameters, metrics, timing, environment, provenance and paths | Existing shared result schema |
| `outputs/clustering/cluster_profiles.csv` | Cluster sizes and observed numerical/category summaries | Report and analysis display |
| `outputs/clustering/category_profiles.csv` | Complete within-cluster category frequencies | Profile interpretation |
| `outputs/clustering/silhouette_sample.csv` | Exact metric-sample source row IDs | Reproducibility; not clustering inputs |
| `outputs/clustering/preprocessing_checks.json` | Diagnostic evidence | Review |
| `outputs/clustering/cluster_sizes.png` | Size distribution | Report figure |
| `outputs/clustering/pca_clusters.png` | Training-only PCA view | Report figure |
| `reports/clustering_m1.md` | Full feature decisions, audit, interpretation and environment | Detailed task report |

Local fitted artifacts are under `artifacts/clustering/`: the K-Means pipeline,
visualization PCA and row assignments. Keep them out of the submitted ZIP and
GitHub. The source regeneration command creates them for local use.

The existing main.py route already works; no router change or dependency on
regression/classification predictions is required. The runner writes the common
result record through `save_result`. It does not edit the integrated team report.

## Reproduction

Run from the repository root with unchanged `data/H2.csv`. Its SHA-256 must be
`6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.
The split version is `m1-group-stratified-v1`; feature policy is `m1-draft-1`.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.\env\Scripts\python.exe -m src.clustering.check_preprocessing
.\env\Scripts\python.exe main.py baseline --task clustering
.\env\Scripts\python.exe -m pytest -q
```

The working environment uses Python 3.12.14, NumPy 2.3.5, pandas 3.0.1,
scikit-learn 1.6.1, SciPy 1.15.3, PyYAML 6.0.3, matplotlib 3.11.2 and joblib
1.6.0. The local env uses bundled base packages; the detailed report explains
the environment setup limitation. A fresh independent-machine installation has
not been verified. Rerunning replaces task outputs; timing can differ.

## Questions prepared for team review

These questions have not been sent.

**Michael:** Can we approve the current booking-time purpose, all training rows
without task exclusions, and the 11-input policy? The snapshot cannot establish
every original booking-entry value. Do we retain those assumptions or change
the analysis point? Please review the strong repeated-guest split and the
disclosure that the inherited manifest used IsCanceled stratification.

**Makarious:** The existing clustering baseline CLI route runs and writes the
common result record. Can you review the metrics, profiles and figures for the
integrated M1 report? For M2, should the application offer analysis viewing
only, or also new-booking cluster assignment? This handoff remains local under
the owner's instruction to postpone GitHub updates.

**Zeina:** Can you confirm the interpretation and snapshot timing limitations
of the active booking attributes? We preserved all source rows and the exact
course source; no EDA-based outlier deletion was introduced.

## Review and remaining work

- Completed: integrated local command, baseline output files, preliminary
  interpretation, feature table, leakage audit, tests and artifact reload check.
- Completed handoff checks: cluster sizes agree with metrics and sum to 55,532;
  shares sum to one; category proportions sum to one within cluster/feature;
  all 3,000 metric sample IDs are unique training IDs; referenced figures and
  artifacts exist.
- Pending: Aya's review and ability to explain the work; scientific review with
  Michael; Makarious's integrated-report assembly and team acceptance.
- Pending under the owner's instruction: identifiable Git commits and remote
  integration. Local completion does not satisfy the course Git-history rule.
  Commit/push/PR actions require explicit later authorization.
- M2 work has not started: compare at least three algorithms before and after
  optimization, investigate k and representation sensitivity, assess stability,
  create a practical dendrogram, and agree the final application workflow.

## Discussion preparation

1. `get_task_data(..., "train")` retrieves the frozen training inputs and None
   as the target: no correct cluster labels are supplied.
2. `preprocessor.fit_transform(X)` learns medians, numeric scales and category
   vocabulary from those inputs and creates the numerical matrix.
3. `model.fit_predict(Z)` learns centers and returns one cluster ID per booking.
   IDs are arbitrary; cluster 0 does not mean not canceled.
4. `n_clusters=3` requests three groups; `n_init=10` tries ten initializations
   and retains the one with the lowest K-Means objective. They are different
   parameters.
5. `evaluate(...)` measures distances in Z, not in the PCA plot. Its fixed
   silhouette sample limits cost and introduces sampling uncertainty.
6. `profiles(...)` describes original observed booking values; these profiles
   support cautious descriptions rather than predetermined business labels.
7. `PCA(...).fit_transform(Z)` creates a plot representation only. The discarded
   variance explains why the image cannot fully prove separation.
8. `joblib.dump(...)` preserves fitted preparation and centers for later loading.
   A fresh prediction should reuse that pipeline rather than fit again.

Understanding checks: Why is k=3 not yet the best k? Why might a rare binary
feature dominate standardized distance? Why are metrics computed before PCA?
Why can a small cluster be meaningful without automatically being better?

## Subsequent local completion

The M2 clustering implementation has now run locally. The earlier M1 results
and limitations remain a historical record. For current commands and completed
experiments, see `reports/clustering_final.md` and `docs/clustering_guide.md`.
GitHub and scientific approval remain pending; no remote action was taken.
