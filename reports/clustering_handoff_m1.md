# Member 5 Milestone 1 handoff

Status: implementation complete; M1-only contribution under review, not merged. Scientific
approval of the shared booking-entry assumptions remains pending.

## Progress report contribution

Using unchanged H2 and all 55,532 frozen training bookings, the provisional
clustering baseline uses the current 11 inputs and fresh shared distance
preprocessing (45 numeric output dimensions). Untuned K-Means uses k=3,
n_init=10 and seed 42. Silhouette is 0.244930 on a fixed
3,000-row training sample; Davies-Bouldin is 1.834343
on all training rows. Model fitting took 1.361128 seconds and
preprocessing took 0.892652 seconds in the current run.

Cluster sizes are 39,945, 1,328 and 14,259. Median lead times are 131, 5 and
11 days. The smallest group contains all repeated guests, indicating strong
influence of the standardized rare binary feature. The groups overlap and
are not validated business segments or cancellation-risk categories. Cluster
IDs are arbitrary. Training-only PCA retains 32.02%
of variance in its 2D view. Exact features, profiles and leakage limitations
are in `reports/clustering_m1.md`.

Reserved test rows were not used for fitting, parameter selection, metric
evaluation or PCA fitting. The shared loader reads the source CSV before
selecting the requested subset. No row exclusions or clipping were applied.

## Deliverables and reproduction

- `src/clustering/baseline.py`: existing `run(config, project_root)` interface.
- `src/clustering/check_preprocessing.py` and `tests/test_clustering.py`.
- `outputs/clustering/`: metrics JSON, numeric/category profiles, fixed metric
  sample IDs, preprocessing checks, cluster-size and PCA figures.
- `reports/clustering_m1.md`, this handoff, concise guide and tested dependencies.
- Fitted artifacts stay ignored and can be regenerated locally.

```powershell
python -m src.clustering.check_preprocessing
python main.py baseline --task clustering
python -m pytest tests/test_clustering.py -q
python -m pytest -q
```

Verified on 2 November 2026 against current integrated code based on origin/main
`1cc934a`: all 15 preprocessing checks passed; focused clustering tests **11
passed**; complete repository suite **122 passed and 2 subtests passed**
(29.55 seconds), with no failures or skips. `git diff --check` passed.
Source/checksum, uncapped 45D representation, training-only sample IDs,
metrics/profile totals, figure paths and protected shared/supervised files
were checked for consistency.

Michael reviews scientific assumptions and feature timing. Makarious receives
these results for integrated M1 report assembly; that report is unchanged.
The future M2 work is planned only and excluded from this PR.
