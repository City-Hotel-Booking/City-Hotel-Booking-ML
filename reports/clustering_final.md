# Clustering final report

Owner: Aya
Member 5

Task C is implemented and exercised locally: three algorithms, baseline and
optimized comparisons, systematic searches, internal metrics, stability checks,
profiles, PCA views, dendrogram, saved models and runnable analysis/inference
commands. This is the clustering contribution, not completion of teammates'
regression/classification tasks. Team scientific review and remote integration
are pending; no commit, push or pull request has been made.

## Purpose and source

Discover booking segments using the existing 11-input feature policy,
`m1-draft-1`. The detailed all-31-column feature table and preliminary audit
remain in `reports/clustering_m1.md`. IsCanceled, ADR, ReservationStatus and
ReservationStatusDate are never inputs or optimization labels. Prior-guest
status is allowed only under the assumption that it reflects history available
at initial booking entry. H2 is a later snapshot; this historical assumption
is not established by the code.

Use City Hotel H2 only, 79,330 source records and 31 columns, SHA-256
`6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`. Read through the shared loader and frozen manifest
`m1-group-stratified-v1`; use all 55,532 training records with no task filters.
The loader reads all records for identity checking, but no validation or final
test rows are used to fit transformations, fit clusters, select parameters,
compute scores or fit visualization PCA. The shared manifest used outcome
stratification and duplicate grouping on all source fields: disclose this
inherited design rather than call the population wholly outcome-independent.

## Preprocessing and protocol revision

The shared unfitted builder supplies training-median imputation, LeadTime
log1p, numeric standardization, missing categorical markers and one-hot
encoding. Eleven inputs become 45 numeric dimensions. Fit this separately
from supervised experiments.

An initial uncapped K-Means subset diagnostic isolated one booking in every
k=2..8 candidate. None passed the predeclared balance rule. At k=3 the
uncapped subset silhouette was about 0.577, despite a singleton and a dominant
group: a concrete example of why internal metrics alone can mislead.
`uncapped_diagnostic.json` preserves the diagnostic. It is not substituted for
the original M1 baseline, which remains unchanged.

The M2 task-specific representation `m2-distance-cap3-v1` caps standardized
values to [-3, 3], after the shared transformation. One-hot indicators remain
unchanged; numeric extremes have bounded distance influence. This fixed rule
was chosen in response to the failed geometry diagnostic, not from target
labels. No record is removed. This changes the geometry, so M1 metrics and
M2 metrics must not be ranked as though their representations were identical.
The shared policy/configuration is unchanged; the fitted cap is saved inside
each M2 prediction pipeline. Capping also sacrifices distinctions among
extreme values and requires future sensitivity review.

## Algorithms and optimization

- K-Means minimizes squared distance to cluster centers. It is fast and
  supports new-booking assignment, but favors roughly center-shaped groups
  and is sensitive to representation and outliers.
- Agglomerative clustering merges nearby groups and exposes a hierarchy.
  It supports several linkage choices but can require quadratic memory and
  does not implement new-booking prediction.
- Gaussian mixtures estimate soft membership in probabilistic components.
  Diagonal covariance permits different spreads with bounded cost, and
  regularization reduces singularity risks. Gaussian assumptions on one-hot
  fields are an approximation; the groups are not a proven generative model.
  GMM is the third permitted algorithm; DBSCAN is not required by the course.

Use one seed-42 uniform sample of 5,000 training bookings for every algorithm
and candidate. Silhouette uses the same seed-43 sample of 1,500 of those rows;
Davies-Bouldin uses all 5,000. Exact IDs are in `comparison_sample.csv` and
`metric_sample.csv`. This supports comparable geometry, not held-out model
generalization. Repeated selection on one sample can overfit that sample.

The systematic search evaluates 49 candidates:

| Algorithm | Search space | Candidates |
| --- | --- | ---: |
| K-Means | k=2..8; n_init=10 or 20; fixed max_iter=300 | 14 |
| Agglomerative | k=2..8; ward, average or complete linkage; Euclidean distance | 21 |
| GMM | components=2..8; diagonal covariance; reg_covar=0.001 or 0.01; n_init=3, max_iter=300 | 14 |

Baseline settings are k=3/n_init=10 for K-Means, k=3/Ward for agglomerative,
and three diagonal components/reg_covar=0.001 for GMM. Eligible solutions
must converge, have at least two groups, every group at least 1% of the subset,
and no group over 95%. Undefined metrics are unavailable rather than zero.
Choose greatest sampled silhouette, then lowest Davies-Bouldin, then fewer
clusters. Eligibility and tie-breaking rules are explicit in the source and
saved results. Inspect all candidates with `python main.py cluster search`.

## Baseline and optimized comparison

All values below refer to the same capped-distance common subset.

| Algorithm | Stage | Parameters | Silhouette | Davies-Bouldin | Mean stability ARI | Fit seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| kmeans | baseline | `{"n_clusters": 3, "n_init": 10}` | 0.142415 | 2.139433 | 0.956664 | 0.238 |
| kmeans | optimized | `{"n_clusters": 2, "n_init": 10}` | 0.222790 | 1.854206 | 0.943080 | 0.166 |
| agglomerative | baseline | `{"linkage": "ward", "n_clusters": 3}` | 0.242120 | 1.618698 | 0.706423 | 2.360 |
| agglomerative | optimized | `{"linkage": "ward", "n_clusters": 3}` | 0.242120 | 1.618698 | 0.706423 | 1.906 |
| gmm | baseline | `{"covariance_type": "diag", "n_components": 3, "reg_covar": 0.001}` | 0.118967 | 2.834893 | 0.860416 | 0.450 |
| gmm | optimized | `{"covariance_type": "diag", "n_components": 2, "reg_covar": 0.01}` | 0.213275 | 3.012208 | 0.927550 | 0.562 |

Optimization does not guarantee every metric improves. Agglomerative search
retained the baseline Ward/k=3 configuration: there was no improvement under
the stated criterion. GMM improved silhouette while Davies-Bouldin worsened;
report both. Silhouette closer to 1 indicates better separation, whereas lower
Davies-Bouldin indicates more compact/separated groups. Neither certifies
business meaning, correctness or causal relationships.

Stability refits each baseline and optimized configuration on three independent
80% subsets (seeds 53, 64 and 75), then computes label-invariant adjusted Rand
agreement against the original labels on overlapping rows. It uses cluster
labels from other unsupervised fits, not supervised targets. Preprocessing
is fixed, so this does not measure variability from relearning preparation.
The mean values in the table summarize three replicates, not confidence bounds.

## Model choice and full-training fits

The analysis winner is **agglomerative**, selected among
eligible solutions with mean stability ARI at least 0.7. Its ARI of about
0.706 indicates moderate sensitivity to the subset; do not call it perfectly
stable. The sample hierarchy is an analysis model only.

The assignment model is **kmeans**, the eligible
predictive algorithm with the best subset silhouette. K-Means supports
predict, is fast, and had mean ARI about 0.943. This is an explicit integration
tradeoff rather than pretending agglomerative clustering predicts new rows.
Labels from different algorithms or refits are not interchangeable identities.

K-Means and GMM each also ran baseline and optimized configurations on all
55,532 training rows. These use a separate fixed 3,000-row silhouette sample,
seed 44, and full-training Davies-Bouldin. They are not directly comparable
to the common-subset table. Agglomerative remains subset-only because a full
55,532-row hierarchy has a much larger quadratic cost. A 5,000-square float64
distance matrix alone is about 200 MB; this is an estimate, not measured RSS.

| Algorithm | Stage | Silhouette | Davies-Bouldin | Fit seconds |
| --- | --- | ---: | ---: | ---: |
| kmeans | baseline | 0.162434 | 2.153020 | 2.109 |
| kmeans | optimized | 0.217067 | 2.095301 | 2.134 |
| gmm | baseline | 0.105801 | 2.845836 | 3.367 |
| gmm | optimized | 0.198776 | 2.910919 | 2.107 |

Dropping only the capped repeat-guest dimension in a separate K-Means
sensitivity check gave ARI **0.998218**
against capped K-Means, silhouette
**0.219995**. The capped solution is much
less dependent on that single field than the M1 solution. This diagnostic
did not change the shared input policy or select the final model.

## Observed segments

The selected three-group hierarchy describes the 5,000-booking subset:

| Cluster | Bookings | Share | Median lead time | Mean Adults | Mean Children |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3,692 | 73.84% | 102 days | 2.057 | 0.002 |
| 1 | 1,075 | 21.50% | 40 days | 1.012 | 0.000 |
| 2 | 233 | 4.66% | 50 days | 1.828 | 1.429 |

These support cautious descriptions of longer-lead larger adult parties,
smaller adult parties, and bookings with children. Group 2 is 75.11% Online TA;
group 1's most frequent market segment is Offline TA/TO at only 28.84%, so
calling it exclusively a corporate/business group would be unsupported.
No segment is called high-value, risky or canceled. Profiles summarize
original observed units, with missing counts separate from imputed inputs.

The full-training optimized K-Means assignment model has 40,285 and 15,247
bookings. Their median lead times are 111 and 18 days, respectively, and mean
adult counts are 2.064 and 1.216. They are two different assignment groups,
not the three hierarchy groups. Complete profile/category tables are saved
for each algorithm/stage and both full predictive models.

## Visual evidence

`comparison_pca.png`, `comparison_sizes.png` and `comparison_metrics.png`
show all six baseline/optimized results on the common subset.
`feature_distributions.png` shows original-unit LeadTime, Adults and Children
distributions of the selected analysis groups; outliers are hidden for display
only and all observations remain in the fits.

PCA was fitted on all training prepared inputs for visualization only. Its
two components retain **39.22%** of variance, so the plot
does not expose the complete 45-dimensional geometry. Cluster fitting and
metric calculation use the prepared representation before PCA.

`dendrogram.png` is a truncated Ward tree on a fixed 1,000-row subset of the
comparison population, seed 47; `dendrogram_sample.csv` stores its IDs.
It visualizes the hierarchy at manageable cost but does not depict every
5,000-row merge. Its vertical axis is linkage distance, not a probability.

## Application and artifacts

The existing CLI now supports:

```powershell
python main.py cluster run
python main.py cluster view
python main.py cluster search
python main.py cluster predict --input examples/clustering_bookings.csv --output outputs/tmp/assignments.csv
```

Run fits/tunes all methods; view/search read saved results without training.
Predict loads the saved full-training K-Means pipeline and selects the required
11 raw input columns. Missing columns and empty input are rejected. The
example CSV contains synthetic illustrative bookings, not real training rows.
The output lists input_row, cluster and model. The application supports CSV
entry rather than requiring edits to Python source.

Saved pipelines, PCA, subset models and row assignments are under ignored
`artifacts/clustering/m2/`. K-Means and GMM support assignment; the hierarchy
does not. Code and commands regenerate artifacts; fitted objects are local
for discussion and excluded from the submission ZIP.

## Leakage, validation and remaining reviews

The complete source checksum, input lists, split/policy versions, parameters,
sampling, training time, environments, candidate history and limitations are
saved in `results.json` and `search.json`. The dedicated all-column leakage
audit in the M1 report still applies. M2 adds a fixed target-free cap, fixed
training-only search populations, training-only visualization, and separate
saved inference components. No outcome-based names, thresholds or parameter
choices were introduced.

No final supervised test set was evaluated. There is no independent held-out
clustering-quality claim. Team approval of snapshot timing/population remains
pending. Remote contribution history is also pending under the owner's
instruction; local completion does not satisfy the course Git-history rule.

Verification includes 81 tests plus 2 subtests; saved candidate/profile/sample
consistency checks; reload reproduction of all subset assignments and all
55,532 assignments for each predictive pipeline; PCA reload; and CLI viewing
and synthetic-booking inference. The test command uses a project-local temp
directory because the previous Windows temp fixture directory was inaccessible.
See `verification.json` and `docs/clustering_guide.md` for reproduction.

The source ZIP packages this local clustering contribution and the existing
team framework; it is not an official final team submission or a committed
revision. Team members must integrate their own supervised work and approve
scientific decisions before official submission.

## Final bundle location and acceptance

The final source/evidence bundle was assembled under the chat-authorized
workspace after the moved OneDrive path triggered automatic permission-review
timeouts for final writes. The OneDrive experimental results are preserved.
Use this bundle as the completed local review copy; final documentation/source
updates have not all been merged back into the OneDrive checkout. The final
bundle suite passed 81 tests and 2 subtests in 32.23 seconds. Python TEMP/TMP
and pytest base-temp are workspace-local for this run.
