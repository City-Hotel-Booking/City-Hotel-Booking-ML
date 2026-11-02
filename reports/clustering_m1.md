# Milestone 1 clustering baseline

Owner: Aya, Member 5. Verified: 2 November 2026.
Implementation and baseline evidence are complete; scientific approval under
`m1-draft-1` is pending. This contribution is under review as an M1-only contribution and is not
merged. No optimized cluster count or validated business segments are claimed.

## Purpose and population

Discover similar booking groups at initial PMS entry using planning horizon,
season, party composition, requested products, booking channels and prior-guest
status. These are booking groups, not unique customers: H2 has no customer ID.
Recorded values come from a later snapshot and may include amendments.

Use the unchanged City Hotel H2: 79,330 rows, 31 source columns, SHA-256
`6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`. Preserve source ordering and row_id. Use all 55,532
training rows in `m1-group-stratified-v1`, seed 42, without task exclusions,
deduplication or outcome-based filtering. The shared split uses cancellation
stratification and grouping on all normalized source fields; disclose that
inherited design rather than call sampling fully outcome-independent.

Reserved test rows were not used for fitting, parameter selection, metric
evaluation or PCA fitting. The shared loader reads the source CSV before
selecting the requested subset. Validation rows are likewise not used here.

## Inputs and preprocessing

The 11 active inputs are: `LeadTime`, `ArrivalDateMonth`, `Adults`, `Children`, `Babies`, `Meal`, `MarketSegment`, `DistributionChannel`, `IsRepeatedGuest`, `ReservedRoomType`, `CustomerType`.

A fresh shared `build_preprocessor("clustering", config, model_family="distance")`
parses numbers, learns training medians, applies configured LeadTime log1p,
standardizes numeric fields, imputes categorical missing markers and one-hot
encodes with unknown-category handling. Dense output is 55,532 x 45. The M1
representation has no additional cap, clipping or engineered fields. Learned
preparation is separate from every supervised experiment.

All source-column decisions follow below. Include remains conditional on
booking-entry timing; defer/exclude means absent from this baseline.

| Source feature | Type | Baseline decision | Purpose and timing or leakage assessment | Planned treatment if active |
| --- | --- | --- | --- | --- |
| IsCanceled | Binary outcome | Exclude | Current booking's eventual outcome; must not define groups or select parameters. | None |
| LeadTime | Numeric count | Include | Planning horizon from entry to planned arrival; amended arrival dates remain a snapshot caveat. | Training median, log1p, standard scaling |
| ArrivalDateYear | Calendar integer | Defer | Planned year can be known at entry, but may create extraction-era cohorts rather than useful booking segments. | None |
| ArrivalDateMonth | Categorical calendar | Include | Seasonal booking context, assuming recorded planned arrival is appropriate at entry. | Missing marker and one-hot encoding |
| ArrivalDateWeekNumber | Calendar integer | Defer | Overlaps with month and can overweight seasonality; no justified extra role yet. | None |
| ArrivalDateDayOfMonth | Calendar integer | Defer | Adds day-level calendar detail without a current segmentation purpose. | None |
| StaysInWeekendNights | Numeric count | Defer | Potentially useful stay pattern, but booked/stayed duration may change after entry; verify initial-value meaning. | None |
| StaysInWeekNights | Numeric count | Defer | Same timing concern; do not derive total stay until both source inputs are justified. | None |
| Adults | Numeric count | Include | Party composition normally recorded at entry; later amendments cannot be ruled out. | Training median and standard scaling |
| Children | Numeric count | Include | Family composition under the same timing assumption; four source values are unknown, not established zero. | Training median and standard scaling |
| Babies | Numeric count | Include | Party composition under the same timing assumption; rare counts may influence scaled distance. | Training median and standard scaling |
| Meal | Categorical | Include | Requested meal package; assume entry-time selection while acknowledging amendments. | Missing marker and one-hot encoding |
| Country | Categorical | Defer | May only be completed at check-in; 24 unknown entries and many categories add timing and distance concerns. | None |
| MarketSegment | Categorical | Include | Commercial booking context normally assigned at entry; verify recorded semantics. | Missing marker and one-hot encoding |
| DistributionChannel | Categorical | Include | Booking route normally known at entry; may overlap with MarketSegment. | Missing marker and one-hot encoding |
| IsRepeatedGuest | Binary indicator | Include | Loyalty context only if based on history before the current booking; future history must not be used. | Training median and standard scaling, following existing builder |
| PreviousCancellations | Numeric history count | Defer | Legitimate prior history is not automatically leakage; omitted because current purpose is not risk/history segmentation and cutoff needs verification. | None |
| PreviousBookingsNotCanceled | Numeric history count | Defer | Could support loyalty segmentation if computed strictly before entry; purpose and historical cutoff remain unresolved. | None |
| ReservedRoomType | Categorical | Include | Requested product, unlike final assignment; assume recorded reservation reflects entry-time preference. | Missing marker and one-hot encoding |
| AssignedRoomType | Categorical | Exclude | Ultimate room assignment may occur or change after entry. | None |
| BookingChanges | Numeric count | Exclude | Counts amendments occurring after entry. | None |
| DepositType | Categorical | Defer | Could be an entry-time contract term; timing/meaning not confirmed. Correlation with cancellation alone would not prove leakage. | None |
| Agent | Categorical identifier | Defer | Potentially known at entry, but identity, timing and many levels need justification; numeric IDs are not distances. NULL is not applicable. | None |
| Company | Categorical identifier | Defer | Same concerns as Agent; NULL is not applicable, not a missing quantity. | None |
| DaysInWaitingList | Numeric duration | Exclude | Accumulated waiting time is unavailable at first entry. | None |
| CustomerType | Categorical | Include | Booking arrangement normally assigned at entry; confirm snapshot timing. | Missing marker and one-hot encoding |
| ADR | Numeric supervised target | Exclude | Explicit task scope excludes ADR from inputs and selection; recorded rate may also differ from initial quote. Shared policy marks it deferred, but this task keeps it excluded. | None |
| RequiredCarParkingSpaces | Numeric count | Defer | Requests can be added after entry; initial request value not established. | None |
| TotalOfSpecialRequests | Numeric count | Defer | Requests can accumulate after entry; initial count not established. | None |
| ReservationStatus | Categorical outcome | Exclude | Final booking status supplies an outcome. | None |
| ReservationStatusDate | Date | Exclude | Final-status timing is future information at entry. | None |

## Untuned experiment and metrics

K-Means uses k=3, k-means++ initialization, n_init=10, seed 42, max_iter=300,
tol=0.0001 and Lloyd updates. These are fixed starting settings; k=3 is neither
optimized nor scientifically approved. K-Means receives no target labels.

| Measure | Current result | Population |
| --- | ---: | --- |
| Silhouette | 0.244930 | Uniform sample of 3,000 training rows, without replacement, seed 42 |
| Davies-Bouldin | 1.834343 | All 55,532 training rows |
| Inertia | 337341.391467 | All training rows |
| K-Means fitting | 1.361128 seconds | fit_predict only |
| Preprocessing | 0.892652 seconds | fit_transform only |
| Metric evaluation | 0.263574 seconds | Both internal metrics |
| Visualization PCA | 0.276519 seconds | Training-only fit_transform |

The model stopped after 20 iterations. Metrics use the prepared 45D
representation before PCA. Sample IDs in `silhouette_sample.csv` are fixed,
unique training IDs; they are not redrawn to favor a result. The silhouette
value is positive but modest, indicating overlap. Lower Davies-Bouldin is
better, but there is no universal pass threshold. Inertia depends on k and
representation and cannot alone establish useful segments.

## Original-unit profiles and figures

| Cluster ID | Bookings | Share | Median lead time | Mean Adults | Mean Children | Repeated guests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 39,945 | 71.93% | 131 days | 1.940 | 0.004 | 0% |
| 1 | 1,328 | 2.39% | 5 days | 1.373 | 0.023 | 100% |
| 2 | 14,259 | 25.68% | 11 days | 1.571 | 0.253 | 0% |

Cluster 0 has longer lead times; cluster 2 has shorter lead times and a higher
mean child count. Cluster 1 contains every repeated-guest training booking.
This indicates sensitivity to standardizing a rare binary feature, not proof
of a validated loyalty segment. Its most frequent market segment is Corporate
at 42.02%, which is not a majority. IDs are arbitrary; no group is described
as canceled, high-risk or high-value. Missing observed numerical values are
excluded from profile means/medians and counted separately.

- [Cluster sizes](../outputs/clustering/cluster_sizes.png): unequal group sizes.
- [PCA view](../outputs/clustering/pca_clusters.png): 10,000 uniformly sampled
  training bookings, plot seed 43. PCA fits all training representations;
  PC1 retains 17.92% and PC2 retains
  14.09%, totaling
  32.02%. The projection discards the
  remaining variance and cannot prove separation in the full feature space.
- `cluster_profiles.csv` and `category_profiles.csv` supply complete measured
  numerical summaries, missing counts and within-group category shares.

## Preliminary leakage audit

| Risk | Control and remaining limitation |
| --- | --- |
| Outcome-driven groups | IsCanceled, ADR and final-status fields are absent from fit, metrics and PCA; no outcome-based parameter choices or names. |
| Snapshot timing | Entry-time availability is assumed, not established; guest counts, products and booking-type fields can change. Scientific review remains pending. |
| Prior-history timing | IsRepeatedGuest must reflect history before this booking; the snapshot does not establish its computation cutoff. |
| Learned preparation | Fit fresh shared preprocessing and PCA only on training rows; never reuse the fitted clustering transformer in supervised tasks. |
| Sampling | Preserve the inherited outcome-stratified manifest; use no outcome filters. Silhouette sampling may underrepresent small groups. |
| Distance weighting | Rare standardized values and correlated channel inputs can dominate; month one-hot ignores cyclic adjacency. No new treatment is introduced in M1. |
| Full-source EDA | Use summaries descriptively; no all-source learned threshold, deletion or clipping rule is applied. |

## Reproduction and current verification

From the repository root, using the verified Python environment:

```powershell
python -m src.clustering.check_preprocessing
python main.py baseline --task clustering
python -m pytest tests/test_clustering.py -q
python -m pytest -q
git diff --check
```

An alternative configuration is passed before the subcommand:
`python main.py --config experiment.yaml baseline --task clustering`.
The saved command retains --config and the record contains effective settings,
actual parameters, population, source/policy versions, sampling and timings.
Actual current environment: python 3.12.14, numpy 2.3.5, pandas 3.0.1, scikit-learn 1.6.1, scipy 1.15.3, matplotlib 3.11.2, joblib 1.6.0, PyYAML 6.0.3, threadpoolctl 3.7.0.
See `requirements-clustering.txt` for the tested package constraints.

Verified on 2 November 2026 against current integrated code based on origin/main
`1cc934a`: all 15 preprocessing checks passed; focused clustering tests **11
passed**; complete repository suite **122 passed and 2 subtests passed**
(29.55 seconds), with no failures or skips. `git diff --check` passed.
Source/checksum, uncapped 45D representation, training-only sample IDs,
metrics/profile totals, figure paths and protected shared/supervised files
were checked for consistency.

The shared CSV, frozen assignments, config, feature policy, regression and
classification implementations/tests/reports/outputs and integrated report
were preserved. Local fitted artifacts remain ignored under
`artifacts/clustering/`; no fitted object, environment, raw dataset or scratch
output belongs in this PR. No supervised test performance is reported.

## Contribution and future plan

Member 5 supplies the baseline runner, preprocessing diagnostic, synthetic
integration tests, metrics, profiles, sample IDs, two figures and M1 notes.
Member 6 retains ownership of integrated-report assembly. Implementation is
complete; scientific review and PR merge remain pending.

Planned for M2 only: compare at least three algorithms before/after systematic
optimization, investigate k and representation sensitivity, evaluate stability,
and agree dimensionality-reduction, hierarchy visualization and application
requirements with the team. No M2 implementation is included in this PR.
