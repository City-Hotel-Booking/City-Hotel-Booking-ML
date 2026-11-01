# Clustering feature and population decisions

Owner: Aya, Member 5.

Status: source verification, feature review and preprocessing verification
completed locally; the untuned K-Means baseline has now run. The scientific
decisions below are provisional under `m1-draft-1`, pending review with Michael.
They do not record team approval or an optimized clustering solution.

## Purpose and analysis point

Discover groups of similar bookings using planning horizon, season, party
composition, booking products, channels and prior-guest status. The proposed
analysis point is initial entry into the hotel PMS. These are booking segments;
without a customer identifier, they must not be described as unique customer
groups or evidence of individual customer behavior over time.

H2 is a later snapshot. A field normally entered at booking can still have been
amended before extraction. The active policy assumes the recorded values are
reasonable proxies for entry-time attributes; it does not establish their
historical availability. Guest counts, meal, room preferences and booking-type
fields require particular caution. If initial values cannot be established,
retain this limitation or explicitly reconsider the analysis point with Michael.
Do not describe the baseline as proven leakage-free.

## Source identity and population

Use the unchanged course-mirror H2 at `data/H2.csv`:

- SHA-256: `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.
- Source shape: 79,330 records and 31 source columns.
- Preserve source order and zero-based `row_id`; never use row_id as a feature.
- The supplied alternative CSV has SHA-256
  `ea4ef1bce2fc0d819e0abbbc1a475e789bca3a675be60cbc3313762aaff542ae`.
  A record-by-record comparison found the same values in the same order after
  trimming outer whitespace and parsing its ReservationStatusDate as M/D/YYYY
  versus YYYY-MM-DD in the course source. The comparison did not rewrite either
  file. Use the exact course source to satisfy the frozen checksum contract.

For the provisional M1 experiment, use all 55,532 frozen training records from
`m1-group-stratified-v1`, seed 42. Keep duplicate bookings and unusual records;
apply no cancellation, ADR, zero-night or zero-guest filter. There are no
task-specific exclusions at this stage. Canceled bookings remain eligible, but
their cancellation outcomes must not enter the clustering matrix or selection.

The shared split was stratified by IsCanceled and grouped on all 31 source
fields. It therefore used outcomes during split construction, even though the
saved manifest contains only row IDs and assignments. Disclose this provenance:
the proposed clustering fit and parameter selection receive no target labels,
but the population inherits the team's outcome-stratified sampling design.
Do not describe this as a wholly outcome-independent sampling procedure.

Validation and reserved test rows are outside the M1 clustering fit, PCA fit
and metric sample. The training subset describes that subset of H2, not a
completed full-H2 segmentation. Agree a documented M2 population before any
broader refit; do not silently change the supervised split or test policy.

## Feature selection for all source columns

The table follows source-column order. Include means active in the current
11-input policy. Defer means absent from the baseline pending a justified
decision. Exclude means absent for the stated analysis point or task scope.
An active inclusion is still conditional on the timing assumptions above.

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

Active inputs, in the existing policy's order:

`LeadTime`, `ArrivalDateMonth`, `Adults`, `Children`, `Babies`, `Meal`,
`MarketSegment`, `DistributionChannel`, `IsRepeatedGuest`, `ReservedRoomType`,
`CustomerType`.

This report changes no active columns, shared feature-policy version or config.
ADR remains outside the active list despite the stronger task-specific exclusion.

## Preprocessing proposal and distance limitations

Reuse `build_preprocessor("clustering", config, model_family="distance")` as
a fresh, separately fitted transformer. Use the passed configuration and
`get_task_data("clustering", project_root, "train", config=config)`; its target
is None. Fit learned medians, category vocabulary and scaling on training rows
only. No fitted supervised transformer is required.

The existing builder parses numeric fields, imputes training medians, applies
log1p to nonnegative LeadTime and standardizes numeric inputs. It imputes a
categorical missing marker and uses one-hot encoding with unknown categories
ignored. Clustering receives dense output. No additional engineered feature or
learned clipping rule is proposed in this step.

One-hot categories avoid artificial ordering between labels. They do not make
the resulting Euclidean distance automatically appropriate: numeric and
categorical blocks have different weights; rare standardized binary/count
values can have large effects; MarketSegment and DistributionChannel may
double-count similar information. Month one-hot treats every distinct month
as equally different and does not capture December/January adjacency.
Ignoring unseen categories can produce an all-zero category block on later
data. Inspect these limitations in preprocessing verification and record any
later sensitivity experiment rather than claiming equal feature influence.

Median imputation is a starting rule, not proof that unknown children equal
the median. Preserve all rows for now. Check rare guest counts and zero-guest
records within training data before making any task-specific treatment change.

## Preliminary leakage audit

| Risk | Proposed control | Remaining evidence or limitation |
| --- | --- | --- |
| Current outcomes or price define segments | Use exactly the 11 active inputs; no IsCanceled, ADR, ReservationStatus or ReservationStatusDate in clustering or PCA. | Verify actual model inputs during implementation. |
| Mutable snapshot fields stand in for entry values | Document each timing assumption and review with Michael. | Original entry-time values have not been established. |
| Future history enters IsRepeatedGuest | Require history cutoff before current entry. | Snapshot alone does not establish calculation provenance. |
| Outcomes affect row selection | No cancellation/ADR filtering; preserve current training assignments. | Existing manifest was outcome-stratified and grouped using all fields; disclose this design. |
| Learned transformations see reserved rows | Fit preprocessing and visualization PCA only on training inputs. | Verify in Step 3 and baseline checks; not yet executed. |
| Metric or feature selection uses target labels | Use internal clustering criteria and booking-attribute profiles only. | Record metric sample, candidate settings and selection rationale when run. |
| Full-source EDA becomes learned treatment | Treat existing summaries as descriptions; estimate any learned rules on training data. | No clipping or outlier deletion is proposed. |
| Row identity or sampling changes silently | Require exact source checksum and frozen manifest; record any metric sample and seed. | Compare result provenance when baseline exists. |

## Results and verification status

Source verification ran locally on 30 October 2026 using `read_dataset` and
`inspect_dataset`. It returned `passed_with_findings` with no blocking issues:
79,330 rows, 31 expected columns, 4 unknown Children, 24 unknown Country,
25,902 excess identical rows, 331 zero-night records and 167 complete
zero-guest records. No records were removed or imputed.

The initial data unittest run had 10 passes and two missing-matplotlib errors.
After local environment setup, the full existing repository suite completed:
**73 passed and 2 subtests passed in 18.37 seconds**. This supersedes the earlier
environment-limited result; it does not test an implemented clustering model.

### Training preprocessing verification

`src/clustering/check_preprocessing.py` reuses the shared task-data interface
and builder. Its run passed all 15 checks; saved evidence is
`outputs/clustering/preprocessing_checks.json`. It confirms exact training
row IDs, absent targets/outcomes, fresh independent transformers, finite dense
output, training-derived medians and scaler statistics, near-zero numeric
means, missing-value handling, unseen-category handling without learning the
probe, and identical repeated-fit representations.

- Input: 55,532 training rows and 11 source inputs.
- Output: 55,532 rows and 45 numeric columns, using 19,991,520 bytes for one
  transformed float64 matrix (not total process memory).
- Numeric inputs: five. Their transformed standard deviations are approximately
  one. Six categorical inputs expand to 40 indicator columns.
- Training category counts: month 12, meal 4, market segment 7, distribution
  channel 5, reserved room 8 and customer type 4.
- One training Children value is unknown. Training medians are LeadTime 88,
  Adults 2, Children 0, Babies 0 and IsRepeatedGuest 0. Imputation to zero here
  follows a learned median; it does not establish the missing booking's true
  child count.
- An unseen Meal value produces an all-zero block for that field without
  changing fitted categories or output width. This avoids an exception but
  leaves a distance limitation for future data.
- The loader reads the full CSV for source identity; preprocessing fits and
  transformations use only training features and synthetic probes. No
  validation or reserved test representation was evaluated by this diagnostic.

### Local environment and commands

The working local environment is `env/`, based on bundled Python 3.12.14 with
access to its bundled packages. The ignored `.venv/` is an unsuccessful earlier
Espressif-Python setup and is not the environment used for these results.
The working versions are NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.6.1,
SciPy 1.15.3 and PyYAML 6.0.3. The first installations with newer ML packages
stalled loading native extensions; replacing local SciPy/scikit-learn with the
versions above allowed checks to finish. The exact native-loading cause has
not been established. Shared requirement files were not changed.

Run from the repository root in PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.\env\Scripts\python.exe -m src.clustering.check_preprocessing
.\env\Scripts\python.exe -m pytest -q
```

For a separate Python 3.12 environment, install the repository's development
requirements and constrain the verified core versions before rerunning:

```powershell
python -m pip install -r requirements-dev.txt "numpy==2.3.5" "pandas==3.0.1" "scikit-learn==1.6.1" "scipy==1.15.3" "PyYAML==6.0.3"
```

That fresh-environment recipe has not itself been tested; the completed run
used the local bundled-base environment described above.

### Untuned K-Means results

The integrated command completed on all 55,532 training bookings. The runner
uses k=3, k-means++ initialization, 10 initializations, seed 42, Lloyd updates,
max_iter=300 and tolerance 0.0001. These are fixed starting settings, not a
search or an approved optimal cluster count. Three provides a small readable
reference whose limitations can be compared with other k values in M2.

K-Means fits the 45-dimensional prepared matrix, not PCA coordinates. It
converged in 20 iterations. Fitting K-Means took 1.039 seconds; preprocessing
took 0.644 seconds in the measured run. Numerical operations are limited to
one thread in the runner to control resource use.

| Measure | Actual result | Evaluation population |
| --- | --- | --- |
| Silhouette | 0.244930 | Fixed uniform sample of 3,000 training bookings, seed 42 |
| Davies-Bouldin | 1.834343 | All training rows, original prepared representation |
| Inertia | 337341.391467 | All training rows; sum of squared center distances |

Silhouette ranges from -1 to 1; higher values generally indicate better
cohesion/separation. This positive but modest value suggests overlap rather
than extremely distinct groups. Davies-Bouldin is lower for more compact,
separated groups, but has no universal pass threshold. Inertia depends on
representation and k and is not a stand-alone quality verdict. These scores
are descriptive training results, not held-out predictive performance.

The silhouette sample is not redrawn to favor a result. Its source IDs are
saved in `outputs/clustering/silhouette_sample.csv`. No outcomes were used in
feature, parameter or metric selection. Metrics are in
`outputs/clustering/metrics.json`.

### Measured booking profiles

| Cluster ID | Bookings | Share | Median lead time | Repeated-guest share |
| --- | ---: | ---: | ---: | ---: |
| 0 | 39,945 | 71.93% | 131 days | 0% |
| 1 | 1,328 | 2.39% | 5 days | 100% |
| 2 | 14,259 | 25.68% | 11 days | 0% |

Cluster 0 contains longer-lead bookings, with mean Adults 1.94 and mean
Children 0.0036. Cluster 1 contains every repeated-guest training booking;
Corporate is its most frequent market segment at 42.02%, not a majority of
that group. Cluster 2 has shorter lead times and more children on average
(mean 0.253); Online TA accounts for 55.15% of that group. These are observed
characteristics, not definitions imposed before fitting. Cluster IDs are
arbitrary and do not establish cancellation risk, price or customer value.

The perfect repeated-guest split is a warning about distance sensitivity:
standardizing a rare binary input can strongly separate its rare values.
This baseline may emphasize loyalty status more than the intended broad
booking patterns. Keep the measured result as the untuned reference, then
investigate alternative weighting/representation and k using unsupervised
criteria in M2. Do not claim a validated business segmentation yet.

Profiles use original observed units, excluding missing values from numerical
means/medians and reporting missing counts separately. The unknown child value
is therefore not silently treated as an observed zero in the profiles. Detailed
outputs are `cluster_profiles.csv` and `category_profiles.csv` under
`outputs/clustering/`.

### Figures and saved components

- `outputs/clustering/cluster_sizes.png` shows the strongly imbalanced groups.
- `outputs/clustering/pca_clusters.png` shows 10,000 uniformly sampled training
  bookings (plot seed 43). PCA is fitted on all training representations only.
  PC1 explains 17.92% and PC2 explains 14.09%, totaling 32.02%. The remaining
  variance is not displayed, so apparent 2D overlap or separation cannot fully
  describe the 45-dimensional clusters.
- Fitted pipeline, visualization PCA and per-booking assignments are local in
  ignored `artifacts/clustering/`. They must be excluded from the submission
  ZIP. Loading the pipeline reproduced all 55,532 saved training assignments;
  loading PCA also transformed a training probe successfully. This verifies
  serialization consistency, not new-booking generalization.

Both figures were visually inspected. Three new behavioral tests cover
undefined scores, reproducible metric sampling and original-unit profiles.
The full suite passed: **76 tests and 2 subtests in 11.24 seconds**. No final
supervised test performance was evaluated.

## Reproduction and contribution

Working local baseline command from the repository root:

```powershell
.\env\Scripts\python.exe main.py baseline --task clustering
```

The existing CLI route calls `src.clustering.baseline.run(config, project_root)`.
No router changes were needed. The null shared baseline setting uses the
runner's documented defaults; an optional `baselines.clustering.n_clusters`
overrides k. Actual settings are always saved with the result.

The present changes replace the report template with the source comparison,
provisional population, all 31 feature decisions, preprocessing rationale and
preliminary leakage audit, and add the training-only diagnostic module and
its JSON evidence. Baseline work adds the K-Means runner, three behavioral
tests, measured metrics, profiles, metric sample IDs and two figures. Shared
code, settings, split and existing evidence files
remain unchanged. A local task branch isolates this work; no commit, push or
pull request has been made. Git contribution remains pending the owner's
review and explicitly approved commits under their account.

## Next steps and Milestone 2

1. Completed: verify the existing clustering preprocessor on training rows and
   establish a working local environment, recording shape, missing values,
   scaling and category behavior.
2. Completed: implement one documented untuned baseline and measure cluster sizes,
   silhouette and Davies-Bouldin scores with a reproducible evaluation sample
   where needed, plus a separate training-only PCA view, explained variance,
   profiles, plots and interpretation. The local M1 handoff is prepared in
   `reports/clustering_handoff_m1.md`; owner review, team acceptance and remote
   integration remain pending.
3. For M2, evaluate at least three justified algorithms with baseline and
   optimized settings, document every search, investigate cluster count,
   compare stability and computational cost, and add a practical hierarchical
   dendrogram. Agree the full-data population and application behavior with
   the team. Keep required fitted artifacts local and outside the submission
   ZIP; supply regeneration instructions.
