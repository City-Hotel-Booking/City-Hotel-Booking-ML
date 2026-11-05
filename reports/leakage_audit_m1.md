# Milestone 1 preliminary leakage audit

Owner: Member 2, for review with Members 1 and 3-5. Updated 4 November 2026.
Status: **provisional** (`m1-draft-1`); the team has not yet approved the
feature lists or ADR population.

## Prediction point and scope

For regression and classification, assume a prediction **when the booking
first enters the hotel property-management system (PMS)**. Clustering describes
bookings using fields available at that same point. H2 is a later snapshot
and cannot prove that every stored value was unchanged since entry. Member 1
and the model owners must review the timing assumptions.

The source has 31 official labels. `src/data/split.py` maps supported
snake_case or official CamelCase headers to these labels explicitly; `row_id`
is a new zero-based source position, not a feature. H1 is never used. This
table reflects `src/features/policy.py`: **U** = use provisionally,
**X** = exclude, **D** = needs team decision and is excluded for now.

| Source column | ADR regression | Cancellation classification | Clustering | Booking-time rationale / concern |
| --- | :---: | :---: | :---: | --- |
| IsCanceled | X | X | X | Later outcome; target for classification, never an input or cluster criterion. |
| LeadTime | U | U | U | Computable from entry and intended arrival dates. |
| ArrivalDateYear | U | U | D | Known planned year; clustering might group by calendar cohort. |
| ArrivalDateMonth | U | U | U | Planned month is known at entry. |
| ArrivalDateWeekNumber | U | U | D | Known week, but overlaps with month and can dominate seasonal clusters. |
| ArrivalDateDayOfMonth | U | U | D | Known planned day; may add noise to segments. |
| StaysInWeekendNights | D | D | D | Stored nights may reflect amendments or actual stay. |
| StaysInWeekNights | D | D | D | Same timing concern as weekend nights. |
| Adults | U | U | U | Initial party size is normally entered; verify amendments. |
| Children | U | U | U | Initial count is normally entered; missing values need imputation. |
| Babies | U | U | U | Initial count is normally entered; verify amendments. |
| Meal | U | U | U | Initial meal package is normally part of the reservation. |
| Country | D | D | D | Could be completed at check-in; high cardinality and missing meaning need review. |
| MarketSegment | U | U | U | Normally assigned at entry; verify timing. |
| DistributionChannel | U | U | U | Normally determined by how the booking arrived. |
| IsRepeatedGuest | U | U | U | Eligible only if calculated from history before this booking. |
| PreviousCancellations | U | U | D | Prior history is eligible for prediction; clusters need a history-led purpose. |
| PreviousBookingsNotCanceled | U | U | D | Same history cutoff and cluster-purpose concern. |
| ReservedRoomType | U | U | U | Requested room type is known at entry. |
| AssignedRoomType | X | X | X | Final room assignment can occur or change later. |
| BookingChanges | X | X | X | Count accumulates after entry. |
| DepositType | D | D | D | Confirm entry-time value and cancellation-policy proxy risk. |
| Agent | D | D | D | Agent ID may be known; missing may mean no agent and IDs are numerous. |
| Company | D | D | D | Company ID may be known; missing may mean no company and IDs are numerous. |
| DaysInWaitingList | X | X | X | Waiting duration accumulates after entry. |
| CustomerType | U | U | U | Normally assigned at entry; verify later changes. |
| ADR | X | D | D | Regression target; recorded value may differ from initial quote or define price-led clusters. |
| RequiredCarParkingSpaces | D | D | D | May be requested or changed after entry. |
| TotalOfSpecialRequests | D | D | D | Count may accumulate after entry. |
| ReservationStatus | X | X | X | Final status reveals the later outcome. |
| ReservationStatusDate | X | X | X | Date of final status is unavailable at booking. |

The policy yields 16 regression inputs, 16 classification inputs, and 11
clustering inputs. These counts come from policy code, not model selection.
Changing any D field to U needs evidence, a policy version change, and new
experiments.

## Regression: predict ADR

| Risk | Current decision | What the team must verify |
| --- | --- | --- |
| ADR as an input | Excluded; it is the target. | Whether source ADR is the price at booking or a later amended/final rate. |
| IsCanceled, ReservationStatus, ReservationStatusDate | Excluded; cancellation/final status happens later. | Do not use cancellation as a predictor or population filter without a stated business task. |
| AssignedRoomType, BookingChanges, DaysInWaitingList | Excluded because they can be updated after entry. | A later prediction point would need a new policy. |
| Stay nights, parking, requests, DepositType, Country, Agent, Company | Disputed and excluded for M1. | Confirm initial values, missing semantics, and availability when price is quoted. |
| Canceled, zero-night, zero-ADR records | No row removal approved. | Decide the ADR modeling population before Member 3 trains; record exclusions by row_id. |

Among the **55,532 training rows only**, all ADR entries parse as numbers;
785 have ADR = 0, none have ADR < 0, 224 have zero total nights, and 23,172
have `IsCanceled = 1`. These sets overlap: 53 training rows have both zero
ADR and cancellation. These are descriptive counts from the frozen manifest
and confirmed local H2, not a decision to exclude rows. If the team excludes
any target population, keep original row_id assignments and record counts
within each split without moving rows.

## Classification: predict IsCanceled

| Risk | Current decision | What the team must verify |
| --- | --- | --- |
| IsCanceled | Target only, never an input. | Keep it out of X and learned preprocessing. |
| ReservationStatus, ReservationStatusDate | Excluded; reveal later outcomes. | No derived status/date field should enter a model. |
| AssignedRoomType, BookingChanges, DaysInWaitingList | Excluded; can reflect post-booking events. | Changing prediction time needs a new audit. |
| ADR | Disputed and excluded; recorded ADR may be final rate. | Allow only a verified booking-time quote with the same meaning. |
| Deposit, stay nights, parking, requests, Country, Agent, Company | Disputed and excluded. | Confirm availability and whether missing means unknown or not applicable. |
| Previous history | Used provisionally. | Ensure history stops strictly before this booking. |

The cancellation label is used to make the stratified shared split, never as
a model input. The manifest contains only row_id and split, allowing Members
3 and 4 to share assignments without labels in the split file.

## Clustering: booking-time segments

There is no target. Exclude `IsCanceled`, `ADR` (pending an agreed
price-segmentation purpose), `ReservationStatus`, and `ReservationStatusDate`
from fitting and parameter choice. Exclude late change, assignment, and
waiting-list fields. Other disputed fields stay out. The 11 active inputs
describe planned timing, party, meal, channel, repeat-guest flag, requested
room, and customer type. Member 5 should state the segment purpose before
interpreting clusters.

For M1, fit cluster preprocessing and candidate algorithms on training rows
only. Choose k/parameters using unsupervised criteria such as silhouette and
Davies-Bouldin on that development population. Outcome labels can describe
chosen clusters afterwards, never select k or features. Fit optional PCA
visualization on the same development population, label it as a projection,
and record the training subsample and seed for any dendrogram.

## Workflow and order checks

1. Read confirmed H2 only. Assign row_id before any filter. The shared
   manifest uses seed 42, requested 70/15/15 proportions, cancellation
   stratification, and SHA-256 groups based on all 31 trimmed/`NULL`-normalized
   fields. Exact identical groups stay intact. There is no booking ID, so
   this cannot prove that every repeated real booking is grouped.
2. Members 3 and 4 join by row_id and use the same frozen assignments.
   Approved task-specific target exclusions happen *within* each split; they
   do not regenerate or reshuffle the manifest.
3. `build_preprocessor(task, config)` returns a fresh **unfitted**
   `ColumnTransformer`. Put it in a scikit-learn `Pipeline` with the model;
   fit the whole pipeline only on each training fold. Numeric median/scaling,
   categorical imputation/one-hot, and later feature engineering learn there.
   Unknown categories are ignored at transform time. Member 1 verified four
   unknown `Children` values; preprocessing treats them as missing, not zero.
   `Agent` and `Company` NULL means not applicable and is preserved as a
   categorical token in the builder. Agent, Company, and Country remain
   excluded until approved; rare-category one-hot is dormant. Scaled model
   families use a train-fold `log1p` transform on LeadTime after imputation;
   tree models can use unscaled inputs. Clustering uses a separate dense,
   scaled transformer instance.
4. Cross-validation and search use development rows only. Put preprocessing,
   feature selection, PCA, and LDA *inside* each fold's pipeline. LDA uses
   that fold's training labels only. Never fit these on all H2 before a
   split. Record tuning search space and random seed.
5. Use validation data for M1 comparison. Reserve final test rows for one M2
   evaluation after model choice. Split creation necessarily read all rows
   and cancellation labels to assign groups and stratify; the metadata also
   includes aggregate test class balance. No model, learned preprocessing,
   feature choice, tuning, or test metric has used final test rows.

The manifest contains 79,330 source positions: 55,532 train, 11,899
validation, and 11,899 test. Cancellation share is 0.4172729237 in train and
0.4172619548 in validation/test. There are 5,846 exact duplicate groups and
25,902 extra rows within these groups; an independent hash check found zero
exact groups spanning splits. `outputs/data/split_metadata.json` records the
figures. The random grouped split does **not** test drift across arrival
years; a chronological holdout could later be a separate robustness check,
not a replacement for this frozen M1 manifest.

## Open review items

- Members 1, 2, and 3: agree what ADR means and which bookings have usable
  targets. Do not remove zero-ADR, zero-night, or canceled rows silently.
- Members 1 and 3-5: verify timing of U fields and decide whether any D
  field is safe and useful. Document source evidence and approval date.
- Member 5: agree clustering population and purpose before selecting
  features or using any outcomes to interpret groups.
- All model owners: cite policy and split versions in result records.

## Member 1 evidence and new leakage checks

The following are **full-source descriptions**, not training counts or
selection thresholds. `outputs/data/validation.json`, `outputs/data/review.json`,
and `outputs/data/schema.csv` record the source SHA-256, counts and meanings;
`reports/data_review.md` records overlaps. The source has 79,330 rows and 31
columns, 4 unknown Children, 24 unknown Country, 8,131 Agent NULL and 75,641
Company NULL values. Country has 166 known categories, Agent 224 and Company
208. Agent and Company IDs are categories even when their tokens look numeric.
Do not interpret ID magnitude as distance, or fit rare-category boundaries on
all rows. The source also has 331 zero-night, 1,208 zero-ADR, 167 complete
zero-guest rows, and an ADR maximum of 5,400. These flags overlap. Applying a
full-source outlier threshold or filtering rows *before* the shared split
would change the scientific population and may leak evaluation information.

The shared loader keeps raw tokens separate from its normalized view. The
frozen split groups after trimming outer whitespace and normalizing blank and
NULL. The analysis view also recognizes NA/N/A/NAN and preserves Agent/Company
NULL as not applicable. The new split-loader path uses Member 1's **raw** view
and the old grouping rule; the regenerated manifest matches the saved file
exactly. Do not feed the normalized view directly into group hashing without
versioning and verifying the result. `row_id` is the original zero-based CSV
position and must be assigned before any task-specific exclusion.

Potential timing risks remain: stored guest counts and stay length may reflect
changes after first booking entry; Country may be filled at check-in; recorded
ADR may have been amended. `ReservationStatus` and `ReservationStatusDate`
encode final outcomes. These findings support keeping disputed fields out for
Milestone 1 but do not prove the initial values of the current active fields.
The arrival period, 2015-07-01 through 2017-08-31, also makes a random split
insufficient to measure future-year drift. Any chronological robustness check
must be separate from the frozen common manifest.

`outputs/data/preprocessing_checks.json` records real-H2 checks: train-only
transformer fitting, unknown-category handling, forbidden outcomes absent,
label-free manifest, deterministic assignments, finite transformed train and
validation arrays, and zero duplicate-group crossings. No model or learned
transformer has been fitted on final test rows.

## Evidence from merged Milestone 1 runners (4 November 2026)

This review covers merged `main` revision `229f456` and its committed result
files. It establishes what the inspected code and saved runs did, not that
every active field was available in its initial form at booking entry. All
three result records use the source checksum recorded in
`outputs/data/split_metadata.json`, split version `m1-group-stratified-v1`,
and feature-policy version `m1-draft-1`. Their saved feature lists match
`get_features(task, config)` exactly: 16 regression, 16 classification and
11 clustering columns. `src/data/prepare.py:99-130` checks version, source
checksum and manifest row IDs before it selects a task's subset and columns.
The nine checks in `outputs/data/preprocessing_checks.json` all pass,
including label-free manifest, zero exact duplicate-group crossings,
train-only transformer fitting and unseen-category handling. A fresh Python
3.11.9 environment rerun on 4 November returned `122 passed, 2 subtests
passed` from `python -m pytest -q`.

| Task | Found or ruled-out risk in actual M1 code | Evidence | Remaining limit |
| --- | --- | --- | --- |
| ADR regression | Requests only frozen `train` and `validation`; gets the 16 active policy inputs; fits shared preprocessing and `LinearRegression` in one pipeline on training rows. No final-status feature or test evaluation appears in this runner. | `src/regression/baseline.py:284-325`; `outputs/regression/metrics.json` (`evaluation_subset=validation`, `test_set_evaluated=false`) | The all-finite-ADR population includes zero-rate, zero-night and canceled bookings. Their business meaning and timing of mutable active inputs remain unapproved. The validation ADR 5,400 was scored, not removed. |
| Cancellation classification | Requests frozen `train` and `validation`; fits shared preprocessing and logistic regression in one pipeline on training rows. `IsCanceled` is the target. Final status/date, recorded ADR and later-change fields are absent from the result's feature list. | `src/classification/baseline.py:313-377`; `outputs/classification/metrics.json` (`evaluation_subset=validation`, 16 features, `test_set_evaluated=false`) | Initial values of guest counts, meal, customer type and prior-history cutoff need source-system evidence. The 0.5 threshold is fixed, not tuned. |
| Booking clustering | Requests frozen `train` only, rejects an outcome target/column and fits the shared dense distance transformer and K-Means there. Silhouette uses a fixed 3,000-row training sample; Davies-Bouldin uses all training rows. PCA fits training representations for the plot only. | `src/clustering/baseline.py:77-113,134-175`; `outputs/clustering/metrics.json` (`evaluation_subset=train`, `test_set_evaluated=false`) | The common training membership was stratified by `IsCanceled` during split creation. This is outcome-informed sample selection even though the clustering runner never receives labels. Purpose and k=3 remain provisional; do not select k or describe groups by cancellation/status. |

The optional regression comparison also uses policy features, frozen
train/validation subsets, and a fresh shared preprocessor in each fixed
candidate pipeline (`src/regression/improve.py:260-347`). It chooses among
two candidates by full-validation MAE (RMSE for ties) without a test score.
Repeated choices on the same validation subset can still overfit it. The
untuned M1 reference remains `outputs/regression/metrics.json`.

### Separation of stages in the inspected code

1. **Source and assignment:** `src/data/load.py` preserves source row IDs.
   `src/data/split.py` creates the label-free manifest after grouping exact
   normalized rows. `src/data/prepare.py:107-130` checks version and checksum
   before handing out a subset. `IsCanceled` stratified the assignment but is
   absent from the manifest and active inputs.
2. **Feature policy and engineering:** `src/features/policy.py` fixes active
   columns before modeling. `src/preprocessing/builders.py:76-159` puts
   numeric parsing, median imputation, optional `LeadTime` `log1p`, scaling,
   categorical cleaning and one-hot encoding in a reusable transformer.
   There is no separate fitted feature-engineering step in the three M1
   baseline runners.
3. **Learned preprocessing:** supervised runners put a fresh transformer
   inside a scikit-learn `Pipeline` and fit on training rows
   (`src/regression/baseline.py:311-321`;
   `src/classification/baseline.py:350-375`). Clustering calls
   `fit_transform` only after requesting frozen training rows
   (`src/clustering/baseline.py:86-106`). Validation is transformed by
   already-fitted supervised pipelines. The real-H2 check records three
   training-index transformer fits, one per task.
4. **Dimensionality reduction:** M1 clustering PCA fits prepared training
   data only for its 2D plot. K-Means and internal metrics use the original
   45-dimensional representation (`src/clustering/baseline.py:109-114,134-172`).
   Supervised PCA and LDA comparisons are **not implemented** in these M1
   runners. In M2, both must fit inside each training fold; LDA must receive
   only that fold's training labels.
5. **Tuning and selection:** the three M1 baselines use fixed settings;
   K-Means uses k=3. No search-based tuning is claimed. The optional
   regression comparison uses validation to choose between two fixed
   candidates (`src/regression/improve.py:313-347`). In M2, keep preprocessing,
   dimensionality reduction and estimator inside each group-aware training
   fold, and record the search space and seed.
6. **Final evaluation:** both supervised M1 results identify `validation` as
   the evaluation subset and explicitly record `test_set_evaluated=false`.
   Both runners request only training and validation. Reserve final test
   scoring until all choices are frozen. Reading source rows for identity or
   aggregate split metadata is different from fitting/scoring test rows;
   do not let the metadata's test-class balance guide model selection.

### Normalization consistency and frozen grouping

EDA and the shared modeling builder preserve `Agent`/`Company` `NULL` as
a categorical not-applicable value. The preparation check records
`not_applicable_ids_preserved: PASS`; both columns remain inactive M1
features. Their scientific approval is still pending.

Literal `NA` is treated as missing in analysis/model views, while frozen
split hashing retains its documented trim/blank/NULL normalization rule.
Preserve the current manifest. Changing group hashing requires review of
its effect on group membership and assignments, with a new split version
where affected.
