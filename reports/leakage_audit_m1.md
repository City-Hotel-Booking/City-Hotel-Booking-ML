# Milestone 1 preliminary leakage audit

Owner: Member 2, for review with Members 1 and 3-5. Updated 29 October 2026.
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
