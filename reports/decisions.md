# Milestone 1 experiment policy
.
This document defines the shared starting assumptions for Milestone 1.
Task owners may develop documented alternatives within their experiments.
Starting assumptions describe the intended experiment; they do not establish
that every stored H2 value was available unchanged at booking entry.

## Responsibilities

| Member | Owner | Responsibility |
| --- | --- | --- |
| 1 | Zeina | Source validation, EDA and data-quality evidence |
| 2 | Michael | Shared splitting, feature policy and preprocessing |
| 3 | Paula | ADR regression experiments |
| 4 | Youssef | Cancellation classification experiments |
| 5 | Aya | Clustering experiments |
| 6 | Makarious | Integration, result format, documentation and report assembly |

## Shared starting settings

| Area | Starting setting | Reason and limitation |
| --- | --- | --- |
| Source | City Hotel H2, 79,330 rows and 31 source columns | Use the exact source associated with the frozen manifest. |
| Prediction point | Initial entry of the booking into the PMS | Provides a consistent feature-availability assumption. H2 contains later snapshots, so mutable fields require caution. |
| Features | Current `m1-draft-1` policy: 16 regression, 16 classification and 11 clustering inputs | Disputed fields remain excluded from the starting lists. |
| Duplicate handling | Preserve all source rows; keep exact normalized duplicate groups together | Identical attributes do not prove erroneous duplicate entry. |
| Split | Frozen grouped, cancellation-stratified 70/15/15 assignment, seed 42 | Supports reproducibility; does not measure future-year drift. |
| Regression population | All bookings with finite ADR | Initially retain canceled, zero-rate, zero-night and extreme-rate records. This predicts recorded ADR across that population rather than realized revenue. |
| Classification population | All source bookings with a valid binary IsCanceled target | Positive class is IsCanceled = 1. |
| Clustering population | All frozen training rows | Discover booking-attribute segments using the current outcome-free inputs. |
| Invalid or unusual records | Flag and investigate; no automatic removal or clipping | Review flags are evidence rather than confirmed errors. |
| Final supervised test set | Reserved for Milestone 2 final evaluation | Do not use test performance to choose features, transformations or models. |
| Initial algorithms | One untuned algorithm selected by each task owner | Record the algorithm, fixed parameters and scientific rationale. |

## Source identity and frozen assignments

Source file: `data/H2.csv`

SHA-256:

`6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`

Split version: `m1-group-stratified-v1`

| Subset | Rows | Canceled rows |
| --- | ---: | ---: |
| Train | 55,532 | 23,172 |
| Validation | 11,899 | 4,965 |
| Test | 11,899 | 4,965 |

The manifest contains only `row_id` and `split`. A row_id is the zero-based
source record position before filtering and is meaningful with the source
checksum. Preserve row identity and assignments throughout every experiment.

The frozen grouping rule hashes all 31 source fields after trimming outer
whitespace and normalizing blank and NULL tokens. Existing checks report
zero exact groups crossing subsets.

The analysis loader additionally recognizes NA/N/A/NAN and preserves
Agent/Company NULL as not applicable. This differs deliberately from the
frozen grouping representation. Do not replace the grouping input with the
normalized modeling view or regenerate assignments silently.

## Preprocessing

- Create a fresh unfitted preprocessor for each experiment.
- Fit learned preprocessing only on training rows.
- If cross-validation is used, fit preprocessing inside every training fold
  and keep duplicate groups together across folds.
- Numeric inputs use training-fold median imputation.
- Linear and distance-based preprocessing uses standard scaling and the
  current log1p LeadTime transformation.
- The LeadTime transformation is a starting choice for compressing a
  nonnegative time interval. Retain or change it using domain reasoning
  and training/development evidence, without consulting final test results.
- Tree preprocessing leaves numerical inputs unscaled and does not apply
  the LeadTime log transformation.
- Categorical inputs use a missing marker and one-hot encoding.
- Unknown categories must transform without refitting the encoder.
- Agent and Company identifiers are categorical. NULL means not applicable.
- Country, Agent and Company remain outside the starting feature lists.
- Clustering uses a separate dense, scaled preprocessing instance.

## Feature availability and leakage

IsCanceled is the classification target and is excluded from model inputs.
ADR is the regression target and is excluded from all starting input lists.
ReservationStatus and ReservationStatusDate describe final outcomes and are
excluded. AssignedRoomType, BookingChanges and DaysInWaitingList are also
excluded under the initial-entry assumption.

Other disputed fields remain outside the starting lists. Current active
fields still depend on assumptions about their availability and possible
amendments. A field's presence in the CSV does not establish its availability
at prediction time.

Changing the prediction point requires a new availability assessment for
every affected input. Higher validation scores alone do not justify future
information.

## Independent task decisions

Each task owner may choose and justify:

- Algorithm and fixed baseline parameters.
- Task-specific population and exclusions.
- Model-appropriate preprocessing.
- Alternative features supported by prediction-time evidence.
- An alternative prediction or analysis point.
- Task-specific figures, interpretations and diagnostic checks.

Routine task decisions do not require separate permission from Members 1
and 2. Owners remain responsible for scientific justification and review.

Implement alternatives within the task's experiment configuration or module.
Do not silently overwrite shared defaults or another owner's results.
Changes to shared modules or policies are reviewed through a pull request.

Record each departure in the task report and saved experiment settings:

1. Previous setting and new setting.
2. Scientific reason and supporting evidence.
3. Prediction point and target meaning.
4. Input feature list and preprocessing.
5. Population rule and row counts before and after exclusions in each subset.
6. Data checksum, split version and feature-policy version.
7. Exact command, random seed and model parameters.
8. Consequences for comparison with other experiments.

Use an explicit experiment identifier for alternatives. A changed feature
list requires an identifiable feature-policy version. Record the actual
features as well as the version. Do not label modified inputs as the unchanged
shared policy.

Task-specific exclusions happen within existing subsets. Never move rows
between subsets. Experiments with different populations or prediction points
must be identified as different tasks or protocols when comparing results.

## Evaluation and result records

For Milestone 1, supervised baselines train on the training subset and report
validation metrics. Cross-validation, if added, must respect duplicate groups.
Do not evaluate final supervised test performance.

Classification records its positive class and decision threshold. Threshold
selection and class weighting must use development data only.

Clustering selects parameters using unsupervised criteria. Outcome labels may
describe already fixed clusters but must not select inputs, parameters or
cluster counts. Document sampling and seeds for expensive metrics and plots.

Save results through `src/evaluation/results.py`. Include task, model,
parameters, checksum, policy versions, evaluation subset, seed, metrics,
runtime, command and figure paths. Also record preprocessing settings,
population and row counts. Use null with a reason for unavailable metrics.

Clustering experiments using the shared training assignment record the
shared split version.

## Existing data evidence

The following are full-source descriptions from Member 1:

- Four unknown Children values and 24 unknown Country values.
- 8,131 Agent NULL and 75,641 Company NULL values mean not applicable.
- 25,902 excess identical rows in 5,846 repeated groups.
- 31,748 records belong to repeated groups.
- 331 zero-night records, all with zero ADR.
- 1,208 zero-ADR records and 167 complete zero-guest records.
- Their zero-night/zero-rate/zero-guest union contains 1,238 distinct records.
- ADR maximum is 5,400; the next-highest value is 510.
- 786 ADR values exceed the descriptive source 99th percentile of 228.
- Country has 166 known categories, Agent 224 and Company 208.
- Arrival dates span 2015-07-01 through 2017-08-31.

Training-only evidence records 785 zero-ADR rows, 224 zero-night rows,
23,172 canceled rows and no negative ADR. Fifty-three training records
are both zero-ADR and canceled.

These counts overlap. Do not add them as disjoint populations. Full-source
percentiles are descriptive and must not become fitted clipping thresholds.

Supporting evidence:

- `outputs/data/validation.json`
- `outputs/data/schema.csv`
- `outputs/data/review.json`
- `outputs/data/split_metadata.json`
- `outputs/data/preprocessing_checks.json`
- `outputs/eda/tables/review_flag_overlaps.csv`
- `reports/data_review.md`
- `reports/leakage_audit_m1.md`

No records have been removed by the shared loader or split implementation.
No baseline model results are claimed