# Cancellation classification baseline

Status: provisional under the current shared draft scientific policy.

## Purpose and prediction point

Predict whether a booking will be canceled.
The target is IsCanceled; cancellation is the positive class, labeled 1.
The starting prediction point is initial booking entry into the hotel PMS.

H2 is a later snapshot. Availability of the active inputs at booking entry
remains an assumption requiring the review recorded in reports/decisions.md.
This experiment does not establish scientific approval of that assumption.

## Population and frozen assignments

Use all frozen training and validation rows with valid binary targets.
Invalid targets stop the experiment rather than being silently excluded.
No task-specific filtering, clipping, duplicate removal or resampling is used.
Canceled, zero-rate, zero-night and extreme-rate records are retained.

| Subset | Rows | Not canceled | Canceled |
| --- | ---: | ---: | ---: |
| Train | 55532 | 32360 | 23172 |
| Validation | 11899 | 6934 | 4965 |

Source positions and frozen subset assignments are preserved.
The task runner requests only train and validation through get_task_data.
Final test performance is not evaluated.

Source SHA-256: `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.

Split version: `m1-group-stratified-v1`.

Feature-policy version: `m1-draft-1`.

## Baseline algorithm and fixed settings

Model: LogisticRegression.

Fixed parameters:

{
  "C": 1.0,
  "class_weight": null,
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": 0.0,
  "max_iter": 2000,
  "n_jobs": null,
  "penalty": "deprecated",
  "random_state": 42,
  "solver": "lbfgs",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}

Logistic regression supplies a simple probabilistic baseline for a binary
target. The shared linear preprocessing supports its numeric scale assumptions.
C=1.0 and the lbfgs solver are fixed starting choices. max_iter=2000 is an
optimization limit. No hyperparameter search or class weighting is performed.

Decision threshold: 0.5.
A probability equal to the threshold is classified as cancellation.
The threshold was fixed before evaluation and was not optimized on validation.

Training time including preprocessing: 2.040999 seconds.
Optimizer iterations: [97].

## Inputs and preprocessing

Exact input features:

- `LeadTime`
- `ArrivalDateYear`
- `ArrivalDateMonth`
- `ArrivalDateWeekNumber`
- `ArrivalDateDayOfMonth`
- `Adults`
- `Children`
- `Babies`
- `Meal`
- `MarketSegment`
- `DistributionChannel`
- `IsRepeatedGuest`
- `PreviousCancellations`
- `PreviousBookingsNotCanceled`
- `ReservedRoomType`
- `CustomerType`

A fresh shared linear preprocessor and LogisticRegression are fitted together
inside one scikit-learn Pipeline on training rows only.

Numeric processing parses values, imputes training medians, applies the
configured log1p transformation to LeadTime and standardizes numeric features.
Categorical processing uses a missing marker and one-hot encoding.
Unseen validation categories are ignored without refitting.

Recorded preprocessing configuration:

{
  "categorical_encoding": "one_hot",
  "categorical_imputation": "constant_missing",
  "high_cardinality_encoding": "one_hot_rare",
  "high_cardinality_min_frequency": 20,
  "numeric_imputation": "median",
  "numeric_scaling": "standard",
  "skewed_numeric_features": [
    "LeadTime"
  ],
  "unknown_categories": "ignore"
}

IsCanceled, ADR, ReservationStatus, ReservationStatusDate, AssignedRoomType,
BookingChanges and DaysInWaitingList are absent from the active inputs.
No learned preprocessing is fitted on validation or final test rows.

## Validation results

| Metric | Value |
| --- | ---: |
| accuracy | 0.575258 |
| precision | 0.463909 |
| recall | 0.115206 |
| f1 | 0.184576 |
| roc_auc | 0.639443 |
| average_precision | 0.519296 |

Average precision is calculated using average_precision_score.
It is not described as trapezoidal area under the precision-recall curve.
Undefined metrics are stored as null with explanations.

Metric notes:

- All requested metrics are available.

Confusion matrix, with true labels on rows and predicted labels on columns:

| True / Predicted | Not canceled (0) | Canceled (1) |
| --- | ---: | ---: |
| Not canceled (0) | 6273 | 661 |
| Canceled (1) | 4393 | 572 |

[Metrics and settings](../outputs/classification/metrics.json)

[Confusion matrix figure](../outputs/classification/confusion_matrix.png)

## Interpretation and limitations

At the fixed threshold, the model correctly identifies 572 canceled bookings
and misses 4393. It flags 661 noncanceled bookings as cancellations and
correctly identifies 6273 noncanceled bookings.

Precision describes the reliability of cancellation alerts. Recall describes
the fraction of cancellations identified. F1 summarizes their balance.
ROC-AUC and average precision evaluate probability ranking independently of
the fixed threshold. Accuracy should be interpreted alongside class balance.

These are development results from one untuned model. No superiority claim
over other classifiers is supported by this experiment alone.

The grouped random split limits exact-duplicate overlap, but it does not
measure future-year drift or identify all repeated real bookings without a
booking identifier. Mutable feature timing and scientific approvals remain
open as documented in the shared decision tracker.

The validation majority-class reference achieves
58.27%
accuracy, compared with 57.53%
for this logistic regression baseline.
At threshold 0.5, cancellation recall is
11.52%.

The current model is therefore a weak initial baseline. These results
do not support deployment or a claim of improvement over the majority
reference. Future experiments should examine recall, precision, F1 and
ranking metrics together, while preserving this baseline for comparison.

## Reproduction and environment

Run from the project root:

    & C:\Users\youss\Downloads\City-Hotel-Booking-ML-main\City-Hotel-Booking-ML-main\.venv\Scripts\python.exe main.py baseline --task classification

Python: 3.12.10.
scikit-learn: 1.9.1.
pandas: 3.1.0.
NumPy: 2.5.4.

The JSON record contains the effective configuration, parameters, row counts,
source identity, policy versions and random seed.
Runtime can vary across machines.

The runner generates metrics.json, confusion_matrix.png and this report.
The report records experiment execution; test-suite status is verified
separately using pytest.

## Milestone 2 plan

Compare additional algorithms and tuned configurations using development data.
Any cross-validation must preserve duplicate groups and fit preprocessing
inside each training fold. Investigate class weighting, threshold selection,
PCA/LDA comparisons and an ensemble or boosting classifier.

After model selection, evaluate the reserved test subset and save the final
inference pipeline for application use. These extensions are planned work.
