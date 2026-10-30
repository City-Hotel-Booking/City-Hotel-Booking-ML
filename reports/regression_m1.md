# Milestone 1 — ADR regression baseline (Paula, Member 3)

## Objective and setup
Predict the recorded **ADR (Average Daily Rate)** for a City Hotel H2
booking, provisionally at the moment it enters the PMS.
Only source features marked `use` in the shared booking-time policy
were included. This is one **untuned** baseline, not the M2 comparison.

- Model: `LinearRegression()` (scikit-learn; no parameter search).
- Prediction point: `at_booking_entry` (starting_assumption).
- ADR population: `all_bookings_with_finite_adr`; retain canceled,
  zero-night, zero-ADR and extreme ADR records if the target is finite.
- No target clipping, target imputation, deduplication, or resplitting.
- Learned preprocessing: numeric median imputation, `log1p(LeadTime)`
  and StandardScaler; categorical missing marker and one-hot encoding
  with `handle_unknown='ignore'`, fitted only on training rows.
- Seed: `42` (used for reproducible residual plot sampling).
- Feature-policy version: `m1-draft-1`.
- Split version: `m1-group-stratified-v1`.
- Source SHA-256: `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.
- Exact command: `python main.py baseline --task regression`.
- Fit time: 3.1186 seconds (pipeline fit only).
- Runtime library versions: `python=3.11.9`, `numpy=2.4.6`, `pandas=3.1.0`, `scipy=1.17.1`, `scikit-learn=1.9.1`, `matplotlib=3.11.2`, `PyYAML=6.0.3`.

## Exact input features

1. `LeadTime`
2. `ArrivalDateYear`
3. `ArrivalDateMonth`
4. `ArrivalDateWeekNumber`
5. `ArrivalDateDayOfMonth`
6. `Adults`
7. `Children`
8. `Babies`
9. `Meal`
10. `MarketSegment`
11. `DistributionChannel`
12. `IsRepeatedGuest`
13. `PreviousCancellations`
14. `PreviousBookingsNotCanceled`
15. `ReservedRoomType`
16. `CustomerType`

## Population checks

| Subset | Original rows | Eligible rows | Excluded nonfinite ADR | Zero ADR | Negative ADR | Max ADR |
|---|---:|---:|---:|---:|---:|---:|
| Train | 55,532 | 55,532 | 0 | 785 | 0 | 451.5000 |
| Validation | 11,899 | 11,899 | 0 | 215 | 0 | 5,400.0000 |

## Results

| Metric | Training | Validation |
|---|---:|---:|
| MAE | 17.4007 | 18.1992 |
| RMSE | 24.4205 | 54.9850 |
| R2 | 0.5755 | 0.2689 |

- Validation figure: `outputs/regression/residuals.png`.
- Machine-readable result: `outputs/regression/metrics.json`.
- **Reserved test rows were not used for fitting, model selection, or evaluation.**
  The shared loader may read the complete source CSV before filtering by split.

## Interpretation and limitations

MAE is the average absolute ADR error in the recorded rate units;
RMSE penalizes larger misses more heavily; R² compares this model to
the validation-mean benchmark. Training versus validation errors can
indicate overfitting or data/feature mismatch; inspect the residual plot
for skew, systematic bias and extreme errors before drawing conclusions.
The all-finite-ADR policy intentionally includes unusual and canceled
bookings. The 5,400 ADR source outlier and zero-ADR records can strongly
influence RMSE and least squares. No source-based clipping was applied.
Validation/train RMSE ratio: 2.252;
validation/train MAE ratio: 1.046.
The validation maximum ADR is far outside the training range;
therefore high validation RMSE cannot be attributed to overfitting alone.
The maximum-ADR validation row contributes 78.81%
of the full validation sum of squared errors, but was not excluded.
Negative validation ADR predictions: 58.
This is a limitation of unconstrained least squares at booking entry.
This grouped *random* split cannot demonstrate performance on unseen
calendar periods or different hotels. Original recorded feature values
may also differ from what was known at booking entry.

## Leakage audit and review status

`ADR` is excluded from X. `IsCanceled`, `AssignedRoomType`,
`BookingChanges`, `DaysInWaitingList`, `ReservationStatus`, and
`ReservationStatusDate` are also excluded. Disputed fields remain out
under Member 2's current policy. The preprocessor and estimator were
fitted together on training rows only. Validation rows were scored
only after fitting. Reserved test rows were not used for fitting, model selection, or evaluation.
**Scientific sign-off remains required:** Zeina, Michael and Paula
must approve the ADR population, prediction-point interpretation and
feature policy in `reports/decisions.md`; current policy is provisional.

## Reproduction, ownership, and M2 continuation

Run `python main.py baseline --task regression` after placing the exact H2 source file
at `data/H2.csv`, installing `requirements.txt`, and validating the
frozen manifest/checksum with `python main.py prepare`.
Member 3 owns `src/regression/baseline.py`,
`tests/test_regression.py`, this report, and
`outputs/regression/metrics.json` / `residuals.png`.
Milestone 2 will compare at least three regression model families in
untuned and tuned configurations, use only validation/CV data for
selection, and evaluate the frozen test once after final selection.
