# Milestone 1 — ADR performance enhancement (Paula, Member 3)

## Scope and integrity

This is an **additional fixed-configuration comparison**; it does not replace
the mandatory untuned linear baseline in `reports/regression_m1.md`.
The original `m1-group-stratified-v1` source assignments, `m1-draft-1`
feature policy and all-finite-ADR population were retained.
Reserved test rows were **not used for fitting, model selection, or evaluation**.
The shared loader reads the full CSV prior to selecting split rows.

- Source checksum: `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.
- Training/validation rows: 55,532 / 11,899.
- Exact input features: `LeadTime, ArrivalDateYear, ArrivalDateMonth, ArrivalDateWeekNumber, ArrivalDateDayOfMonth, Adults, Children, Babies, Meal, MarketSegment, DistributionChannel, IsRepeatedGuest, PreviousCancellations, PreviousBookingsNotCanceled, ReservedRoomType, CustomerType`.
- Prediction point: booking entry (provisional pending team approval).
- Seed: `42`; run: `python -m src.regression.improve`.
- Preprocessing: Member 2's unfitted `tree` builder, fitted separately
  with each model on training rows only; categorical unknowns ignored.
- Models: original LinearRegression baseline; fixed HistGradientBoostingRegressor
  with absolute-error loss and `early_stopping=False` (no internal holdout);
  RandomForestRegressor with 100 trees and
  minimum five training samples per leaf (predeclared, not searched).

## Runtime versions and reproducibility

- Runtime versions: `python=3.11.9`, `numpy=2.4.6`, `pandas=3.1.0`, `scipy=1.17.1`, `scikit-learn=1.9.1`, `matplotlib=3.11.2`, `PyYAML=6.0.3`.
- `early_stopping=False` prevents a second, non-group-aware HGB internal split.
- All candidate transformations are fitted solely on the frozen training rows.

## Full validation comparison

| Model | Train MAE | Train RMSE | Train R² | Validation MAE | Validation RMSE | Validation R² | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| `linear_regression` | 17.4007 | 24.4205 | 0.5755 | 18.1992 | 54.9850 | 0.2689 | 3.1186 |
| `hist_gradient_boosting_mae` | 11.8819 | 19.8696 | 0.7190 | 15.0025 | 53.7516 | 0.3013 | 3.6552 |
| `random_forest_regularized` | 5.7381 | 12.0968 | 0.8958 | 12.5223 | 52.8052 | 0.3257 | 6.7181 |

Model choice here uses **full-validation MAE** (RMSE breaks ties);
test results and outlier-excluded diagnostics do not enter selection.
The lowest-MAE experimental candidate is `random_forest_regularized` (validation
MAE 12.5223; RMSE 52.8052; R² 0.3257).
This selection is preliminary and is not a tuned/deployed M2 model.

## Extreme validation ADR diagnostic (not model selection)

- Source row ID: 8455.
- Recorded ADR: 5,400.0000.
- Every model was evaluated on this row for the primary metrics.
- The next table measures how much this single row contributes to each
  model's total validation squared error. The last column is a diagnostic
  sensitivity estimate only; do not substitute it for official RMSE.

| Model | Prediction on largest ADR | Squared-error share | RMSE without largest row (diagnostic only) | Negative predictions |
|---|---:|---:|---:|---:|
| `linear_regression` | 75.3506 | 78.8103% | 25.3119 | 58 |
| `hist_gradient_boosting_mae` | 76.2516 | 82.4406% | 22.5250 | 35 |
| `random_forest_regularized` | 76.4305 | 85.4165% | 20.1663 | 0 |

## Evidence and interpretation

- Metric table: `outputs/regression/comparison.csv`.
- Structured comparison: `outputs/regression/comparison.json`.
- Model comparison figure: `outputs/regression/comparison.png`.
- HGB-only residuals: `outputs/regression/hist_gradient_boosting_mae_residuals.png`.
- Random Forest-only residuals: `outputs/regression/random_forest_regularized_residuals.png`.
- Backward-compatible enhanced residual alias: `outputs/regression/enhanced_residuals.png` (shows `random_forest_regularized`).
- Individual candidate results: `outputs/regression/*_metrics.json`.
- Large ADR events can dominate RMSE even when typical errors improve.
  The score gap alone does not prove ordinary overfitting.
- The extreme record must not be removed post hoc to improve the score.
- The maximum training ADR is 451.50, compared
  with the maximum validation ADR of 5400.00.
  Tree ensembles cannot reliably infer a rate outside the observed
  training target range; their predictions are constrained
  by observed training patterns, not guaranteed to extrapolate.
- Original source values may change after booking entry; feature-timing
  assumptions remain provisional pending Zeina/Michael review.

## Next actions and reproducibility

1. Preserve original baseline metrics and report for Milestone 1.
2. Have Michael and Zeina review ADR population/feature timing and the
   extreme value's validity; document approval in `reports/decisions.md`.
3. For M2, tune at least three families using train-only group-aware CV
   where suitable; use the frozen validation set for selection and
   evaluate the reserved test once after model choices are frozen.
4. Reproduce via `python -m src.regression.improve` after running the linear baseline.
5. Record the source SHA, tested package versions, commit SHA and actual
   author contributions in Makarious's integrated report.
