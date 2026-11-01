# Member 3 Regression Technical Note — Milestone 1

**Owner:** Paula (GitHub: `PaulaAdelDev`)  
**Task:** City Hotel H2 — Average Daily Rate (ADR) Prediction  
**Evidence date:** October 30, 2026  
**Status:** Baseline and enhancement experiments completed; scientific policy review pending.

This document supplements the two regression reports:

- `reports/regression_m1.md`
- `reports/regression_enhancement_m1.md`

The course project specification defines the requirements. The team's shared prediction-point and feature-availability decisions remain provisional.

## 1. Scientific Question

At the time a booking first enters the hotel's Property Management System (PMS), how accurately can the available booking attributes predict its recorded Average Daily Rate (ADR)?

Because the dataset represents a PMS-derived snapshot, some recorded attributes may have changed after booking creation. Their availability at the assumed prediction point must therefore be scientifically reviewed.

## 2. Dataset and Evaluation Contract

- **Dataset:** City Hotel H2 only.
- **Source records:** 79,330.
- **Source variables:** 31.
- **SHA-256:** `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`
- **Split method:** Grouped, cancellation-stratified random split.
- **Training records:** 55,532.
- **Validation records:** 11,899.
- **Reserved test records:** 11,899.
- **Random seed:** 42.
- **Feature-policy version:** `m1-draft-1`.
- **Split version:** `m1-group-stratified-v1`.
- **Regression inputs:** 16 features selected through the shared feature policy.

The shared implementation uses `get_task_data()` for selecting the frozen training and validation assignments and `build_preprocessor()` for constructing unfitted task-specific preprocessing pipelines.

The initial population includes all bookings with finite ADR targets, including canceled bookings, zero-night bookings, zero-ADR records and extreme ADR observations.

No automatic target clipping, outlier removal, duplicate deletion or resplitting was performed.

The shared loader reads the complete H2 source CSV before selecting split rows. Reserved test rows were not used for fitting, model selection, or evaluation.

## 3. Stage 1 — Original Regression Baseline

### Model and preprocessing

The initial experiment used an untuned `LinearRegression()` estimator.

Preprocessing consisted of:

- Median imputation for numerical features.
- `log1p` transformation of `LeadTime`.
- Standard scaling of numerical features.
- Constant missing-value handling for categorical features.
- One-hot encoding with unknown-category handling.

All learned preprocessing and the regression estimator were fitted exclusively on training records through a scikit-learn Pipeline.

### Baseline results

| Metric | Training | Validation |
|---|---:|---:|
| MAE | 17.4007 | 18.1992 |
| RMSE | 24.4205 | 54.9850 |
| R² | 0.5755 | 0.2689 |

The validation MAE was relatively close to the training MAE, whereas validation RMSE increased substantially.

The largest training ADR was 451.50, compared with 5,400.00 in validation. This extreme validation observation contributed approximately 78.81% of the baseline's total validation squared error.

The baseline also generated 58 negative validation ADR predictions.

These observations demonstrate limitations involving extreme rates, extrapolation and unconstrained linear predictions. The RMSE gap should not automatically be interpreted as evidence of overfitting.

**Baseline evidence:**

- `outputs/regression/metrics.json`
- `outputs/regression/residuals.png`
- `reports/regression_m1.md`

## 4. Stage 2 — Regression Performance Enhancement

The original baseline was preserved.

Two additional fixed-configuration models were evaluated using the same training and validation partitions, feature policy and target population.

### Models

**Histogram Gradient Boosting Regressor**

- Loss function: `absolute_error`
- Random seed: 42
- Early stopping: `False` (no internal validation holdout).
- Purpose: Evaluate nonlinear relationships while emphasizing absolute prediction error.

**Random Forest Regressor**

- Number of estimators: 100
- Minimum samples per leaf: 5
- Random seed: 42
- Parallel execution: `n_jobs=-1`
- Purpose: Evaluate nonlinear relationships through an ensemble of decision trees.

Both models used Member 2's tree preprocessing configuration, fitted exclusively on training records.

These were fixed experimental configurations, not results of systematic hyperparameter optimization.

### Enhanced regression results

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Linear Regression | 18.1992 | 54.9850 | 0.2689 |
| Histogram Gradient Boosting | 15.0025 | 53.7516 | 0.3013 |
| Random Forest | **12.5223** | **52.8052** | **0.3257** |

Random Forest achieved the lowest validation MAE among the three evaluated configurations.

Relative to Linear Regression, Random Forest reduced validation MAE by approximately **31.2%**.

The improvement in RMSE was smaller, partly because of the extreme validation ADR observation.

Random Forest is therefore the preferred preliminary experimental candidate under the stated full-validation MAE selection criterion. It has not been selected as the final tuned or deployed model.

## 5. Extreme-Value Analysis

The validation dataset contains an ADR observation of 5,400.00 at source row ID 8455.

All three models were evaluated against this observation in their official validation metrics.

| Model | Percentage of validation squared error attributable to the extreme observation |
|---|---:|
| Linear Regression | 78.81% |
| Histogram Gradient Boosting | 82.44% |
| Random Forest | 85.42% |

The larger percentage for Random Forest does not mean its total error was worse. Random Forest reduced errors elsewhere, making the extreme observation account for a greater proportion of the remaining squared error.

Sensitivity analyses excluding this single observation were diagnostic only. They were not substituted for official metrics and were not used for model selection.

## 6. Leakage Prevention and Experimental Integrity

The regression experiments excluded the target `ADR` and information considered unavailable or unsafe at initial booking entry, including:

- `IsCanceled`
- `AssignedRoomType`
- `BookingChanges`
- `DaysInWaitingList`
- `ReservationStatus`
- `ReservationStatusDate`

All inputs were obtained from the shared feature policy.

Learned preprocessing and estimators were fitted exclusively using training records.

The frozen training and validation assignments were preserved across experiments.

Reserved test rows were not used for fitting, model selection, or evaluation. The shared loader reads the full source CSV before selecting the requested split.

**Scientific approval is still required** for the regression population, prediction-point interpretation, and feature-availability policy. These decisions must not be presented as already approved.

## 7. Testing and Reproducibility

The focused regression suite completed successfully with 18 passing tests.

The full repository test suite subsequently reported:

**91 tests passed and 2 subtests passed.**

The verified local environment used Python 3.11.9, NumPy 2.4.6, pandas 3.1.0, SciPy 1.17.1, scikit-learn 1.9.1, Matplotlib 3.11.2, and PyYAML 6.0.3.

### Reproduction commands

```powershell
.\.venv\Scripts\python.exe main.py prepare
.\.venv\Scripts\python.exe main.py baseline --task regression
.\.venv\Scripts\python.exe -m src.regression.improve
.\.venv\Scripts\python.exe -m pytest tests/test_regression.py tests/test_regression_improve.py -v
.\.venv\Scripts\python.exe -m pytest -q
```

### Generated evidence

- `outputs/regression/metrics.json`
- `outputs/regression/residuals.png`
- `outputs/regression/comparison.json`
- `outputs/regression/comparison.csv`
- `outputs/regression/comparison.png`
- `outputs/regression/enhanced_residuals.png` (Random Forest alias)
- `outputs/regression/hist_gradient_boosting_mae_residuals.png` (HGB)
- `outputs/regression/random_forest_regularized_residuals.png` (Random Forest)
- `outputs/regression/hist_gradient_boosting_mae_metrics.json`
- `outputs/regression/random_forest_regularized_metrics.json`

## 8. Remaining Scientific Decisions

The following issues require review rather than assumptions of approval:

1. Whether the recorded attributes truly represent information available at booking entry.
2. Whether zero-ADR, canceled, zero-night and extreme-rate records should remain in the regression population.
3. Whether the extreme 5,400 ADR observation represents a valid unusual booking rate or a source-data error.
4. Whether the current random split sufficiently represents realistic temporal generalization.

Any approved changes to the population, features or prediction point must be documented and evaluated reproducibly.

## 9. Milestone 2 Continuation

Milestone 2 requires at least three regression model families, each evaluated in baseline and tuned configurations.

Further work includes documented hyperparameter optimization, appropriate training-only cross-validation, model comparisons, and final test evaluation after the model-selection process is frozen.

The final system must also support saved inference pipelines.

## 10. Team Handoff

Member 3 supplies the regression implementations, tests, verified metrics, generated figures and experiment reports.

Michael reviews the shared split, preprocessing and regression feature policy.

Zeina reviews the ADR population and unusual-value interpretation.

Makarious integrates the verified regression findings into the overall Milestone 1 progress report.

Scientific approvals and member contributions should be recorded by the responsible people after their actual review.

## Reference

António, N., Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49.

https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/