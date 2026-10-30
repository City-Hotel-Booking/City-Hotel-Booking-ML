# Member 3 regression technical note — Milestone 1

**Owner:** Paula (GitHub `PaulaAdelDev`) · **Task:** predict City Hotel H2 `ADR` · **Evidence date:** October 30, 2026.

This note supplements the two executable Markdown reports `reports/regression_m1.md` and `reports/regression_enhancement_m1.md`. The shared course specification controls requirements; the team's scientific policy is still provisional.

## Scientific question

At initial booking entry, how well can the initially recorded booking attributes estimate eventual recorded ADR? The dataset is a PMS-derived snapshot, so several fields can change after booking. Prediction-point validity must be reviewed, not assumed from CSV presence.

## Immutable input and evaluation contract

- Data: H2.csv only, 79,330 records × 31 source variables; SHA-256 `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.
- Grouped, cancellation-stratified random split: 55,532 train, 11,899 validation, 11,899 **untouched test**, seed 42.
- Shared interface: `get_task_data('regression', root, 'train'|'validation', config=config)` and `build_preprocessor('regression',config,model_family='linear'|'tree')`.
- Initial model: untuned LinearRegression with scaled numeric and one-hot features.
- Feature policy: 16 inputs, version m1-draft-1; `ADR`, cancellation outcome and post-booking fields excluded.
- Population: all finite ADR targets; no automatic cancellation/zero-night/zero-rate/outlier deletion.

## Reproduced baseline evidence

| Measure | Train | Validation |
|---|---:|---:|
| MAE | 17.4007 | 18.1992 |
| RMSE | 24.4205 | 54.9850 |
| R² | 0.5755 | 0.2689 |

The training maximum ADR was 451.50 while the validation maximum was 5,400.00. The extreme residual affects squared-loss metrics much more than average absolute error; high RMSE alone should not be labeled proof of overfitting. Negative predictions were observed in the baseline scatter and should be treated as a model limitation, not silently clipped without review.

## Improvement protocol

Preserve the mandatory baseline. Run two new fixed candidates on the **same** train/validation partitions: HistGradientBoostingRegressor (absolute-error loss) and RandomForestRegressor (100 trees, at least 5 samples per leaf). Use Member 2's tree preprocessing trained only on train. Compare on complete validation MAE, RMSE, R², plus training metrics and fit time. For diagnostics, show the maximum-rate row's squared-error contribution and a sensitivity-only one-row-omitted RMSE; neither can replace official scores. **Improved scores are pending a real H2 run.**

## Commands and handoff

```powershell
.\.venv\Scripts\python.exe main.py prepare
.\.venv\Scripts\python.exe main.py baseline --task regression
.\.venv\Scripts\python.exe -m src.regression.improve
.\.venv\Scripts\python.exe -m pytest tests/test_regression.py tests/test_regression_improve.py -v
```

Outputs are under `outputs/regression/`; reports under `reports/`. Do not commit raw H2, fitted models or virtual environments. GitHub feature-branch review is required; Michael verifies split and feature compliance, Zeina verifies the ADR-population decision, and Makarious integrates report evidence.

## Open issues

1. Booking-entry interpretation vs later PMS snapshots.
2. Whether zero-ADR, canceled, zero-night or ADR=5,400 records should have task-specific exclusion rules approved *before* new experiments (not retroactively score-optimize).
3. Temporal generalization is not estimated by the current random split.
4. Milestone 2 still needs 3+ regression models each baseline/tuned and a one-time untouched test evaluation.

### Reference

António, Almeida and Nunes. *Hotel booking demand datasets*, Data in Brief 22 (2019), 41–49. https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/
