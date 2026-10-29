# Shared experiment decisions

Status: **Member 2 draft, awaiting team review**. Updated **29 October 2026**.
The booking-time feature policy is m1-draft-1. Proposed settings are recorded
in config.yaml; they are not evidence that the team has approved or run them.
Record the reviewers and approval date here before calling a decision final.
Change the policy or split version and rerun affected experiments when a rule changes.

| Decision | Chosen option or current recommendation | Reason | Alternatives considered | Open concerns | Version | Date |
| --- | --- | --- | --- | --- | --- | --- |
| Prediction/analysis point | **Provisional:** when a booking first enters the PMS, for all three tasks. | Gives a concrete rule for checking when each input exists. | Before arrival; after check-in; retrospective clustering. | H2 is a later snapshot, so some recorded values may have changed after entry. Team approval needed. | m1-draft-1 | 2026-10-27 |
| ADR regression population | **Open:** examine all rows with a usable ADR target and document exclusions before training. | Canceled, zero-night, zero-ADR, and extreme-rate rows can have different meanings. | Drop all canceled or zero-rate rows; cap or remove outliers. | Training-only evidence below; Members 1, 2, and 3 must agree on the target population. | pending | 2026-10-27 |
| Import cleaning | **Implemented, pending review:** trim outer whitespace; treat blank/NA/N/A/NAN as unknown; preserve `Agent`/`Company` `NULL` as not applicable in modeling. Keep source row positions. | Member 1 found 4 unknown `Children`, 24 unknown `Country`, 8,131 not-applicable `Agent`, and 75,641 not-applicable `Company`. | Replace all NULL with missing; assume missing Children means zero. | The frozen split still groups after trim/blank/NULL normalization; its group definition must be reviewed before changing. No row removals are approved. | m1-draft-1 | 2026-10-29 |
| Duplicate policy | **Requested:** keep all rows; group identical records on all 31 normalized source fields for splitting. | An identical record may still be a distinct booking, but duplicates across splits can inflate validation scores. | Delete duplicates; split rows independently. | Without a booking ID, exact matching cannot find every repeated real booking. Record the group definition. | m1-draft-1 | 2026-10-27 |
| Supervised split | **Implemented, pending team approval:** seed 42, approximately 70/15/15, stratified by IsCanceled and kept intact by duplicate group. | Shares one reproducible assignment across regression and classification while limiting duplicate leakage. | Chronological arrival-date holdout; ordinary stratified random split. | Exact ratios can shift with indivisible groups. A random split does not measure 2015-2017 time drift; report this limitation. | m1-group-stratified-v1 | 2026-10-27 |
| Task feature lists | **Provisional:** use only columns marked use in src/features/policy.py; keep disputed columns out. | A column is eligible only when its value is available at the selected point. | Use every non-target column; add conditional fields without an audit. | Members 1 and 3-5 must review the timing and meaning of each disputed field. | m1-draft-1 | 2026-10-27 |
| Shared preprocessing | **Implemented, pending team approval:** fresh unfitted ColumnTransformer per task; numeric parsing and training-fold median imputation; scaling for linear/distance models, unscaled tree option; `log1p` on nonnegative LeadTime for scaled models; categorical missing marker and one-hot with unknown-category ignore. Clustering has dense scaled output. | Member 1 reports LeadTime skew 1.39; learned statistics/categories must come from training folds. Real H2 checks pass for train-only fitting and unseen categories. | Pre-fit on all rows; treat Agent/Company IDs as numbers; drop missing rows. | Agent/Company/Country remain excluded by the active policy; rare-category one-hot is dormant. Model owners must record `model_family` and preprocessing settings. | m1-draft-1 | 2026-10-29 |
| ADR and cancellation cross-use | **Provisional:** IsCanceled is excluded from ADR inputs; recorded ADR is held out of cancellation inputs. | Cancellation is a later outcome; recorded ADR may differ from the initial quote. | Use recorded ADR as an initial price proxy. | ADR could be allowed for cancellation only if a same-meaning booking-time quote is verified. | m1-draft-1 | 2026-10-27 |
| High-cardinality fields | **Open:** Agent, Company, and Country stay out until timing, missing semantics, and encoding are agreed. | They can have many levels and missing values may mean different things. | One-hot with rare-category grouping; ordinal encoding; training-fold target encoding. | Any learned encoding must fit within training folds. Target encoding needs extra care against leakage. | pending | 2026-10-27 |
| Clustering population and purpose | **Open:** propose booking-time segments without outcome or final-status labels. | Clusters should reflect booking attributes rather than known answers. | Retrospective customer segmentation; risk-led segmentation. | Agree whether canceled bookings and prior-outcome history are in scope. Do not use outcome labels for parameter choice. | pending | 2026-10-27 |
| Final supervised test set | **Requested:** freeze test membership and reserve model evaluation for Milestone 2. | Prevents tuning or feature choices based on final performance. | Repeatedly check test scores during Milestone 1. | Split metadata will include one-time aggregate class balance for integrity checking; it must not guide model selection. | m1-group-stratified-v1 | 2026-10-27 |
| Baseline algorithms | **Open:** Members 3-5 choose one untuned model or algorithm each after shared decisions. | Baselines need the agreed population, features, and preprocessing. | Pick models before the data policy is settled. | Record settings, validation results, and exact commands once implemented. | pending | 2026-10-27 |

## Approval record

- Team reviewers: **pending**.
- Approval date: **pending**.
- Source data quality counts: **Member 1's full-source evidence is recorded below**;
  treatments and prediction-time assumptions still await scientific approval.
- A change to prediction time, ADR population, split policy, or active columns
  requires a new version and rerun of affected experiments.

## Observed split evidence

The team-confirmed H2 source was run locally on 27 October 2026. Its raw-file
SHA-256 is 6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4.
The generated manifest has 79,330 unique zero-based row IDs and only the
row_id and split columns. The assignments are 55,532 train, 11,899 validation,
and 11,899 test. The canceled-class share is 0.4172729237 in train and
0.4172619548 in each of validation and test. There are 5,846 groups of
identical normalized source rows; an independent hash check found zero groups
spanning splits. These figures describe the current draft split, not team
approval or final test performance. The machine-readable evidence is in
outputs/data/split_metadata.json and outputs/data/split_manifest.csv.

## Training-only ADR population evidence

Using the frozen manifest to select only the 55,532 training rows from the
confirmed local H2 file, all ADR entries parse numerically. There are 785
zero-ADR rows, zero negative-ADR rows, 224 zero-night rows, and 23,172 canceled
rows. Fifty-three rows are both zero-ADR and canceled, so these groups must
not be added as disjoint counts. No records were removed. Members 1, 2, and 3
must decide whether any of these groups belong in the regression target
population before Member 3's baseline is accepted.

## Member 1 evidence reconciled on 29 October 2026

These are **full-source descriptions**, distinct from the training-only counts
above. The source has 79,330 rows, 31 official columns, no blocking validation
issues, and SHA-256 `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.
Member 1 reports 25,902 excess identical rows in 5,846 groups (31,748 group
members); the split keeps every group intact. There are 331 zero-night rows,
1,208 zero-ADR rows, 167 complete zero-guest rows, and 1,238 distinct rows in
their union. All 331 zero-night rows have zero ADR. ADR ranges from 0 to 5,400;
the next-highest rate is 510. The 786 values above the *source-wide* 99th
percentile of 228 are descriptive only; do not use that percentile as a fitted
clipping threshold. The 4 missing Children and 24 missing Country entries are
unknown; 8,131 Agent and 75,641 Company NULL entries mean not applicable.
All figures come from `outputs/data/validation.json`, `outputs/data/review.json`,
`outputs/data/schema.csv`, and `reports/data_review.md`; overlap details are in
`outputs/eda/tables/review_flag_overlaps.csv`.

Member 1's `read_dataset` supplies raw tokens for the split and a separate
normalized analysis view. The split now uses its raw-token loader but retains
the frozen trim/blank/NULL group rule. Regenerating the manifest gave an
identical `row_id` and `split` table. The shared feature lists remain 16 ADR,
16 cancellation, and 11 clustering inputs, so `feature_policy_version` stays
`m1-draft-1`. A change to active columns requires a new feature-policy version;
a change to group definition, source identity/order, seed, ratios, or assignment
requires a reviewed split-version change. Task-specific row exclusions stay
within each frozen split and need a documented population rule.

## Pending decisions before Milestone 1

The dates below are proposed team response dates, not recorded approvals.
If a response is late, use the stated **provisional** default for development
and mark the result provisional; do not silently call a decision approved.

| Decision | Member 2 recommendation | Approvers | Status | Proposed response date | Provisional default if unanswered |
| --- | --- | --- | --- | --- | --- |
| Prediction point and mutable fields | Predict at initial PMS entry; keep later and disputed fields out until initial-value evidence exists. | Zeina, Paula, Youssef, Aya, Michael | Open | 2026-10-31 | Keep `m1-draft-1` active lists; label all baselines provisional. |
| ADR target population | Agree meaning of ADR; inspect zero nights, zero ADR, cancellation and row 8455 before any exclusions. | Zeina, Paula, Michael | Open | 2026-10-31 | Train a clearly labeled provisional baseline on all rows with finite ADR within each frozen split; no silent deletions or clipping. |
| Agent/Company and NA normalization | Preserve not-applicable NULL in modeling; retain frozen grouping until group-hash impact is reviewed. | Zeina, Michael | Open | 2026-11-01 | Keep Agent/Company excluded and the existing manifest unchanged. |
| Duplicate interpretation and split | Keep all records and existing grouped, stratified assignment; document temporal-drift limitation. | Zeina, Michael, Paula, Youssef | Open | 2026-11-01 | Keep the frozen `m1-group-stratified-v1` manifest. |
| Clustering purpose/population | Use booking-time segments on training rows; exclude outcome and final-status fields. | Aya, Michael | Open | 2026-11-02 | Use the current 11-input policy and all training rows; mark cluster meaning provisional. |
| Baseline choices and result schema | Owners select one untuned method each; record data and policy versions and validation results. | Paula, Youssef, Aya, Makarious | Open | 2026-11-03 | Do not invent a model or metric; report the missing baseline explicitly. |

Approval record: reviewer names, decision, date, and resulting version are
**pending**. Record them here when the owners respond.
