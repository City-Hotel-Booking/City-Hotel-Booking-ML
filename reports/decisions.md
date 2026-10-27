# Shared experiment decisions

Status: **Member 2 draft, awaiting team review**. Date: **27 October 2026**.
The booking-time feature policy is m1-draft-1. Proposed settings are recorded
in config.yaml; they are not evidence that the team has approved or run them.
Record the reviewers and approval date here before calling a decision final.
Change the policy or split version and rerun affected experiments when a rule changes.

| Decision | Chosen option or current recommendation | Reason | Alternatives considered | Open concerns | Version | Date |
| --- | --- | --- | --- | --- | --- | --- |
| Prediction/analysis point | **Provisional:** when a booking first enters the PMS, for all three tasks. | Gives a concrete rule for checking when each input exists. | Before arrival; after check-in; retrospective clustering. | H2 is a later snapshot, so some recorded values may have changed after entry. Team approval needed. | m1-draft-1 | 2026-10-27 |
| ADR regression population | **Open:** examine all rows with a usable ADR target and document exclusions before training. | Canceled, zero-night, zero-ADR, and extreme-rate rows can have different meanings. | Drop all canceled or zero-rate rows; cap or remove outliers. | Training-only evidence below; Members 1, 2, and 3 must agree on the target population. | pending | 2026-10-27 |
| Import cleaning | **Implemented for preprocessing, pending review:** convert literal NULL, blank, and NA strings to missing, and strip padded text without changing source row positions. | H2 has four literal NA values in Children; numeric parsing otherwise fails. | Leave text unchanged; replace missing with a numeric sentinel. | Split grouping currently normalizes blank/NULL; Member 1 should confirm NA semantics and whether split normalization needs a new version. Distinguish unknown from not-applicable Agent/Company values. No row removals are approved. | m1-draft-1 | 2026-10-27 |
| Duplicate policy | **Requested:** keep all rows; group identical records on all 31 normalized source fields for splitting. | An identical record may still be a distinct booking, but duplicates across splits can inflate validation scores. | Delete duplicates; split rows independently. | Without a booking ID, exact matching cannot find every repeated real booking. Record the group definition. | m1-draft-1 | 2026-10-27 |
| Supervised split | **Implemented, pending team approval:** seed 42, approximately 70/15/15, stratified by IsCanceled and kept intact by duplicate group. | Shares one reproducible assignment across regression and classification while limiting duplicate leakage. | Chronological arrival-date holdout; ordinary stratified random split. | Exact ratios can shift with indivisible groups. A random split does not measure 2015-2017 time drift; report this limitation. | m1-group-stratified-v1 | 2026-10-27 |
| Task feature lists | **Provisional:** use only columns marked use in src/features/policy.py; keep disputed columns out. | A column is eligible only when its value is available at the selected point. | Use every non-target column; add conditional fields without an audit. | Members 1 and 3-5 must review the timing and meaning of each disputed field. | m1-draft-1 | 2026-10-27 |
| Shared preprocessing | **Implemented, pending team approval:** create an unfitted ColumnTransformer per task; numeric parsing, median imputation and standard scaling; categorical cleaning, missing marker and one-hot with unknown-category ignore. Fit it only inside the model Pipeline on each training fold. | Learned statistics and categories must come only from the training fold; unfamiliar categories must not stop validation or inference. | Pre-fit a transformer on all rows; ordinal encode nominal IDs; drop missing rows. | H2 has four literal `NA` entries in Children; the transformer treats these as missing. Member 1 should confirm import semantics. High-cardinality inputs remain excluded by policy. | m1-draft-1 | 2026-10-27 |
| ADR and cancellation cross-use | **Provisional:** IsCanceled is excluded from ADR inputs; recorded ADR is held out of cancellation inputs. | Cancellation is a later outcome; recorded ADR may differ from the initial quote. | Use recorded ADR as an initial price proxy. | ADR could be allowed for cancellation only if a same-meaning booking-time quote is verified. | m1-draft-1 | 2026-10-27 |
| High-cardinality fields | **Open:** Agent, Company, and Country stay out until timing, missing semantics, and encoding are agreed. | They can have many levels and missing values may mean different things. | One-hot with rare-category grouping; ordinal encoding; training-fold target encoding. | Any learned encoding must fit within training folds. Target encoding needs extra care against leakage. | pending | 2026-10-27 |
| Clustering population and purpose | **Open:** propose booking-time segments without outcome or final-status labels. | Clusters should reflect booking attributes rather than known answers. | Retrospective customer segmentation; risk-led segmentation. | Agree whether canceled bookings and prior-outcome history are in scope. Do not use outcome labels for parameter choice. | pending | 2026-10-27 |
| Final supervised test set | **Requested:** freeze test membership and reserve model evaluation for Milestone 2. | Prevents tuning or feature choices based on final performance. | Repeatedly check test scores during Milestone 1. | Split metadata will include one-time aggregate class balance for integrity checking; it must not guide model selection. | m1-group-stratified-v1 | 2026-10-27 |
| Baseline algorithms | **Open:** Members 3-5 choose one untuned model or algorithm each after shared decisions. | Baselines need the agreed population, features, and preprocessing. | Pick models before the data policy is settled. | Record settings, validation results, and exact commands once implemented. | pending | 2026-10-27 |

## Approval record

- Team reviewers: **pending**.
- Approval date: **pending**.
- Source data quality counts: **training-only initial checks recorded below**;
  Member 1's full validation and treatment decisions are still pending.
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
