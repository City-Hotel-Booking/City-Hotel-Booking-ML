# H2 unusual records and duplicate review

The review identifies records that require interpretation before the team fixes its experimental population. Counts are source-wide descriptions. Flags can overlap and are not exclusion rules. No records were removed or imputed.

Dataset SHA-256: `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`

## Findings and recommendations

| Topic | Verified evidence | Proposed action and owner |
|---|---|---|
| Missing Children | 4 records | Keep unknown distinct from zero. Michael should select a justified strategy; any learned imputation must be fitted on training only. |
| Missing Country | 24 records | Propose a distinct unknown category; Michael reviews semantics and implementation. |
| Agent and Company NULL | Not-applicable categories, preserved as NULL | Retain as categorical IDs; do not treat their numeric codes as quantities. |
| Zero nights | 331 records; 331 also have zero ADR; 23 are canceled | Paula and Michael should clarify the meaning of a nightly rate for zero-night records before defining the ADR population. |
| Zero ADR | 1,208 records; 129 are canceled; 137 also have zero guests | Check booking context. A recorded zero does not alone establish an error or a complimentary stay. |
| Zero total guests | 167 complete guest-count records; 137 also have zero ADR; 67 also have zero nights | Investigate possible administrative or data-quality explanations. These are hypotheses, not confirmed causes. Coordinate any exclusion across task populations. |
| Extreme ADR | Maximum 5400; next highest 510 | Flag the extreme for Paula and Michael. Do not silently delete or cap it; decide a policy using domain evidence and training-only model selection. |
| Rare room P | 10 records; 10 canceled, 10 zero ADR and 10 zero guests | This overlap deserves review. It does not establish what the anonymous room code means or justify a rule for future bookings. |
| Identical records | 25,902 excess rows in 5,846 repeated groups | Preserve records pending agreement. Michael should assess duplicate-group overlap when establishing the split. |

## Extreme ADR record

Source row_id 8455 has ADR 5400, IsCanceled 1, 2 adults, 0 children and 0 babies, reserved room A, and segment Offline TA/TO. Its total nights are 1. These fields do not establish why the rate is extreme. A rate-entry error and an unusual genuine transaction are possibilities requiring external evidence.

## Overlaps and affected populations

The union of zero-night, zero-ADR and zero-guest records contains **1,238 distinct rows**. Adding their separate counts would double-count overlapping records. Exact mutually exclusive combinations are in `outputs/eda/tables/review_flag_combinations.csv`; pairwise overlaps are in `review_flag_overlaps.csv`.

There are 786 records strictly above the descriptive source ADR 99th percentile of 228. This threshold is for EDA only. It must not become a full-data learned clipping rule.

The babies-above-two flag identifies 2 records. This transparent review threshold is not a declared source-validity limit. Unknown guest counts are excluded from the zero-total-guests flag rather than assumed to be zero.

## Duplicate interpretation

There are 53,428 distinct raw records among 79,330 rows. Of these, 5,846 groups repeat; 31,748 rows belong to those groups. The largest group contains 180 identical records. The first occurrence is included in group membership counts but excluded from the excess-duplicate count.

No reliable booking identifier is supplied in the 31-column schema. Distinct bookings can share recorded attributes, so equality alone does not prove duplicate data entry. Conversely, repeated records distributed across training and evaluation can make performance appear too optimistic. Michael's existing provisional split keeps exact normalized groups together and preserves source positions. This work does not regenerate that split. Scientific approval and any task-specific population changes must be recorded without reassigning row IDs.

`duplicate_group_examples.csv` gives representative row IDs for the ten largest repeated groups. `duplicate_profiles_by_cancellation.csv` separates duplicated records by the observed outcome. These are descriptive summaries and do not select a split or model.

## Handoff packages

| Recipient | Evidence to review | Decision or next action |
|---|---|---|
| Michael | Validation, schema, this review, missingness, duplicate sizes and overlap tables | Review cleaning proposals and implications for row identity, split and task policies. |
| Paula | ADR distribution, extreme examples, cancellation-stratified ADR populations, review flags and overlaps | Agree the ADR population and proposed outlier handling with Michael. |
| Youssef | Cancellation balance, market summary, category frequencies and rare-category findings | Use class-aware metrics and review feature timing with Michael. |
| Aya | Numeric distributions/skew, category cardinalities, repeated groups and unusual-population findings | Agree segmentation purpose and feature/population policy, then build separate preprocessing. |
| Makarious | Source checksum, generated reports, manifest, all result tables/figures and exact commands | Integrate checked material and coordinate shared entry-point/dependency changes. |

These packages are prepared locally. They have not been sent, reviewed or approved by the recipients.

## Open scientific decisions

1. How should unknown Country and Children values be represented or imputed?
2. Should zero-night, zero-rate, zero-guest or canceled records belong to the ADR population, and why?
3. What is the justified response to the isolated extreme ADR?
4. How should repeated groups be handled in the split without assuming they are erroneous bookings?
5. Which fields are available at each approved prediction or clustering analysis point?
6. How should Agent/Company NULL categories and literal NA values be normalized consistently across EDA, modeling and duplicate grouping? Agent and Company remain excluded from active inputs. Members 1 and 2 must agree the representations and version the relevant policy; a change to group hashing also requires an explicitly reviewed split version rather than silently overwriting the frozen manifest.

Michael coordinates these decisions with Zeina and the model owners. Any learned imputation, scaling, encoding, dimensionality reduction or threshold selection belongs inside the agreed training protocol. This contribution creates no model or test-performance results and preserves the existing shared split assignments.

## Reproduction

Run `python main.py analyze` from the repository root. The command regenerates this report, `outputs/data/review.json` and all six review tables. Representative records retain source row IDs; the original CSV and all source rows remain unchanged. Findings can be checked against the independently implemented audit in `scripts/verify_outputs.py`.
