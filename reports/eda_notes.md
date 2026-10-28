# H2 data validation and EDA handoff

This report records the H2 validation and descriptive EDA for Zeina's Member 1 work package. It supplies evidence for the team's preparation and baseline experiments. Cleaning, split, feature and population decisions still require scientific review.

## Scope and source identity

The source contains **79,330 bookings and 31 source variables**. Arrival dates range from 2015-07-01 through 2017-08-31. Validation status: `passed_with_findings`. There are 0 blocking validation issues; see `outputs/data/validation.json` for their details.

SHA-256: `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`

`row_id` is the zero-based source record position. It is the DataFrame index and is not an extra source feature. Row IDs are meaningful only with this checksum and unchanged source ordering. No rows were removed, no values imputed, and no preprocessing or models fitted.

This is full-source descriptive EDA. The shared repository already supplies Michael's provisional grouped split in `outputs/data/split_manifest.csv` and `split_metadata.json`; this command preserves those files. No test performance was evaluated. Use the existing assignments for later model experiments and fit learned preprocessing on training only. The descriptive IQR and percentile values here must not become all-data fitted preprocessing rules. Approval of scientific decisions remains pending in `reports/decisions.md`.

## Schema and missingness

All 31 required source labels are present. Meanings and semantic types in `outputs/data/schema.csv` follow Section 4 of the course specification. The original [dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) establishes the not-applicable meaning of NULL in Agent and Company and warns that booking attributes can change over time. These definitions are not an approved modeling feature policy.

The raw loader retains string tokens exactly. The analysis view strips outer whitespace and maps the explicitly documented tokens `N/A, NA, NAN, NULL` and blank fields to missing values, case-insensitively, **except that Agent and Company retain NULL as a not-applicable category**. Numerical conversion failures are counted separately. Labels such as `Undefined` survive as categories for review. Numerical ID tokens in Agent and Company remain categorical strings.

| Column | Normalized missing | Not applicable | Padded tokens | Distinct nonmissing |
|---|---:|---:|---:|---:|
| IsCanceled | 0 | 0 | 0 | 2 |
| LeadTime | 0 | 0 | 0 | 453 |
| ArrivalDateYear | 0 | 0 | 0 | 3 |
| ArrivalDateMonth | 0 | 0 | 0 | 12 |
| ArrivalDateWeekNumber | 0 | 0 | 0 | 53 |
| ArrivalDateDayOfMonth | 0 | 0 | 0 | 31 |
| StaysInWeekendNights | 0 | 0 | 0 | 14 |
| StaysInWeekNights | 0 | 0 | 0 | 29 |
| Adults | 0 | 0 | 0 | 5 |
| Children | 4 | 0 | 0 | 4 |
| Babies | 0 | 0 | 0 | 5 |
| Meal | 0 | 0 | 79,330 | 4 |
| Country | 24 | 0 | 0 | 166 |
| MarketSegment | 0 | 0 | 0 | 8 |
| DistributionChannel | 0 | 0 | 0 | 5 |
| IsRepeatedGuest | 0 | 0 | 79,330 | 2 |
| PreviousCancellations | 0 | 0 | 0 | 10 |
| PreviousBookingsNotCanceled | 0 | 0 | 0 | 73 |
| ReservedRoomType | 0 | 0 | 79,330 | 8 |
| AssignedRoomType | 0 | 0 | 79,330 | 9 |
| BookingChanges | 0 | 0 | 0 | 21 |
| DepositType | 0 | 0 | 79,330 | 3 |
| Agent | 0 | 8,131 | 79,330 | 224 |
| Company | 0 | 75,641 | 79,330 | 208 |
| DaysInWaitingList | 0 | 0 | 0 | 115 |
| CustomerType | 0 | 0 | 0 | 4 |
| ADR | 0 | 0 | 0 | 5,405 |
| RequiredCarParkingSpaces | 0 | 0 | 0 | 4 |
| TotalOfSpecialRequests | 0 | 0 | 0 | 6 |
| ReservationStatus | 0 | 0 | 0 | 3 |
| ReservationStatusDate | 0 | 0 | 0 | 864 |

The original paper explicitly treats NULL in Agent and Company as not applicable; for example, Agent NULL means the booking did not come through an agency. These are categorical values, not missing quantities, and are preserved as `NULL` after trimming whitespace. Do not median-impute these identifiers or drop Company on the mistaken premise that its NULL category is missing data. Children and Country need explicit missing-value decisions; missing Children does not prove zero children.

## Duplicate records

There are **25,902 excess exact duplicate rows** beyond each group's first record. **31,748 records** belong to duplicate groups. After the documented normalization there are 25,902 excess duplicate rows. `duplicate_group_sizes.csv` reports the size distribution of these groups.

Recommendation pending team approval: preserve all source records initially. Identical booking attributes alone do not establish duplicate data entry. Michael's existing provisional split groups identical normalized records together. Preserve that manifest and review its documented policy rather than silently regenerating assignments. Removing rows or assigning groups changes the experimental population and must be documented.

## Targets and unusual observations

IsCanceled contains 33,102 canceled bookings (41.73%) and 46,228 non-canceled bookings. These are source proportions, not validation results. Youssef should report class-aware metrics alongside accuracy.

ADR ranges from 0 to 5400; the median is 99.9. The descriptive 1.5-IQR interval is [9, 196.2], flagging 3,377 records for review. Such flags do not establish errors. Inspect `adr_extreme_examples.csv` and `adr_population_by_cancellation.csv` before deciding on the regression population, any rate exclusions, or transformations.

| Check or descriptive flag | Records |
|---|---:|
| IsCanceled_negative | 0 |
| IsCanceled_noninteger | 0 |
| LeadTime_negative | 0 |
| LeadTime_noninteger | 0 |
| ArrivalDateYear_negative | 0 |
| ArrivalDateYear_noninteger | 0 |
| ArrivalDateWeekNumber_negative | 0 |
| ArrivalDateWeekNumber_noninteger | 0 |
| ArrivalDateDayOfMonth_negative | 0 |
| ArrivalDateDayOfMonth_noninteger | 0 |
| StaysInWeekendNights_negative | 0 |
| StaysInWeekendNights_noninteger | 0 |
| StaysInWeekNights_negative | 0 |
| StaysInWeekNights_noninteger | 0 |
| Adults_negative | 0 |
| Adults_noninteger | 0 |
| Children_negative | 0 |
| Children_noninteger | 0 |
| Babies_negative | 0 |
| Babies_noninteger | 0 |
| IsRepeatedGuest_negative | 0 |
| IsRepeatedGuest_noninteger | 0 |
| PreviousCancellations_negative | 0 |
| PreviousCancellations_noninteger | 0 |
| PreviousBookingsNotCanceled_negative | 0 |
| PreviousBookingsNotCanceled_noninteger | 0 |
| BookingChanges_negative | 0 |
| BookingChanges_noninteger | 0 |
| DaysInWaitingList_negative | 0 |
| DaysInWaitingList_noninteger | 0 |
| RequiredCarParkingSpaces_negative | 0 |
| RequiredCarParkingSpaces_noninteger | 0 |
| TotalOfSpecialRequests_negative | 0 |
| TotalOfSpecialRequests_noninteger | 0 |
| IsCanceled_outside_binary | 0 |
| IsRepeatedGuest_outside_binary | 0 |
| invalid_arrival_dates | 0 |
| arrival_outside_documented_period | 0 |
| unrecognized_arrival_months | 0 |
| invalid_reservation_status_dates | 0 |
| arrival_week_outside_1_to_53 | 0 |
| zero_night_bookings | 331 |
| zero_total_guests_complete_records | 167 |
| zero_adults | 390 |
| babies_above_2_review_only | 2 |
| ADR_zero | 1,208 |
| ADR_negative | 0 |
| unrecognized_reservation_status | 0 |
| cancellation_status_disagreements | 0 |

Zero-night, zero-ADR, zero-guest and unusually large guest-count records require interpretation. Zero adults alone is a review flag, not proof of an invalid booking. The babies-above-two threshold is a transparent review aid, not a source validity rule. No outlier deletion or winsorization has been applied.

## EDA figures and interpretation

- [View cancellation_balance](../outputs/eda/figures/cancellation_balance.png)
- [View adr_distribution](../outputs/eda/figures/adr_distribution.png)
- [View missing_values](../outputs/eda/figures/missing_values.png)
- [View lead_time_and_stay_length](../outputs/eda/figures/lead_time_and_stay_length.png)
- [View arrivals_by_month](../outputs/eda/figures/arrivals_by_month.png)
- [View cancellation_by_market_segment](../outputs/eda/figures/cancellation_by_market_segment.png)
- [View adr_by_reserved_room](../outputs/eda/figures/adr_by_reserved_room.png)

The full ADR histogram uses a log count axis so sparse extremes remain visible. Its companion view shows only values at or below the source 99th percentile for display; every record remains in the data. The room-type boxplots hide fliers for readability, and small category sizes must be checked in the frequency table.

Monthly arrival counts include partial endpoint years: 2015 starts in July and 2017 ends in August. Do not compare annual totals as though all three years had full coverage. Market-segment cancellation rates and room-type ADR distributions describe associations and do not prove causation or establish feature availability at prediction time.

## Proposed decisions for scientific review

1. Accept deterministic whitespace and missing-token normalization after reviewing raw-token evidence. It does not require fitting on data.
2. Agree meanings and treatments for each missing field; fit any learned imputation only inside training pipelines.
3. Agree duplicate interpretation and split strategy before baseline experiments. Preserve stable source identity through every task-specific filter.
4. Paula and Michael should agree whether canceled, zero-night, zero-rate and extreme-rate bookings belong in the ADR prediction population. Report each affected count and overlap before exclusion.
5. Define each task's prediction/analysis point before approving features. Final ReservationStatus and ReservationStatusDate encode lifecycle outcomes. AssignedRoomType, BookingChanges and other mutable fields need timing review. The original paper warns that guest counts, stay duration and room preferences can change during the booking lifecycle, and nationality may be unknown until check-in. This document does not approve the remaining fields automatically.
6. Aya and Michael should select a justified, separately prepared clustering population and feature set. Agent and Company have high cardinality; numerical IDs are not quantities or meaningful distances. Outcome labels must not select clusters or parameters.
7. Members 1 and 2 must reconcile normalization before enabling Agent/Company or changing duplicate grouping. EDA preserves their NULL category; the modeling builder converts NULL to missing, and these columns are currently excluded. EDA also treats literal NA as missing, whereas the split reader normalizes only blank/NULL tokens. Agree the representations, version the feature/preprocessing policy and any affected split rule, and preserve the current frozen manifest until that review is complete.

## Handoff recipients

| Recipient | Evidence supplied and decision needed |
|---|---|
| Michael | Validation JSON, schema, missingness, duplicates and unusual values; scientifically review treatments, source semantics and split implications. |
| Paula | ADR summaries, extreme examples, cancellation-stratified populations, zero-night and zero-rate evidence; agree regression population. |
| Youssef | Class balance, category frequencies and observed segment rates; review class-aware evaluation and prediction-time availability. |
| Aya | Numeric scales/skew, category cardinalities, duplicate/population findings; agree segmentation purpose and separate preprocessing. |
| Makarious | Checksummed evidence, figures, tables and commands; integrate and review the generated notes with the owners. |

## Reproduction

Validation command: `python main.py validate`

Analysis command: `python main.py analyze`

Outputs: `outputs/data/validation.json`, `outputs/data/schema.csv`, and the tables/figures listed in `outputs/eda/manifest.json`. Tables are in `outputs/eda/tables/` and figures are in `outputs/eda/figures/`. This generated report is `reports/eda_notes.md`. [Data review](data_review.md) documents unusual records, overlaps, duplicates and proposed treatments. Use the unchanged source checksum to reproduce the counts.

## Remaining human review

Zeina should review the figures, verify business meanings against the original article, and explain the findings. Michael should scientifically review this handoff. A split and provisional feature policy are already implemented; their scientific approval, cleaning choices and baseline choices remain pending team review. No GitHub commits, pull requests or approvals are claimed by these generated files.
