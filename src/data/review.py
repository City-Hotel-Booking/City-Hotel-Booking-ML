"""Investigate unusual records and duplicate overlap without changing the data."""

import json
from pathlib import Path

import pandas as pd

from .load import Dataset


def build_review(dataset: Dataset, validation: dict) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Return source-wide descriptive evidence, not an exclusion policy or split."""
    data = dataset.data
    nights = data["StaysInWeekendNights"] + data["StaysInWeekNights"]
    # Unknown guest counts must not silently become zero.
    guests = data[["Adults", "Children", "Babies"]].sum(axis=1, min_count=3)
    threshold = float(data["ADR"].quantile(.99))
    masks = pd.DataFrame({
        "zero_nights": nights.eq(0), "zero_ADR": data["ADR"].eq(0),
        "zero_guests": guests.eq(0), "ADR_above_source_p99": data["ADR"].gt(threshold),
        "babies_above_2_review_only": data["Babies"].gt(2),
    }, index=data.index).fillna(False).astype(bool)
    summaries = []
    for flag in masks:
        selected = masks[flag]
        values = data.loc[selected]
        count = int(selected.sum())
        summaries.append({
            "flag": flag, "records": count,
            "canceled": int(values["IsCanceled"].eq(1).sum()),
            "not_canceled": int(values["IsCanceled"].eq(0).sum()),
            "zero_ADR": int(values["ADR"].eq(0).sum()),
            "zero_nights": int(nights.loc[selected].eq(0).sum()),
            "zero_guests": int(guests.loc[selected].eq(0).sum()),
            "median_ADR": float(values["ADR"].median()) if count else None,
        })
    overlaps = []
    for position, first in enumerate(masks):
        for second in list(masks)[position+1:]:
            overlaps.append({"first_flag": first, "second_flag": second,
                             "records": int((masks[first] & masks[second]).sum())})
    combinations = masks.value_counts(sort=False).rename("records").reset_index()
    example_frames = []
    columns = ["IsCanceled", "ADR", "Adults", "Children", "Babies", "ReservedRoomType", "MarketSegment", "LeadTime"]
    for flag in masks:
        examples = data.loc[masks[flag], columns].head(10).copy()
        examples["total_nights"] = nights.loc[examples.index]
        examples["total_guests"] = guests.loc[examples.index]
        examples.insert(0, "flag", flag)
        example_frames.append(examples.reset_index())

    raw_duplicates = dataset.raw.duplicated()
    duplicate_members = dataset.raw.duplicated(keep=False)
    duplicate_profiles = []
    for label in [0, 1]:
        mask = data["IsCanceled"].eq(label)
        duplicate_profiles.append({"IsCanceled": label, "source_records": int(mask.sum()),
            "excess_duplicate_records": int((mask & raw_duplicates).sum()),
            "records_in_duplicate_groups": int((mask & duplicate_members).sum())})
    group_ids = dataset.raw.groupby(list(dataset.raw.columns), sort=False, dropna=False).ngroup()
    group_sizes = group_ids.value_counts()
    grouped = pd.DataFrame({"group_id": group_ids, "row_id": data.index})
    representatives = grouped.groupby("group_id")["row_id"].min().to_frame("first_row_id")
    representatives["identical_group_size"] = group_sizes
    representatives = representatives.loc[representatives["identical_group_size"].gt(1)].sort_values(
        ["identical_group_size", "first_row_id"], ascending=[False, True]).head(10)
    duplicate_examples = data.loc[representatives["first_row_id"], columns].copy()
    duplicate_examples["identical_group_size"] = representatives["identical_group_size"].to_numpy()

    p_room = data["ReservedRoomType"].eq("P")
    summary = {
        "dataset_sha256": dataset.checksum, "source_rows": len(data),
        "source_p99_ADR": threshold,
        "flag_profiles": summaries,
        "distinct_rows_with_any_review_flag": int(masks.any(axis=1).sum()),
        "distinct_rows_with_zero_nights_ADR_or_guests": int(masks[["zero_nights", "zero_ADR", "zero_guests"]].any(axis=1).sum()),
        "duplicate_groups": int(group_sizes.gt(1).sum()),
        "unique_raw_records": int(len(group_sizes)),
        "largest_identical_group": int(group_sizes.max()),
        "raw_excess_duplicates": int(raw_duplicates.sum()),
        "raw_duplicate_group_members": int(duplicate_members.sum()),
        "room_P": {"records": int(p_room.sum()), "canceled": int((p_room & data["IsCanceled"].eq(1)).sum()),
                   "zero_ADR": int((p_room & data["ADR"].eq(0)).sum()), "zero_guests": int((p_room & guests.eq(0)).sum())},
        "max_ADR_row_id": int(data["ADR"].idxmax()),
        "max_ADR": float(data["ADR"].max()), "second_highest_ADR": float(data["ADR"].nlargest(2).iloc[-1]),
        "rows_removed": 0, "scientific_decision_status": "Proposals; awaiting team review.",
    }
    assert summary["raw_excess_duplicates"] == validation["duplicates"]["raw_excess_rows"]
    tables = {
        "review_flag_summary.csv": pd.DataFrame(summaries),
        "review_flag_overlaps.csv": pd.DataFrame(overlaps),
        "review_flag_combinations.csv": combinations,
        "review_record_examples.csv": pd.concat(example_frames, ignore_index=True),
        "duplicate_profiles_by_cancellation.csv": pd.DataFrame(duplicate_profiles),
        "duplicate_group_examples.csv": duplicate_examples.reset_index(),
    }
    return summary, tables


def write_review(dataset: Dataset, validation: dict, eda_directory: Path, destination: Path, command: str,
                 *, data_directory: Path) -> list[str]:
    summary, tables = build_review(dataset, validation)
    target = eda_directory / "tables"
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(target / name, index=False)
    (data_directory / "review.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    profiles = {row["flag"]: row for row in summary["flag_profiles"]}
    zero_nights, zero_rate, zero_guests = (profiles[name] for name in ["zero_nights", "zero_ADR", "zero_guests"])
    highest = dataset.data.loc[summary["max_ADR_row_id"]]
    text = f"""# H2 unusual records and duplicate review

The review identifies records that require interpretation before the team fixes its experimental population. Counts are source-wide descriptions. Flags can overlap and are not exclusion rules. No records were removed or imputed.

Dataset SHA-256: `{dataset.checksum}`

## Findings and recommendations

| Topic | Verified evidence | Proposed action and owner |
|---|---|---|
| Missing Children | {validation['columns']['Children']['normalized_missing_count']} records | Keep unknown distinct from zero. Michael should select a justified strategy; any learned imputation must be fitted on training only. |
| Missing Country | {validation['columns']['Country']['normalized_missing_count']} records | Propose a distinct unknown category; Michael reviews semantics and implementation. |
| Agent and Company NULL | Not-applicable categories, preserved as NULL | Retain as categorical IDs; do not treat their numeric codes as quantities. |
| Zero nights | {zero_nights['records']:,} records; {zero_nights['zero_ADR']:,} also have zero ADR; {zero_nights['canceled']:,} are canceled | Paula and Michael should clarify the meaning of a nightly rate for zero-night records before defining the ADR population. |
| Zero ADR | {zero_rate['records']:,} records; {zero_rate['canceled']:,} are canceled; {zero_rate['zero_guests']:,} also have zero guests | Check booking context. A recorded zero does not alone establish an error or a complimentary stay. |
| Zero total guests | {zero_guests['records']:,} complete guest-count records; {zero_guests['zero_ADR']:,} also have zero ADR; {zero_guests['zero_nights']:,} also have zero nights | Investigate possible administrative or data-quality explanations. These are hypotheses, not confirmed causes. Coordinate any exclusion across task populations. |
| Extreme ADR | Maximum {summary['max_ADR']:g}; next highest {summary['second_highest_ADR']:g} | Flag the extreme for Paula and Michael. Do not silently delete or cap it; decide a policy using domain evidence and training-only model selection. |
| Rare room P | {summary['room_P']['records']} records; {summary['room_P']['canceled']} canceled, {summary['room_P']['zero_ADR']} zero ADR and {summary['room_P']['zero_guests']} zero guests | This overlap deserves review. It does not establish what the anonymous room code means or justify a rule for future bookings. |
| Identical records | {summary['raw_excess_duplicates']:,} excess rows in {summary['duplicate_groups']:,} repeated groups | Preserve records pending agreement. Michael should assess duplicate-group overlap when establishing the split. |

## Extreme ADR record

Source row_id {summary['max_ADR_row_id']} has ADR {highest['ADR']:g}, IsCanceled {highest['IsCanceled']}, {highest['Adults']} adults, {highest['Children']} children and {highest['Babies']} babies, reserved room {highest['ReservedRoomType']}, and segment {highest['MarketSegment']}. Its total nights are {highest['StaysInWeekNights'] + highest['StaysInWeekendNights']}. These fields do not establish why the rate is extreme. A rate-entry error and an unusual genuine transaction are possibilities requiring external evidence.

## Overlaps and affected populations

The union of zero-night, zero-ADR and zero-guest records contains **{summary['distinct_rows_with_zero_nights_ADR_or_guests']:,} distinct rows**. Adding their separate counts would double-count overlapping records. Exact mutually exclusive combinations are in `outputs/eda/tables/review_flag_combinations.csv`; pairwise overlaps are in `review_flag_overlaps.csv`.

There are {profiles['ADR_above_source_p99']['records']:,} records strictly above the descriptive source ADR 99th percentile of {summary['source_p99_ADR']:g}. This threshold is for EDA only. It must not become a full-data learned clipping rule.

The babies-above-two flag identifies {profiles['babies_above_2_review_only']['records']} records. This transparent review threshold is not a declared source-validity limit. Unknown guest counts are excluded from the zero-total-guests flag rather than assumed to be zero.

## Duplicate interpretation

There are {summary['unique_raw_records']:,} distinct raw records among {summary['source_rows']:,} rows. Of these, {summary['duplicate_groups']:,} groups repeat; {summary['raw_duplicate_group_members']:,} rows belong to those groups. The largest group contains {summary['largest_identical_group']} identical records. The first occurrence is included in group membership counts but excluded from the excess-duplicate count.

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

Michael coordinates these decisions with Zeina and the model owners. Any learned imputation, scaling, encoding, dimensionality reduction or threshold selection belongs inside the agreed training protocol. This contribution creates no model or test-performance results and preserves the existing shared split assignments.

## Reproduction

Run `{command}` from the repository root. The command regenerates this report, `outputs/data/review.json` and all six review tables. Representative records retain source row IDs; the original CSV and all source rows remain unchanged. Findings can be checked against the independently implemented audit in `scripts/verify_outputs.py`.
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return [f"tables/{name}" for name in tables]
