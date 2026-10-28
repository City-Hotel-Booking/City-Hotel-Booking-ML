"""Reproducible full-source descriptions, with no model fitting or row removal."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .load import Dataset, arrival_dates
from .schema import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, MISSING_TOKENS


def run_eda(dataset: Dataset, validation: dict, eda_directory: Path, command: str) -> dict:
    data = dataset.data
    target = eda_directory
    (target / "tables").mkdir(parents=True, exist_ok=True)
    (target / "figures").mkdir(parents=True, exist_ok=True)
    figures = []
    tables = []

    def table(name, frame, index=False):
        frame.to_csv(target / "tables" / name, index=index)
        tables.append(f"tables/{name}")

    def save(name, fig):
        fig.savefig(target / "figures" / name, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.append(f"figures/{name}")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    color, accent = "#315b83", "#c67535"
    numeric = data[NUMERIC_COLUMNS].astype(float)
    summary = numeric.describe(percentiles=[.01, .25, .5, .75, .95, .99]).T
    summary["skew"] = numeric.skew()
    summary.index.name = "column"
    table("numeric_summary.csv", summary.reset_index())

    missing = pd.DataFrame([
        {"column": c, "missing_count": i["normalized_missing_count"],
         "missing_fraction": i["normalized_missing_count"] / len(data),
         "not_applicable_count": i["not_applicable_count"],
         "padded_count": i["whitespace_padded_count"],
         "distinct_nonmissing": i["distinct_nonmissing"]}
        for c, i in validation["columns"].items()
    ])
    table("missingness_and_cardinality.csv", missing)
    frequencies = []
    for c in CATEGORICAL_COLUMNS:
        for value, count in data[c].value_counts(dropna=False).items():
            frequencies.append({"column": c, "value": "<MISSING>" if pd.isna(value) else value,
                                "count": int(count), "fraction": int(count) / len(data)})
    table("category_frequencies.csv", pd.DataFrame(frequencies))
    table("cancellation_balance.csv", pd.DataFrame([
        {"IsCanceled": label, "count": int(data["IsCanceled"].eq(label).sum()),
         "fraction": float(data["IsCanceled"].eq(label).mean())} for label in [0, 1]
    ]))

    work = data.copy()
    work["total_nights"] = data["StaysInWeekendNights"] + data["StaysInWeekNights"]
    work["total_guests"] = data[["Adults", "Children", "Babies"]].sum(axis=1, min_count=3)
    work["arrival_month"] = arrival_dates(data).dt.to_period("M").astype(str)
    monthly = work.groupby("arrival_month").agg(bookings=("ADR", "size"), median_ADR=("ADR", "median"),
                                               cancellation_fraction=("IsCanceled", "mean")).reset_index()
    table("arrivals_by_month.csv", monthly)
    adr_population = work.groupby("IsCanceled").agg(
        bookings=("ADR", "size"), median_ADR=("ADR", "median"), mean_ADR=("ADR", "mean"),
        zero_ADR=("ADR", lambda s: int(s.eq(0).sum())),
        zero_nights=("total_nights", lambda s: int(s.eq(0).sum())),
        zero_guests=("total_guests", lambda s: int(s.eq(0).sum())),
    ).reset_index()
    table("adr_population_by_cancellation.csv", adr_population)
    # Individual records needed to inspect the most extreme recorded values.
    table("adr_extreme_examples.csv", work.nlargest(15, "ADR")[[
        "ADR", "IsCanceled", "total_nights", "Adults", "Children", "Babies",
        "ReservedRoomType", "MarketSegment", "arrival_month",
    ]].reset_index())
    group_sizes = dataset.raw.groupby(list(dataset.raw.columns), dropna=False, sort=False).size()
    duplicate_summary = group_sizes[group_sizes.gt(1)].value_counts().sort_index().rename_axis("records_per_identical_group").reset_index(name="number_of_groups")
    table("duplicate_group_sizes.csv", duplicate_summary)
    markets = work.groupby("MarketSegment").agg(bookings=("IsCanceled", "size"),
              cancellation_fraction=("IsCanceled", "mean"), median_ADR=("ADR", "median")).sort_values("bookings", ascending=False).reset_index()
    table("market_segment_summary.csv", markets)

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = [validation["cancellation"]["not_canceled"], validation["cancellation"]["canceled"]]
    bars = ax.bar(["Not canceled (0)", "Canceled (1)"], counts, color=[color, accent])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, count, f"{count:,}\n({count/len(data):.1%})", ha="center", va="bottom")
    ax.set(title="Cancellation balance in the source data", ylabel="Bookings", ylim=(0, max(counts)*1.22))
    save("cancellation_balance.png", fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(numeric["ADR"].dropna(), bins=70, color=color)
    axes[0].set(title="ADR across the full recorded range", xlabel="Recorded ADR", ylabel="Bookings")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Bookings (log scale)")
    cutoff = float(numeric["ADR"].quantile(.99))
    axes[1].hist(numeric.loc[numeric["ADR"].le(cutoff), "ADR"], bins=50, color=color)
    axes[1].set(title=f"ADR at or below its 99th percentile ({cutoff:g})", xlabel="Recorded ADR", ylabel="Bookings")
    fig.suptitle("ADR distribution; zoom changes the display only", y=1.04)
    fig.tight_layout()
    save("adr_distribution.png", fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    visible = missing.loc[missing["missing_count"].gt(0)].sort_values("missing_count")
    ax.barh(visible["column"], visible["missing_count"], color=color)
    for i, count in enumerate(visible["missing_count"]):
        ax.text(count + 0.3, i, f"{count:,}", va="center")
    maximum_missing = int(visible["missing_count"].max()) if not visible.empty else 1
    ax.set(title="Missing values", xlabel="Bookings", xlim=(0, maximum_missing*1.3))
    visible = missing.loc[missing["not_applicable_count"].gt(0)].sort_values("not_applicable_count")
    axes[1].barh(visible["column"], visible["not_applicable_count"], color=accent)
    for i, count in enumerate(visible["not_applicable_count"]):
        axes[1].text(count + 1000, i, f"{count:,}", va="center")
    axes[1].set(title="Not applicable (NULL category)", xlabel="Bookings", xlim=(0, len(data)*1.25))
    fig.suptitle("Missingness and not-applicable identifiers are different", y=1.03)
    fig.tight_layout()
    save("missing_values.png", fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(numeric["LeadTime"], bins=45, color=color)
    axes[0].set(title="Lead time", xlabel="Days", ylabel="Bookings")
    axes[1].hist(work["total_nights"], bins=np.arange(0, int(work["total_nights"].max())+2)-.5, color=color)
    axes[1].set(title="Total booked or stayed nights", xlabel="Nights", ylabel="Bookings (log scale)", yscale="log")
    fig.tight_layout()
    save("lead_time_and_stay_length.png", fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(monthly["arrival_month"], monthly["bookings"], color=color)
    ax.set(title="Arrivals by calendar month", ylabel="Bookings", xlabel="Arrival month")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    save("arrivals_by_month.png", fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(markets["MarketSegment"], markets["cancellation_fraction"]*100, color=color)
    ax.set(title="Observed cancellations by market segment", ylabel="Canceled bookings (%)", ylim=(0, 110))
    ax.tick_params(axis="x", rotation=30)
    for i, row in markets.iterrows():
        ax.text(i, row["cancellation_fraction"]*100+2, f"n={row['bookings']:,}", ha="center", fontsize=8)
    fig.text(.01, -.04, "Descriptive associations only. Small groups can have unstable rates.", fontsize=9)
    fig.tight_layout()
    save("cancellation_by_market_segment.png", fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    rooms = sorted(work["ReservedRoomType"].dropna().unique())
    ax.boxplot([work.loc[work["ReservedRoomType"].eq(room), "ADR"].astype(float).to_numpy() for room in rooms],
               tick_labels=rooms, showfliers=False)
    ax.set(title="Recorded ADR by reserved room type", xlabel="Reserved room type", ylabel="Recorded ADR")
    fig.text(.01, -.03, "Boxplots hide fliers for readability; no rows are removed. Review group sizes in category_frequencies.csv.", fontsize=8)
    save("adr_by_reserved_room.png", fig)

    manifest = {"dataset_sha256": dataset.checksum, "source_rows": len(data),
                "scope": "Full-source descriptive EDA. No training, split creation, or test scoring.",
                "command": command, "tables": tables, "figures": figures}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_notes(dataset: Dataset, validation: dict, manifest: dict, destination: Path) -> None:
    """Generate a factual first handoff; recommendations stay explicitly pending."""
    v = validation
    missing_lines = "\n".join(
        f"| {column} | {info['normalized_missing_count']:,} | {info['not_applicable_count']:,} | {info['whitespace_padded_count']:,} | {info['distinct_nonmissing']:,} |"
        for column, info in v["columns"].items())
    checks = "\n".join(f"| {name} | {count:,} |" for name, count in v["checks"].items())
    figure_links = "\n".join(f"- [View {Path(name).stem}](../outputs/eda/{name})" for name in manifest["figures"])
    token_list = ', '.join(sorted(t for t in MISSING_TOKENS if t))
    text = f"""# H2 data validation and EDA handoff

This report records the H2 validation and descriptive EDA for Zeina's Member 1 work package. It supplies evidence for the team's preparation and baseline experiments. Cleaning, split, feature and population decisions still require scientific review.

## Scope and source identity

The source contains **{v['source_rows']:,} bookings and {v['source_columns']} source variables**. Arrival dates range from {v['arrival_period']['first']} through {v['arrival_period']['last']}. Validation status: `{v['status']}`. There are {len(v['blocking_issues'])} blocking validation issues; see `outputs/data/validation.json` for their details.

SHA-256: `{dataset.checksum}`

`row_id` is the zero-based source record position. It is the DataFrame index and is not an extra source feature. Row IDs are meaningful only with this checksum and unchanged source ordering. No rows were removed, no values imputed, and no preprocessing or models fitted.

This is full-source descriptive EDA. The shared repository already supplies Michael's provisional grouped split in `outputs/data/split_manifest.csv` and `split_metadata.json`; this command preserves those files. No test performance was evaluated. Use the existing assignments for later model experiments and fit learned preprocessing on training only. The descriptive IQR and percentile values here must not become all-data fitted preprocessing rules. Approval of scientific decisions remains pending in `reports/decisions.md`.

## Schema and missingness

All 31 required source labels are present. Meanings and semantic types in `outputs/data/schema.csv` follow Section 4 of the course specification. The original [dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) establishes the not-applicable meaning of NULL in Agent and Company and warns that booking attributes can change over time. These definitions are not an approved modeling feature policy.

The raw loader retains string tokens exactly. The analysis view strips outer whitespace and maps the explicitly documented tokens `{token_list}` and blank fields to missing values, case-insensitively, **except that Agent and Company retain NULL as a not-applicable category**. Numerical conversion failures are counted separately. Labels such as `Undefined` survive as categories for review. Numerical ID tokens in Agent and Company remain categorical strings.

| Column | Normalized missing | Not applicable | Padded tokens | Distinct nonmissing |
|---|---:|---:|---:|---:|
{missing_lines}

The original paper explicitly treats NULL in Agent and Company as not applicable; for example, Agent NULL means the booking did not come through an agency. These are categorical values, not missing quantities, and are preserved as `NULL` after trimming whitespace. Do not median-impute these identifiers or drop Company on the mistaken premise that its NULL category is missing data. Children and Country need explicit missing-value decisions; missing Children does not prove zero children.

## Duplicate records

There are **{v['duplicates']['raw_excess_rows']:,} excess exact duplicate rows** beyond each group's first record. **{v['duplicates']['raw_rows_in_duplicate_groups']:,} records** belong to duplicate groups. After the documented normalization there are {v['duplicates']['normalized_excess_rows']:,} excess duplicate rows. `duplicate_group_sizes.csv` reports the size distribution of these groups.

Recommendation pending team approval: preserve all source records initially. Identical booking attributes alone do not establish duplicate data entry. Michael's existing provisional split groups identical normalized records together. Preserve that manifest and review its documented policy rather than silently regenerating assignments. Removing rows or assigning groups changes the experimental population and must be documented.

## Targets and unusual observations

IsCanceled contains {v['cancellation']['canceled']:,} canceled bookings ({v['cancellation']['canceled_fraction']:.2%}) and {v['cancellation']['not_canceled']:,} non-canceled bookings. These are source proportions, not validation results. Youssef should report class-aware metrics alongside accuracy.

ADR ranges from {v['adr']['min']:g} to {v['adr']['max']:g}; the median is {v['adr']['median']:g}. The descriptive 1.5-IQR interval is [{v['adr']['iqr_lower_fence']:g}, {v['adr']['iqr_upper_fence']:g}], flagging {v['adr']['outside_iqr_fences']:,} records for review. Such flags do not establish errors. Inspect `adr_extreme_examples.csv` and `adr_population_by_cancellation.csv` before deciding on the regression population, any rate exclusions, or transformations.

| Check or descriptive flag | Records |
|---|---:|
{checks}

Zero-night, zero-ADR, zero-guest and unusually large guest-count records require interpretation. Zero adults alone is a review flag, not proof of an invalid booking. The babies-above-two threshold is a transparent review aid, not a source validity rule. No outlier deletion or winsorization has been applied.

## EDA figures and interpretation

{figure_links}

The full ADR histogram uses a log count axis so sparse extremes remain visible. Its companion view shows only values at or below the source 99th percentile for display; every record remains in the data. The room-type boxplots hide fliers for readability, and small category sizes must be checked in the frequency table.

Monthly arrival counts include partial endpoint years: 2015 starts in July and 2017 ends in August. Do not compare annual totals as though all three years had full coverage. Market-segment cancellation rates and room-type ADR distributions describe associations and do not prove causation or establish feature availability at prediction time.

## Proposed decisions for scientific review

1. Accept deterministic whitespace and missing-token normalization after reviewing raw-token evidence. It does not require fitting on data.
2. Agree meanings and treatments for each missing field; fit any learned imputation only inside training pipelines.
3. Agree duplicate interpretation and split strategy before baseline experiments. Preserve stable source identity through every task-specific filter.
4. Paula and Michael should agree whether canceled, zero-night, zero-rate and extreme-rate bookings belong in the ADR prediction population. Report each affected count and overlap before exclusion.
5. Define each task's prediction/analysis point before approving features. Final ReservationStatus and ReservationStatusDate encode lifecycle outcomes. AssignedRoomType, BookingChanges and other mutable fields need timing review. The original paper warns that guest counts, stay duration and room preferences can change during the booking lifecycle, and nationality may be unknown until check-in. This document does not approve the remaining fields automatically.
6. Aya and Michael should select a justified, separately prepared clustering population and feature set. Agent and Company have high cardinality; numerical IDs are not quantities or meaningful distances. Outcome labels must not select clusters or parameters.

## Handoff recipients

| Recipient | Evidence supplied and decision needed |
|---|---|
| Michael | Validation JSON, schema, missingness, duplicates and unusual values; scientifically review treatments, source semantics and split implications. |
| Paula | ADR summaries, extreme examples, cancellation-stratified populations, zero-night and zero-rate evidence; agree regression population. |
| Youssef | Class balance, category frequencies and observed segment rates; review class-aware evaluation and prediction-time availability. |
| Aya | Numeric scales/skew, category cardinalities, duplicate/population findings; agree segmentation purpose and separate preprocessing. |
| Makarious | Checksummed evidence, figures, tables and commands; integrate and review the generated notes with the owners. |

## Reproduction

Validation command: `{v['command']}`

Analysis command: `{manifest['command']}`

Outputs: `outputs/data/validation.json`, `outputs/data/schema.csv`, and the tables/figures listed in `outputs/eda/manifest.json`. Tables are in `outputs/eda/tables/` and figures are in `outputs/eda/figures/`. This generated report is `reports/eda_notes.md`. [Data review](data_review.md) documents unusual records, overlaps, duplicates and proposed treatments. Use the unchanged source checksum to reproduce the counts.

## Remaining human review

Zeina should review the figures, verify business meanings against the original article, and explain the findings. Michael should scientifically review this handoff. A split and provisional feature policy are already implemented; their scientific approval, cleaning choices and baseline choices remain pending team review. No GitHub commits, pull requests or approvals are claimed by these generated files.
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def run(config: dict, project_root: Path) -> None:
    """Shared CLI runner; preserve all source rows and the existing split."""
    from .validate import validate_data
    from .review import write_review

    dataset, validation = validate_data(config, project_root)
    command = validation["command"]
    eda_directory = project_root / config["paths"]["eda_outputs"]
    reports = project_root / config["paths"]["reports"]
    manifest = run_eda(dataset, validation, eda_directory, command)
    manifest["tables"].extend(write_review(
        dataset, validation, eda_directory, reports / "data_review.md", command,
        data_directory=project_root / config["paths"]["data_outputs"],
    ))
    manifest["review_summary"] = "../data/review.json"
    (eda_directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_notes(dataset, validation, manifest, reports / "eda_notes.md")
    print(f"EDA: {len(manifest['tables'])} tables and {len(manifest['figures'])} figures in {eda_directory}")
    print(f"Reports: {reports / 'eda_notes.md'} and {reports / 'data_review.md'}")
