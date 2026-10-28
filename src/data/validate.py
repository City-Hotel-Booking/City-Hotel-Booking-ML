"""Descriptive validation only: no row deletion or modeling decisions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .load import Dataset, arrival_dates, read_dataset
from .schema import (SCHEMA, SOURCE_COLUMNS, EXPECTED_ROWS, NUMERIC_COLUMNS,
                     INTEGER_COLUMNS, MISSING_TOKENS, MONTHS, NOT_APPLICABLE_COLUMNS)


def _count(mask) -> int:
    return int(mask.fillna(False).sum())


def inspect_dataset(dataset: Dataset) -> dict:
    raw, data = dataset.raw, dataset.data
    missing = [c for c in SOURCE_COLUMNS if c not in raw]
    extra = [c for c in raw if c not in SOURCE_COLUMNS]
    report = {
        "dataset_sha256": dataset.checksum,
        "source_file": dataset.path.name,
        "source_rows": len(raw), "source_columns": len(raw.columns),
        "row_id_definition": "Zero-based CSV data-record position before any filtering; tied to dataset_sha256.",
        "expected_rows": EXPECTED_ROWS, "expected_columns": len(SOURCE_COLUMNS),
        "schema": {"missing_columns": missing, "unexpected_columns": extra,
                   "column_order_matches": list(raw.columns) == SOURCE_COLUMNS},
        "normalization": {"strip_outer_whitespace": True, "case_insensitive_missing_tokens": sorted(MISSING_TOKENS),
                          "exception": "Agent/Company NULL is preserved as a not-applicable category per the original paper.",
                          "rows_removed": 0, "imputation_performed": False},
        "columns": {}, "checks": {},
        "duplicates": {
            "raw_excess_rows": int(raw.duplicated().sum()),
            "normalized_excess_rows": int(data.duplicated().sum()),
            "raw_rows_in_duplicate_groups": int(raw.duplicated(keep=False).sum()),
            "policy": "Report only; identical records are not automatically errors. Team decision pending.",
        },
    }
    for column in raw:
        stripped = raw[column].str.strip()
        info = {
            "raw_dtype": str(raw[column].dtype), "normalized_dtype": str(data[column].dtype),
            "raw_blank_count": _count(raw[column].eq("")),
            "whitespace_padded_count": _count(raw[column].ne(stripped)),
            "missing_token_count_after_strip": _count(stripped.str.upper().isin(MISSING_TOKENS)),
            "normalized_missing_count": int(data[column].isna().sum()),
            "not_applicable_count": _count(stripped.str.upper().eq("NULL")) if column in NOT_APPLICABLE_COLUMNS else 0,
            "distinct_nonmissing": int(data[column].nunique(dropna=True)),
        }
        if column in NUMERIC_COLUMNS:
            nonmissing_tokens = ~stripped.str.upper().isin(MISSING_TOKENS)
            info["invalid_numeric_tokens"] = _count(nonmissing_tokens & data[column].isna())
            info["nonfinite_numeric_values"] = _count(data[column].notna() & ~np.isfinite(data[column].astype(float)))
        report["columns"][column] = info
    if missing or extra:
        report["status"] = "failed"
        report["blocking_issues"] = ["Source labels do not match the required H2 schema."]
        return report

    checks = report["checks"]
    for column in INTEGER_COLUMNS:
        checks[f"{column}_negative"] = _count(data[column].lt(0))
        finite = data[column].where(np.isfinite(data[column].astype(float)))
        checks[f"{column}_noninteger"] = _count(finite.mod(1).ne(0) & finite.notna())
    for column in ["IsCanceled", "IsRepeatedGuest"]:
        checks[f"{column}_outside_binary"] = _count(data[column].notna() & ~data[column].isin([0, 1]))
    checks["invalid_arrival_dates"] = int(arrival_dates(data).isna().sum())
    checks["arrival_outside_documented_period"] = _count(
        arrival_dates(data).lt("2015-07-01") | arrival_dates(data).gt("2017-08-31"))
    checks["unrecognized_arrival_months"] = _count(~data["ArrivalDateMonth"].isin(MONTHS))
    # The source mirror uses ISO dates; the supplied spreadsheet export used
    # US month/day/year dates. Parse each explicit format without guessing.
    status_tokens = data["ReservationStatusDate"]
    status_dates = pd.to_datetime(status_tokens, format="%Y-%m-%d", errors="coerce")
    status_dates = status_dates.fillna(pd.to_datetime(status_tokens, format="%m/%d/%Y", errors="coerce"))
    checks["invalid_reservation_status_dates"] = int(status_dates.isna().sum())
    checks["arrival_week_outside_1_to_53"] = _count(~data["ArrivalDateWeekNumber"].between(1, 53))
    checks["zero_night_bookings"] = _count((data["StaysInWeekendNights"] + data["StaysInWeekNights"]).eq(0))
    checks["zero_total_guests_complete_records"] = _count(data[["Adults", "Children", "Babies"]].sum(axis=1, min_count=3).eq(0))
    checks["zero_adults"] = _count(data["Adults"].eq(0))
    checks["babies_above_2_review_only"] = _count(data["Babies"].gt(2))
    checks["ADR_zero"] = _count(data["ADR"].eq(0))
    checks["ADR_negative"] = _count(data["ADR"].lt(0))
    checks["unrecognized_reservation_status"] = _count(~data["ReservationStatus"].isin(["Canceled", "No-Show", "Check-Out"]))
    expected_canceled = data["ReservationStatus"].map({"Canceled": 1, "No-Show": 1, "Check-Out": 0})
    checks["cancellation_status_disagreements"] = _count(expected_canceled.notna() & expected_canceled.ne(data["IsCanceled"]))
    finite_adr = data.loc[np.isfinite(data["ADR"].astype(float)), "ADR"].astype(float)
    if finite_adr.empty:
        report["status"] = "failed"
        report["blocking_issues"] = ["ADR has no finite observed numerical values."]
        return report
    q1, q3 = finite_adr.quantile([0.25, 0.75]).astype(float)
    lower, upper = round(q1 - 1.5 * (q3 - q1), 12), round(q3 + 1.5 * (q3 - q1), 12)
    report["adr"] = {
        "min": float(finite_adr.min()), "max": float(finite_adr.max()),
        "median": float(finite_adr.median()), "q1": q1, "q3": q3,
        "iqr_lower_fence": lower, "iqr_upper_fence": upper,
        "outside_iqr_fences": _count(data["ADR"].lt(lower) | data["ADR"].gt(upper)),
        "interpretation": "Full-source descriptive flags, not automatic exclusion rules or learned preprocessing thresholds.",
    }
    report["cancellation"] = {
        "not_canceled": _count(data["IsCanceled"].eq(0)),
        "canceled": _count(data["IsCanceled"].eq(1)),
        "canceled_fraction": float(data["IsCanceled"].mean()),
    }
    report["arrival_period"] = {"first": str(arrival_dates(data).min().date()), "last": str(arrival_dates(data).max().date())}
    blocking = []
    if len(raw) != EXPECTED_ROWS:
        blocking.append(f"Expected {EXPECTED_ROWS} source rows; found {len(raw)}.")
    for column, info in report["columns"].items():
        if info.get("invalid_numeric_tokens", 0) or info.get("nonfinite_numeric_values", 0):
            blocking.append(f"{column} has malformed or nonfinite numerical values.")
    for key, count in checks.items():
        if count and (key.endswith(("_negative", "_noninteger", "_outside_binary")) or key.startswith(("invalid_", "unrecognized_", "arrival_week_outside")) or key == "cancellation_status_disagreements"):
            blocking.append(f"{key}: {count}")
    report["blocking_issues"] = blocking
    report["status"] = "failed" if blocking else "passed_with_findings"
    return report


def write_validation(dataset: Dataset, data_directory: Path, command: str) -> dict:
    target = data_directory
    target.mkdir(parents=True, exist_ok=True)
    report = inspect_dataset(dataset)
    report["command"] = command
    (target / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    schema = []
    definitions = {name: (role, meaning) for name, role, meaning in SCHEMA}
    for column, info in report["columns"].items():
        role, meaning = definitions.get(column, ("unexpected", "Not in the required source schema."))
        schema.append({"source_label": column, "semantic_role": role, "meaning": meaning, **info})
    pd.DataFrame(schema).to_csv(target / "schema.csv", index=False)
    return report


def validate_data(config: dict, project_root: Path) -> tuple[Dataset, dict]:
    """Use the team's configured source/output paths without changing decisions."""
    import shlex
    import sys

    dataset = read_dataset(project_root / config["data"]["raw_path"])
    command = "python main.py " + shlex.join(sys.argv[1:])
    report = write_validation(dataset, project_root / config["paths"]["data_outputs"], command)
    if report["blocking_issues"]:
        raise ValueError("Validation failed: " + "; ".join(report["blocking_issues"]))
    expected_rows = int(config["data"]["expected_rows"])
    expected_columns = int(config["data"]["expected_columns"])
    if expected_rows != EXPECTED_ROWS or expected_columns != len(SOURCE_COLUMNS):
        raise ValueError("The configured H2 dimensions must be 79330 rows and 31 columns.")
    print(f"{len(dataset.raw):,} rows x {len(dataset.raw.columns)} source columns | {report['status']}")
    print(f"SHA-256: {dataset.checksum}")
    return dataset, report


def run(config: dict, project_root: Path) -> None:
    """Shared CLI runner called by main.py validate."""
    validate_data(config, project_root)
    print(f"Validation: {config['paths']['data_outputs']}/validation.json")
