"""Independently reconcile saved evidence with CSV records, then run unit tests.

Run from the repository root: python scripts/verify_outputs.py
"""

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(condition, description):
    if not condition:
        raise ValueError(description)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(data_path, output_root, *, data_directory=None, eda_directory=None):
    data_directory = data_directory or output_root / "data"
    eda_directory = eda_directory or output_root / "eda"
    rows = read_csv(data_path)
    checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
    validation = json.loads((data_directory / "validation.json").read_text(encoding="utf-8"))
    review = json.loads((data_directory / "review.json").read_text(encoding="utf-8"))
    manifest = json.loads((eda_directory / "manifest.json").read_text(encoding="utf-8"))
    count = len(rows)
    require(checksum == validation["dataset_sha256"] == review["dataset_sha256"] == manifest["dataset_sha256"], "Dataset checksum mismatch.")
    require(count == validation["source_rows"] == review["source_rows"] == manifest["source_rows"], "Source counts disagree.")
    repeated = Counter(tuple(row.values()) for row in rows)
    excess = sum(size - 1 for size in repeated.values())
    require(excess == review["raw_excess_duplicates"] == validation["duplicates"]["raw_excess_rows"], "Duplicate count mismatch.")
    require(sum(size > 1 for size in repeated.values()) == review["duplicate_groups"], "Repeated-group count mismatch.")
    require(max(repeated.values()) == review["largest_identical_group"], "Largest repeated group mismatch.")
    flags = Counter()
    zero_population = 0
    for row in rows:
        rate = Decimal(row["ADR"].strip())
        nights = int(row["StaysInWeekNights"]) + int(row["StaysInWeekendNights"])
        guest_tokens = [row[c].strip() for c in ["Adults", "Children", "Babies"]]
        complete = all(token.upper() not in {"", "NULL", "NA", "N/A", "NAN"} for token in guest_tokens)
        zero_guests = complete and sum(Decimal(token) for token in guest_tokens) == 0
        flags["zero_nights"] += nights == 0
        flags["zero_ADR"] += rate == 0
        flags["zero_guests"] += zero_guests
        flags["babies_above_2_review_only"] += Decimal(row["Babies"]) > 2
        zero_population += nights == 0 or rate == 0 or zero_guests
    profiles = {item["flag"]: item for item in review["flag_profiles"]}
    for name, value in flags.items():
        require(value == profiles[name]["records"], f"Review flag mismatch: {name}")
    require(zero_population == review["distinct_rows_with_zero_nights_ADR_or_guests"], "Flag union double-counted.")
    for column in ["Agent", "Company"]:
        require(sum(row[column].strip().upper() == "NULL" for row in rows) == validation["columns"][column]["not_applicable_count"], f"Not-applicable count mismatch: {column}")
    tables = eda_directory / "tables"
    for filename, field in [("cancellation_balance.csv", "count"), ("arrivals_by_month.csv", "bookings"),
                            ("adr_population_by_cancellation.csv", "bookings"), ("review_flag_combinations.csv", "records")]:
        require(sum(int(row[field]) for row in read_csv(tables / filename)) == count, f"Population does not reconcile: {filename}")
    groups = read_csv(tables / "duplicate_group_sizes.csv")
    require(sum((int(r["records_per_identical_group"]) - 1) * int(r["number_of_groups"]) for r in groups) == excess, "Duplicate size table does not reconcile.")
    for relative in manifest["tables"] + manifest["figures"]:
        require((eda_directory / relative).is_file(), f"Missing output: {relative}")
    require(all((eda_directory / relative).stat().st_size > 0 for relative in manifest["tables"] + manifest["figures"]), "An output is empty.")
    split_metadata = data_directory / "split_metadata.json"
    if split_metadata.exists():
        split_info = json.loads(split_metadata.read_text(encoding="utf-8"))
        require(split_info["data_checksum_sha256"] == checksum, "Source differs from the frozen split source.")
        assignments = read_csv(data_directory / "split_manifest.csv")
        ids = [int(row["row_id"]) for row in assignments]
        require(len(ids) == count and len(set(ids)) == count and set(ids) == set(range(count)), "Split row IDs are not complete and unique.")
        require(dict(Counter(row["split"] for row in assignments)) == split_info["row_counts"], "Frozen split counts disagree.")
    require(hashlib.sha256(data_path.read_bytes()).hexdigest() == checksum, "Source changed during verification.")
    return {"status": "passed", "dataset_sha256": checksum, "source_rows": count,
            "independent_csv_reconciliation": True, "source_unchanged_during_verification": True,
            "tables": len(manifest["tables"]), "figures": len(manifest["figures"]),
            "source_rows_removed": review["rows_removed"], "python": platform.python_version()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        from src.settings import load_config
        config_path = args.config if args.config.is_absolute() else ROOT / args.config
        config = load_config(config_path)
        source = args.data or ROOT / config["data"]["raw_path"]
        data_directory = args.output_root / "data" if args.output_root else ROOT / config["paths"]["data_outputs"]
        eda_directory = args.output_root / "eda" if args.output_root else ROOT / config["paths"]["eda_outputs"]
        output_root = args.output_root or data_directory.parent
        summary = verify(source, output_root, data_directory=data_directory, eda_directory=eda_directory)
    except (ValueError, OSError, KeyError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    tests = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful():
        return 1
    summary["tests_passed"] = result.testsRun
    summary["tests_skipped"] = len(result.skipped)
    summary["dependencies"] = {name: version(name) for name in ["pandas", "numpy", "matplotlib", "PyYAML", "scikit-learn", "scipy", "seaborn", "joblib", "pytest"]}
    (output_root / "verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
