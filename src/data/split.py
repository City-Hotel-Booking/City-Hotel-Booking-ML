"""Independent H2 reader and reproducible grouped, stratified split.

row_id is the zero-based data-row position in the source CSV. Exact duplicate
groups use SHA-256 over all 31 source fields after whitespace/NULL normalization.
The raw file is never edited and no duplicate row is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from src.features.policy import OFFICIAL_COLUMNS


# Explicit one-to-one aliases for the second supported header style.
SNAKE_TO_OFFICIAL: dict[str, str] = {
    "is_canceled": "IsCanceled",
    "lead_time": "LeadTime",
    "arrival_date_year": "ArrivalDateYear",
    "arrival_date_month": "ArrivalDateMonth",
    "arrival_date_week_number": "ArrivalDateWeekNumber",
    "arrival_date_day_of_month": "ArrivalDateDayOfMonth",
    "stays_in_weekend_nights": "StaysInWeekendNights",
    "stays_in_week_nights": "StaysInWeekNights",
    "adults": "Adults",
    "children": "Children",
    "babies": "Babies",
    "meal": "Meal",
    "country": "Country",
    "market_segment": "MarketSegment",
    "distribution_channel": "DistributionChannel",
    "is_repeated_guest": "IsRepeatedGuest",
    "previous_cancellations": "PreviousCancellations",
    "previous_bookings_not_canceled": "PreviousBookingsNotCanceled",
    "reserved_room_type": "ReservedRoomType",
    "assigned_room_type": "AssignedRoomType",
    "booking_changes": "BookingChanges",
    "deposit_type": "DepositType",
    "agent": "Agent",
    "company": "Company",
    "days_in_waiting_list": "DaysInWaitingList",
    "customer_type": "CustomerType",
    "adr": "ADR",
    "required_car_parking_spaces": "RequiredCarParkingSpaces",
    "total_of_special_requests": "TotalOfSpecialRequests",
    "reservation_status": "ReservationStatus",
    "reservation_status_date": "ReservationStatusDate",
}
HEADER_TO_OFFICIAL = {column: column for column in OFFICIAL_COLUMNS} | SNAKE_TO_OFFICIAL
SPLITS = ("train", "validation", "test")


class SplitResult(NamedTuple):
    manifest: pd.DataFrame
    metadata: dict[str, Any]


def _read_h2(path: Path) -> pd.DataFrame:
    """Read H2 without Member 1's loader; keep every source row and its position."""
    if not path.is_file():
        raise FileNotFoundError(f"Place the official H2 CSV at {path}")
    raw = pd.read_csv(
        path, dtype=str, keep_default_na=False, na_filter=False,
        skip_blank_lines=False, encoding="utf-8-sig",
    )
    original_headers = [str(column).strip() for column in raw.columns]
    unknown = [column for column in original_headers if column not in HEADER_TO_OFFICIAL]
    if unknown:
        raise ValueError(f"Unknown H2 source headers: {unknown}")
    mapped = [HEADER_TO_OFFICIAL[column] for column in original_headers]
    if len(mapped) != len(OFFICIAL_COLUMNS) or set(mapped) != set(OFFICIAL_COLUMNS):
        raise ValueError("H2 must contain each of the 31 official columns exactly once.")
    if len(set(mapped)) != len(mapped):
        raise ValueError("Two H2 headers map to the same official source column.")

    raw.columns = mapped
    raw = raw.loc[:, list(OFFICIAL_COLUMNS)]
    raw.insert(0, "row_id", range(len(raw)))
    for column in OFFICIAL_COLUMNS:
        values = raw[column].str.strip()
        raw[column] = values.mask(values.isin(["", "NULL"]))
    raw.attrs["header_mapping"] = dict(zip(original_headers, mapped))
    return raw


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_hashes(data: pd.DataFrame) -> list[str]:
    """Hash all 31 normalized source values; row_id is never part of the hash."""
    groups: list[str] = []
    for row in data.loc[:, list(OFFICIAL_COLUMNS)].itertuples(index=False, name=None):
        digest = hashlib.sha256()
        for value in row:
            if pd.isna(value):
                digest.update(b"\xff")
            else:
                encoded = str(value).encode("utf-8")
                digest.update(b"\x00" + len(encoded).to_bytes(4, "big") + encoded)
        groups.append(digest.hexdigest())
    return groups


def _split_settings(config: dict) -> tuple[int, dict[str, float], str]:
    decisions = config["decisions"]
    if decisions["split_method"] != "stratified_group_random":
        raise ValueError("This splitter requires split_method=stratified_group_random.")
    params = decisions["split_parameters"]
    ratios = {name: float(params[name]) for name in SPLITS}
    if any(ratio <= 0 for ratio in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError("Train, validation, and test ratios must be positive and sum to 1.")
    if params["stratify_column"] != "IsCanceled":
        raise ValueError("This splitter stratifies on IsCanceled.")
    if params["group_columns"] != "all_31_source_columns":
        raise ValueError("This splitter groups identical values in all 31 source columns.")
    return int(config["project"]["random_seed"]), ratios, str(decisions["split_version"])


def create_split(data: pd.DataFrame, config: dict) -> SplitResult:
    """Assign duplicate groups to approximately 70/15/15 splits by class.

    Each class is allocated separately, so class balance stays close across
    splits. Whole groups are indivisible, so exact ratios are not guaranteed.
    """
    if not set(OFFICIAL_COLUMNS).issubset(data.columns):
        raise ValueError("Missing official H2 source columns.")
    ids = pd.to_numeric(data["row_id"], errors="raise")
    if (
        ids.isna().any()
        or ids.duplicated().any()
        or not (ids >= 0).all()
        or not (ids % 1 == 0).all()
    ):
        raise ValueError("row_id must contain unique nonnegative source positions.")
    if set(ids.astype(int)) != set(range(len(data))):
        raise ValueError("row_id must cover every original data-row position.")
    labels = pd.to_numeric(data["IsCanceled"], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("IsCanceled must contain only 0 and 1.")
    labels = labels.astype(int)
    seed, ratios, split_version = _split_settings(config)

    group_ids = _group_hashes(data)
    groups = pd.DataFrame({"group_id": group_ids, "label": labels.to_numpy()})
    summary = groups.groupby("group_id", sort=True).agg(
        size=("label", "size"),
        class_count=("label", "nunique"),
        label=("label", "first"),
    )
    if (summary["class_count"] != 1).any():
        raise ValueError("A duplicate group contains different cancellation labels.")

    assignment: dict[str, str] = {}
    for label in (0, 1):
        class_groups = [
            (str(group_id), int(size))
            for group_id, size in summary.loc[summary["label"] == label, "size"].items()
        ]
        if len(class_groups) < len(SPLITS):
            raise ValueError(f"Class {label} has too few distinct groups for three splits.")
        random.Random(seed + label).shuffle(class_groups)
        class_groups.sort(key=lambda item: item[1], reverse=True)
        target = {
            name: sum(size for _, size in class_groups) * ratios[name] for name in SPLITS
        }
        assigned = {name: 0 for name in SPLITS}
        for group_id, size in class_groups:
            chosen = max(SPLITS, key=lambda name: target[name] - assigned[name])
            assignment[group_id] = chosen
            assigned[chosen] += size

    split_for_rows = [assignment[group_id] for group_id in group_ids]
    manifest = pd.DataFrame({
        "row_id": ids.astype(int).to_numpy(),
        "split": split_for_rows,
    }).sort_values("row_id").reset_index(drop=True)
    if any((manifest["split"] == name).sum() == 0 for name in SPLITS):
        raise ValueError("The group sizes prevent three nonempty splits.")

    class_balance: dict[str, dict[str, float]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    row_counts: dict[str, int] = {}
    for name in SPLITS:
        split_labels = labels[[split_name == name for split_name in split_for_rows]]
        counts = split_labels.value_counts().reindex([0, 1], fill_value=0)
        row_counts[name] = int(len(split_labels))
        class_counts[name] = {str(label): int(counts[label]) for label in (0, 1)}
        class_balance[name] = {
            str(label): float(counts[label] / len(split_labels)) for label in (0, 1)
        }
    metadata: dict[str, Any] = {
        "seed": seed,
        "requested_ratios": ratios,
        "achieved_ratios": {
            name: row_counts[name] / len(manifest) for name in SPLITS
        },
        "row_counts": row_counts,
        "class_counts": class_counts,
        "class_balance": class_balance,
        "duplicate_group_count": int((summary["size"] > 1).sum()),
        "duplicate_extra_row_count": int((summary["size"] - 1).sum()),
        "grouping": "SHA-256 of all 31 source fields after trim/NULL normalization",
        "assignment_method": (
            "Shuffle groups within each class with seed+class, sort largest first, "
            "then place each group in the split with the largest remaining row quota."
        ),
        "split_version": split_version,
    }
    return SplitResult(manifest, metadata)


def run(config: dict, project_root: Path) -> None:
    """Write the label-free manifest and its reproducibility metadata."""
    data_path = project_root / config["data"]["raw_path"]
    data = _read_h2(data_path)
    expected = int(config["data"]["expected_rows"])
    if len(data) != expected:
        raise ValueError(f"Expected {expected} H2 rows; found {len(data)}.")
    result = create_split(data, config)
    output_dir = project_root / config["paths"]["data_outputs"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.csv"
    metadata_path = output_dir / "split_metadata.json"
    metadata = {
        **result.metadata,
        "data_checksum_sha256": _file_sha256(data_path),
        "source_rows": len(data),
        "source_columns": len(OFFICIAL_COLUMNS),
        "source_header_mapping": data.attrs["header_mapping"],
    }
    if manifest_path.exists() or metadata_path.exists():
        if not (manifest_path.exists() and metadata_path.exists()):
            raise ValueError("An incomplete previous split exists; inspect it before rerunning.")
        previous_manifest = pd.read_csv(manifest_path)
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not previous_manifest.equals(result.manifest)
            or previous_metadata.get("data_checksum_sha256") != metadata["data_checksum_sha256"]
            or previous_metadata.get("split_version") != metadata["split_version"]
            or previous_metadata.get("requested_ratios") != metadata["requested_ratios"]
        ):
            raise ValueError("A different split already exists; preserve and version it before replacing.")
    result.manifest.to_csv(manifest_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(result.manifest):,} assignments to {manifest_path}")
    print(f"Split counts: {metadata['row_counts']}")


def main() -> None:
    """Standalone entry: python -m src.data.split."""
    parser = argparse.ArgumentParser(description="Prepare the City Hotel H2 split.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    import yaml

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The configuration must contain a YAML mapping.")
    run(config, project_root)


if __name__ == "__main__":
    main()
