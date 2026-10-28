"""Read source tokens separately from a conservatively normalized view."""

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path

import pandas as pd

from .schema import MISSING_TOKENS, NUMERIC_COLUMNS, NOT_APPLICABLE_COLUMNS


@dataclass
class Dataset:
    path: Path
    checksum: str
    raw: pd.DataFrame
    data: pd.DataFrame


def read_dataset(path: str | Path) -> Dataset:
    """Keep all rows. The zero-based CSV record position becomes the row_id index.

    raw retains every field as text, including padded NULL tokens. data strips
    outside whitespace, preserves Agent/Company NULL as not applicable, maps
    other documented missing tokens, and parses known numerical columns.
    No imputation, row filtering, or learning
    takes place. Non-numeric tokens are reported by validation, not hidden.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"H2 CSV not found: {path}. Set data.raw_path in config.yaml.")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = csv.reader(stream)
        header = next(records, [])
        if not header or len(header) != len(set(header)):
            raise ValueError("CSV header is empty or contains duplicate column labels.")
        for record_number, record in enumerate(records, start=2):
            if len(record) != len(header):
                raise ValueError(f"CSV record {record_number} has {len(record)} fields; expected {len(header)}.")
    if "row_id" in header:
        raise ValueError("Source must not already contain the reserved row_id field.")
    raw = pd.read_csv(path, dtype="string", keep_default_na=False, na_filter=False, encoding="utf-8-sig")
    raw.index = pd.RangeIndex(len(raw), name="row_id")
    data = raw.copy()
    for column in data.columns:
        values = data[column].str.strip()
        missing_tokens = MISSING_TOKENS - {"NULL"} if column in NOT_APPLICABLE_COLUMNS else MISSING_TOKENS
        if column in NOT_APPLICABLE_COLUMNS:
            values = values.mask(values.str.upper().eq("NULL"), "NULL")
        data[column] = values.mask(values.str.upper().isin(missing_tokens), pd.NA)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return Dataset(path, checksum, raw, data)


def load_h2(path: str | Path) -> pd.DataFrame:
    """Shared loader interface: a DataFrame with row_id and 31 source columns.

    Numerical fields are parsed, outer whitespace is trimmed, and unknown
    values are preserved as missing. Agent/Company NULL remains not applicable.
    The index and explicit row_id column both preserve source record positions.
    Validation and EDA use read_dataset when they also need original tokens.
    """
    dataset = read_dataset(path)
    frame = dataset.data.reset_index()
    frame.attrs["data_checksum_sha256"] = dataset.checksum
    frame.attrs["source_columns"] = list(dataset.raw.columns)
    return frame


def arrival_dates(data: pd.DataFrame) -> pd.Series:
    from .schema import MONTHS
    month_numbers = data["ArrivalDateMonth"].map({name: i + 1 for i, name in enumerate(MONTHS)})
    return pd.to_datetime(pd.DataFrame({
        "year": data["ArrivalDateYear"], "month": month_numbers,
        "day": data["ArrivalDateDayOfMonth"],
    }, index=data.index), errors="coerce")
