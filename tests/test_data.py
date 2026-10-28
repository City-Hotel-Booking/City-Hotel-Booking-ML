"""Behavioral checks for source preservation and evidence that affects handoffs."""

import csv
from contextlib import redirect_stdout, redirect_stderr
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data.load import read_dataset, load_h2
from src.data.schema import SOURCE_COLUMNS
from src.data.validate import inspect_dataset
from src.data.review import build_review


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "H2.csv"

    def row(self, **updates):
        record = dict.fromkeys(SOURCE_COLUMNS, "0")
        record.update({"ArrivalDateYear": "2015", "ArrivalDateMonth": "July", "ArrivalDateWeekNumber": "27",
                       "ArrivalDateDayOfMonth": "1", "StaysInWeekNights": "2", "Adults": "2",
                       "Meal": "BB       ", "Country": "PRT", "MarketSegment": "Online TA",
                       "DistributionChannel": "TA/TO", "ReservedRoomType": "A   ", "AssignedRoomType": "A   ",
                       "DepositType": "No Deposit  ", "Agent": "001", "Company": "       NULL",
                       "CustomerType": "Transient", "ADR": "100", "ReservationStatus": "Check-Out",
                       "ReservationStatusDate": "7/3/2015"})
        record.update(updates)
        return record

    def write(self, rows, fields=None):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or SOURCE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def test_loader_preserves_file_records_and_categorical_ids(self):
        self.write([self.row(Children="NA", Country=""), self.row(), self.row()])
        before = self.path.read_bytes()
        dataset = read_dataset(self.path)
        self.assertEqual(dataset.checksum, hashlib.sha256(before).hexdigest())
        self.assertEqual(dataset.raw.loc[0, "Company"], "       NULL")
        self.assertEqual(dataset.data.loc[0, "Agent"], "001")
        self.assertTrue(pd.isna(dataset.data.loc[0, "Children"]))
        self.assertTrue(pd.isna(dataset.data.loc[0, "Country"]))
        self.assertTrue(dataset.data["Company"].eq("NULL").all())
        self.assertEqual(dataset.data.index.tolist(), [0, 1, 2])
        self.assertEqual(dataset.data.index.name, "row_id")
        self.assertEqual(len(dataset.data), 3)
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_counts_padding_and_duplicates_are_distinct(self):
        self.write([self.row(), self.row(), self.row(Meal="BB")])
        report = inspect_dataset(read_dataset(self.path))
        self.assertEqual(report["duplicates"]["raw_excess_rows"], 1)
        self.assertEqual(report["duplicates"]["raw_rows_in_duplicate_groups"], 2)
        self.assertEqual(report["duplicates"]["normalized_excess_rows"], 2)
        self.assertEqual(report["columns"]["Meal"]["whitespace_padded_count"], 2)
        self.assertEqual(report["columns"]["Company"]["normalized_missing_count"], 0)
        self.assertEqual(report["columns"]["Company"]["not_applicable_count"], 3)
        self.assertEqual(report["normalization"]["rows_removed"], 0)

    def test_malformed_numbers_dates_and_outcomes_are_flagged(self):
        self.write([self.row(Children="oops", ArrivalDateMonth="February", ArrivalDateDayOfMonth="31",
                             IsCanceled="1", ReservationStatus="Check-Out", ADR="-5"), self.row(Adults="1.5")])
        report = inspect_dataset(read_dataset(self.path))
        self.assertEqual(report["columns"]["Children"]["invalid_numeric_tokens"], 1)
        self.assertEqual(report["checks"]["invalid_arrival_dates"], 1)
        self.assertEqual(report["checks"]["Adults_noninteger"], 1)
        self.assertEqual(report["checks"]["cancellation_status_disagreements"], 1)
        self.assertEqual(report["checks"]["ADR_negative"], 1)
        self.assertEqual(report["status"], "failed")

    def test_schema_failure_is_written_and_analysis_stops(self):
        self.write([{"ADR": "100"}], fields=["ADR"])
        from src.data.eda import run
        config = {
            "data": {"raw_path": "H2.csv", "expected_rows": 79330, "expected_columns": 31},
            "paths": {"data_outputs": "out/data", "eda_outputs": "out/eda", "reports": "reports"},
        }
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            with self.assertRaisesRegex(ValueError, "Validation failed"):
                run(config, self.root)
        report = json.loads((self.root / "out/data/validation.json").read_text())
        self.assertIn("IsCanceled", report["schema"]["missing_columns"])
        self.assertFalse((self.root / "out/eda").exists())

    def test_duplicate_headers_and_ragged_records_are_rejected(self):
        self.path.write_text("ADR,ADR\n1,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_dataset(self.path)
        self.path.write_text("ADR,Adults\n1,2,3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fields"):
            read_dataset(self.path)

    def test_unknown_categories_are_retained_for_review(self):
        self.write([self.row(MarketSegment="Undefined", Country="XYZ")])
        dataset = read_dataset(self.path)
        self.assertEqual(dataset.data.loc[0, "MarketSegment"], "Undefined")
        self.assertEqual(dataset.data.loc[0, "Country"], "XYZ")

    def test_nonfinite_rates_are_reported_without_invalid_json(self):
        self.write([self.row(ADR="inf"), self.row()])
        report = inspect_dataset(read_dataset(self.path))
        self.assertEqual(report["columns"]["ADR"]["nonfinite_numeric_values"], 1)
        self.assertEqual(report["status"], "failed")
        json.dumps(report, allow_nan=False)

    def test_review_preserves_unknown_guest_counts_and_does_not_double_count(self):
        self.write([self.row(Adults="0", Children="NA", Babies="0", ADR="0", StaysInWeekNights="0"),
                    self.row(Adults="0", Children="0", Babies="0", ADR="0"), self.row()])
        dataset = read_dataset(self.path)
        summary, tables = build_review(dataset, inspect_dataset(dataset))
        flags = {row["flag"]: row for row in summary["flag_profiles"]}
        self.assertEqual(flags["zero_guests"]["records"], 1)
        self.assertEqual(flags["zero_ADR"]["records"], 2)
        self.assertEqual(summary["distinct_rows_with_zero_nights_ADR_or_guests"], 2)
        self.assertEqual(tables["review_flag_combinations.csv"]["records"].sum(), 3)
        self.assertEqual(len(dataset.data), 3)

    def test_review_distinguishes_duplicate_groups_members_and_excess(self):
        self.write([self.row(), self.row(), self.row(), self.row(ADR="80"), self.row(ADR="80"), self.row(ADR="150")])
        dataset = read_dataset(self.path)
        summary, tables = build_review(dataset, inspect_dataset(dataset))
        self.assertEqual(summary["duplicate_groups"], 2)
        self.assertEqual(summary["raw_duplicate_group_members"], 5)
        self.assertEqual(summary["raw_excess_duplicates"], 3)
        self.assertEqual(summary["largest_identical_group"], 3)
        self.assertEqual(tables["duplicate_group_examples.csv"]["row_id"].tolist(), [0, 3])


    def test_public_loader_returns_dataframe_with_stable_source_ids(self):
        self.write([self.row(), self.row(Children="NA")])
        frame = load_h2(self.path)
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(list(frame.columns), ["row_id", *SOURCE_COLUMNS])
        self.assertEqual(frame["row_id"].tolist(), [0, 1])
        self.assertTrue(frame["Company"].eq("NULL").all())
        self.assertTrue(pd.isna(frame.loc[1, "Children"]))
        self.assertEqual(frame.attrs["data_checksum_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_status_date_supports_original_mirror_and_supplied_export(self):
        self.write([self.row(ReservationStatusDate="2015-07-03"), self.row(ReservationStatusDate="7/3/2015")])
        report = inspect_dataset(read_dataset(self.path))
        self.assertEqual(report["checks"]["invalid_reservation_status_dates"], 0)


if __name__ == "__main__":
    unittest.main()
