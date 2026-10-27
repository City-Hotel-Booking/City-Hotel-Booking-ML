"""Focused tests for the independent Member 2 split workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.split import SNAKE_TO_OFFICIAL, _read_h2, create_split, run
from src.features.policy import OFFICIAL_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TEMP_DIR = PROJECT_ROOT / "outputs" / "tmp"
TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _config(row_count: int = 110) -> dict:
    return {
        "project": {"random_seed": 42},
        "data": {"raw_path": "data/H2.csv", "expected_rows": row_count},
        "decisions": {
            "split_method": "stratified_group_random",
            "split_parameters": {
                "train": 0.70,
                "validation": 0.15,
                "test": 0.15,
                "stratify_column": "IsCanceled",
                "group_columns": "all_31_source_columns",
            },
            "split_version": "test-v1",
        },
        "paths": {"data_outputs": "outputs/data"},
    }


def _sample_data() -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    rows: list[dict[str, str]] = []
    duplicate_pairs: list[tuple[int, int]] = []
    for number in range(100):
        row = {column: "fixed" for column in OFFICIAL_COLUMNS}
        row["IsCanceled"] = str(number % 2)
        row["LeadTime"] = str(number)
        row["ADR"] = str(number + 100)
        first_id = len(rows)
        rows.append(row.copy())
        if number % 11 == 0:
            duplicate_pairs.append((first_id, len(rows)))
            rows.append(row.copy())
    data = pd.DataFrame(rows, columns=OFFICIAL_COLUMNS)
    data.insert(0, "row_id", range(len(data)))
    return data, duplicate_pairs


class SplitTests(unittest.TestCase):
    def test_header_aliases_are_one_to_one(self) -> None:
        self.assertEqual(len(SNAKE_TO_OFFICIAL), 31)
        self.assertEqual(set(SNAKE_TO_OFFICIAL.values()), set(OFFICIAL_COLUMNS))

    def test_manifest_is_disjoint_complete_and_label_free(self) -> None:
        data, _ = _sample_data()
        result = create_split(data, _config(len(data)))
        manifest = result.manifest
        self.assertEqual(list(manifest.columns), ["row_id", "split"])
        self.assertEqual(len(manifest), len(data))
        self.assertEqual(set(manifest["row_id"]), set(range(len(data))))
        self.assertFalse(manifest["row_id"].duplicated().any())
        self.assertEqual(set(manifest["split"]), {"train", "validation", "test"})

    def test_identical_rows_never_cross_splits(self) -> None:
        data, pairs = _sample_data()
        manifest = create_split(data, _config(len(data))).manifest.set_index("row_id")
        for first_id, duplicate_id in pairs:
            self.assertEqual(manifest.loc[first_id, "split"], manifest.loc[duplicate_id, "split"])

    def test_class_balance_is_similar_across_splits(self) -> None:
        data, _ = _sample_data()
        manifest = create_split(data, _config(len(data))).manifest
        overall = (data["IsCanceled"] == "1").mean()
        labels = data.set_index("row_id")["IsCanceled"]
        for name in ("train", "validation", "test"):
            row_ids = manifest.loc[manifest["split"] == name, "row_id"]
            share = (labels.loc[row_ids] == "1").mean()
            self.assertLess(abs(share - overall), 0.06)

    def test_same_seed_reproduces_the_manifest(self) -> None:
        data, _ = _sample_data()
        first = create_split(data, _config(len(data))).manifest
        second = create_split(data, _config(len(data))).manifest
        pd.testing.assert_frame_equal(first, second)

    def test_reader_accepts_official_and_snake_case_headers(self) -> None:
        data, _ = _sample_data()
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_DIR) as folder:
            root = Path(folder)
            for snake_case in (False, True):
                path = root / ("snake.csv" if snake_case else "official.csv")
                source = data.drop(columns="row_id").iloc[:8].copy()
                source.loc[source.index[0], "Meal"] = " HB       "
                source.loc[source.index[0], "Company"] = "   NULL  "
                if snake_case:
                    reverse = {official: snake for snake, official in SNAKE_TO_OFFICIAL.items()}
                    source = source.rename(columns=reverse)
                source.to_csv(path, index=False)
                loaded = _read_h2(path)
                self.assertEqual(list(loaded.columns), ["row_id", *OFFICIAL_COLUMNS])
                self.assertEqual(loaded["row_id"].tolist(), list(range(8)))
                self.assertEqual(loaded.loc[0, "Meal"], "HB")
                self.assertTrue(pd.isna(loaded.loc[0, "Company"]))
                self.assertEqual(len(loaded.attrs["header_mapping"]), 31)

    def test_run_writes_manifest_and_metadata(self) -> None:
        data, _ = _sample_data()
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_DIR) as folder:
            root = Path(folder)
            (root / "data").mkdir()
            data.drop(columns="row_id").to_csv(root / "data" / "H2.csv", index=False)
            run(_config(len(data)), root)
            manifest = pd.read_csv(root / "outputs" / "data" / "split_manifest.csv")
            metadata = json.loads(
                (root / "outputs" / "data" / "split_metadata.json").read_text()
            )
            self.assertEqual(list(manifest.columns), ["row_id", "split"])
            self.assertEqual(sum(metadata["row_counts"].values()), len(data))
            self.assertEqual(metadata["split_version"], "test-v1")
            self.assertEqual(len(metadata["data_checksum_sha256"]), 64)
            self.assertEqual(metadata["duplicate_group_count"], 10)
            run(_config(len(data)), root)
            changed = _config(len(data))
            changed["project"]["random_seed"] = 43
            with self.assertRaisesRegex(ValueError, "different split"):
                run(changed, root)

    @unittest.skipUnless((PROJECT_ROOT / "data" / "H2.csv").is_file(), "H2.csv unavailable")
    def test_official_h2_union_has_79330_source_rows(self) -> None:
        data = _read_h2(PROJECT_ROOT / "data" / "H2.csv")
        self.assertEqual(len(data), 79_330)
        result = create_split(data, _config(79_330))
        manifest = result.manifest
        self.assertEqual(len(manifest), 79_330)
        self.assertEqual(set(manifest["row_id"]), set(range(79_330)))
        self.assertFalse(manifest["row_id"].duplicated().any())
        independent_groups = pd.util.hash_pandas_object(
            data.loc[:, list(OFFICIAL_COLUMNS)], index=False
        )
        grouped = pd.DataFrame({
            "group": independent_groups.to_numpy(),
            "split": manifest["split"].to_numpy(),
        })
        self.assertTrue((grouped.groupby("group")["split"].nunique() == 1).all())
        balances = [result.metadata["class_balance"][name]["1"] for name in (
            "train", "validation", "test",
        )]
        self.assertLess(max(balances) - min(balances), 0.02)


if __name__ == "__main__":
    unittest.main()
