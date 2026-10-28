"""Check Member 1's integration with the existing shared project contracts."""

import hashlib
import json
from pathlib import Path
import unittest

import pandas as pd

from main import RUNNERS, build_parser
from src.data.load import load_h2
from src.data.schema import SOURCE_COLUMNS
from src.features.policy import OFFICIAL_COLUMNS, get_features
from src.preprocessing.builders import build_preprocessor
from src.settings import load_config


ROOT = Path(__file__).resolve().parents[1]


class Member1IntegrationTests(unittest.TestCase):
    def test_shared_command_routes_and_config_option_are_preserved(self):
        self.assertEqual(tuple(SOURCE_COLUMNS), OFFICIAL_COLUMNS)
        self.assertEqual(RUNNERS["prepare"], "src.data.split")
        self.assertEqual(RUNNERS["validate"], "src.data.validate")
        args = build_parser().parse_args(["--config", "config.yaml", "baseline", "--task", "regression"])
        self.assertEqual(args.task, "regression")
        self.assertEqual(args.config, "config.yaml")

    @unittest.skipUnless((ROOT / "data/H2.csv").is_file(), "Local H2 source not supplied")
    def test_shared_loader_source_identity_and_preprocessor_compatibility(self):
        config = load_config(ROOT / "config.yaml")
        source = ROOT / config["data"]["raw_path"]
        frame = load_h2(source)
        metadata = json.loads((ROOT / "outputs/data/split_metadata.json").read_text())
        self.assertEqual(frame["row_id"].tolist(), list(range(79330)))
        self.assertEqual(frame.attrs["data_checksum_sha256"], metadata["data_checksum_sha256"])
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), metadata["data_checksum_sha256"])
        manifest = pd.read_csv(ROOT / "outputs/data/split_manifest.csv")
        train_ids = manifest.loc[manifest["split"].eq("train"), "row_id"].head(12).tolist()
        for task in ["regression", "classification", "clustering"]:
            features = get_features(task, config)
            # A small compatibility fit, not a model experiment or evaluation.
            transformer = build_preprocessor(task, config)
            selected = frame.loc[train_ids, features]
            output = transformer.fit_transform(selected)
            self.assertEqual(output.shape[0], 12)
            self.assertGreater(output.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
