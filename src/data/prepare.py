"""Shared, unfitted data access for the three hotel-booking tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.load import load_h2
from src.data.split import run as create_manifest
from src.features.policy import get_features, get_target
from src.settings import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = {"train", "validation", "test"}


def load_manifest(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Read the frozen label-free assignments without generating a new split."""
    root = Path(project_root)
    config = load_config(root / "config.yaml")
    path = root / config["paths"]["data_outputs"] / "split_manifest.csv"
    manifest = pd.read_csv(path)
    if list(manifest.columns) != ["row_id", "split"]:
        raise ValueError("Split manifest must contain only row_id and split.")
    if manifest["row_id"].isna().any() or manifest["row_id"].duplicated().any():
        raise ValueError("Split manifest has missing or repeated row_id values.")
    if not manifest["split"].isin(SPLITS).all():
        raise ValueError("Split manifest contains an unknown split name.")
    return manifest


def get_row_ids(split: str, project_root: Path = PROJECT_ROOT) -> list[int]:
    """Return source positions assigned to one frozen split."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    manifest = load_manifest(project_root)
    return manifest.loc[manifest["split"].eq(split), "row_id"].astype(int).tolist()


def get_task_data(
    task: str, project_root: Path, split: str
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Select approved inputs and target; callers fit only on train folds."""
    root = Path(project_root)
    config = load_config(root / "config.yaml")
    metadata_path = root / config["paths"]["data_outputs"] / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["split_version"] != config["decisions"]["split_version"]:
        raise ValueError("Split version differs from the frozen manifest.")
    frame = load_h2(root / config["data"]["raw_path"])
    if frame.attrs["data_checksum_sha256"] != metadata["data_checksum_sha256"]:
        raise ValueError("H2 source checksum differs from the frozen manifest.")
    ids = get_row_ids(split, root)
    indexed = frame.set_index("row_id")
    if not indexed.index.is_unique:
        raise ValueError("Source row_id values are not unique.")
    data = indexed.loc[ids]
    features = get_features(task, config)
    target = get_target(task)
    X = data.loc[:, features].copy()
    y = data[target].copy() if target is not None else None
    return X, y


def run(config: dict, project_root: Path) -> None:
    """Create or verify the shared split through Member 1's source reader."""
    create_manifest(config, project_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the shared H2 split.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    path = Path(args.config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    run(load_config(path), PROJECT_ROOT)


if __name__ == "__main__":
    main()
