"""Shared data access using the frozen H2 split and task feature policy."""

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


def load_manifest(
    project_root: Path = PROJECT_ROOT,
    *,
    config: dict | None = None,
) -> pd.DataFrame:
    """Read and validate complete, label-free source assignments."""
    root = Path(project_root)
    if config is None:
        config = load_config(root / "config.yaml")

    path = root / config["paths"]["data_outputs"] / "split_manifest.csv"
    manifest = pd.read_csv(path)

    if list(manifest.columns) != ["row_id", "split"]:
        raise ValueError("Split manifest must contain only row_id and split.")

    if manifest.empty:
        raise ValueError("Split manifest must not be empty.")

    ids = pd.to_numeric(manifest["row_id"], errors="coerce")
    valid_ids = (
        ids.notna()
        & ids.ge(0)
        & ids.lt(len(manifest))
        & ids.mod(1).eq(0)
    )
    if not valid_ids.all():
        raise ValueError(
            "Manifest row_id values must be integer source positions "
            "from zero to the number of rows minus one."
        )

    ids = ids.astype("int64")
    if ids.duplicated().any():
        raise ValueError("Split manifest contains repeated row_id values.")

    if set(ids) != set(range(len(manifest))):
        raise ValueError("Split manifest does not cover every source position.")

    if not manifest["split"].isin(SPLITS).all():
        raise ValueError("Split manifest contains an unknown split name.")

    if set(manifest["split"]) != SPLITS:
        raise ValueError("Train, validation and test splits must all be nonempty.")

    manifest["row_id"] = ids
    return manifest.sort_values("row_id").reset_index(drop=True)


def get_row_ids(
    split: str,
    project_root: Path = PROJECT_ROOT,
    *,
    config: dict | None = None,
) -> list[int]:
    """Return source positions assigned to one frozen split."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")

    manifest = load_manifest(project_root, config=config)
    return manifest.loc[manifest["split"].eq(split), "row_id"].tolist()


def get_task_data(
    task: str,
    project_root: Path,
    split: str,
    *,
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return task inputs and target while preserving source row IDs."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")

    root = Path(project_root)
    if config is None:
        config = load_config(root / "config.yaml")

    features = get_features(task, config)
    target = get_target(task)

    metadata_path = (
        root / config["paths"]["data_outputs"] / "split_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata["split_version"] != config["decisions"]["split_version"]:
        raise ValueError("Split version differs from the frozen manifest.")

    frame = load_h2(root / config["data"]["raw_path"])
    if frame.attrs["data_checksum_sha256"] != metadata["data_checksum_sha256"]:
        raise ValueError("H2 source checksum differs from the frozen manifest.")

    manifest = load_manifest(root, config=config)
    indexed = frame.set_index("row_id")

    if not indexed.index.is_unique:
        raise ValueError("Source row_id values are not unique.")

    if (
        len(manifest) != len(indexed)
        or set(manifest["row_id"]) != set(indexed.index)
    ):
        raise ValueError("Split manifest does not match the source row IDs.")

    ids = manifest.loc[manifest["split"].eq(split), "row_id"].tolist()
    data = indexed.loc[ids]

    X = data.loc[:, features].copy()
    y = data[target].copy() if target is not None else None
    return X, y


def run(config: dict, project_root: Path) -> None:
    """Create or verify the split using the shared source reader."""
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