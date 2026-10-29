"""Checks for configuration handling and frozen split validation."""

import json

import pandas as pd
import pytest

from src.data import prepare


def write_manifest(root, rows):
    folder = root / "custom_outputs"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "split_manifest.csv", index=False)
    return {"paths": {"data_outputs": "custom_outputs"}}


def test_selected_config_controls_manifest_path(tmp_path):
    config = write_manifest(tmp_path, {
        "row_id": [0, 1, 2],
        "split": ["train", "validation", "test"],
    })

    ids = prepare.get_row_ids("validation", tmp_path, config=config)

    assert ids == [1]


def test_manifest_is_returned_in_source_order(tmp_path):
    config = write_manifest(tmp_path, {
        "row_id": [2, 0, 1],
        "split": ["test", "train", "validation"],
    })

    manifest = prepare.load_manifest(tmp_path, config=config)

    assert manifest["row_id"].tolist() == [0, 1, 2]
    assert manifest["split"].tolist() == ["train", "validation", "test"]


@pytest.mark.parametrize("ids", [
    [0, 1, 1],
    [-1, 1, 2],
    [0, 1.5, 2],
    [0, 1, 3],
    [0, None, 2],
    [0, "invalid", 2],
])
def test_invalid_source_ids_are_rejected(tmp_path, ids):
    config = write_manifest(tmp_path, {
        "row_id": ids,
        "split": ["train", "validation", "test"],
    })

    with pytest.raises(ValueError):
        prepare.load_manifest(tmp_path, config=config)


@pytest.mark.parametrize("splits", [
    ["train", "train", "test"],
    ["train", "validation", "unknown"],
    ["train", None, "test"],
])
def test_invalid_or_missing_splits_are_rejected(tmp_path, splits):
    config = write_manifest(tmp_path, {
        "row_id": [0, 1, 2],
        "split": splits,
    })

    with pytest.raises(ValueError):
        prepare.load_manifest(tmp_path, config=config)


def test_empty_manifest_is_rejected(tmp_path):
    config = write_manifest(tmp_path, {
        "row_id": [],
        "split": [],
    })

    with pytest.raises(ValueError, match="must not be empty"):
        prepare.load_manifest(tmp_path, config=config)


def test_label_column_is_rejected(tmp_path):
    config = write_manifest(tmp_path, {
        "row_id": [0, 1, 2],
        "split": ["train", "validation", "test"],
        "IsCanceled": [0, 1, 0],
    })

    with pytest.raises(ValueError, match="only row_id and split"):
        prepare.load_manifest(tmp_path, config=config)


@pytest.fixture
def task_setup(tmp_path, monkeypatch):
    config = write_manifest(tmp_path, {
        "row_id": [0, 1, 2],
        "split": ["validation", "train", "test"],
    })
    config.update({
        "data": {"raw_path": "custom_source.csv"},
        "decisions": {"split_version": "test-split"},
    })

    metadata = {
        "split_version": "test-split",
        "data_checksum_sha256": "test-checksum",
    }
    metadata_path = tmp_path / "custom_outputs" / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    frame = pd.DataFrame({
        "row_id": [0, 1, 2],
        "LeadTime": [10, 20, 30],
        "ADR": [80.0, 100.0, 120.0],
        "IsCanceled": [0, 1, 0],
    })
    frame.attrs["data_checksum_sha256"] = "test-checksum"

    def fake_load(path):
        assert path == tmp_path / "custom_source.csv"
        return frame

    def fake_features(task, selected_config):
        assert selected_config is config
        return ["LeadTime"]

    monkeypatch.setattr(prepare, "load_h2", fake_load)
    monkeypatch.setattr(prepare, "get_features", fake_features)

    return config, frame, metadata_path


@pytest.mark.parametrize("task,target", [
    ("regression", "ADR"),
    ("classification", "IsCanceled"),
    ("clustering", None),
])
def test_task_data_preserves_ids_and_returns_correct_target(
    tmp_path, task_setup, task, target
):
    config, frame, _ = task_setup

    X, y = prepare.get_task_data(
        task, tmp_path, "train", config=config
    )

    assert X.index.tolist() == [1]
    assert X.columns.tolist() == ["LeadTime"]
    assert X["LeadTime"].tolist() == [20]

    if target is None:
        assert y is None
    else:
        assert y.index.tolist() == [1]
        assert y.tolist() == [frame.loc[1, target]]


def test_source_checksum_mismatch_is_rejected(tmp_path, task_setup):
    config, frame, _ = task_setup
    frame.attrs["data_checksum_sha256"] = "different-source"

    with pytest.raises(ValueError, match="checksum"):
        prepare.get_task_data(
            "regression", tmp_path, "train", config=config
        )


def test_split_version_mismatch_is_rejected(tmp_path, task_setup):
    config, _, _ = task_setup
    config["decisions"]["split_version"] = "different-version"

    with pytest.raises(ValueError, match="Split version"):
        prepare.get_task_data(
            "regression", tmp_path, "train", config=config
        )


def test_incomplete_source_coverage_is_rejected(tmp_path, task_setup):
    config, frame, _ = task_setup
    frame.loc[3] = [3, 40, 140.0, 1]

    with pytest.raises(ValueError, match="source row IDs"):
        prepare.get_task_data(
            "regression", tmp_path, "train", config=config
        )


def test_unknown_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown split"):
        prepare.get_row_ids("unknown", tmp_path, config={})