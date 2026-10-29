"""Checks for the shared experiment-result writer."""

import json

import pytest

from src.evaluation.results import save_result


@pytest.fixture
def result():
    """Synthetic record used only for testing, never as project evidence."""
    return {
        "task": "regression",
        "model": "test_model",
        "parameters": {},
        "data_checksum": "test-checksum",
        "feature_policy_version": "test-policy",
        "split_version": "test-split",
        "evaluation_subset": "validation",
        "seed": 42,
        "metrics": {"MAE": 1.5, "R2": None},
        "metric_notes": {"R2": "Unavailable in this synthetic test."},
        "train_seconds": 0.25,
        "command": "test command",
        "figure_paths": [],
    }


def test_saves_record_and_creates_output_directory(tmp_path, result):
    path = tmp_path / "nested" / "metrics.json"

    save_result(result, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == result


def test_missing_field_is_rejected(tmp_path, result):
    del result["data_checksum"]
    path = tmp_path / "metrics.json"

    with pytest.raises(ValueError, match="Missing result fields"):
        save_result(result, path)

    assert not path.exists()


@pytest.mark.parametrize("seconds", [-1, float("inf"), float("nan"), True])
def test_invalid_runtime_is_rejected(tmp_path, result, seconds):
    result["train_seconds"] = seconds

    with pytest.raises(ValueError, match="train_seconds"):
        save_result(result, tmp_path / "metrics.json")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_invalid_metric_preserves_existing_file(tmp_path, result, value):
    path = tmp_path / "metrics.json"
    save_result(result, path)
    original = path.read_bytes()

    result["metrics"]["MAE"] = value

    with pytest.raises(ValueError, match="JSON-compatible"):
        save_result(result, path)

    assert path.read_bytes() == original


@pytest.mark.parametrize("task", ["regression", "classification", "clustering"])
def test_accepts_each_task_and_preserves_extra_fields(tmp_path, result, task):
    result["task"] = task
    result["population"] = "synthetic test population"
    path = tmp_path / "metrics.json"

    save_result(result, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["task"] == task
    assert saved["population"] == "synthetic test population"