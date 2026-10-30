"""Focused Member 3 regression tests; no H2.csv required for these fixtures."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.features.policy import get_features
from src.regression import baseline
from src.settings import load_config


@pytest.fixture
def config():
    return copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config.yaml"))


def _synthetic_data(config):
    """Controlled toy bookings with 16 approved features, disjoint IDs."""
    rng = np.random.default_rng(2026)
    features = get_features("regression", config)
    numeric = {
        "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber",
        "ArrivalDateDayOfMonth", "Adults", "Children", "Babies",
        "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    }
    n = 100
    columns = {}
    for key in features:
        if key == "LeadTime":
            columns[key] = rng.integers(0, 200, n).astype(float)
        elif key == "ArrivalDateYear":
            columns[key] = rng.choice([2015, 2016, 2017], n)
        elif key in numeric:
            columns[key] = rng.integers(0, 4, n).astype(float)
        else:
            columns[key] = rng.choice(["A", "B", "C"], n)
    x = pd.DataFrame(columns, index=pd.Index(range(100, 200), name="row_id"))
    y = pd.Series(
        60 + x["LeadTime"].to_numpy() * 0.3
        + (x["Meal"].eq("A")).to_numpy() * 10
        + rng.normal(0, 2, n),
        index=x.index, name="ADR",
    )
    return (x.iloc[:80], y.iloc[:80]), (x.iloc[80:], y.iloc[80:])


@pytest.fixture
def experiment(tmp_path, config, monkeypatch):
    (tmp_path / "outputs" / "data").mkdir(parents=True)
    metadata = {
        "split_version": config["decisions"]["split_version"],
        "source_rows": 79330,
        "source_columns": 31,
        "data_checksum_sha256": "a" * 64,
    }
    (tmp_path / "outputs" / "data" / "split_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    train, validation = _synthetic_data(config)
    calls = []

    def fake_task_data(task, project_root, split, *, config):
        calls.append(split)
        assert task == "regression"
        assert Path(project_root) == tmp_path
        if split == "train":
            return train
        if split == "validation":
            return validation
        raise AssertionError("Test split must never be requested by Member 3")

    monkeypatch.setattr(baseline, "get_task_data", fake_task_data)
    return tmp_path, config, train, validation, calls


def test_one_untuned_model_generates_real_outputs_without_touching_test(experiment):
    root, config, train, validation, calls = experiment
    observed_fit_ids = []
    original_fit_transform = ColumnTransformer.fit_transform

    def check_fit_ids(transformer, X, y=None, **kwargs):
        observed_fit_ids.append(X.index.to_list())
        return original_fit_transform(transformer, X, y=y, **kwargs)

    with patch.object(ColumnTransformer, "fit_transform", check_fit_ids):
        baseline.run(config, root)

    assert calls == ["train", "validation"]
    assert observed_fit_ids == [train[0].index.to_list()]
    output = root / "outputs" / "regression"
    result = json.loads((output / "metrics.json").read_text())
    assert (output / "residuals.png").stat().st_size > 1000
    report = (root / "reports" / "regression_m1.md").read_text()
    assert result["task"] == "regression"
    assert result["model"] == "linear_regression"
    assert result["evaluation_subset"] == "validation"
    assert result["training_subset"] == "train"
    assert result["test_set_evaluated"] is False
    assert result["seed"] == 42
    assert result["data_checksum"] == "a" * 64
    assert result["features"] == get_features("regression", config)
    assert len(result["features"]) == 16
    assert result["population"]["train"]["eligible_rows"] == 80
    assert result["population"]["validation"]["eligible_rows"] == 20
    assert set(result["metrics"]) == {"mae", "rmse", "r2"}
    assert all(np.isfinite(v) for v in result["metrics"].values())
    assert "Validation" in report and "MAE" in report
    assert "not loaded, predicted or evaluated" in report
    assert result["figure_paths"] == ["outputs/regression/residuals.png"]


def test_invalid_and_zero_adr_are_handled_without_clipping():
    features = pd.DataFrame({"LeadTime": [1, 2, 3, 4, 5]}, index=[10, 11, 12, 13, 14])
    target = pd.Series([0, 120, "NaN", 5400, np.inf], index=features.index)
    x, y, counts = baseline._finite_target_rows(features, target, "train")
    assert x.index.tolist() == [10, 11, 13]
    assert y.tolist() == [0.0, 120.0, 5400.0]
    assert counts["excluded_nonfinite_adr"] == 2
    assert counts["zero_adr_rows"] == 1
    assert counts["maximum_adr"] == 5400


def test_no_automatic_reindexing_of_misaligned_labels():
    x = pd.DataFrame({"LeadTime": [1, 2]}, index=[10, 11])
    y = pd.Series([1, 2], index=[11, 10])
    with pytest.raises(ValueError, match="alignment"):
        baseline._finite_target_rows(x, y, "train")


def test_disjoint_subset_requirement(experiment):
    root, config, train, validation, calls = experiment
    overlapping = (validation[0].copy(), validation[1].copy())
    overlapping[0].index = pd.Index([100] + list(range(181, 200)))
    overlapping[1].index = overlapping[0].index

    def bad_task_data(task, project_root, split, *, config):
        return train if split == "train" else overlapping

    with patch.object(baseline, "get_task_data", bad_task_data):
        with pytest.raises(ValueError, match="overlap"):
            baseline.run(config, root)
    assert not (root / "outputs" / "regression" / "metrics.json").exists()


@pytest.mark.parametrize("config_change", [
    lambda c: c["decisions"].update(adr_population="drop_canceled"),
    lambda c: c["decisions"]["prediction_point"].update(regression="after_checkin"),
    lambda c: c["targets"].update(regression="IsCanceled"),
    lambda c: c["baselines"].update(regression="random_forest"),
])
def test_scientific_contract_changes_are_rejected(config, config_change):
    config_change(config)
    with pytest.raises(ValueError):
        baseline._require_contract(config)


def test_forbidden_feature_policy_is_rejected(experiment, monkeypatch):
    root, config, train, validation, calls = experiment
    monkeypatch.setattr(baseline, "get_features", lambda task, conf: ["ADR"])
    with pytest.raises(ValueError, match="future-information"):
        baseline.run(config, root)
    assert calls == []


def test_split_metadata_inconsistency_prevents_output(experiment):
    root, config, train, validation, calls = experiment
    metadata_path = root / "outputs" / "data" / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["split_version"] = "different"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata split version"):
        baseline.run(config, root)
    assert not (root / "outputs" / "regression" / "metrics.json").exists()
