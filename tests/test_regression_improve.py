"""Focused integration tests for optional regression improvement experiment."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.policy import get_features
from src.regression import baseline, improve
from src.settings import load_config


def _fixture(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config.yaml"))
    rng = np.random.default_rng(7)
    feats = get_features("regression", cfg)
    numeric = {"LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "Adults", "Children", "Babies", "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled"}
    n = 120
    cols = {}
    for feature in feats:
        if feature == "LeadTime":
            cols[feature] = rng.integers(0, 365, size=n)
        elif feature == "ArrivalDateYear":
            cols[feature] = rng.choice([2015, 2016, 2017], size=n)
        elif feature in numeric:
            cols[feature] = rng.integers(0, 4, size=n)
        else:
            cols[feature] = rng.choice(["A", "B", "C"], size=n)
    x = pd.DataFrame(cols, index=pd.Index(range(120), name="row_id"))
    y = pd.Series(65 + 0.12 * x.LeadTime + 12 * (x.Meal == "A") + rng.normal(0, 4, n), index=x.index)
    training, validation = (x.iloc[:95], y.iloc[:95]), (x.iloc[95:], y.iloc[95:])
    meta = {"split_version": cfg["decisions"]["split_version"], "source_rows": 79330, "source_columns": 31, "data_checksum_sha256": "a" * 64, "row_counts": {"train": 95, "validation": 25, "test": 11900}}
    path = tmp_path / "outputs" / "data"
    path.mkdir(parents=True, exist_ok=True)
    (path / "split_metadata.json").write_text(json.dumps(meta))
    calls = []
    def get_data(task, root, subset, *, config):
        calls.append(subset)
        assert subset in {"train", "validation"}, "The test split must be untouched"
        return training if subset == "train" else validation
    monkeypatch.setattr(baseline, "get_task_data", get_data)
    monkeypatch.setattr(improve, "get_task_data", get_data)
    baseline.run(cfg, tmp_path)
    calls.clear()
    return cfg, meta, calls


def test_models_and_reports_with_frozen_validation(tmp_path, monkeypatch):
    cfg, meta, calls = _fixture(tmp_path, monkeypatch)
    results = improve.run(cfg, tmp_path)
    assert calls == ["train", "validation"]
    assert len(results["comparison"]) == 3
    assert all(np.isfinite([r["mae"], r["rmse"], r["r2"]]).all() for r in results["comparison"])
    assert results["selected_model"] == min(results["comparison"], key=lambda r:(r["mae"],r["rmse"]))["model"]
    assert results["test_set_evaluated"] is False
    for p in ["comparison.json", "comparison.csv", "comparison.png", "enhanced_residuals.png", "hist_gradient_boosting_mae_metrics.json", "random_forest_regularized_metrics.json"]:
        assert (tmp_path / "outputs" / "regression" / p).stat().st_size > 100
    report = (tmp_path / "reports" / "regression_enhancement_m1.md").read_text()
    assert "FULL" not in report or "Full" in report  # Smoke check that report was generated.
    assert "validation" in report.lower()
    assert "not loaded or evaluated" in report
    # Baseline record remains unmodified by enhancement.
    baseline_result = json.loads((tmp_path/"outputs"/"regression"/"metrics.json").read_text())
    assert baseline_result["model"] == "linear_regression"


def test_mismatched_source_checksum_fails_before_loading_data(tmp_path, monkeypatch):
    cfg, _, calls = _fixture(tmp_path, monkeypatch)
    path = tmp_path / "outputs" / "data" / "split_metadata.json"
    meta = json.loads(path.read_text())
    meta["data_checksum_sha256"] = "b" * 64
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Baseline data_checksum mismatch"):
        improve.run(cfg, tmp_path)
    assert calls == []


def test_forbidden_input_fails_before_loading_data(tmp_path, monkeypatch):
    cfg, _, calls = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(improve, "get_features", lambda *_: ["ADR", "LeadTime"])
    with pytest.raises(ValueError, match="leakage-prone"):
        improve.run(cfg, tmp_path)
    assert calls == []
