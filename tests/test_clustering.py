"""Behavior checks for unsupervised baseline evaluation and interpretation."""
import numpy as np
import pandas as pd
import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

import main as cli
from src.clustering import baseline
from src.clustering.baseline import evaluate, profiles
from src.evaluation.results import REQUIRED_FIELDS
from src.features.policy import get_features
from src.preprocessing.builders import NUMERIC_COLUMNS
from src.settings import load_config


def test_single_cluster_scores_are_unavailable():
    scores, notes, ids = evaluate(np.zeros((5, 2)), np.zeros(5), 42)
    assert scores == {"silhouette": None, "davies_bouldin": None}
    assert notes and len(ids) == 0


def test_metric_sample_is_reproducible_and_does_not_use_pca():
    Z = np.r_[np.zeros((20, 3)), np.full((20, 3), 10.)]
    labels = np.repeat([0, 1], 20)
    scores, _, ids = evaluate(Z, labels, 42, sample_size=15)
    scores2, _, ids2 = evaluate(Z, labels, 42, sample_size=15)
    assert np.array_equal(ids, ids2) and scores == scores2
    assert len(ids) == len(set(ids)) == 15
    assert scores["silhouette"] == 1 and scores["davies_bouldin"] == 0


def test_profiles_keep_original_units_and_missing_counts():
    X = pd.DataFrame({"LeadTime": [10., 30., 100.], "Children": [0., np.nan, 2.], "Meal": ["BB", "BB", "HB"]})
    summary, frequencies = profiles(X, np.array([0, 0, 1]))
    assert summary.loc[0, "LeadTime_mean"] == 20
    assert summary.loc[0, "Children_missing"] == 1
    assert summary["size"].sum() == len(X)
    assert summary["share"].sum() == 1
    assert frequencies.groupby(["cluster", "feature"])["share"].sum().eq(1).all()


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """Small synthetic bookings; no source dataset or reserved rows are used."""
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config.yaml"))
    config["paths"]["clustering_outputs"] = "custom/results"
    config["paths"]["artifacts"] = "custom/artifacts"
    config["data"]["raw_path"] = "custom/source.csv"
    source = tmp_path / config["data"]["raw_path"]
    source.parent.mkdir(parents=True)
    source.write_text("synthetic source identity for a mocked task loader\n")
    ids = pd.Index(range(101, 161), name="row_id")
    X = pd.DataFrame(index=ids)
    for column in get_features("clustering", config):
        X[column] = np.arange(60) % 3 if column in NUMERIC_COLUMNS else np.tile(["A", "B", "C"], 20)
    X["LeadTime"] = np.arange(60, dtype=float)
    X["IsRepeatedGuest"] = np.r_[1, np.zeros(59)]
    X.loc[ids[0], "Children"] = np.nan
    calls, fitted_ids, builders, pca_shapes = [], [], [], []

    def task_data(task, root, split, *, config):
        assert task == "clustering" and Path(root) == tmp_path
        calls.append((split, config))
        assert split == "train", "Only frozen training rows may be requested"
        return X.copy(), None

    original_fit = ColumnTransformer.fit_transform
    original_pca = PCA.fit_transform
    original_builder = baseline.build_preprocessor

    def capture_fit(transformer, frame, y=None, **kwargs):
        fitted_ids.append(frame.index.tolist())
        Z = original_fit(transformer, frame, y=y, **kwargs)
        assert np.max(np.abs(Z)) > 3, "M1 representation must not acquire the M2 cap"
        return Z

    def capture_builder(task, passed, *, model_family):
        builders.append((task, passed, model_family))
        return original_builder(task, passed, model_family=model_family)

    def capture_pca(transformer, values, y=None):
        pca_shapes.append(values.shape)
        return original_pca(transformer, values, y=y)

    monkeypatch.setattr(baseline, "get_task_data", task_data)
    monkeypatch.setattr(baseline, "build_preprocessor", capture_builder)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", capture_fit)
    monkeypatch.setattr(PCA, "fit_transform", capture_pca)
    monkeypatch.setattr(baseline.sys, "argv", ["pytest"])
    return tmp_path, config, X, calls, fitted_ids, builders, pca_shapes


def test_runner_is_train_only_and_writes_shared_provenance_and_outputs(experiment):
    root, config, X, calls, fitted_ids, builders, pca_shapes = experiment
    baseline.run(config, root)
    output = root / config["paths"]["clustering_outputs"]
    record = json.loads((output / "metrics.json").read_text())
    assert [split for split, _ in calls] == ["train"]
    assert calls[0][1] is config and builders == [("clustering", config, "distance")]
    assert fitted_ids == [X.index.tolist()]
    assert pca_shapes == [tuple(record["transformed_shape"])]
    assert REQUIRED_FIELDS <= record.keys()
    assert record["data_checksum"] == hashlib.sha256((root / config["data"]["raw_path"]).read_bytes()).hexdigest()
    assert record["feature_policy_version"] == config["decisions"]["feature_policy_version"]
    assert record["split_version"] == config["decisions"]["split_version"]
    assert record["features"] == list(X.columns)
    assert record["row_counts"] == {"train": len(X)} and record["task_specific_exclusions"] == 0
    assert record["evaluation_subset"] == "train" and record["test_set_evaluated"] is False
    assert record["preprocessing"] == config["preprocessing"]
    assert record["effective_config"] == config
    assert record["parameters"]["n_clusters"] == 3
    assert record["parameters"]["n_init"] == 10 and record["seed"] == 42
    assert record["command_source"] == "programmatic_default"
    assert sum(record["cluster_sizes"].values()) == len(X)
    assert record["train_seconds"] >= 0 and record["preprocess_seconds"] >= 0
    assert record["environment"]["python"] and record["environment"]["scikit-learn"]
    assert pd.read_csv(output / "cluster_profiles.csv")["size"].sum() == len(X)
    assert not pd.read_csv(output / "category_profiles.csv").empty
    sample = pd.read_csv(output / "silhouette_sample.csv").row_id
    assert sample.is_unique and set(sample) <= set(X.index)
    for name in record["figure_paths"]:
        assert (root / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("outcome", ["ADR", "IsCanceled", "ReservationStatus", "ReservationStatusDate"])
def test_runner_rejects_outcome_features_before_fitting(experiment, monkeypatch, outcome):
    root, config, X, _, fitted_ids, _, _ = experiment
    unsafe = X.assign(**{outcome: 0})
    monkeypatch.setattr(baseline, "get_task_data", lambda *args, **kwargs: (unsafe, None))
    with pytest.raises(ValueError, match="exclude supervised outcomes"):
        baseline.run(config, root)
    assert fitted_ids == []


def test_runner_rejects_any_supplied_target(experiment, monkeypatch):
    root, config, X, _, fitted_ids, _, _ = experiment
    monkeypatch.setattr(baseline, "get_task_data", lambda *args, **kwargs: (X, pd.Series(0, index=X.index)))
    with pytest.raises(ValueError, match="exclude supervised outcomes"):
        baseline.run(config, root)
    assert fitted_ids == []


def test_cli_propagates_alternative_config_and_records_actual_command(experiment, monkeypatch):
    root, config, X, calls, fitted_ids, _, _ = experiment
    # Removing the optional log transform is supported by the shared builder.
    config["preprocessing"]["skewed_numeric_features"] = []
    config_path = root / "custom configuration.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    monkeypatch.setattr(baseline.sys, "argv", ["main.py", "--config", config_path.name, "baseline", "--task", "clustering"])
    assert cli.main() == 0
    record = json.loads((root / config["paths"]["clustering_outputs"] / "metrics.json").read_text())
    assert record["command"] == "python main.py --config 'custom configuration.yaml' baseline --task clustering"
    assert record["command_source"] == "cli"
    assert record["effective_config"] == config
    assert record["preprocessing"]["skewed_numeric_features"] == []
    assert calls[0][1] == config and fitted_ids == [X.index.tolist()]
    assert not (root / "outputs/clustering").exists()


def test_m1_cli_has_baseline_route_and_no_m2_route():
    args = cli.build_parser().parse_args(["baseline", "--task", "clustering"])
    assert args.command == "baseline" and args.task == "clustering"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["cluster", "view"])
