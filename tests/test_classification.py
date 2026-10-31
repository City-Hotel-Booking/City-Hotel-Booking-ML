"""Classification checks using synthetic data, never project evidence."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classification import baseline
from src.features.policy import get_features
from src.preprocessing.builders import NUMERIC_COLUMNS, build_preprocessor
from src.settings import load_config


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    config = copy.deepcopy(load_config(project_root / "config.yaml"))
    config["paths"]["data_outputs"] = "outputs/data"
    config["paths"]["classification_outputs"] = "outputs/classification"
    config["paths"]["reports"] = "reports"
    features = get_features("classification", config)

    def sample(size, first_id):
        row_ids = pd.Index(range(first_id, first_id + size), name="row_id")
        labels = np.arange(size) % 2
        frame = pd.DataFrame(index=row_ids)

        for feature in features:
            if feature in NUMERIC_COLUMNS:
                frame[feature] = labels + 1
            else:
                frame[feature] = np.where(
                    labels == 1, "synthetic_a", "synthetic_b"
                )

        target = pd.Series(labels, index=row_ids, name="IsCanceled")
        return frame, target

    subsets = {
        "train": sample(40, 100),
        "validation": sample(12, 200),
    }
    calls = []
    fitted_rows = []
    families = []
    transformers = []

    def task_data(task, root, split, *, config):
        assert task == "classification"
        assert Path(root) == tmp_path
        assert config["project"]["random_seed"] == 42
        calls.append(split)
        if split not in subsets:
            pytest.fail("The runner requested a forbidden subset.")
        X, y = subsets[split]
        return X.copy(), y.copy()

    def recording_builder(task, passed_config, *, model_family):
        families.append(model_family)
        transformer = build_preprocessor(
            task, passed_config, model_family=model_family
        )
        transformers.append(transformer)
        original_fit_transform = transformer.fit_transform

        def fit_transform(X, y=None, **kwargs):
            fitted_rows.append(X.index.tolist())
            return original_fit_transform(X, y, **kwargs)

        monkeypatch.setattr(transformer, "fit_transform", fit_transform)
        return transformer

    monkeypatch.setattr(baseline, "get_task_data", task_data)
    monkeypatch.setattr(baseline, "build_preprocessor", recording_builder)
    monkeypatch.setattr(
        baseline.sys,
        "argv",
        ["main.py", "baseline", "--task", "classification"],
    )

    data_directory = tmp_path / "outputs/data"
    data_directory.mkdir(parents=True)
    (data_directory / "split_metadata.json").write_text(
        json.dumps({
            "data_checksum_sha256": "synthetic-checksum",
            "split_version": config["decisions"]["split_version"],
        }),
        encoding="utf-8",
    )

    return {
        "config": config,
        "subsets": subsets,
        "calls": calls,
        "fitted_rows": fitted_rows,
        "families": families,
        "transformers": transformers,
    }


def test_runner_fits_only_train_and_saves_consistent_outputs(
    tmp_path, experiment
):
    baseline.run(experiment["config"], tmp_path)

    assert experiment["calls"] == ["train", "validation"]
    assert experiment["fitted_rows"] == [
        experiment["subsets"]["train"][0].index.tolist()
    ]
    assert experiment["families"] == ["linear"]

    output = tmp_path / "outputs/classification"
    result = json.loads(
        (output / "metrics.json").read_text(encoding="utf-8")
    )

    assert result["evaluation_subset"] == "validation"
    assert result["positive_class"] == 1
    assert result["decision_threshold"] == 0.5
    assert result["row_counts"] == {"train": 40, "validation": 12}
    assert result["class_counts"]["train"] == {"0": 20, "1": 20}
    assert result["data_checksum"] == "synthetic-checksum"
    assert result["status"] == "provisional"
    assert result["parameters"]["class_weight"] is None
    assert result["train_seconds"] >= 0
    assert result["task_specific_exclusions"] == 0
    assert result["policy_departures"] == []

    forbidden = {
        "IsCanceled", "ADR", "ReservationStatus",
        "ReservationStatusDate", "AssignedRoomType",
        "BookingChanges", "DaysInWaitingList",
    }
    assert not forbidden.intersection(result["features"])

    matrix = np.asarray(result["confusion_matrix"])
    assert matrix.shape == (2, 2)
    assert matrix.sum() == 12
    assert matrix.sum(axis=1).tolist() == [6, 6]

    for value in result["metrics"].values():
        assert value is not None
        assert np.isfinite(value)
        assert 0 <= value <= 1

    tn, fp, fn, tp = matrix.ravel()
    metrics = result["metrics"]
    assert metrics["accuracy"] == pytest.approx((tn + tp) / 12)
    assert metrics["precision"] == pytest.approx(tp / (tp + fp))
    assert metrics["recall"] == pytest.approx(tp / (tp + fn))
    assert metrics["f1"] == pytest.approx(
        2 * tp / (2 * tp + fp + fn)
    )

    figure = output / "confusion_matrix.png"
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result["figure_paths"] == [
        "outputs/classification/confusion_matrix.png"
    ]

    report = (
        tmp_path / "reports/classification_m1.md"
    ).read_text(encoding="utf-8")
    assert "provisional" in report
    assert "Final test performance is not evaluated." in report
    assert "synthetic-checksum" in report
    for value in metrics.values():
        assert f"{value:.6f}" in report


def test_training_medians_ignore_validation_values(tmp_path, experiment):
    validation_X = experiment["subsets"]["validation"][0]
    validation_X["LeadTime"] = 100000

    baseline.run(experiment["config"], tmp_path)

    transformer = experiment["transformers"][0]
    numeric_columns = next(
        columns
        for name, _, columns in transformer.transformers_
        if name == "numeric"
    )
    lead_index = list(numeric_columns).index("LeadTime")
    median = transformer.named_transformers_[
        "numeric"
    ].named_steps["impute"].statistics_[lead_index]

    expected = experiment["subsets"]["train"][0]["LeadTime"].median()
    assert median == pytest.approx(expected)


def test_metrics_use_class_one_and_inclusive_threshold():
    target = pd.Series([0, 0, 1, 1])
    scores = np.array([0.1, 0.5, 0.4, 0.9])

    metrics, matrix, notes = baseline.evaluate(target, scores)

    assert matrix == [[1, 1], [1, 1]]
    for name in ("accuracy", "precision", "recall", "f1"):
        assert metrics[name] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert notes == {}


def test_no_positive_predictions_preserves_defined_zero_metrics():
    metrics, _, notes = baseline.evaluate(
        pd.Series([0, 1]), np.array([0.1, 0.2])
    )

    assert metrics["precision"] is None
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert "precision" in notes


def test_single_class_undefined_metrics_have_explanations():
    metrics, matrix, notes = baseline.evaluate(
        pd.Series([0, 0]), np.array([0.1, 0.2])
    )

    assert matrix == [[2, 0], [0, 0]]
    assert metrics["accuracy"] == 1.0

    for name in (
        "precision", "recall", "f1", "roc_auc", "average_precision"
    ):
        assert metrics[name] is None
        assert name in notes


@pytest.mark.parametrize(
    "scores",
    [
        np.array([np.nan, 0.2]),
        np.array([0.1, np.inf]),
        np.array([-0.1, 0.2]),
        np.array([0.1, 1.1]),
        np.array([0.1]),
        np.array([[0.1], [0.2]]),
    ],
)
def test_invalid_probabilities_are_rejected(scores):
    with pytest.raises(ValueError, match="probability"):
        baseline.evaluate(pd.Series([0, 1]), scores)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        baseline.evaluate(
            pd.Series([0, 1]), np.array([0.1, 0.9]), threshold
        )


@pytest.mark.parametrize("invalid_label", [2, np.nan])
def test_invalid_targets_do_not_produce_results(
    tmp_path, experiment, invalid_label
):
    target = experiment["subsets"]["train"][1].astype(float)
    target.iloc[0] = invalid_label
    X, _ = experiment["subsets"]["train"]
    experiment["subsets"]["train"] = X, target

    with pytest.raises(ValueError, match="binary"):
        baseline.run(experiment["config"], tmp_path)

    assert not (tmp_path / "outputs/classification/metrics.json").exists()


def test_single_class_training_is_rejected(tmp_path, experiment):
    experiment["subsets"]["train"][1].iloc[:] = 0

    with pytest.raises(ValueError, match="both target classes"):
        baseline.run(experiment["config"], tmp_path)


def test_overlapping_source_rows_are_rejected(tmp_path, experiment):
    X, y = experiment["subsets"]["validation"]
    X.index = pd.Index(range(100, 112), name="row_id")
    y.index = X.index

    with pytest.raises(ValueError, match="overlap"):
        baseline.run(experiment["config"], tmp_path)


def test_target_index_misalignment_is_rejected(tmp_path, experiment):
    target = experiment["subsets"]["train"][1]
    target.index = pd.Index(range(300, 340), name="row_id")

    with pytest.raises(ValueError, match="misaligned"):
        baseline.run(experiment["config"], tmp_path)