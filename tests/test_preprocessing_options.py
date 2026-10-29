"""Checks for model-family preprocessing and categorical ID semantics."""

import numpy as np
import pandas as pd
import pytest

from src.features.policy import get_features
from src.preprocessing.builders import (
    NUMERIC_COLUMNS,
    _categorical_array,
    build_preprocessor,
)
from src.settings import load_config
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config.yaml")


def sample_rows(task):
    features = get_features(task, CONFIG)
    values = {
        column: [1, 2, 3]
        if column in NUMERIC_COLUMNS
        else ["A", "B", "A"]
        for column in features
    }
    frame = pd.DataFrame(values)
    frame["LeadTime"] = [0, 3, 15]
    return frame


@pytest.mark.parametrize("family", ["linear", "distance"])
def test_scaled_families_transform_lead_time(family):
    rows = sample_rows("regression")
    transformer = build_preprocessor(
        "regression", CONFIG, model_family=family
    )
    transformer.fit(rows)

    numeric = transformer.named_transformers_["numeric"]
    columns = transformer.transformers_[0][2]
    position = columns.index("LeadTime")
    actual = numeric.transform(rows[columns])[:, position]

    logged = np.log1p(rows["LeadTime"].to_numpy(dtype=float))
    expected = (logged - logged.mean()) / logged.std()

    np.testing.assert_allclose(actual, expected)


def test_tree_family_preserves_numeric_units():
    rows = sample_rows("regression")
    transformer = build_preprocessor(
        "regression", CONFIG, model_family="tree"
    )
    transformer.fit(rows)

    numeric = transformer.named_transformers_["numeric"]
    columns = transformer.transformers_[0][2]
    actual = numeric.transform(rows[columns])

    np.testing.assert_allclose(
        actual, rows[columns].to_numpy(dtype=float)
    )


def test_clustering_returns_dense_finite_output():
    rows = sample_rows("clustering")
    transformer = build_preprocessor("clustering", CONFIG)

    output = transformer.fit_transform(rows)

    assert isinstance(output, np.ndarray)
    assert output.shape[0] == len(rows)
    assert np.isfinite(output).all()


def test_not_applicable_ids_are_distinct_from_unknown_values():
    rows = pd.DataFrame({
        "Agent": [" NULL ", "NA", "001"],
        "Company": ["null", "", "007"],
        "Meal": ["NULL", "BB", "HB"],
    })

    output = _categorical_array(rows)

    assert output[0, 0] == "NULL"
    assert output[0, 1] == "NULL"
    assert pd.isna(output[0, 2])
    assert pd.isna(output[1, 0])
    assert pd.isna(output[1, 1])
    assert output[2, 0] == "001"
    assert output[2, 1] == "007"


@pytest.mark.parametrize("family", ["linear", "distance"])
def test_negative_lead_time_is_rejected_for_log_transform(family):
    rows = sample_rows("regression")
    rows.loc[0, "LeadTime"] = -1
    transformer = build_preprocessor(
        "regression", CONFIG, model_family=family
    )

    with pytest.raises(ValueError, match="nonnegative"):
        transformer.fit_transform(rows)


@pytest.mark.parametrize("family", ["linear", "tree"])
def test_clustering_rejects_incompatible_model_family(family):
    with pytest.raises(ValueError, match="Clustering"):
        build_preprocessor(
            "clustering", CONFIG, model_family=family
        )


def test_unknown_model_family_is_rejected():
    with pytest.raises(ValueError, match="model_family"):
        build_preprocessor(
            "regression", CONFIG, model_family="unknown"
        )