"""Behavior checks for unsupervised baseline evaluation and interpretation."""
import numpy as np
import pandas as pd
from src.clustering.baseline import evaluate, profiles


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
