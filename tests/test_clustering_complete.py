"""Selection, invalid solutions and inference behavior beyond baseline tests."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from src.clustering.complete import choose, estimator, predict, score_solution, search_space


def test_degenerate_solutions_are_unavailable_and_ineligible():
    scores = score_solution(np.zeros((10, 2)), np.zeros(10, dtype=int), np.arange(5))
    assert scores["silhouette"] is None
    assert scores["davies_bouldin"] is None
    assert not scores["eligible"]


def test_balance_rule_rejects_tiny_high_scoring_group():
    labels = np.r_[np.zeros(199, dtype=int), 1]
    Z = np.c_[labels * 100, np.arange(200) * .001]
    scores = score_solution(Z, labels, np.arange(200))
    assert scores["silhouette"] > .9
    assert not scores["eligible"]


def test_selection_ignores_ineligible_and_unconverged_candidates():
    def candidate(sil, eligible, converged=True):
        return {"scores": {"silhouette": sil, "eligible": eligible, "davies_bouldin": 1., "n_clusters": 2}, "converged": converged}
    good = candidate(.3, True)
    assert choose([candidate(.9, False), candidate(.8, True, False), good]) is good
    with pytest.raises(ValueError, match="No eligible"):
        choose([candidate(.9, False)])


def test_all_algorithms_have_systematic_search_and_supported_prediction():
    baselines, grids = search_space()
    assert set(grids) == {"kmeans", "agglomerative", "gmm"}
    assert [len(grids[k]) for k in grids] == [14, 21, 14]
    assert all(baselines[k] in grids[k] for k in grids)
    assert hasattr(estimator("kmeans", baselines["kmeans"], 42), "predict")
    assert hasattr(estimator("gmm", baselines["gmm"], 42), "predict")
    assert not hasattr(estimator("agglomerative", baselines["agglomerative"], 42), "predict")


def test_predict_reuses_saved_model_and_validates_schema(tmp_path, monkeypatch):
    X = pd.DataFrame({"LeadTime": [0., 1., 100., 101.]})
    model = Pipeline([("model", KMeans(n_clusters=2, random_state=42, n_init=2))]).fit(X)
    artifacts = tmp_path / "artifacts/clustering/m2"
    artifacts.mkdir(parents=True)
    joblib.dump(model, artifacts / "kmeans_pipeline.joblib")
    output = tmp_path / "outputs/clustering/m2"
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps({"features": ["LeadTime"], "artifact_directory": "artifacts/clustering/m2", "selection": {"assignment_model": "kmeans"}}))
    incoming = tmp_path / "bookings.csv"
    X.to_csv(incoming, index=False)
    monkeypatch.setattr(Pipeline, "fit", lambda *args, **kwargs: pytest.fail("Inference must not retrain"))
    predict(tmp_path, incoming, tmp_path / "assigned.csv")
    assert pd.read_csv(tmp_path / "assigned.csv").cluster.tolist() == model.predict(X).tolist()
    pd.DataFrame({"wrong": [1]}).to_csv(incoming, index=False)
    with pytest.raises(ValueError, match="missing required"):
        predict(tmp_path, incoming, tmp_path / "invalid.csv")
