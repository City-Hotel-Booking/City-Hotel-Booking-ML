"""Audit saved M2 evidence and reloaded models without retraining."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.prepare import get_task_data
from src.settings import load_config


def run(root):
    root = Path(root)
    out = root / "outputs/clustering/m2"
    artifacts = root / "artifacts/clustering/m2"
    result = json.loads((out / "results.json").read_text())
    config = load_config(root / "config.yaml")
    X, target = get_task_data("clustering", root, "train", config=config)
    sample = pd.read_csv(out / "comparison_sample.csv").row_id
    metric = pd.read_csv(out / "metric_sample.csv").row_id
    search = json.loads((out / "search.json").read_text())
    assert target is None and not set(result["features"]) & {"ADR", "IsCanceled", "ReservationStatus", "ReservationStatusDate"}
    assert len(sample) == 5000 and sample.is_unique and set(sample) <= set(X.index)
    assert len(metric) == 1500 and metric.is_unique and set(metric) <= set(sample)
    assert len(search) == 49
    assert len(result["comparison"]) == 6
    assert len(result["full_training_runs"]) == 4
    assert all((root / p).is_file() for p in result["figure_paths"])
    preprocessor = joblib.load(artifacts / "preprocessor.joblib")
    with threadpool_limits(limits=1):
        Z = preprocessor.transform(X)
        assert np.isfinite(Z).all() and np.max(np.abs(Z)) <= 3.
        XS = X.loc[sample]
        S = preprocessor.transform(XS)
        for algorithm in ("kmeans", "gmm"):
            pipeline = joblib.load(artifacts / f"{algorithm}_pipeline.joblib")
            assignments = pd.read_csv(artifacts / f"{algorithm}_full_assignments.csv")
            assert np.array_equal(assignments.row_id, X.index)
            assert np.array_equal(pipeline.predict(X), assignments.cluster)
            profiles = pd.read_csv(out / f"{algorithm}_full_profiles.csv")
            assert profiles["size"].sum() == len(X)
        for algorithm in ("kmeans", "gmm", "agglomerative"):
            model = joblib.load(artifacts / f"{algorithm}_subset.joblib")
            assignments = pd.read_csv(artifacts / f"{algorithm}_subset_assignments.csv")
            assert np.array_equal(assignments.row_id, sample)
            labels = model.labels_ if algorithm == "agglomerative" else model.predict(S)
            assert np.array_equal(labels, assignments.cluster)
            for stage in ("baseline", "optimized"):
                profile = pd.read_csv(out / f"{algorithm}_{stage}_profiles.csv")
                assert profile["size"].sum() == len(sample)
                assert np.isclose(profile.share.sum(), 1)
        pca = joblib.load(artifacts / "pca.joblib")
        assert np.isfinite(pca.transform(S)).all()
    verification = {"status": "PASS", "search_candidates": 49, "comparison_records": 6,
                    "full_training_records": 4, "training_rows": len(X), "subset_rows": len(sample),
                    "full_pipeline_assignments_reproduced": ["kmeans", "gmm"],
                    "all_subset_assignments_reproduced": True,
                    "no_validation_or_test_evaluation": True,
                    "command": "python -m src.clustering.verify_complete"}
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
