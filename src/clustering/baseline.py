"""Member 5: one provisional, untuned M1 K-Means experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.data.prepare import get_task_data
from src.evaluation.results import save_result
from src.features.policy import get_features
from src.preprocessing.builders import build_preprocessor, NUMERIC_COLUMNS



def actual_command():
    """Keep CLI options such as --config without changing the shared router."""
    if Path(sys.argv[0]).name.lower() == "main.py" and "baseline" in sys.argv[1:]:
        return "python " + shlex.join(["main.py", *sys.argv[1:]])
    return "python main.py baseline --task clustering"

def evaluate(Z, labels, seed, sample_size=3000):
    """Use original prepared distances, never distances in the 2D plot."""
    unique = np.unique(labels)
    notes = {}
    if not 1 < len(unique) < len(labels):
        return {"silhouette": None, "davies_bouldin": None}, {"internal_scores": "Need between 2 and n-1 distinct clusters."}, np.array([], dtype=int)
    ids = np.sort(np.random.default_rng(seed).choice(len(Z), min(sample_size, len(Z)), replace=False))
    if 1 < len(np.unique(labels[ids])) < len(ids):
        silhouette = float(silhouette_score(Z[ids], labels[ids]))
    else:
        silhouette = None
        notes["silhouette"] = "Fixed uniform sample has fewer than two clusters or only singleton clusters; sample was not redrawn."
    return {"silhouette": silhouette, "davies_bouldin": float(davies_bouldin_score(Z, labels))}, notes, ids


def profiles(X, labels):
    """Describe measured groups using original units and observed categories."""
    frame = X.copy()
    frame["cluster"] = labels
    summary, frequencies = [], []
    for cluster, group in frame.groupby("cluster", sort=True):
        record = {"cluster": int(cluster), "size": len(group), "share": len(group) / len(frame)}
        for column in X:
            if column in NUMERIC_COLUMNS:
                values = pd.to_numeric(group[column], errors="raise")
                record[column + "_mean"] = float(values.mean()) if values.notna().any() else None
                record[column + "_median"] = float(values.median()) if values.notna().any() else None
                record[column + "_missing"] = int(values.isna().sum())
            else:
                counts = group[column].astype("string").fillna("__MISSING__").value_counts()
                record[column + "_mode"] = str(counts.index[0])
                record[column + "_mode_share"] = int(counts.iloc[0]) / len(group)
                for category, count in counts.items():
                    frequencies.append({"cluster": int(cluster), "feature": column, "category": str(category), "count": int(count), "share": int(count) / len(group)})
        summary.append(record)
    return pd.DataFrame(summary), pd.DataFrame(frequencies)


def run(config: dict, project_root: Path) -> None:
    root = Path(project_root)
    seed = int(config["project"]["random_seed"])
    settings = config.get("baselines", {}).get("clustering") or {}
    if not isinstance(settings, dict):
        raise ValueError("baselines.clustering must be a mapping or null.")
    k = settings.get("n_clusters", 3)
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("n_clusters must be an integer of at least two.")
    X, target = get_task_data("clustering", root, "train", config=config)
    if target is not None or set(X) & {"IsCanceled", "ADR", "ReservationStatus", "ReservationStatusDate"}:
        raise ValueError("Clustering inputs must exclude supervised outcomes.")
    if list(X.columns) != get_features("clustering", config):
        raise ValueError("Inputs differ from the current clustering feature policy.")
    if len(X) <= k or not X.index.is_unique:
        raise ValueError("Need unique source row IDs and more rows than clusters.")
    output = root / config["paths"]["clustering_outputs"]
    artifacts = root / config["paths"]["artifacts"] / "clustering"
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    preprocessor = build_preprocessor("clustering", config, model_family="distance")
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, tol=1e-4, algorithm="lloyd", random_state=seed)
    with threadpool_limits(limits=1):
        start = perf_counter()
        Z = preprocessor.fit_transform(X)
        preprocess_seconds = perf_counter() - start
        if not np.isfinite(Z).all():
            raise ValueError("Prepared data contain nonfinite values.")
        start = perf_counter()
        labels = model.fit_predict(Z)
        train_seconds = perf_counter() - start
        start = perf_counter()
        metrics, metric_notes, sample_ids = evaluate(Z, labels, seed)
        metric_seconds = perf_counter() - start
        pca = PCA(n_components=2, svd_solver="full")
        start = perf_counter()
        coordinates = pca.fit_transform(Z)
        pca_seconds = perf_counter() - start
    summary, frequencies = profiles(X, labels)
    summary.to_csv(output / "cluster_profiles.csv", index=False)
    frequencies.to_csv(output / "category_profiles.csv", index=False)
    pd.DataFrame({"row_id": X.index.to_numpy()[sample_ids]}).to_csv(output / "silhouette_sample.csv", index=False)
    # Per-booking outputs and fitted components remain local, under artifacts.
    pd.DataFrame({"row_id": X.index, "cluster": labels}).to_csv(artifacts / "assignments.csv", index=False)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    joblib.dump(pipeline, artifacts / "kmeans_pipeline.joblib")
    joblib.dump(pca, artifacts / "visualization_pca.joblib")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["cluster"].astype(str), summary["size"], color="#527baa")
    ax.set(xlabel="Cluster (arbitrary ID)", ylabel="Training bookings", title="K-Means baseline cluster sizes")
    for row in summary.itertuples():
        ax.text(str(row.cluster), row.size, f"{row.size:,}", ha="center", va="bottom")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(output / "cluster_sizes.png", dpi=160)
    plt.close(fig)
    plot_ids = np.sort(np.random.default_rng(seed + 1).choice(len(Z), min(10000, len(Z)), replace=False))
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in np.unique(labels):
        selected = plot_ids[labels[plot_ids] == cluster]
        ax.scatter(coordinates[selected, 0], coordinates[selected, 1], s=9, alpha=0.35, label=f"Cluster {cluster}", rasterized=True)
    variance = pca.explained_variance_ratio_
    ax.set(xlabel=f"PC1 ({variance[0]:.1%} variance)", ylabel=f"PC2 ({variance[1]:.1%} variance)", title=f"Training PCA view of K-Means: {len(plot_ids):,} sampled bookings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "pca_clusters.png", dpi=160)
    plt.close(fig)
    source = root / config["data"]["raw_path"]
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    result = {
        "task": "clustering", "model": "KMeans", "parameters": model.get_params(),
        "data_checksum": checksum,
        "feature_policy_version": config["decisions"]["feature_policy_version"],
        "split_version": config["decisions"]["split_version"],
        "evaluation_subset": "train", "seed": seed,
        "metrics": {**metrics, "inertia": float(model.inertia_), "n_clusters": int(len(np.unique(labels)))},
        "metric_notes": metric_notes, "train_seconds": train_seconds,
        "preprocess_seconds": preprocess_seconds, "metric_seconds": metric_seconds, "pca_seconds": pca_seconds,
        "timing_notes": {"train_seconds": "K-Means fit_predict only", "preprocess_seconds": "Shared preprocessor fit_transform only", "metric_seconds": "Silhouette and Davies-Bouldin evaluation", "pca_seconds": "Visualization PCA fit_transform only"},
        "command": actual_command(),
        "command_source": "cli" if Path(sys.argv[0]).name.lower() == "main.py" else "programmatic_default",
        "effective_config": config,
        "figure_paths": [(output / name).relative_to(root).as_posix() for name in ("cluster_sizes.png", "pca_clusters.png")],
        "features": list(X.columns), "model_family": "distance", "preprocessing": config["preprocessing"],
        "population": "All frozen training bookings; no task exclusions; outcomes not provided to clustering.",
        "row_counts": {"train": len(X)}, "task_specific_exclusions": 0,
        "prediction_point": config["decisions"]["prediction_point"]["clustering"],
        "scientific_status": "provisional; shared policy awaiting team review",
        "test_set_evaluated": False,
        "input_shape": list(X.shape), "transformed_shape": list(Z.shape),
        "cluster_sizes": {str(row.cluster): int(row.size) for row in summary.itertuples()},
        "n_iter": int(model.n_iter_), "converged_before_iteration_limit": bool(model.n_iter_ < model.max_iter),
        "silhouette_sampling": {"method": "uniform without replacement, fixed before evaluation", "seed": seed, "rows": len(sample_ids), "row_ids_path": (output / "silhouette_sample.csv").relative_to(root).as_posix()},
        "davies_bouldin_population": "all training rows in the original prepared representation",
        "pca": {"purpose": "visualization only; clustering uses original prepared features", "fit_subset": "train", "explained_variance_ratio": variance.tolist(), "plot_rows": len(plot_ids), "plot_seed": seed + 1},
        "artifact_paths": [(artifacts / name).relative_to(root).as_posix() for name in ("kmeans_pipeline.joblib", "visualization_pca.joblib", "assignments.csv")],
        "environment": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "PyYAML", "threadpoolctl")}},
        "limitations": ["k=3 is a fixed untuned reference, not an optimized cluster count.", "Scores describe training geometry, not held-out predictive performance.", "Uniform silhouette sampling can underrepresent rare clusters.", "Snapshot timing assumptions and outcome-stratified manifest provenance remain provisional.", "Standardized rare counts and correlated channel inputs can dominate distances.", "PCA projection discards information and does not establish cluster separation.", "Reserved test rows were not used for fitting, parameter selection, metric evaluation or PCA fitting. The shared loader reads the source CSV before selecting the requested subset."],
    }
    save_result(result, output / "metrics.json")
    print(json.dumps({"metrics": result["metrics"], "cluster_sizes": result["cluster_sizes"], "train_seconds": train_seconds, "output": str(output)}, indent=2))
