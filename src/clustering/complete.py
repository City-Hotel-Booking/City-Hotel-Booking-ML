"""Reproducible M2 clustering experiments; all learning uses frozen train rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from threadpoolctl import threadpool_limits

from src.clustering.baseline import profiles
from src.data.prepare import get_task_data
from src.evaluation.results import save_result
from src.preprocessing.builders import build_preprocessor
from src.settings import load_config

FORBIDDEN = {"IsCanceled", "ADR", "ReservationStatus", "ReservationStatusDate"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def estimator(algorithm, parameters, seed):
    if algorithm == "kmeans":
        return KMeans(random_state=seed, max_iter=300, **parameters)
    if algorithm == "agglomerative":
        return AgglomerativeClustering(metric="euclidean", **parameters)
    if algorithm == "gmm":
        return GaussianMixture(random_state=seed, n_init=3, max_iter=300, **parameters)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def score_solution(Z, labels, metric_indices):
    """No outcome labels; all scores use original prepared coordinates."""
    counts = np.unique(labels, return_counts=True)[1]
    valid = 1 < len(counts) < len(Z)
    sample_labels = labels[metric_indices]
    sample_valid = 1 < len(np.unique(sample_labels)) < len(metric_indices)
    sil = float(silhouette_score(Z[metric_indices], sample_labels)) if valid and sample_valid else None
    db = float(davies_bouldin_score(Z, labels)) if valid else None
    minimum, maximum = float(counts.min() / len(Z)), float(counts.max() / len(Z))
    eligible = valid and sil is not None and minimum >= .01 and maximum <= .95
    return {"silhouette": sil, "davies_bouldin": db, "n_clusters": len(counts),
            "minimum_share": minimum, "maximum_share": maximum, "noise_fraction": 0.0,
            "eligible": bool(eligible), "cluster_sizes": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}}


def rank_key(record):
    return (record["scores"]["silhouette"], -record["scores"]["davies_bouldin"], -record["scores"]["n_clusters"])


def choose(records):
    eligible = [r for r in records if r["scores"]["eligible"] and r["converged"]]
    if not eligible:
        raise ValueError("No eligible, converged solution; inspect search.json rather than force a winner.")
    return max(eligible, key=rank_key)


def search_space():
    baseline = {
        "kmeans": {"n_clusters": 3, "n_init": 10},
        "agglomerative": {"n_clusters": 3, "linkage": "ward"},
        "gmm": {"n_components": 3, "covariance_type": "diag", "reg_covar": .001},
    }
    grids = {
        "kmeans": [{"n_clusters": k, "n_init": n} for k in range(2, 9) for n in (10, 20)],
        "agglomerative": [{"n_clusters": k, "linkage": link} for k in range(2, 9) for link in ("ward", "average", "complete")],
        "gmm": [{"n_components": k, "covariance_type": "diag", "reg_covar": reg} for k in range(2, 9) for reg in (.001, .01)],
    }
    return baseline, grids


def fit_record(algorithm, params, Z, metric_indices, seed):
    model = estimator(algorithm, params, seed)
    start = perf_counter()
    labels = model.fit_predict(Z)
    seconds = perf_counter() - start
    record = {"algorithm": algorithm, "parameters": params, "fit_seconds": seconds,
              "converged": bool(getattr(model, "converged_", getattr(model, "n_iter_", 0) < getattr(model, "max_iter", 1))),
              "scores": score_solution(Z, labels, metric_indices)}
    return model, labels, record


def stability(algorithm, params, Z, reference_labels, seed):
    """Refit on three 80% subsets; compare labels only on overlapping rows."""
    rows = []
    for offset in (11, 22, 33):
        indices = np.sort(np.random.default_rng(seed + offset).choice(len(Z), int(.8 * len(Z)), replace=False))
        model = estimator(algorithm, params, seed + offset)
        labels = model.fit_predict(Z[indices])
        rows.append({"seed": seed + offset, "rows": len(indices),
                     "ari": float(adjusted_rand_score(reference_labels[indices], labels))})
    return {"method": "80% without replacement; fixed preprocessor; label-invariant ARI on overlap",
            "replicates": rows, "mean_ari": float(np.mean([r["ari"] for r in rows]))}


def make_figures(output, Z, X, models, pca, seed):
    coordinates = pca.transform(Z)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for axis, (name, stage, labels) in zip(axes.flat, models):
        for cluster in np.unique(labels):
            mask = labels == cluster
            axis.scatter(coordinates[mask, 0], coordinates[mask, 1], color=plt.get_cmap("tab10")(int(cluster)), s=4, alpha=.35, label=str(cluster))
        axis.legend(title="Cluster ID", fontsize=7, title_fontsize=7, markerscale=2)
        axis.set_title(f"{name}: {stage}")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    fig.suptitle(f"Common training subset; PCA retains {pca.explained_variance_ratio_.sum():.1%} variance")
    fig.tight_layout()
    fig.savefig(output / "comparison_pca.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for axis, (name, stage, labels) in zip(axes.flat, models):
        ids, counts = np.unique(labels, return_counts=True)
        axis.bar(ids.astype(str), counts)
        axis.set_title(f"{name}: {stage}")
        axis.set_xlabel("Arbitrary cluster ID")
        axis.set_ylabel("Bookings")
    fig.tight_layout()
    fig.savefig(output / "comparison_sizes.png", dpi=140)
    plt.close(fig)
    # A dendrogram of the same feature space, using a recorded smaller subset.
    ids = np.sort(np.random.default_rng(seed + 5).choice(len(Z), min(1000, len(Z)), replace=False))
    pd.DataFrame({"row_id": X.index[ids]}).to_csv(output / "dendrogram_sample.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(linkage(Z[ids], method="ward"), truncate_mode="lastp", p=30, ax=ax)
    ax.set(title="Ward hierarchy: 1,000 training bookings; final 30 merged groups", xlabel="Merged group (parentheses show group size)", ylabel="Ward linkage distance")
    fig.tight_layout()
    fig.savefig(output / "dendrogram.png", dpi=140)
    plt.close(fig)


def profile_figure(output, X, labels):
    clusters = np.unique(labels)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, column in zip(axes, ("LeadTime", "Adults", "Children")):
        values = [pd.to_numeric(X.loc[labels == cluster, column]).dropna().to_numpy() for cluster in clusters]
        axis.boxplot(values, tick_labels=[str(c) for c in clusters], showfliers=False)
        axis.set(title=column, xlabel="Cluster ID", ylabel="Original observed units")
    fig.suptitle("Selected analysis: booking distributions (outliers hidden for display only)")
    fig.tight_layout()
    fig.savefig(output / "feature_distributions.png", dpi=140)
    plt.close(fig)


def run(config, root):
    root = Path(root)
    seed = int(config["project"]["random_seed"])
    output = root / config["paths"]["clustering_outputs"] / "m2"
    artifacts = root / config["paths"]["artifacts"] / "clustering" / "m2"
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    X, target = get_task_data("clustering", root, "train", config=config)
    if target is not None or FORBIDDEN.intersection(X.columns):
        raise ValueError("Outcome columns are forbidden.")
    preprocessor = Pipeline([("shared", build_preprocessor("clustering", config)),
                             ("distance_cap", FunctionTransformer(np.clip, kw_args={"a_min": -3., "a_max": 3.}, feature_names_out="one-to-one"))])
    with threadpool_limits(limits=1):
        Z = preprocessor.fit_transform(X)
        if not np.isfinite(Z).all():
            raise ValueError("Nonfinite representation.")
        # Common subset and metric rows are fixed before any algorithm is fitted.
        indices = np.sort(np.random.default_rng(seed).choice(len(X), min(5000, len(X)), replace=False))
        S, XS = Z[indices], X.iloc[indices]
        metric_indices = np.sort(np.random.default_rng(seed + 1).choice(len(S), min(1500, len(S)), replace=False))
        raw = preprocessor.named_steps["shared"].transform(XS)
        rejected = []
        for k in range(2, 9):
            _, _, record = fit_record("kmeans", {"n_clusters": k, "n_init": 10}, raw, metric_indices, seed)
            rejected.append(record)
        write_json(output / "uncapped_diagnostic.json", rejected)
        pd.DataFrame({"row_id": XS.index}).to_csv(output / "comparison_sample.csv", index=False)
        pd.DataFrame({"row_id": XS.index[metric_indices]}).to_csv(output / "metric_sample.csv", index=False)
        baseline_params, grids = search_space()
        records, comparison, fitted, plots = [], [], {}, []
        for algorithm in grids:
            print(f"Running {algorithm} baseline and {len(grids[algorithm])} candidates", flush=True)
            base_model, base_labels, base = fit_record(algorithm, baseline_params[algorithm], S, metric_indices, seed)
            candidates = []
            for params in grids[algorithm]:
                try:
                    model, labels, record = fit_record(algorithm, params, S, metric_indices, seed)
                    record["error"] = None
                except (ValueError, np.linalg.LinAlgError) as exc:
                    record = {"algorithm": algorithm, "parameters": params, "error": str(exc)}
                records.append(record)
                if record["error"] is None:
                    candidates.append(record)
            write_json(output / "search.json", records)
            best = choose(candidates)
            model, labels, optimized = fit_record(algorithm, best["parameters"], S, metric_indices, seed)
            optimized["stability"] = stability(algorithm, best["parameters"], S, labels, seed)
            base["stability"] = stability(algorithm, baseline_params[algorithm], S, base_labels, seed)
            for stage, rec, cluster_labels in (("baseline", base, base_labels), ("optimized", optimized, labels)):
                rec["stage"] = stage
                comparison.append(rec)
                prof, categories = profiles(XS, cluster_labels)
                prof.to_csv(output / f"{algorithm}_{stage}_profiles.csv", index=False)
                categories.to_csv(output / f"{algorithm}_{stage}_categories.csv", index=False)
                plots.append((algorithm, stage, cluster_labels))
            fitted[algorithm] = (model, labels, optimized)
            joblib.dump(model, artifacts / f"{algorithm}_subset.joblib")
            pd.DataFrame({"row_id": XS.index, "cluster": labels}).to_csv(artifacts / f"{algorithm}_subset_assignments.csv", index=False)
            write_json(output / "search.json", records)
        stable = [r for _, _, r in fitted.values() if r["scores"]["eligible"] and r["converged"] and r["stability"]["mean_ari"] >= .7]
        winner = choose(stable) if stable else choose([r for _, _, r in fitted.values()])
        stable_fallback = not bool(stable)
        # New-booking assignment uses the strongest eligible predictive algorithm.
        deployable = choose([r for alg, (_, _, r) in fitted.items() if alg in ("kmeans", "gmm")])
        full_records = []
        for algorithm in ("kmeans", "gmm"):
            for stage, params in (("baseline", baseline_params[algorithm]), ("optimized", fitted[algorithm][2]["parameters"])):
                full_metric = np.sort(np.random.default_rng(seed + 2).choice(len(X), min(3000, len(X)), replace=False))
                model, labels, record = fit_record(algorithm, params, Z, full_metric, seed)
                record["stage"] = stage
                full_records.append(record)
                if stage == "optimized":
                    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
                    joblib.dump(pipeline, artifacts / f"{algorithm}_pipeline.joblib")
                    prof, cats = profiles(X, labels)
                    prof.to_csv(output / f"{algorithm}_full_profiles.csv", index=False)
                    cats.to_csv(output / f"{algorithm}_full_categories.csv", index=False)
                    pd.DataFrame({"row_id": X.index, "cluster": labels}).to_csv(artifacts / f"{algorithm}_full_assignments.csv", index=False)
        pd.DataFrame({"row_id": X.index[full_metric]}).to_csv(output / "full_metric_sample.csv", index=False)
        # Sensitivity is explicitly separate from optimization and does not alter shared features.
        names = preprocessor.get_feature_names_out().tolist()
        keep = [i for i, name in enumerate(names) if not name.endswith("__IsRepeatedGuest")]
        alt_model, alt_labels, alt = fit_record("kmeans", fitted["kmeans"][2]["parameters"], S[:, keep], metric_indices, seed)
        sensitivity = {"change": "Drop standardized IsRepeatedGuest dimension for diagnostic only; no shared policy change",
                       "scores": alt["scores"], "ari_vs_original_kmeans": float(adjusted_rand_score(fitted["kmeans"][1], alt_labels))}
        pca = PCA(n_components=2, svd_solver="full").fit(Z)
        joblib.dump(pca, artifacts / "pca.joblib")
        joblib.dump(preprocessor, artifacts / "preprocessor.joblib")
        make_figures(output, S, XS, plots, pca, seed)
        profile_figure(output, XS, fitted[winner["algorithm"]][1])
    flattened = [{"algorithm": r["algorithm"], "stage": r["stage"], "parameters": json.dumps(r["parameters"], sort_keys=True),
                  **{k: v for k, v in r["scores"].items() if k != "cluster_sizes"},
                  "fit_seconds": r["fit_seconds"], "stability_mean_ari": r["stability"]["mean_ari"]} for r in comparison]
    table = pd.DataFrame(flattened)
    table.to_csv(output / "comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for stage in ("baseline", "optimized"):
        part = table[table.stage == stage]
        x = np.arange(3) + (-.18 if stage == "baseline" else .18)
        axes[0].bar(x, part.silhouette, width=.36, label=stage)
        axes[1].bar(x, part.davies_bouldin, width=.36, label=stage)
    for axis, title in zip(axes, ("Silhouette (higher is better)", "Davies-Bouldin (lower is better)")):
        axis.set_xticks(range(3), list(grids))
        axis.set_title(title)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison_metrics.png", dpi=140)
    plt.close(fig)
    checksum = hashlib.sha256((root / config["data"]["raw_path"]).read_bytes()).hexdigest()
    result = {"task": "clustering", "model": "three_algorithm_comparison", "parameters": {"baselines": baseline_params, "search_space": grids},
              "data_checksum": checksum, "feature_policy_version": config["decisions"]["feature_policy_version"],
              "split_version": config["decisions"]["split_version"], "evaluation_subset": "train",
              "seed": seed, "metrics": {"comparison_rows": len(S), "winner_silhouette": winner["scores"]["silhouette"]},
              "train_seconds": sum(r["fit_seconds"] for r in records if r.get("error") is None),
              "command": "python main.py cluster run", "figure_paths": [(output / f"{name}.png").relative_to(root).as_posix() for name in ("comparison_pca", "comparison_sizes", "comparison_metrics", "dendrogram", "feature_distributions")],
              "selection": {"criterion": "eligible converged solutions: maximize sampled silhouette, then minimize DB, then fewer clusters; prefer stability ARI >= 0.7",
                            "eligibility": "Every cluster >=1% and largest <=95%; no outcome labels",
                            "analysis_winner": winner["algorithm"], "assignment_model": deployable["algorithm"], "stability_threshold_fallback": stable_fallback},
              "comparison": comparison, "full_training_runs": full_records, "sensitivity": sensitivity,
              "features": list(X.columns), "preprocessing": {"shared": config["preprocessing"], "task_variant": "m2-distance-cap3-v1", "fixed_standardized_cap": [-3., 3.]},
              "population": {"train_rows": len(X), "comparison_rows": len(S), "silhouette_rows": len(metric_indices), "subset_seed": seed, "metric_seed": seed + 1, "no_task_exclusions": True},
              "pca_variance": pca.explained_variance_ratio_.tolist(),
              "artifact_directory": artifacts.relative_to(root).as_posix(),
              "environment": {"python": platform.python_version(), **{p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib")}},
              "limitations": ["All selection uses training geometry, not held-out generalization.", "Fixed subset and repeated metric use can favor sample-specific structure.", "Stability fixes preprocessing; it does not measure full preprocessing uncertainty.", "Snapshot timing and inherited outcome-stratified manifest remain provisional.", "Gaussian densities on one-hot features are an approximation; diagonal covariance and regularization control singularities.", "Agglomerative model is subset-only and has no new-row predict method.", "Full-data predictive refits can differ from subset ranking; compare separately.", "No supervised final test set evaluation or GitHub update performed."]}
    save_result(result, output / "results.json")
    print(table.to_string(index=False), flush=True)
    print(json.dumps(result["selection"], indent=2), flush=True)
    return result


def predict(root, input_path, output_path, result_path=None):
    root = Path(root)
    summary = json.loads(Path(result_path or root / "outputs/clustering/m2/results.json").read_text())
    algorithm = summary["selection"]["assignment_model"]
    pipeline = joblib.load(root / summary["artifact_directory"] / f"{algorithm}_pipeline.joblib")
    X = pd.read_csv(input_path)
    features = summary["features"]
    missing = set(features) - set(X.columns)
    if missing:
        raise ValueError(f"Booking CSV is missing required features: {sorted(missing)}")
    if X.empty:
        raise ValueError("Booking CSV must contain at least one row.")
    with threadpool_limits(limits=1):
        labels = pipeline.predict(X[features])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"input_row": range(len(X)), "cluster": labels, "model": algorithm}).to_csv(output_path, index=False)
    print(f"Assigned {len(X)} bookings using saved {algorithm}; no retraining.")


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Clustering comparisons and saved-model assignment")
    parser.add_argument("--config", default="config.yaml")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("run", help="Run all baseline, tuning, stability and full-training experiments")
    commands.add_parser("view", help="Display saved comparisons and selected models without retraining")
    commands.add_parser("search", help="Inspect all saved hyperparameter candidates without retraining")
    assign = commands.add_parser("predict", help="Assign new bookings with the saved predictive pipeline")
    assign.add_argument("--input", required=True)
    assign.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = Path(args.config)
    config = load_config(path if path.is_absolute() else root / path)
    output = root / config["paths"]["clustering_outputs"] / "m2"
    if args.action == "run":
        run(config, root)
    elif args.action == "view":
        print(pd.read_csv(output / "comparison.csv").to_string(index=False))
        print(json.dumps(json.loads((output / "results.json").read_text())["selection"], indent=2))
    elif args.action == "search":
        records = json.loads((output / "search.json").read_text())
        print(pd.DataFrame([{"algorithm": r["algorithm"], "parameters": json.dumps(r["parameters"]),
                             "error": r.get("error"), **{k: v for k, v in r.get("scores", {}).items() if k != "cluster_sizes"}}
                            for r in records]).to_string(index=False))
    else:
        predict(root, args.input, args.output, output / "results.json")


if __name__ == "__main__":
    main()
