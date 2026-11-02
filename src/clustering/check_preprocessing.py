

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from src.data.prepare import get_task_data, load_manifest
from src.features.policy import get_features
from src.preprocessing.builders import build_preprocessor
from src.settings import load_config


def run(config: dict, project_root: Path) -> dict:
    root = Path(project_root)
    X, target = get_task_data("clustering", root, "train", config=config)
    manifest = load_manifest(root, config=config)
    train_ids = manifest.loc[manifest["split"].eq("train"), "row_id"].tolist()
    forbidden = {"IsCanceled", "ADR", "ReservationStatus", "ReservationStatusDate"}
    checks = {
        "no_target": target is None,
        "exact_training_row_ids": X.index.tolist() == train_ids,
        "active_features_only": list(X.columns) == get_features("clustering", config),
        "outcomes_absent": not bool(set(X.columns) & forbidden),
    }
    if not all(checks.values()):
        raise ValueError(f"Unsafe clustering inputs: {checks}")

    transformer = build_preprocessor("clustering", config, model_family="distance")
    fresh = build_preprocessor("clustering", config, model_family="distance")
    checks["separate_instances"] = transformer is not fresh
    try:
        check_is_fitted(fresh)
    except NotFittedError:
        checks["fresh_builder_unfitted"] = True
    else:
        checks["fresh_builder_unfitted"] = False

    Z = transformer.fit_transform(X)
    checks["dense_finite_output"] = isinstance(Z, np.ndarray) and bool(np.isfinite(Z).all())
    checks["one_output_row_per_training_booking"] = Z.shape[0] == len(X)

    numeric = transformer.named_transformers_["numeric"]
    numeric_columns = next(cols for name, _, cols in transformer.transformers_ if name == "numeric")
    parsed = numeric.named_steps["parse"].transform(X[numeric_columns])
    medians = np.nanmedian(parsed, axis=0)
    checks["medians_match_training_only"] = bool(np.allclose(
        numeric.named_steps["impute"].statistics_, medians
    ))
    imputed = numeric.named_steps["impute"].transform(parsed)
    expected_scaled_input = imputed.copy()
    if "LeadTime" in config["preprocessing"]["skewed_numeric_features"]:
        pos = numeric_columns.index("LeadTime")
        expected_scaled_input[:, pos] = np.log1p(expected_scaled_input[:, pos])
    scaler = numeric.named_steps["scale"]
    checks["scaler_matches_training_transforms"] = bool(
        np.allclose(scaler.mean_, expected_scaled_input.mean(axis=0))
        and np.allclose(scaler.var_, expected_scaled_input.var(axis=0))
        and np.all(np.asarray(scaler.n_samples_seen_) == len(X))
    )
    scaled = numeric.transform(X[numeric_columns])
    checks["numeric_means_near_zero"] = bool(np.allclose(scaled.mean(axis=0), 0, atol=1e-10))

    categorical = transformer.named_transformers_["categorical"]
    category_columns = next(cols for name, _, cols in transformer.transformers_ if name == "categorical")
    encoder = categorical.named_steps["encode"]
    categories_before = [values.copy() for values in encoder.categories_]
    probe = X.iloc[[0]].copy()
    token = "__CLUSTERING_UNSEEN_MEAL__"
    while token in encoder.categories_[category_columns.index("Meal")]:
        token += "_"
    probe["Meal"] = token
    probe["Children"] = np.nan
    probe_z = transformer.transform(probe)
    checks["unseen_category_and_missing_numeric_supported"] = bool(
        probe_z.shape == (1, Z.shape[1]) and np.isfinite(probe_z).all()
    )
    checks["transform_does_not_learn_probe_category"] = all(
        np.array_equal(before, after)
        for before, after in zip(categories_before, encoder.categories_)
    )
    category_probe = categorical.transform(probe[category_columns])
    if hasattr(category_probe, "toarray"):
        category_probe = category_probe.toarray()
    meal_position = category_columns.index("Meal")
    start = sum(len(values) for values in encoder.categories_[:meal_position])
    stop = start + len(encoder.categories_[meal_position])
    checks["unseen_meal_block_is_zero"] = bool(np.all(category_probe[:, start:stop] == 0))

    # Repeat fitting a fresh object on exactly X to check deterministic
    # representation construction. This does not test historical availability.
    repeated = fresh.fit_transform(X)
    checks["repeat_fit_same_representation"] = bool(np.allclose(Z, repeated))

    source = root / config["data"]["raw_path"]
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "task": "clustering_preprocessing",
        "data_checksum": hashlib.sha256(source.read_bytes()).hexdigest(),
        "feature_policy_version": config["decisions"]["feature_policy_version"],
        "split_version": config["decisions"]["split_version"],
        "evaluation_subset": "train",
        "features": list(X.columns),
        "input_shape": list(X.shape),
        "transformed_shape": list(Z.shape),
        "transformed_bytes": int(Z.nbytes),
        "missing_training_inputs": {column: int(X[column].isna().sum()) for column in X},
        "numeric_training_medians": dict(zip(numeric_columns, medians.tolist())),
        "numeric_scaled_standard_deviations": dict(zip(numeric_columns, scaled.std(axis=0).tolist())),
        "categorical_levels": {column: len(values) for column, values in zip(category_columns, encoder.categories_)},
        "checks": checks,
        "environment": {"python": platform.python_version(), **{
            package: version(package) for package in ("numpy", "pandas", "scikit-learn", "scipy", "PyYAML")
        }},
        "command": "python -m src.clustering.check_preprocessing",
        "limitations": [
            "Source loader reads the complete CSV to verify identity; only training feature rows are transformed or fitted.",
            "Synthetic probes derive from training inputs; no validation/test evaluation occurred.",
            "Passing checks does not establish historical booking-entry availability or validate distance weighting.",
            "No clustering estimator or PCA was fitted; no fitted artifacts are saved.",
        ],
    }
    path = root / config["paths"]["clustering_outputs"] / "preprocessing_checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    report = run(load_config(config_path), root)
    print(f"{report['status']}: {sum(report['checks'].values())}/{len(report['checks'])} checks")
    print(f"Training representation: {report['input_shape']} -> {report['transformed_shape']}")
    for name, passed in report["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
