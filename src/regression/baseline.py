"""Member 3: reproducible, untuned booking-entry ADR regression baseline.

Uses Member 1's H2 loader through Member 2's frozen split interface,
Member 2's feature policy and fresh preprocessing builder, and Member 6's
result-record writer.  NEVER requests or evaluates the final test split.
"""

from __future__ import annotations

import json
import math
import platform
import shlex
import sys
from importlib.metadata import version as package_version
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")  # Safe in headless CI and command-line environments.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data.prepare import get_task_data
from src.evaluation.results import save_result
from src.features.policy import get_features
from src.preprocessing.builders import build_preprocessor


TASK = "regression"
MODEL_NAME = "linear_regression"
COMMAND = "python main.py baseline --task regression"
POPULATION_POLICY = "all_bookings_with_finite_adr"
FORBIDDEN_FEATURES = frozenset({
    "ADR", "IsCanceled", "AssignedRoomType", "BookingChanges",
    "DaysInWaitingList", "ReservationStatus", "ReservationStatusDate",
})


def _software_versions() -> dict[str, str]:
    """Capture actual runtime versions, not guessed library versions."""
    return {
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scipy": package_version("scipy"),
        "scikit-learn": package_version("scikit-learn"),
        "matplotlib": package_version("matplotlib"),
        "PyYAML": package_version("PyYAML"),
    }


def _actual_baseline_command() -> str:
    """Preserve --config and other passed options when launched via main.py.

    Direct unit-test calls to run() have no CLI invocation; retain the
    documented default in that case. No changes to the shared CLI are needed.
    """
    argv = sys.argv
    if Path(argv[0]).name.lower() != "main.py" or "baseline" not in argv[1:]:
        return COMMAND
    return "python " + shlex.join(["main.py", *argv[1:]])


def _require_contract(config: dict) -> None:
    """Refuse to silently use a different shared scientific configuration."""
    try:
        decisions = config["decisions"]
        if decisions["adr_population"] != POPULATION_POLICY:
            raise ValueError(
                "Regression code supports only the documented finite-ADR population. "
                "A different policy needs an explicit reviewed implementation."
            )
        if decisions["prediction_point"][TASK] != "at_booking_entry":
            raise ValueError("Regression prediction point is not at_booking_entry.")
        if config["targets"][TASK] != "ADR":
            raise ValueError("The regression target must be ADR.")
        if config["data"]["expected_rows"] != 79330 or config["data"]["expected_columns"] != 31:
            raise ValueError("Expected source contract is 79,330 City Hotel rows and 31 columns.")
        if not decisions["feature_policy_version"] or not decisions["split_version"]:
            raise ValueError("Missing required feature/split version in config.yaml.")
        if config["baselines"][TASK] not in (None, MODEL_NAME):
            raise ValueError("config.yaml selects a different regression baseline.")
        int(config["project"]["random_seed"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Incomplete regression configuration: {exc}") from exc


def _finite_target_rows(
    features: pd.DataFrame, target: pd.Series, subset: str
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Exclude invalid ADR targets *inside* the existing split, never re-split."""
    if not isinstance(features, pd.DataFrame) or not isinstance(target, pd.Series):
        raise ValueError(f"{subset}: expected DataFrame features and Series target.")
    if features.empty or len(features) != len(target):
        raise ValueError(f"{subset}: empty or mismatched target/features.")
    if not features.index.is_unique or not target.index.is_unique:
        raise ValueError(f"{subset}: source row IDs must be unique.")
    if not features.index.equals(target.index):
        raise ValueError(f"{subset}: target/source row ID alignment mismatch.")

    # String conversion is allowed only for assessing target validity; no imputation
    # of supervised labels, no winsorizing, no target-based feature generation.
    numeric = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(numeric)
    retained = features.loc[valid].copy()
    values = numeric[valid]
    if len(values) < 2:
        raise ValueError(f"{subset}: fewer than two finite ADR targets remain.")
    return retained, values, {
        "source_rows": int(len(features)),
        "eligible_rows": int(valid.sum()),
        "excluded_nonfinite_adr": int((~valid).sum()),
        "zero_adr_rows": int(np.count_nonzero(values == 0)),
        "negative_adr_rows": int(np.count_nonzero(values < 0)),
        "maximum_adr": float(np.max(values)),
    }


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    # Do not use mean_squared_error(..., squared=False): removed in newer sklearn.
    output = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }
    if any(not math.isfinite(value) for value in output.values()):
        raise ValueError("Non-finite regression metric; investigate source/model values.")
    return output


def _save_residual_figure(
    predictions: np.ndarray, actual: np.ndarray, path: Path, seed: int
) -> None:
    """Full-population residual histogram plus reproducibly sampled scatter."""
    errors = actual - predictions
    rng = np.random.default_rng(seed)
    indices = np.arange(len(actual))
    if len(indices) > 8000:
        indices = np.sort(rng.choice(indices, size=8000, replace=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].scatter(predictions[indices], errors[indices], s=8, alpha=0.32)
    axes[0].axhline(0.0, linestyle="--", color="black", linewidth=1)
    axes[0].set(xlabel="Predicted ADR", ylabel="Actual - predicted ADR",
                title="Validation residuals (up to 8,000 sampled points)")
    axes[0].grid(alpha=0.15)

    # Use the entire validation population for histogram; do not hide outliers.
    axes[1].hist(errors, bins=75, edgecolor="none")
    axes[1].axvline(0.0, linestyle="--", color="black", linewidth=1)
    axes[1].set(xlabel="Actual - predicted ADR", ylabel="Validation row count",
                title="Validation residual distribution (all eligible rows)")
    axes[1].grid(alpha=0.15)
    fig.suptitle("City Hotel H2: untuned Linear Regression | no test evaluation")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _write_report(path: Path, record: dict) -> None:
    """Generate the report from measured numbers; never use invented metrics."""
    train = record["training_metrics"]
    val = record["metrics"]
    sizes = record["population"]
    features = record["features"]
    assumptions = record["decision_status"]
    fmt = lambda number: f"{number:,.4f}"
    lines = [
        "# Milestone 1 — ADR regression baseline (Paula, Member 3)",
        "",
        "## Objective and setup",
        "Predict the recorded **ADR (Average Daily Rate)** for a City Hotel H2",
        "booking, provisionally at the moment it enters the PMS.",
        "Only source features marked `use` in the shared booking-time policy",
        "were included. This is one **untuned** baseline, not the M2 comparison.",
        "",
        f"- Model: `LinearRegression()` (scikit-learn; no parameter search).",
        f"- Prediction point: `{record['prediction_point']}` ({assumptions}).",
        f"- ADR population: `{record['population_policy']}`; retain canceled,",
        "  zero-night, zero-ADR and extreme ADR records if the target is finite.",
        "- No target clipping, target imputation, deduplication, or resplitting.",
        "- Learned preprocessing: numeric median imputation, `log1p(LeadTime)`",
        "  and StandardScaler; categorical missing marker and one-hot encoding",
        "  with `handle_unknown='ignore'`, fitted only on training rows.",
        f"- Seed: `{record['seed']}` (used for reproducible residual plot sampling).",
        f"- Feature-policy version: `{record['feature_policy_version']}`.",
        f"- Split version: `{record['split_version']}`.",
        f"- Source SHA-256: `{record['data_checksum']}`.",
        f"- Exact command: `{record['command']}`.",
        f"- Fit time: {record['train_seconds']:.4f} seconds (pipeline fit only).",
        "- Runtime library versions: " + ", ".join(
            f"`{name}={ver}`" for name, ver in record["software_versions"].items()
        ) + ".",
        "",
        "## Exact input features",
        "",
        *[f"{i}. `{feature}`" for i, feature in enumerate(features, start=1)],
        "",
        "## Population checks",
        "",
        "| Subset | Original rows | Eligible rows | Excluded nonfinite ADR | Zero ADR | Negative ADR | Max ADR |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {subset.title()} | {sizes[subset]['source_rows']:,} | "
            f"{sizes[subset]['eligible_rows']:,} | "
            f"{sizes[subset]['excluded_nonfinite_adr']:,} | "
            f"{sizes[subset]['zero_adr_rows']:,} | "
            f"{sizes[subset]['negative_adr_rows']:,} | "
            f"{fmt(sizes[subset]['maximum_adr'])} |"
            for subset in ("train", "validation")
        ],
        "",
        "## Results",
        "",
        "| Metric | Training | Validation |",
        "|---|---:|---:|",
        *[
            f"| {name.upper()} | {fmt(train[name])} | {fmt(val[name])} |"
            for name in ("mae", "rmse", "r2")
        ],
        "",
        "- Validation figure: `outputs/regression/residuals.png`.",
        "- Machine-readable result: `outputs/regression/metrics.json`.",
        "- **Reserved test rows were not used for fitting, model selection, or evaluation.**",
        "  The shared loader may read the complete source CSV before filtering by split.",
        "",
        "## Interpretation and limitations",
        "",
        "MAE is the average absolute ADR error in the recorded rate units;",
        "RMSE penalizes larger misses more heavily; R² compares this model to",
        "the validation-mean benchmark. Training versus validation errors can",
        "indicate overfitting or data/feature mismatch; inspect the residual plot",
        "for skew, systematic bias and extreme errors before drawing conclusions.",
        "The all-finite-ADR policy intentionally includes unusual and canceled",
        "bookings. The 5,400 ADR source outlier and zero-ADR records can strongly",
        "influence RMSE and least squares. No source-based clipping was applied.",
        f"Validation/train RMSE ratio: {val['rmse'] / train['rmse']:.3f};",
        f"validation/train MAE ratio: {val['mae'] / train['mae']:.3f}.",
        "The validation maximum ADR is far outside the training range;",
        "therefore high validation RMSE cannot be attributed to overfitting alone.",
        f"The maximum-ADR validation row contributes {100*record['diagnostics']['largest_adr_squared_error_share']:.2f}%",
        "of the full validation sum of squared errors, but was not excluded.",
        f"Negative validation ADR predictions: {record['diagnostics']['negative_prediction_count']}.",
        "This is a limitation of unconstrained least squares at booking entry.",
        "This grouped *random* split cannot demonstrate performance on unseen",
        "calendar periods or different hotels. Original recorded feature values",
        "may also differ from what was known at booking entry.",
        "",
        "## Leakage audit and review status",
        "",
        "`ADR` is excluded from X. `IsCanceled`, `AssignedRoomType`,",
        "`BookingChanges`, `DaysInWaitingList`, `ReservationStatus`, and",
        "`ReservationStatusDate` are also excluded. Disputed fields remain out",
        "under Member 2's current policy. The preprocessor and estimator were",
        "fitted together on training rows only. Validation rows were scored",
        "only after fitting. Reserved test rows were not used for fitting, model selection, or evaluation.",
        "**Scientific sign-off remains required:** Zeina, Michael and Paula",
        "must approve the ADR population, prediction-point interpretation and",
        "feature policy in `reports/decisions.md`; current policy is provisional.",
        "",
        "## Reproduction, ownership, and M2 continuation",
        "",
        f"Run `{record['command']}` after placing the exact H2 source file",
        "at `data/H2.csv`, installing `requirements.txt`, and validating the",
        "frozen manifest/checksum with `python main.py prepare`.",
        "Member 3 owns `src/regression/baseline.py`,",
        "`tests/test_regression.py`, this report, and",
        "`outputs/regression/metrics.json` / `residuals.png`.",
        "Milestone 2 will compare at least three regression model families in",
        "untuned and tuned configurations, use only validation/CV data for",
        "selection, and evaluate the frozen test once after final selection.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: dict, project_root: Path) -> None:
    """Run the frozen training/validation protocol and publish the M1 handoff."""
    _require_contract(config)
    root = Path(project_root).resolve()
    features = get_features(TASK, config)
    if not features or len(features) != len(set(features)):
        raise ValueError("Regression feature list is empty or contains duplicates.")
    if FORBIDDEN_FEATURES.intersection(features):
        raise ValueError("Regression policy contains target or future-information features.")

    # Request only train and validation assignments. The shared data loader
    # reads the source CSV before filtering; test rows are never used for fit,
    # model selection, or evaluation by this runner.
    X_train_raw, y_train_raw = get_task_data(TASK, root, "train", config=config)
    X_valid_raw, y_valid_raw = get_task_data(TASK, root, "validation", config=config)
    if list(X_train_raw.columns) != features or list(X_valid_raw.columns) != features:
        raise ValueError("Shared task data columns differ from approved feature list.")
    if not X_train_raw.index.intersection(X_valid_raw.index).empty:
        raise ValueError("Training and validation row IDs overlap.")

    X_train, y_train, train_population = _finite_target_rows(
        X_train_raw, y_train_raw, "train"
    )
    X_valid, y_valid, validation_population = _finite_target_rows(
        X_valid_raw, y_valid_raw, "validation"
    )
    seed = int(config["project"]["random_seed"])
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(TASK, config, model_family="linear")),
        ("model", LinearRegression()),
    ])

    # The only fit occurs here. NEVER fit/fit_transform on validation rows.
    start = perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = perf_counter() - start
    predicted_train = np.asarray(pipeline.predict(X_train), dtype=float)
    predicted_valid = np.asarray(pipeline.predict(X_valid), dtype=float)
    if not np.isfinite(predicted_train).all() or not np.isfinite(predicted_valid).all():
        raise ValueError("Model produced non-finite predictions.")
    train_metrics = _metrics(y_train, predicted_train)
    validation_metrics = _metrics(y_valid, predicted_valid)

    metadata_path = root / config["paths"]["data_outputs"] / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["split_version"] != config["decisions"]["split_version"]:
        raise ValueError("Output metadata split version differs from configuration.")
    if metadata.get("source_rows") != 79330 or metadata.get("source_columns") != 31:
        raise ValueError("Output metadata does not describe the required H2 dataset.")
    checksum = metadata.get("data_checksum_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("Frozen source checksum is missing or invalid.")

    output_dir = root / config["paths"]["regression_outputs"]
    image_path = output_dir / "residuals.png"
    metrics_path = output_dir / "metrics.json"
    report_path = root / config["paths"]["reports"] / "regression_m1.md"
    relative_figure = image_path.relative_to(root).as_posix()
    record = {
        "task": TASK,
        "experiment_id": "regression_m1_linear_v1",
        "model": MODEL_NAME,
        "parameters": pipeline.named_steps["model"].get_params(deep=False),
        "data_checksum": checksum,
        "feature_policy_version": config["decisions"]["feature_policy_version"],
        "split_version": config["decisions"]["split_version"],
        "evaluation_subset": "validation",
        "seed": seed,
        "metrics": validation_metrics,
        "train_seconds": float(elapsed),
        "command": _actual_baseline_command(),
        "software_versions": _software_versions(),
        "figure_paths": [relative_figure],
        "training_metrics": train_metrics,
        "training_subset": "train",
        "diagnostics": {
            "largest_validation_adr_row_id": int(X_valid.index[int(np.argmax(y_valid))]),
            "largest_validation_adr": float(np.max(y_valid)),
            "prediction_for_largest_adr": float(predicted_valid[int(np.argmax(y_valid))]),
            "largest_adr_squared_error_share": float(
                np.square(y_valid[int(np.argmax(y_valid))] - predicted_valid[int(np.argmax(y_valid))])
                / max(float(np.square(y_valid - predicted_valid).sum()), np.finfo(float).tiny)
            ),
            "negative_prediction_count": int(np.count_nonzero(predicted_valid < 0)),
            "no_outliers_dropped": True,
        },
        "features": features,
        "prediction_point": config["decisions"]["prediction_point"][TASK],
        "decision_status": config["decisions"].get(
            "prediction_point_status", "not_recorded"
        ),
        "population_policy": POPULATION_POLICY,
        "population": {"train": train_population, "validation": validation_population},
        "preprocessing_model_family": "linear",
        "preprocessing": dict(config["preprocessing"]),
        "test_set_evaluated": False,
        "fitted_artifact_saved": False,
        "metric_notes": {"mae": "Lower is better.", "rmse": "Lower is better.",
                         "r2": "Higher is better; negative values are possible."},
    }
    # Write generated figures/results only after *all* metrics and contracts pass.
    _save_residual_figure(predicted_valid, y_valid, image_path, seed)
    save_result(record, metrics_path)  # Member 6's shared schema/JSON writer.
    _write_report(report_path, record)

    print(f"ADR regression: {MODEL_NAME} (untuned); {len(features)} features")
    print(f"Train: {len(y_train):,} | validation: {len(y_valid):,}; test untouched")
    print("Validation: " + ", ".join(
        f"{key.upper()}={value:.4f}" for key, value in validation_metrics.items()
    ))
    print(f"Saved {metrics_path}, {image_path}, and {report_path}")
