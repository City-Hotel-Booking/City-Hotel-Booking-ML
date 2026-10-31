"""Member 3 optional M1 ADR improvement experiment (train/validation ONLY).

Usage: python -m src.regression.improve

Keeps the mandatory `python main.py baseline --task regression` linear baseline
unchanged. Runs fixed, untuned tree-family candidates using Member 2's frozen
split, leakage policy and unfitted tree preprocessor. A final test split is
never requested. All metric selection uses the original full validation set.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.prepare import get_task_data
from src.evaluation.results import save_result
from src.features.policy import get_features
from src.preprocessing.builders import build_preprocessor
from src.regression.baseline import (
    FORBIDDEN_FEATURES, POPULATION_POLICY, _finite_target_rows,
    _metrics, _require_contract, _software_versions,
)
from src.settings import load_config

TASK = "regression"
COMMAND = "python -m src.regression.improve"
MODEL_OPTIONS = ("hist_gradient_boosting_mae", "random_forest_regularized")


def _dense(values):
    """HistGradientBoosting requires dense input; no fitting is performed."""
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def _models(seed: int, config: dict):
    """Predeclared, fixed candidates. These are NOT tuned M2 experiments."""
    return {
        "hist_gradient_boosting_mae": Pipeline([
            ("preprocess", build_preprocessor(TASK, config, model_family="tree")),
            ("dense", FunctionTransformer(_dense, validate=False)),
            ("model", HistGradientBoostingRegressor(
                loss="absolute_error", random_state=seed, early_stopping=False
            )),
        ]),
        "random_forest_regularized": Pipeline([
            ("preprocess", build_preprocessor(TASK, config, model_family="tree")),
            ("model", RandomForestRegressor(
                n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=-1
            )),
        ]),
    }


def _verify_baseline(baseline: dict, config: dict, metadata: dict, features: list[str]) -> None:
    """Fail before fitting if the baseline cannot be fairly compared."""
    checks = {
        "task": TASK,
        "model": "linear_regression",
        "data_checksum": metadata["data_checksum_sha256"],
        "feature_policy_version": config["decisions"]["feature_policy_version"],
        "split_version": config["decisions"]["split_version"],
        "evaluation_subset": "validation",
        "seed": int(config["project"]["random_seed"]),
        "population_policy": POPULATION_POLICY,
        "test_set_evaluated": False,
    }
    for key, expected in checks.items():
        if baseline.get(key) != expected:
            raise ValueError(f"Baseline {key} mismatch: {baseline.get(key)!r} != {expected!r}")
    if baseline.get("features") != features:
        raise ValueError("Baseline features differ; cannot compare fairly.")
    for split, expected in metadata["row_counts"].items():
        if split in ("train", "validation") and baseline["population"][split]["source_rows"] != expected:
            raise ValueError(f"Baseline {split} sample count differs from frozen split.")
    for key in ("mae", "rmse", "r2"):
        score = baseline["metrics"].get(key)
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Baseline validation metric {key} is invalid.")


def _diagnostic(actual: np.ndarray, predicted: np.ndarray, row_ids: pd.Index) -> dict:
    """Diagnostic only: maximum observed validation ADR is never removed for selection."""
    errors = actual - predicted
    largest = int(np.argmax(actual))
    squared = np.square(errors)
    max_share = float(squared[largest] / squared.sum()) if squared.sum() else 0.0
    remaining = np.ones(len(actual), dtype=bool)
    remaining[largest] = False
    return {
        "largest_validation_adr_row_id": int(row_ids[largest]),
        "largest_validation_adr": float(actual[largest]),
        "prediction_for_largest_adr": float(predicted[largest]),
        "largest_row_share_of_total_squared_error": max_share,
        "rmse_without_largest_validation_adr_diagnostic_only": float(
            math.sqrt(float(np.mean(squared[remaining])))
        ),
        "negative_prediction_count": int(np.count_nonzero(predicted < 0)),
        "note": "The largest ADR row was INCLUDED in all primary validation metrics; removal is diagnostic only.",
    }


def _plot_comparison(rows: list[dict], destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    names = [row["model"].replace("_", "\n") for row in rows]
    for ax, metric, label in zip(axes, ("mae", "rmse"), ("Validation MAE", "Validation RMSE")):
        values = [row[metric] for row in rows]
        ax.bar(range(len(names)), values)
        ax.set_xticks(range(len(names)), names, fontsize=8)
        ax.set_ylabel("ADR units")
        ax.set_title(label + " (lower is better)")
        ax.grid(axis="y", alpha=0.2)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Full-validation regression comparison — reserved test not evaluated")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(destination, dpi=150)
    finally:
        plt.close(fig)


def _plot_residuals(actual: np.ndarray, predicted: np.ndarray, destination: Path,
                    seed: int, model_name: str) -> None:
    residuals = actual - predicted
    rng = np.random.default_rng(seed)
    sample = np.arange(len(actual))
    if len(sample) > 8000:
        sample = np.sort(rng.choice(sample, size=8000, replace=False))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(predicted[sample], residuals[sample], s=7, alpha=.28)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Predicted ADR", ylabel="Actual − predicted ADR",
                title="Residuals, reproducible sample")
    # Full histogram restricted to central 99% FOR DISPLAY ONLY; show tail separately.
    low, high = np.quantile(residuals, [0.005, 0.995])
    axes[1].hist(residuals, bins=70, range=(low, high))
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Actual − predicted ADR", ylabel="Count",
                title="Central 99% residuals (plot zoom only)")
    fig.suptitle(f"{model_name}: validation diagnostics (all rows scored)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(destination, dpi=150)
    finally:
        plt.close(fig)


def _write_report(path: Path, summary: dict) -> None:
    fmt = lambda v: f"{v:,.4f}"
    rows = summary["comparison"]
    selected = summary["selected_model"]
    chosen = next(item for item in rows if item["model"] == selected)
    lines = [
        "# Milestone 1 — ADR performance enhancement (Paula, Member 3)", "",
        "## Scope and integrity", "",
        "This is an **additional fixed-configuration comparison**; it does not replace",
        "the mandatory untuned linear baseline in `reports/regression_m1.md`.",
        "The original `m1-group-stratified-v1` source assignments, `m1-draft-1`",
        "feature policy and all-finite-ADR population were retained.",
        "Reserved test rows were **not used for fitting, model selection, or evaluation**.",
        "The shared loader reads the full CSV prior to selecting split rows.", "",
        f"- Source checksum: `{summary['data_checksum']}`.",
        f"- Training/validation rows: {summary['n_train']:,} / {summary['n_validation']:,}.",
        f"- Exact input features: `{', '.join(summary['features'])}`.",
        f"- Prediction point: booking entry (provisional pending team approval).",
        f"- Seed: `{summary['seed']}`; run: `{summary['command']}`.",
        "- Preprocessing: Member 2's unfitted `tree` builder, fitted separately",
        "  with each model on training rows only; categorical unknowns ignored.",
        "- Models: original LinearRegression baseline; fixed HistGradientBoostingRegressor",
        "  with absolute-error loss and `early_stopping=False` (no internal holdout);",
        "  RandomForestRegressor with 100 trees and",
        "  minimum five training samples per leaf (predeclared, not searched).", "",
        "## Runtime versions and reproducibility", "",
        "- Runtime versions: " + ", ".join(
            f"`{name}={ver}`" for name, ver in summary["software_versions"].items()
        ) + ".",
        "- `early_stopping=False` prevents a second, non-group-aware HGB internal split.",
        "- All candidate transformations are fitted solely on the frozen training rows.", "",
        "## Full validation comparison", "",
        "| Model | Train MAE | Train RMSE | Train R² | Validation MAE | Validation RMSE | Validation R² | Fit seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| `{row['model']}` | {fmt(row['train_mae'])} | {fmt(row['train_rmse'])} | "
                     f"{fmt(row['train_r2'])} | {fmt(row['mae'])} | {fmt(row['rmse'])} | "
                     f"{fmt(row['r2'])} | {fmt(row['train_seconds'])} |")
    lines += [
        "", "Model choice here uses **full-validation MAE** (RMSE breaks ties);",
        "test results and outlier-excluded diagnostics do not enter selection.",
        f"The lowest-MAE experimental candidate is `{selected}` (validation",
        f"MAE {fmt(chosen['mae'])}; RMSE {fmt(chosen['rmse'])}; R² {fmt(chosen['r2'])}).",
        "This selection is preliminary and is not a tuned/deployed M2 model.", "",
        "## Extreme validation ADR diagnostic (not model selection)", "",
        f"- Source row ID: {summary['extreme_row']['row_id']}.",
        f"- Recorded ADR: {fmt(summary['extreme_row']['actual_adr'])}.",
        "- Every model was evaluated on this row for the primary metrics.",
        "- The next table measures how much this single row contributes to each",
        "  model's total validation squared error. The last column is a diagnostic",
        "  sensitivity estimate only; do not substitute it for official RMSE.", "",
        "| Model | Prediction on largest ADR | Squared-error share | RMSE without largest row (diagnostic only) | Negative predictions |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        d = row["diagnostics"]
        lines.append(f"| `{row['model']}` | {fmt(d['prediction_for_largest_adr']) if d else 'Not recorded'} | "
                     f"{fmt(100*d['largest_row_share_of_total_squared_error'])+'%' if d else 'Not recorded'} | "
                     f"{fmt(d['rmse_without_largest_validation_adr_diagnostic_only']) if d else 'Not recorded'} | "
                     f"{d['negative_prediction_count'] if d else 'Not recorded'} |")
    lines += [
        "", "## Evidence and interpretation", "",
        "- Metric table: `outputs/regression/comparison.csv`.",
        "- Structured comparison: `outputs/regression/comparison.json`.",
        "- Model comparison figure: `outputs/regression/comparison.png`.",
        "- HGB-only residuals: `outputs/regression/hist_gradient_boosting_mae_residuals.png`.",
        "- Random Forest-only residuals: `outputs/regression/random_forest_regularized_residuals.png`.",
        f"- Backward-compatible enhanced residual alias: `outputs/regression/enhanced_residuals.png` (shows `{summary['plotted_enhanced_model']}`).",
        "- Individual candidate results: `outputs/regression/*_metrics.json`.",
        "- Large ADR events can dominate RMSE even when typical errors improve.",
        "  The score gap alone does not prove ordinary overfitting.",
        "- The extreme record must not be removed post hoc to improve the score.",
        f"- The maximum training ADR is {summary['max_training_adr']:.2f}, compared",
        f"  with the maximum validation ADR of {summary['extreme_row']['actual_adr']:.2f}.",
        "  Tree ensembles cannot reliably infer a rate outside the observed",
        "  training target range; their predictions are constrained",
        "  by observed training patterns, not guaranteed to extrapolate.",
        "- Original source values may change after booking entry; feature-timing",
        "  assumptions remain provisional pending Zeina/Michael review.", "",
        "## Next actions and reproducibility", "",
        "1. Preserve original baseline metrics and report for Milestone 1.",
        "2. Have Michael and Zeina review ADR population/feature timing and the",
        "   extreme value's validity; document approval in `reports/decisions.md`.",
        "3. For M2, tune at least three families using train-only group-aware CV",
        "   where suitable; use the frozen validation set for selection and",
        "   evaluate the reserved test once after model choices are frozen.",
        f"4. Reproduce via `{summary['command']}` after running the linear baseline.",
        "5. Record the source SHA, tested package versions, commit SHA and actual",
        "   author contributions in Makarious's integrated report.", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: dict, project_root: Path, *, command: str = COMMAND) -> dict:
    """Evaluate two fixed candidate models; never read or score the final test."""
    _require_contract(config)
    root = Path(project_root).resolve()
    features = get_features(TASK, config)
    if not features or set(features) & FORBIDDEN_FEATURES or len(features) != len(set(features)):
        raise ValueError("Invalid or leakage-prone regression feature list.")
    meta_file = root / config["paths"]["data_outputs"] / "split_metadata.json"
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    if metadata.get("split_version") != config["decisions"]["split_version"]:
        raise ValueError("Frozen split metadata version mismatch.")
    if metadata.get("source_rows") != 79330 or metadata.get("source_columns") != 31:
        raise ValueError("H2 source contract mismatch.")
    out_dir = root / config["paths"]["regression_outputs"]
    baseline_file = out_dir / "metrics.json"
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    _verify_baseline(baseline, config, metadata, features)

    # Strictly train and validation only. get_task_data validates input checksum.
    Xt_raw, yt_raw = get_task_data(TASK, root, "train", config=config)
    Xv_raw, yv_raw = get_task_data(TASK, root, "validation", config=config)
    if list(Xt_raw.columns) != features or list(Xv_raw.columns) != features:
        raise ValueError("Task data differs from frozen approved features.")
    if not Xt_raw.index.intersection(Xv_raw.index).empty:
        raise ValueError("Train and validation source IDs overlap.")
    Xt, yt, pt = _finite_target_rows(Xt_raw, yt_raw, "train")
    Xv, yv, pv = _finite_target_rows(Xv_raw, yv_raw, "validation")
    if pt["source_rows"] != metadata["row_counts"]["train"] or pv["source_rows"] != metadata["row_counts"]["validation"]:
        raise ValueError("Population and frozen split row counts differ.")
    if pt != baseline["population"]["train"] or pv != baseline["population"]["validation"]:
        raise ValueError("Baseline population statistics differ; comparison invalid.")
    seed = int(config["project"]["random_seed"])
    rows = [{
        "model": "linear_regression", "mae": baseline["metrics"]["mae"],
        "rmse": baseline["metrics"]["rmse"], "r2": baseline["metrics"]["r2"],
        "train_mae": baseline["training_metrics"]["mae"],
        "train_rmse": baseline["training_metrics"]["rmse"],
        "train_r2": baseline["training_metrics"]["r2"],
        "train_seconds": baseline["train_seconds"],
        "diagnostics": (None if "diagnostics" not in baseline else {
            "prediction_for_largest_adr": baseline["diagnostics"]["prediction_for_largest_adr"],
            "largest_row_share_of_total_squared_error": baseline["diagnostics"]["largest_adr_squared_error_share"],
            "negative_prediction_count": baseline["diagnostics"]["negative_prediction_count"],
            "rmse_without_largest_validation_adr_diagnostic_only": math.sqrt(
                max(0.0, (len(yv) * baseline["metrics"]["rmse"] ** 2 -
                          (baseline["diagnostics"]["largest_validation_adr"] -
                           baseline["diagnostics"]["prediction_for_largest_adr"]) ** 2) / (len(yv) - 1))
            ),
        }),
    }]
    predictions = {}
    artifacts = {}
    for name, pipeline in _models(seed, config).items():
        start = perf_counter()
        pipeline.fit(Xt, yt)
        seconds = float(perf_counter() - start)
        train_prediction = np.asarray(pipeline.predict(Xt), dtype=float)
        valid_prediction = np.asarray(pipeline.predict(Xv), dtype=float)
        if not np.isfinite(train_prediction).all() or not np.isfinite(valid_prediction).all():
            raise ValueError(f"Model {name} produced nonfinite predictions.")
        tm = _metrics(yt, train_prediction)
        vm = _metrics(yv, valid_prediction)
        diagnostics = _diagnostic(yv, valid_prediction, Xv.index)
        rows.append({
            "model": name, **vm, "train_mae": tm["mae"],
            "train_rmse": tm["rmse"], "train_r2": tm["r2"],
            "train_seconds": seconds, "diagnostics": diagnostics,
        })
        predictions[name] = valid_prediction
        artifact = {
            "task": TASK, "model": name,
            "parameters": pipeline.named_steps["model"].get_params(deep=False),
            "data_checksum": metadata["data_checksum_sha256"],
            "feature_policy_version": config["decisions"]["feature_policy_version"],
            "split_version": config["decisions"]["split_version"],
            "evaluation_subset": "validation", "seed": seed, "metrics": vm,
            "train_seconds": seconds, "command": command,
            "software_versions": _software_versions(),
            "figure_paths": ["outputs/regression/comparison.png", f"outputs/regression/{name}_residuals.png"],
            "training_metrics": tm, "features": features,
            "population": {"train": pt, "validation": pv},
            "population_policy": POPULATION_POLICY,
            "preprocessing_model_family": "tree",
            "diagnostics": diagnostics,
            "test_set_evaluated": False,
            "tuned": False,
        }
        artifacts[name] = artifact
    # Best candidate by full validation MAE, with RMSE as a tie breaker.
    chosen = min(rows, key=lambda row: (row["mae"], row["rmse"]))
    selected_name = chosen["model"]
    # Recreate baseline predictions only if baseline is selected? Avoid a second fit;
    # use the best enhanced candidate for the residual plot and label it accurately.
    enhanced_only = min(rows[1:], key=lambda row: (row["mae"], row["rmse"]))["model"]
    _plot_comparison(rows, out_dir / "comparison.png")
    for name, valid_predictions in predictions.items():
        _plot_residuals(yv, valid_predictions, out_dir / f"{name}_residuals.png", seed, name)
    # Legacy plot always matches the selected enhanced candidate, and is labelled.
    shutil.copyfile(out_dir / f"{enhanced_only}_residuals.png", out_dir / "enhanced_residuals.png")
    for name, artifact in artifacts.items():
        save_result(artifact, out_dir / f"{name}_metrics.json")
    summary = {
        "task": TASK, "source": "H2", "data_checksum": metadata["data_checksum_sha256"],
        "feature_policy_version": config["decisions"]["feature_policy_version"],
        "split_version": config["decisions"]["split_version"],
        "seed": seed, "n_train": int(len(yt)), "n_validation": int(len(yv)),
        "features": features, "population_policy": POPULATION_POLICY,
        "max_training_adr": float(np.max(yt)),
        "selected_model": selected_name, "plotted_enhanced_model": enhanced_only,
        "selected_criterion": "minimum FULL validation MAE; RMSE tie breaker",
        "extreme_row": {"row_id": int(Xv.index[int(np.argmax(yv))]), "actual_adr": float(np.max(yv))},
        "command": command, "software_versions": _software_versions(),
        "residual_figure_paths": {
            name: f"outputs/regression/{name}_residuals.png" for name in predictions
        },
        "test_set_evaluated": False,
        "comparison": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame([{key: value for key, value in row.items() if key != "diagnostics"} for row in rows]).to_csv(out_dir / "comparison.csv", index=False)
    _write_report(root / config["paths"]["reports"] / "regression_enhancement_m1.md", summary)
    print("Validation model comparison (original full population; no test evaluation):")
    for row in rows:
        print(f"  {row['model']}: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, R2={row['r2']:.4f}")
    print(f"Lowest full-validation MAE: {selected_name}")
    print("Saved outputs/regression/comparison.*, per-model residual figures, "
          "enhanced_residuals.png, reports/regression_enhancement_m1.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen-split ADR regression candidates")
    parser.add_argument("--config", default="config.yaml", help="Project config path (relative to repository)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    invocation = "python -m src.regression.improve"
    if sys.argv[1:]:
        invocation += " " + shlex.join(sys.argv[1:])
    run(config, root, command=invocation)


if __name__ == "__main__":
    main()
