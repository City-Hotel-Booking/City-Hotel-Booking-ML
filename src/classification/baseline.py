"""Untuned cancellation classification on the shared frozen split."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data.prepare import get_task_data
from src.evaluation.results import save_result
from src.preprocessing.builders import build_preprocessor


THRESHOLD = 0.5


def _validate_target(values: pd.Series) -> pd.Series:
    """Reject invalid labels without silently excluding source rows."""
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError("Classification requires nonmissing binary targets.")
    return numeric.astype(int)


def evaluate(
    target: pd.Series,
    probabilities: np.ndarray,
    threshold: float = THRESHOLD,
) -> tuple[dict, list[list[int]], dict]:
    """Evaluate cancellation probabilities with positive class equal to one."""
    y = _validate_target(target)
    scores = np.asarray(probabilities, dtype=float)

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one.")

    if (
        scores.ndim != 1
        or len(scores) != len(y)
        or len(y) == 0
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Expected one finite probability per target.")

    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    _, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )

    metrics = {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "average_precision": None,
    }
    notes = {}

    if true_positive + false_positive:
        metrics["precision"] = float(
            precision_score(y, predicted, pos_label=1, zero_division=0)
        )
    else:
        notes["precision"] = "Undefined because no positive predictions exist."

    if true_positive + false_negative:
        metrics["recall"] = float(
            recall_score(y, predicted, pos_label=1, zero_division=0)
        )
    else:
        notes["recall"] = "Undefined because no positive targets exist."

    if 2 * true_positive + false_positive + false_negative:
        metrics["f1"] = float(
            f1_score(y, predicted, pos_label=1, zero_division=0)
        )
    else:
        notes["f1"] = (
            "Undefined because neither positive targets nor positive "
            "predictions exist."
        )

    if y.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, scores))
        metrics["average_precision"] = float(
            average_precision_score(y, scores)
        )
    else:
        reason = (
            "Not reported because validation contains only one target class."
        )
        notes["roc_auc"] = reason
        notes["average_precision"] = reason

    return metrics, matrix.astype(int).tolist(), notes


def _write_report(result: dict, destination: Path) -> None:
    """Generate a report using actual metrics and recorded settings."""
    metric_rows = "\n".join(
        f"| {name} | "
        f"{'Unavailable' if value is None else format(value, '.6f')} |"
        for name, value in result["metrics"].items()
    )
    features = "\n".join(f"- `{name}`" for name in result["features"])
    metric_notes = "\n".join(
        f"- {name}: {reason}"
        for name, reason in result["metric_notes"].items()
    ) or "- All requested metrics are available."

    train_counts = result["class_counts"]["train"]
    validation_counts = result["class_counts"]["validation"]
    tn, fp = result["confusion_matrix"][0]
    fn, tp = result["confusion_matrix"][1]

    text = f"""# Cancellation classification baseline

Status: provisional under the current shared draft scientific policy.

## Purpose and prediction point

Predict whether a booking will be canceled.
The target is IsCanceled; cancellation is the positive class, labeled 1.
The starting prediction point is initial booking entry into the hotel PMS.

H2 is a later snapshot. Availability of the active inputs at booking entry
remains an assumption requiring the review recorded in reports/decisions.md.
This experiment does not establish scientific approval of that assumption.

## Population and frozen assignments

Use all frozen training and validation rows with valid binary targets.
Invalid targets stop the experiment rather than being silently excluded.
No task-specific filtering, clipping, duplicate removal or resampling is used.
Canceled, zero-rate, zero-night and extreme-rate records are retained.

| Subset | Rows | Not canceled | Canceled |
| --- | ---: | ---: | ---: |
| Train | {result["row_counts"]["train"]} | {train_counts["0"]} | {train_counts["1"]} |
| Validation | {result["row_counts"]["validation"]} | {validation_counts["0"]} | {validation_counts["1"]} |

Source positions and frozen subset assignments are preserved.
The task runner requests only train and validation through get_task_data.
Final test performance is not evaluated.

Source SHA-256: `{result["data_checksum"]}`.

Split version: `{result["split_version"]}`.

Feature-policy version: `{result["feature_policy_version"]}`.

## Baseline algorithm and fixed settings

Model: LogisticRegression.

Fixed parameters:

{json.dumps(result["parameters"], indent=2, sort_keys=True)}

Logistic regression supplies a simple probabilistic baseline for a binary
target. The shared linear preprocessing supports its numeric scale assumptions.
C=1.0 and the lbfgs solver are fixed starting choices. max_iter=2000 is an
optimization limit. No hyperparameter search or class weighting is performed.

Decision threshold: {result["decision_threshold"]}.
A probability equal to the threshold is classified as cancellation.
The threshold was fixed before evaluation and was not optimized on validation.

Training time including preprocessing: {result["train_seconds"]:.6f} seconds.
Optimizer iterations: {result["optimizer_iterations"]}.

## Inputs and preprocessing

Exact input features:

{features}

A fresh shared linear preprocessor and LogisticRegression are fitted together
inside one scikit-learn Pipeline on training rows only.

Numeric processing parses values, imputes training medians, applies the
configured log1p transformation to LeadTime and standardizes numeric features.
Categorical processing uses a missing marker and one-hot encoding.
Unseen validation categories are ignored without refitting.

Recorded preprocessing configuration:

{json.dumps(result["preprocessing"], indent=2, sort_keys=True)}

IsCanceled, ADR, ReservationStatus, ReservationStatusDate, AssignedRoomType,
BookingChanges and DaysInWaitingList are absent from the active inputs.
No learned preprocessing is fitted on validation or final test rows.

## Validation results

| Metric | Value |
| --- | ---: |
{metric_rows}

Average precision is calculated using average_precision_score.
It is not described as trapezoidal area under the precision-recall curve.
Undefined metrics are stored as null with explanations.

Metric notes:

{metric_notes}

Confusion matrix, with true labels on rows and predicted labels on columns:

| True / Predicted | Not canceled (0) | Canceled (1) |
| --- | ---: | ---: |
| Not canceled (0) | {tn} | {fp} |
| Canceled (1) | {fn} | {tp} |

[Metrics and settings](../outputs/classification/metrics.json)

[Confusion matrix figure](../outputs/classification/confusion_matrix.png)

## Interpretation and limitations

At the fixed threshold, the model correctly identifies {tp} canceled bookings
and misses {fn}. It flags {fp} noncanceled bookings as cancellations and
correctly identifies {tn} noncanceled bookings.

Precision describes the reliability of cancellation alerts. Recall describes
the fraction of cancellations identified. F1 summarizes their balance.
ROC-AUC and average precision evaluate probability ranking independently of
the fixed threshold. Accuracy should be interpreted alongside class balance.

These are development results from one untuned model. No superiority claim
over other classifiers is supported by this experiment alone.

The grouped random split limits exact-duplicate overlap, but it does not
measure future-year drift or identify all repeated real bookings without a
booking identifier. Mutable feature timing and scientific approvals remain
open as documented in the shared decision tracker.

The validation majority-class reference achieves
{max(validation_counts.values()) / result["row_counts"]["validation"]:.2%}
accuracy, compared with {result["metrics"]["accuracy"]:.2%}
for this logistic regression baseline.
At threshold {result["decision_threshold"]}, cancellation recall is
{format(result["metrics"]["recall"], ".2%") if result["metrics"]["recall"] is not None else "unavailable"}.

The current model is therefore a weak initial baseline. These results
do not support deployment or a claim of improvement over the majority
reference. Future experiments should examine recall, precision, F1 and
ranking metrics together, while preserving this baseline for comparison.

## Reproduction and environment

Run from the project root:

    {result["command"]}

Python: {result["environment"]["python"]}.
scikit-learn: {result["environment"]["scikit_learn"]}.
pandas: {result["environment"]["pandas"]}.
NumPy: {result["environment"]["numpy"]}.

The JSON record contains the effective configuration, parameters, row counts,
source identity, policy versions and random seed.
Runtime can vary across machines.

The runner generates metrics.json, confusion_matrix.png and this report.
The report records experiment execution; test-suite status is verified
separately using pytest.

## Milestone 2 plan

Compare additional algorithms and tuned configurations using development data.
Any cross-validation must preserve duplicate groups and fit preprocessing
inside each training fold. Investigate class weighting, threshold selection,
PCA/LDA comparisons and an ensemble or boosting classifier.

After model selection, evaluate the reserved test subset and save the final
inference pipeline for application use. These extensions are planned work.
"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def run(config: dict, project_root: Path) -> None:
    """Fit the baseline on train and report validation metrics only."""
    root = Path(project_root).resolve()

    X_train, y_train = get_task_data(
        "classification", root, "train", config=config
    )
    X_validation, y_validation = get_task_data(
        "classification", root, "validation", config=config
    )

    y_train = _validate_target(y_train)
    y_validation = _validate_target(y_validation)

    if X_train.empty or X_validation.empty:
        raise ValueError("Training and validation inputs must be nonempty.")
    if y_train.nunique() != 2:
        raise ValueError("Training data must contain both target classes.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("Training features and targets are misaligned.")
    if not X_validation.index.equals(y_validation.index):
        raise ValueError("Validation features and targets are misaligned.")
    if not X_train.index.is_unique or not X_validation.index.is_unique:
        raise ValueError("Source row IDs must be unique within each subset.")
    if not X_train.index.intersection(X_validation.index).empty:
        raise ValueError("Training and validation row IDs overlap.")
    if not X_train.columns.equals(X_validation.columns):
        raise ValueError("Training and validation feature columns differ.")

    seed = int(config["project"]["random_seed"])
    estimator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight=None,
        random_state=seed,
    )
    pipeline = Pipeline([
        (
            "preprocess",
            build_preprocessor(
                "classification", config, model_family="linear"
            ),
        ),
        ("model", estimator),
    ])

    started = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            pipeline.fit(X_train, y_train)
        except ConvergenceWarning as exc:
            raise ValueError(
                "Logistic regression did not converge. Inspect the run "
                "before accepting baseline results."
            ) from exc
    train_seconds = perf_counter() - started

    positive_index = int(
        np.flatnonzero(pipeline.named_steps["model"].classes_ == 1)[0]
    )
    probabilities = pipeline.predict_proba(X_validation)[:, positive_index]
    metrics, matrix, metric_notes = evaluate(
        y_validation, probabilities, THRESHOLD
    )

    metadata_path = (
        root / config["paths"]["data_outputs"] / "split_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    output_directory = root / config["paths"]["classification_outputs"]
    figure_path = output_directory / "confusion_matrix.png"

    result = {
        "task": "classification",
        "experiment_id": "logistic-regression-m1-provisional-v1",
        "status": "provisional",
        "model": "LogisticRegression",
        "parameters": estimator.get_params(deep=False),
        "data_checksum": metadata["data_checksum_sha256"],
        "feature_policy_version": config["decisions"][
            "feature_policy_version"
        ],
        "split_version": metadata["split_version"],
        "evaluation_subset": "validation",
        "seed": seed,
        "metrics": metrics,
        "metric_notes": metric_notes,
        "train_seconds": float(train_seconds),
        "command": "& " + subprocess.list2cmdline([sys.executable, *sys.argv]),
        "figure_paths": [figure_path.relative_to(root).as_posix()],
        "positive_class": 1,
        "decision_threshold": THRESHOLD,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": [0, 1],
        "features": list(X_train.columns),
        "model_family": "linear",
        "preprocessing": config["preprocessing"],
        "effective_config": config,
        "prediction_point": config["decisions"]["prediction_point"][
            "classification"
        ],
        "population": (
            "All frozen training and validation rows with valid binary "
            "IsCanceled; invalid targets are rejected; no row exclusions."
        ),
        "row_counts": {
            "train": int(len(X_train)),
            "validation": int(len(X_validation)),
        },
        "class_counts": {
            "train": {
                str(label): int(y_train.eq(label).sum())
                for label in (0, 1)
            },
            "validation": {
                str(label): int(y_validation.eq(label).sum())
                for label in (0, 1)
            },
        },
        "task_specific_exclusions": 0,
        "policy_departures": [],
        "optimizer_iterations": [
            int(value) for value in estimator.n_iter_
        ],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": version("scikit-learn"),
            "pandas": version("pandas"),
            "numpy": version("numpy"),
        },
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    try:
        display = ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(matrix),
            display_labels=["Not canceled (0)", "Canceled (1)"],
        )
        display.plot(
            ax=axis, cmap="Blues", colorbar=False, values_format="d"
        )
        axis.set_title("Cancellation baseline: validation")
        figure.tight_layout()
        figure.savefig(figure_path, dpi=160)
    finally:
        plt.close(figure)

    save_result(result, output_directory / "metrics.json")
    _write_report(
        result,
        root / config["paths"]["reports"] / "classification_m1.md",
    )

    print("Classification baseline completed (provisional).")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_validation)}")
    print(f"Training seconds: {train_seconds:.6f}")
    print(f"Results: {output_directory / 'metrics.json'}")
    print(f"Figure: {figure_path}")