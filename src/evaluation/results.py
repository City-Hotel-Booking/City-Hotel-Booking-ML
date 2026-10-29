"""Save a consistent JSON record for each model experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path


REQUIRED_FIELDS = {
    "task",
    "model",
    "parameters",
    "data_checksum",
    "feature_policy_version",
    "split_version",
    "evaluation_subset",
    "seed",
    "metrics",
    "train_seconds",
    "command",
    "figure_paths",
}

VALID_TASKS = {"regression", "classification", "clustering"}


def save_result(result: dict, path: Path) -> None:
    """Validate and save an experiment record.

    Use ordinary Python values in the record. Unavailable metrics should
    be None, with explanations in an optional metric_notes dictionary.
    Additional task-specific fields are preserved.
    """
    if not isinstance(result, dict):
        raise ValueError("The result must be a dictionary.")

    missing = REQUIRED_FIELDS - result.keys()
    if missing:
        raise ValueError(f"Missing result fields: {sorted(missing)}")

    if result["task"] not in VALID_TASKS:
        raise ValueError("Unknown experiment task.")

    if not isinstance(result["parameters"], dict):
        raise ValueError("parameters must be a dictionary.")

    if not isinstance(result["metrics"], dict) or not result["metrics"]:
        raise ValueError("metrics must be a nonempty dictionary.")

    if not isinstance(result["figure_paths"], list) or not all(
        isinstance(item, str) for item in result["figure_paths"]
    ):
        raise ValueError("figure_paths must be a list of strings.")

    seconds = result["train_seconds"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds < 0
    ):
        raise ValueError("train_seconds must be a finite nonnegative number.")

    # Serialize before opening the destination so invalid records cannot
    # overwrite an existing result file.
    try:
        text = json.dumps(result, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Result values must be JSON-compatible and contain no NaN "
            "or infinity. Convert NumPy values to Python values first."
        ) from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + "\n", encoding="utf-8")