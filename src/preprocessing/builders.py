"""Build unfitted preprocessing for each provisional booking-time feature list."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.policy import get_features


NUMERIC_COLUMNS = frozenset({
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "BookingChanges",
    "DaysInWaitingList", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
})
CATEGORICAL_COLUMNS = frozenset({
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment",
    "DistributionChannel", "ReservedRoomType", "AssignedRoomType",
    "DepositType", "Agent", "Company", "CustomerType", "ReservationStatus",
})
HIGH_CARDINALITY_COLUMNS = frozenset({"Country", "Agent", "Company"})


def _clean_value(value: Any, preserve_null: bool = False) -> Any:
    """Apply deterministic import cleaning inside the modeling pipeline."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value.upper() == "NULL" and preserve_null:
            return "NULL"
        if value.upper() in ("", "NULL", "NA", "N/A", "NAN"):
            return np.nan
    return value


def _numeric_array(values: Any) -> np.ndarray:
    frame = pd.DataFrame(values).apply(lambda column: column.map(_clean_value))
    numeric = frame.apply(pd.to_numeric, errors="raise")
    return numeric.to_numpy(dtype=float, na_value=np.nan)


def _categorical_array(values: Any) -> np.ndarray:
    frame = pd.DataFrame(values).apply(
        lambda column: column.map(
            lambda value: _clean_value(value, column.name in {"Agent", "Company"})
        )
    )
    categories = frame.apply(
        lambda column: column.map(
            lambda value: np.nan if pd.isna(value) else str(value)
        )
    )
    return categories.to_numpy(dtype=object)


def _log_selected(values: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Reduce right skew in approved nonnegative inputs after imputation."""
    result = values.copy()
    for index in indices:
        if np.any(result[:, index] < 0):
            raise ValueError("Log-transformed booking counts must be nonnegative.")
        result[:, index] = np.log1p(result[:, index])
    return result


def build_preprocessor(
    task: str, config: dict, *, model_family: str = "distance"
) -> ColumnTransformer:
    """Return an unfitted transformer for the task's active source inputs.

    Fit the complete model pipeline on training folds only. Distance and
    linear models scale numeric inputs; trees keep their original units.
    Clustering has its own feature list and a dense, scaled output.
    Disputed high-cardinality IDs stay out until approved; if later enabled,
    rare-category one-hot avoids arbitrary numeric distances between IDs.
    """
    features = get_features(task, config)
    if model_family not in {"distance", "linear", "tree"}:
        raise ValueError("model_family must be distance, linear, or tree.")
    if task == "clustering" and model_family != "distance":
        raise ValueError("Clustering uses the scaled distance preprocessor.")
    if not features:
        raise ValueError(f"No active features for task {task}.")
    unknown = set(features) - NUMERIC_COLUMNS - CATEGORICAL_COLUMNS
    if unknown:
        raise ValueError(f"Features need a documented type: {sorted(unknown)}")

    settings = config.get("preprocessing", {})
    required = {
        "numeric_imputation": "median",
        "numeric_scaling": "standard",
        "categorical_imputation": "constant_missing",
        "categorical_encoding": "one_hot",
        "unknown_categories": "ignore",
    }
    for key, expected in required.items():
        if settings.get(key) != expected:
            raise ValueError(f"preprocessing.{key} must be {expected!r}.")

    numeric_columns = [column for column in features if column in NUMERIC_COLUMNS]
    skewed = tuple(settings.get("skewed_numeric_features", ["LeadTime"]))
    if set(skewed) - {"LeadTime"}:
        raise ValueError("Only LeadTime has an approved log1p input transform.")
    log_indices = tuple(numeric_columns.index(column) for column in skewed if column in numeric_columns)
    category_columns = [
        column for column in features
        if column in CATEGORICAL_COLUMNS - HIGH_CARDINALITY_COLUMNS
    ]
    high_cardinality = [
        column for column in features if column in HIGH_CARDINALITY_COLUMNS
    ]
    transformers = []
    if numeric_columns:
        numeric_steps = [
            ("parse", FunctionTransformer(
                _numeric_array, validate=False, feature_names_out="one-to-one"
            )),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
        if model_family != "tree" and log_indices:
            numeric_steps.append(("log_skewed", FunctionTransformer(
                _log_selected, kw_args={"indices": log_indices}, validate=False,
                feature_names_out="one-to-one",
            )))
        if model_family != "tree":
            numeric_steps.append(("scale", StandardScaler()))
        numeric = Pipeline(numeric_steps)
        transformers.append(("numeric", numeric, numeric_columns))
    if category_columns:
        categorical = Pipeline([
            ("clean", FunctionTransformer(
                _categorical_array, validate=False, feature_names_out="one-to-one"
            )),
            ("impute", SimpleImputer(
                strategy="constant", fill_value="__MISSING__", keep_empty_features=True
            )),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical, category_columns))
    if high_cardinality:
        if settings.get("high_cardinality_encoding") != "one_hot_rare":
            raise ValueError("Approved high-cardinality inputs need one_hot_rare.")
        minimum = int(settings.get("high_cardinality_min_frequency", 0))
        if minimum < 2:
            raise ValueError("high_cardinality_min_frequency must be at least 2.")
        rare_categorical = Pipeline([
            ("clean", FunctionTransformer(
                _categorical_array, validate=False, feature_names_out="one-to-one"
            )),
            ("impute", SimpleImputer(
                strategy="constant", fill_value="__MISSING__", keep_empty_features=True
            )),
            ("encode", OneHotEncoder(
                handle_unknown="ignore", min_frequency=minimum
            )),
        ])
        transformers.append(("high_cardinality", rare_categorical, high_cardinality))
    return ColumnTransformer(
        transformers, remainder="drop", sparse_threshold=0 if task == "clustering" else 0.3
    )
