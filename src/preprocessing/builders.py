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


def _clean_value(value: Any) -> Any:
    """Apply deterministic import cleaning inside the modeling pipeline."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value.upper() in ("", "NULL", "NA"):
            return np.nan
    return value


def _numeric_array(values: Any) -> np.ndarray:
    frame = pd.DataFrame(values).apply(lambda column: column.map(_clean_value))
    numeric = frame.apply(pd.to_numeric, errors="raise")
    return numeric.to_numpy(dtype=float, na_value=np.nan)


def _categorical_array(values: Any) -> np.ndarray:
    frame = pd.DataFrame(values).apply(lambda column: column.map(_clean_value))
    categories = frame.apply(
        lambda column: column.map(
            lambda value: np.nan if pd.isna(value) else str(value)
        )
    )
    return categories.to_numpy(dtype=object)


def build_preprocessor(task: str, config: dict) -> ColumnTransformer:
    """Return an unfitted transformer for the task's active source inputs.

    Members 3 and 4 put this object inside a model Pipeline and fit the whole
    pipeline on training folds only. Clustering uses its own separate instance.
    Disputed fields remain absent until the team approves a new policy version.
    """
    features = get_features(task, config)
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
    category_columns = [
        column for column in features
        if column in CATEGORICAL_COLUMNS - HIGH_CARDINALITY_COLUMNS
    ]
    high_cardinality = [
        column for column in features if column in HIGH_CARDINALITY_COLUMNS
    ]
    transformers = []
    if numeric_columns:
        numeric = Pipeline([
            ("parse", FunctionTransformer(
                _numeric_array, validate=False, feature_names_out="one-to-one"
            )),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
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
    return ColumnTransformer(transformers, remainder="drop")
