"""Behavioral checks for the shared, unfitted preprocessing interface."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.features.policy import get_features
from src.preprocessing.builders import (
    CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, build_preprocessor,
)


CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())


def _rows(task: str) -> pd.DataFrame:
    features = get_features(task, CONFIG)
    values = {
        column: ["1", "1", "1", "1"] if column in NUMERIC_COLUMNS
        else ["A", "A", "A", "A"]
        for column in features
    }
    frame = pd.DataFrame(values, index=[10, 11, 12, 13])
    frame["LeadTime"] = ["1", "3", "NULL", "5"]
    frame["Meal"] = ["BB", "HB", " BB ", "NULL"]
    return frame


class PreprocessorTests(unittest.TestCase):
    def test_builder_is_unfitted_and_excludes_targets(self) -> None:
        for task in ("regression", "classification", "clustering"):
            transformer = build_preprocessor(task, CONFIG)
            with self.assertRaises(NotFittedError):
                check_is_fitted(transformer)
            selected = [column for _, _, columns in transformer.transformers for column in columns]
            self.assertEqual(set(selected), set(get_features(task, CONFIG)))
            self.assertEqual(len(selected), len(set(selected)))
            self.assertFalse(set(selected) & {"IsCanceled", "ADR", "ReservationStatus"})

    def test_training_median_and_categories_ignore_held_out_rows(self) -> None:
        train = _rows("classification")
        held_out = train.iloc[[0]].copy()
        held_out.index = [99]
        held_out["LeadTime"] = "999"
        held_out["Meal"] = "EXOTIC"

        transformer = build_preprocessor("classification", CONFIG)
        transformer.fit(train)
        numeric_columns = transformer.transformers_[0][2]
        lead_time_position = numeric_columns.index("LeadTime")
        median = transformer.named_transformers_["numeric"].named_steps["impute"].statistics_[
            lead_time_position
        ]
        self.assertEqual(median, 3.0)

        category_columns = transformer.transformers_[1][2]
        meal_position = category_columns.index("Meal")
        encoder = transformer.named_transformers_["categorical"].named_steps["encode"]
        self.assertNotIn("EXOTIC", encoder.categories_[meal_position])
        self.assertIn("BB", encoder.categories_[meal_position])
        self.assertNotIn(" BB ", encoder.categories_[meal_position])

        output = transformer.transform(held_out)
        self.assertEqual(output.shape, (1, transformer.transform(train).shape[1]))
        self.assertTrue(np.isfinite(output.toarray() if hasattr(output, "toarray") else output).all())
        category_output = transformer.named_transformers_["categorical"].transform(
            held_out[category_columns]
        )
        category_output = (
            category_output.toarray() if hasattr(category_output, "toarray") else category_output
        )
        meal_offset = sum(len(categories) for categories in encoder.categories_[:meal_position])
        meal_width = len(encoder.categories_[meal_position])
        self.assertEqual(category_output[0, meal_offset:meal_offset + meal_width].sum(), 0)

    def test_numeric_ids_never_get_median_imputed_as_measures(self) -> None:
        # Agent and Company are excluded by policy. If approved later, they
        # belong to the categorical rather than numeric family.
        self.assertFalse({"Agent", "Company", "Country"} & NUMERIC_COLUMNS)
        self.assertTrue({"Agent", "Company", "Country"} <= CATEGORICAL_COLUMNS)
        self.assertFalse({"Agent", "Company", "Country"} & set(get_features("classification", CONFIG)))

    def test_literal_na_in_numeric_source_is_imputed(self) -> None:
        train = _rows("regression")
        train.loc[12, "Children"] = "NA"
        transformer = build_preprocessor("regression", CONFIG)
        result = transformer.fit_transform(train)
        self.assertTrue(np.isfinite(result.toarray() if hasattr(result, "toarray") else result).all())

    def test_unsupported_settings_fail_instead_of_silent_fallback(self) -> None:
        changed = {**CONFIG, "preprocessing": {**CONFIG["preprocessing"], "unknown_categories": "error"}}
        with self.assertRaisesRegex(ValueError, "unknown_categories"):
            build_preprocessor("classification", changed)

    def test_model_pipeline_fits_preprocessor_on_train_indices_only(self) -> None:
        rows = _rows("classification")
        train_ids = [10, 11, 12]
        validation_ids = [13]
        observed_fit_indices: list[list[int]] = []
        original = ColumnTransformer.fit_transform

        def record_fit(transformer, X, y=None, **kwargs):
            observed_fit_indices.append(list(X.index))
            return original(transformer, X, y=y, **kwargs)

        model = Pipeline([
            ("preprocess", build_preprocessor("classification", CONFIG)),
            ("estimator", DummyClassifier()),
        ])
        with patch.object(ColumnTransformer, "fit_transform", record_fit):
            model.fit(rows.loc[train_ids], [0, 1, 0])
        self.assertEqual(observed_fit_indices, [train_ids])
        self.assertEqual(len(model.predict(rows.loc[validation_ids])), 1)


if __name__ == "__main__":
    unittest.main()
