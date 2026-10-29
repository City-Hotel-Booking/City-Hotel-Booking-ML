"""Run real H2 preparation checks without fitting on validation or test rows."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.data.load import load_h2
from src.data.prepare import load_manifest
from src.data.split import _read_h2_shared, create_split
from src.features.policy import get_features
from src.preprocessing.builders import build_preprocessor, _categorical_array
from src.settings import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = {"IsCanceled", "ReservationStatus", "ReservationStatusDate", "ADR"}


def _finite(values: object) -> bool:
    """Check sparse and dense transformer outputs without densifying them."""
    array = values.data if hasattr(values, "toarray") else np.asarray(values)
    return bool(np.isfinite(array).all())


def run_checks(config: dict, project_root: Path) -> dict:
    root = Path(project_root)
    results: dict[str, dict] = {}

    def record(name: str, passed: bool, detail: str) -> None:
        results[name] = {"status": "PASS" if passed else "FAIL", "detail": detail}

    source_path = root / config["data"]["raw_path"]
    source = load_h2(source_path)
    manifest = load_manifest(root, config=config)
    metadata = json.loads((root / config["paths"]["data_outputs"] / "split_metadata.json").read_text())
    ids = {name: manifest.loc[manifest["split"].eq(name), "row_id"].astype(int).tolist()
           for name in ("train", "validation", "test")}

    valid_manifest = (list(manifest.columns) == ["row_id", "split"]
                      and len(manifest) == len(source)
                      and set(manifest["row_id"]) == set(range(len(source))))
    record("label_free_manifest", valid_manifest,
           f"columns={list(manifest.columns)}, rows={len(manifest)}")
    same_source = source.attrs["data_checksum_sha256"] == metadata["data_checksum_sha256"]
    record("source_identity", same_source, f"sha256={source.attrs['data_checksum_sha256']}")

    selected = {task: get_features(task, config)
                for task in ("regression", "classification", "clustering")}
    forbidden_found = {task: sorted(FORBIDDEN & set(columns))
                       for task, columns in selected.items()}
    record("forbidden_outcomes_absent", all(not found for found in forbidden_found.values()),
           str(forbidden_found))

    raw_for_groups = _read_h2_shared(source_path)
    first = create_split(raw_for_groups, config).manifest
    second = create_split(raw_for_groups, config).manifest
    deterministic = first.equals(second) and first.equals(manifest)
    record("same_seed_determinism", deterministic,
           "Two regenerated assignments equal each other and the frozen manifest." if deterministic
           else "Regenerated assignments differ from each other or the frozen manifest.")

    # Independent pandas row hashes check the saved assignments.
    columns = [column for column in raw_for_groups if column != "row_id"]
    row_hashes = pd.util.hash_pandas_object(raw_for_groups[columns], index=False)
    groups = pd.DataFrame({"hash": row_hashes.to_numpy(), "split": manifest["split"].to_numpy()})
    crossings = int(groups.groupby("hash")["split"].nunique().gt(1).sum())
    record("duplicate_groups_stay_together", crossings == 0,
           f"groups_crossing_splits={crossings}")

    indexed = source.set_index("row_id")
    if not indexed.index.is_unique:
        raise ValueError("Source row_id values are not unique.")
    fit_indices: list[list[int]] = []
    original_fit_transform = ColumnTransformer.fit_transform

    def capture_fit(transformer, X, y=None, **kwargs):
        fit_indices.append(list(X.index))
        return original_fit_transform(transformer, X, y=y, **kwargs)

    outputs_finite = True
    output_details: list[str] = []
    unknown_ok = False
    for task, columns in selected.items():
        train = indexed.loc[ids["train"], columns]
        validation = indexed.loc[ids["validation"], columns]
        transformer = build_preprocessor(task, config)
        with patch.object(ColumnTransformer, "fit_transform", capture_fit):
            transformed_train = transformer.fit_transform(train)
        transformed_validation = transformer.transform(validation)
        finite = _finite(transformed_train) and _finite(transformed_validation)
        outputs_finite &= finite
        output_details.append(f"{task}: train={transformed_train.shape}, validation={transformed_validation.shape}, finite={finite}")
        if task == "classification":
            probe = validation.iloc[[0]].copy()
            probe["Meal"] = "__UNSEEN_MEAL__"
            encoder = transformer.named_transformers_["categorical"].named_steps["encode"]
            category_columns = transformer.transformers_[1][2]
            meal_categories = encoder.categories_[category_columns.index("Meal")]
            unknown_ok = ("__UNSEEN_MEAL__" not in meal_categories
                          and transformer.transform(probe).shape[1] == transformed_train.shape[1]
                          and _finite(transformer.transform(probe)))
    record("train_only_fitting", len(fit_indices) == 3
           and all(indices == ids["train"] for indices in fit_indices),
           f"fit_calls={len(fit_indices)}; each must contain exactly {len(ids['train'])} training row_ids")
    record("unknown_category_handling", unknown_ok,
           "An unseen Meal token transforms with the same output width and finite values.")
    record("zero_nan_after_transform", outputs_finite, "; ".join(output_details))

    not_applicable = _categorical_array(pd.DataFrame({"Agent": ["NULL"], "Company": [" NULL "]}))
    record("not_applicable_ids_preserved", bool((not_applicable == "NULL").all()),
           "Agent and Company NULL remain categorical not-applicable values.")
    report = {
        "status": "PASS" if all(item["status"] == "PASS" for item in results.values()) else "FAIL",
        "dataset_sha256": source.attrs["data_checksum_sha256"],
        "split_version": config["decisions"]["split_version"],
        "feature_policy_version": config["decisions"]["feature_policy_version"],
        "rows": {name: len(values) for name, values in ids.items()},
        "checks": results,
    }
    output_path = root / config["paths"]["data_outputs"] / "preprocessing_checks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_checks(load_config(PROJECT_ROOT / "config.yaml"), PROJECT_ROOT)
    for name, result in report["checks"].items():
        print(f"{result['status']} {name}: {result['detail']}")
    print(f"Overall: {report['status']}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
