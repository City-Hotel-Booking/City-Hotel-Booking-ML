"""Member 1: validate H2 and write a source-quality summary."""

from pathlib import Path


def run(config: dict, project_root: Path) -> None:
    """Write outputs/data/validation.json and outputs/data/schema.csv.

    Check expected rows and exact labels, types, missing values, padded strings,
    not-applicable categories, duplicates, dates, and unusual values. Report issues
    without silently removing observations or choosing model preprocessing.
    """
    raise NotImplementedError("The validation runner is not implemented yet (Member 1).")
