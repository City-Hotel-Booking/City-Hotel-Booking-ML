"""Member 1: reproducible exploratory analysis using the shared loader."""

from pathlib import Path


def run(config: dict, project_root: Path) -> None:
    """Write tables/plots to outputs/eda and findings to reports/eda_notes.md.

    Inspect targets, missingness, duplicates, distributions, outliers, and useful
    relationships. Do not fit preprocessing for a supervised model here.
    """
    raise NotImplementedError("The EDA runner is not implemented yet (Member 1).")
