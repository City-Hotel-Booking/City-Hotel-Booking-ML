"""Member 1: shared H2 loader. Preserve source labels and source row identity."""

from pathlib import Path


def load_h2(path: Path):
    """Return a DataFrame; do not silently filter or impute rows."""
    raise NotImplementedError("Member 1 implements load_h2 after the data rules are agreed.")
