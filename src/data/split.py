"""Member 2: one reproducible split used by both supervised owners."""

from pathlib import Path


def create_split(data, config: dict):
    """Return row_id/subset assignments under the agreed policy.

    Preserve source row identities. The manifest contains row_id and subset
    (train, validation, test), not targets. Document seed, checksum, exclusions,
    duplicate/group handling, and any temporal cutoffs alongside the manifest.
    """
    raise NotImplementedError("The split policy is still open (Member 2).")


def run(config: dict, project_root: Path) -> None:
    """Write outputs/data/split_manifest.csv and outputs/data/split_metadata.json."""
    raise NotImplementedError("The split runner is not implemented yet (Member 2).")
