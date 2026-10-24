"""Member 6 with Members 3 to 5: common experiment-result serialization."""

from pathlib import Path


def save_result(result: dict, path: Path) -> None:
    """Write the shared JSON record; implement before model handoffs.

    Fields: task, model, parameters, data_checksum, feature_policy_version,
    split_version (null for clustering), evaluation_subset, seed, metrics,
    train_seconds, command, figure_paths. Use null and a reason for an unavailable
    metric, not a fabricated zero. Add classifier threshold/positive class or
    clustering population/sampling where relevant. No test scores in M1.
    """
    raise NotImplementedError("The common result writer is not implemented yet (Member 6).")
