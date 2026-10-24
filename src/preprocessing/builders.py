"""Member 2: reusable, unfitted task-specific sklearn preprocessing."""


def build_preprocessor(task: str, scale_numeric: bool = False):
    """Return an unfitted transformer using the approved feature policy.

    Handle IDs, not-applicable values, missing values, and unseen categories
    deliberately. Put learned imputation/encoding/scaling inside model pipelines;
    supervised owners fit these on training folds only.
    """
    raise NotImplementedError("Preprocessing rules are not implemented yet (Member 2).")
