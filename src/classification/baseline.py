"""Member 4: one untuned IsCanceled classification experiment."""

from pathlib import Path


def run(config: dict, project_root: Path) -> None:
    """Use the shared split and fit preprocessing on training only.

    Save validation accuracy, precision, recall, F1, confusion matrix, ROC-AUC
    and/or PR-AUC where meaningful, positive class, threshold, and training time
    under outputs/classification. Explain results in reports/classification_m1.md.
    Leave final test rows untouched.
    """
    raise NotImplementedError("The classification baseline is not implemented yet (Member 4).")
