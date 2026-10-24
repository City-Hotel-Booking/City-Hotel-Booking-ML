"""Member 2: agreed task inputs and deterministic feature engineering."""


def get_features(task: str) -> list[str]:
    """Return approved source inputs for regression, classification, or clustering.

    Match reasons in reports/leakage_audit_m1.md. Exclude task targets and future
    outcomes from the relevant inputs. Score improvement alone cannot justify a field.
    """
    raise NotImplementedError("Task feature lists have not been agreed yet (Member 2).")
