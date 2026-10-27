"""Feature-policy checks that protect the first booking-time experiments."""

from __future__ import annotations

import unittest

from src.features.policy import (
    COLUMN_POLICY, OFFICIAL_COLUMNS, POLICY_VERSION, TASKS, get_features, get_policy,
)


class PolicyTests(unittest.TestCase):
    def test_all_source_columns_have_three_explained_decisions(self) -> None:
        self.assertEqual(len(OFFICIAL_COLUMNS), 31)
        self.assertEqual(set(COLUMN_POLICY), set(OFFICIAL_COLUMNS))
        for task in TASKS:
            decisions = get_policy(task)
            self.assertEqual(len(decisions), 31)
            self.assertTrue(all(decision.reason for decision in decisions.values()))
            self.assertTrue(all(
                decision.recommendation
                for decision in decisions.values()
                if decision.role == "needs-team-decision"
            ))

    def test_active_features_exclude_targets_and_late_outcomes(self) -> None:
        late = {
            "IsCanceled", "AssignedRoomType", "BookingChanges",
            "DaysInWaitingList", "ReservationStatus", "ReservationStatusDate",
        }
        for task in TASKS:
            active = set(get_features(task))
            self.assertFalse(active & late)
        self.assertNotIn("ADR", get_features("regression"))
        self.assertNotIn("ADR", get_features("classification"))

    def test_version_mismatch_and_unknown_task_fail(self) -> None:
        self.assertTrue(get_features("regression", {
            "decisions": {"feature_policy_version": POLICY_VERSION},
        }))
        with self.assertRaises(ValueError):
            get_features("regression", {
                "decisions": {"feature_policy_version": "wrong"},
            })
        with self.assertRaises(ValueError):
            get_policy("forecasting")


if __name__ == "__main__":
    unittest.main()
