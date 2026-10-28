"""Provisional booking-time feature policy for the 31 H2 source columns.

Only columns marked "use" enter the initial feature lists. The team must
review every "needs-team-decision" row before changing this policy version.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NamedTuple


Task = Literal["regression", "classification", "clustering"]
Role = Literal["use", "exclude", "needs-team-decision"]
TASKS: tuple[Task, ...] = ("regression", "classification", "clustering")
POLICY_VERSION = "m1-draft-1"
PREDICTION_POINT = "When the booking first enters the hotel PMS (provisional)."

OFFICIAL_COLUMNS: tuple[str, ...] = (
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
)


class ColumnDecision(NamedTuple):
    role: Role
    reason: str
    recommendation: str = ""


def _same(role: Role, reason: str, recommendation: str = "") -> dict[Task, ColumnDecision]:
    """Apply one documented decision to all three tasks."""
    return {task: ColumnDecision(role, reason, recommendation) for task in TASKS}


COLUMN_POLICY: dict[str, dict[Task, ColumnDecision]] = {
    "IsCanceled": {
        "regression": ColumnDecision("exclude", "Cancellation happens after booking and could reveal future information."),
        "classification": ColumnDecision("exclude", "This is the classification target."),
        "clustering": ColumnDecision("exclude", "An outcome label cannot define unsupervised groups."),
    },
    "LeadTime": _same("use", "Lead time is known from booking entry and the planned arrival date."),
    "ArrivalDateYear": {
        "regression": ColumnDecision("use", "Planned arrival year is known at booking; evaluate time drift separately."),
        "classification": ColumnDecision("use", "Planned arrival year is known at booking; evaluate time drift separately."),
        "clustering": ColumnDecision(
            "needs-team-decision", "Year may create calendar cohorts instead of booking segments.",
            "Start without year; add it only for an explicit time-segmentation goal.",
        ),
    },
    "ArrivalDateMonth": _same("use", "Planned arrival month is known when the booking is entered."),
    "ArrivalDateWeekNumber": {
        "regression": ColumnDecision("use", "Planned arrival week is known at booking; it overlaps with month."),
        "classification": ColumnDecision("use", "Planned arrival week is known at booking; it overlaps with month."),
        "clustering": ColumnDecision(
            "needs-team-decision", "Week overlaps with month and may dominate seasonal clusters.",
            "Start with month only; compare representations using unsupervised criteria later.",
        ),
    },
    "ArrivalDateDayOfMonth": {
        "regression": ColumnDecision("use", "The planned arrival day is known at booking."),
        "classification": ColumnDecision("use", "The planned arrival day is known at booking."),
        "clustering": ColumnDecision(
            "needs-team-decision", "Day of month may add calendar noise to segments.",
            "Start without it unless the segment purpose needs day-level timing.",
        ),
    },
    "StaysInWeekendNights": _same(
        "needs-team-decision", "Stay length can change after entry; H2 has 331 zero-night records across both night fields.",
        "Use only if the team can justify the booking-entry value.",
    ),
    "StaysInWeekNights": _same(
        "needs-team-decision", "Stay length can change after entry; H2 has 331 zero-night records across both night fields.",
        "Use only if the team can justify the booking-entry value.",
    ),
    "Adults": _same("use", "Initial party size is normally entered with the booking; verify amendments."),
    "Children": _same("use", "Initial child count is normally entered; four H2 values are unknown, not proven zero."),
    "Babies": _same("use", "Initial baby count is normally entered with the booking; verify amendments."),
    "Meal": _same("use", "The selected meal package is normally part of the initial booking."),
    "Country": _same(
        "needs-team-decision", "Country may be completed at check-in; H2 has 24 unknown values and 166 known categories.",
        "Use only if a booking-time country value is available; keep unknown distinct if approved.",
    ),
    "MarketSegment": _same("use", "The booking's market channel is normally assigned at entry; verify timing."),
    "DistributionChannel": _same("use", "The booking distribution channel is normally known at entry."),
    "IsRepeatedGuest": _same("use", "Prior-guest status should be based only on history before this booking."),
    "PreviousCancellations": {
        "regression": ColumnDecision("use", "Only cancellations before this booking are eligible history."),
        "classification": ColumnDecision("use", "Only cancellations before this booking are eligible history."),
        "clustering": ColumnDecision(
            "needs-team-decision", "Past cancellation outcomes could make segments risk-led.",
            "Start without it unless the team explicitly wants history-based segments.",
        ),
    },
    "PreviousBookingsNotCanceled": {
        "regression": ColumnDecision("use", "Only completed prior bookings are eligible history."),
        "classification": ColumnDecision("use", "Only completed prior bookings are eligible history."),
        "clustering": ColumnDecision(
            "needs-team-decision", "Past booking outcomes may change the meaning of segments.",
            "Start without it; add only for an agreed loyalty/history goal.",
        ),
    },
    "ReservedRoomType": _same("use", "The room type requested at booking entry is available then."),
    "AssignedRoomType": _same("exclude", "The final assigned room can be set or changed after booking."),
    "BookingChanges": _same("exclude", "The count accumulates after booking entry."),
    "DepositType": _same(
        "needs-team-decision", "Deposit policy may be known at entry but can proxy later cancellation.",
        "Use only after confirming its booking-time meaning and documenting the proxy risk.",
    ),
    "Agent": _same(
        "needs-team-decision", "Agent ID may be known at entry; NULL means no agency and IDs are numerous.",
        "Confirm booking-time availability; preserve NULL as not applicable and use bounded categorical encoding if approved.",
    ),
    "Company": _same(
        "needs-team-decision", "Company ID may be known at entry; NULL means not applicable and IDs are numerous.",
        "Confirm booking-time availability; preserve NULL as not applicable and use bounded categorical encoding if approved.",
    ),
    "DaysInWaitingList": _same("exclude", "Waiting duration is accumulated after the booking enters the PMS."),
    "CustomerType": _same("use", "Booking customer type is normally assigned at entry; verify timing."),
    "ADR": {
        "regression": ColumnDecision("exclude", "This is the regression target."),
        "classification": ColumnDecision(
            "needs-team-decision", "Recorded ADR may differ from the initial quote; H2 has 1,208 zero-ADR rows.",
            "Exclude until a booking-time quote with the same meaning is confirmed.",
        ),
        "clustering": ColumnDecision(
            "needs-team-decision", "Recorded ADR may be a later price and would define price-led segments.",
            "Start without ADR; add only for an agreed booking-price segmentation goal.",
        ),
    },
    "RequiredCarParkingSpaces": _same(
        "needs-team-decision", "Requested parking spaces may be added or changed after booking.",
        "Use only if the source value is the initial booking request.",
    ),
    "TotalOfSpecialRequests": _same(
        "needs-team-decision", "Special requests can accumulate after booking entry.",
        "Use only if the count is frozen at booking entry.",
    ),
    "ReservationStatus": _same("exclude", "Final reservation status is recorded after booking and reveals outcomes."),
    "ReservationStatusDate": _same("exclude", "The date of final status is future information at booking entry."),
}


def get_policy(task: str) -> dict[str, ColumnDecision]:
    """Return all 31 decisions for one task in source-column order."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return {column: COLUMN_POLICY[column][task] for column in OFFICIAL_COLUMNS}


def get_features(task: str, config: Mapping[str, object] | None = None) -> list[str]:
    """Return provisional inputs marked "use" for a task.

    Disputed columns stay out until the team approves a new policy version.
    The optional config checks that callers expect this version.
    """
    policy = get_policy(task)
    if config is not None:
        decisions = config.get("decisions")
        if isinstance(decisions, Mapping):
            version = decisions.get("feature_policy_version")
            if version not in (None, POLICY_VERSION):
                raise ValueError(f"Feature policy version mismatch: {version!r}")
    return [column for column, decision in policy.items() if decision.role == "use"]


def get_target(task: str) -> str | None:
    """Return the supervised target, or None for unsupervised clustering."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return {"regression": "ADR", "classification": "IsCanceled", "clustering": None}[task]


if set(COLUMN_POLICY) != set(OFFICIAL_COLUMNS) or any(
    set(COLUMN_POLICY[column]) != set(TASKS) for column in OFFICIAL_COLUMNS
):
    raise RuntimeError("The feature policy must cover all 31 columns and tasks.")
