"""Source labels and meanings transcribed from the supplied course specification.

These are semantic roles, not an approved modeling feature policy. Agent and
Company are categorical IDs even when their non-missing tokens are digits.
"""

EXPECTED_ROWS = 79_330
SCHEMA = [
    ("IsCanceled", "binary target", "1 if canceled; 0 otherwise."),
    ("LeadTime", "integer", "Days between PMS booking entry and arrival."),
    ("ArrivalDateYear", "temporal integer", "Arrival year."),
    ("ArrivalDateMonth", "temporal category", "Arrival month name."),
    ("ArrivalDateWeekNumber", "temporal integer", "Arrival week number."),
    ("ArrivalDateDayOfMonth", "temporal integer", "Arrival day of month."),
    ("StaysInWeekendNights", "integer", "Number of weekend nights booked or stayed."),
    ("StaysInWeekNights", "integer", "Number of weekday nights booked or stayed."),
    ("Adults", "integer", "Number of adults."),
    ("Children", "integer", "Number of children."),
    ("Babies", "integer", "Number of babies."),
    ("Meal", "category", "Meal package."),
    ("Country", "category", "Country of origin code."),
    ("MarketSegment", "category", "Market segment designation."),
    ("DistributionChannel", "category", "Booking distribution channel."),
    ("IsRepeatedGuest", "binary", "1 for a repeated guest; 0 otherwise."),
    ("PreviousCancellations", "integer", "Previous bookings canceled by the customer."),
    ("PreviousBookingsNotCanceled", "integer", "Previous bookings not canceled by the customer."),
    ("ReservedRoomType", "category", "Room type originally reserved."),
    ("AssignedRoomType", "category", "Room type ultimately assigned."),
    ("BookingChanges", "integer", "Booking amendments before check-in or cancellation."),
    ("DepositType", "category", "Deposit arrangement."),
    ("Agent", "categorical ID", "Travel-agency identifier; may contain NULL."),
    ("Company", "categorical ID", "Company or entity identifier; may contain NULL."),
    ("DaysInWaitingList", "integer", "Days on the waiting list before confirmation."),
    ("CustomerType", "category", "Booking or customer type."),
    ("ADR", "numeric target", "Average Daily Rate."),
    ("RequiredCarParkingSpaces", "integer", "Number of requested car-parking spaces."),
    ("TotalOfSpecialRequests", "integer", "Number of special requests."),
    ("ReservationStatus", "outcome category", "Final reservation status."),
    ("ReservationStatusDate", "date", "Date the last reservation status was set."),
]
SOURCE_COLUMNS = [row[0] for row in SCHEMA]
NUMERIC_COLUMNS = [name for name, role, _ in SCHEMA if "integer" in role or "binary" in role or role == "numeric target"]
INTEGER_COLUMNS = [c for c in NUMERIC_COLUMNS if c != "ADR"]
CATEGORICAL_COLUMNS = [name for name, role, _ in SCHEMA if "category" in role or "categorical" in role]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
# Explicit, auditable token normalization; categories such as Undefined survive.
MISSING_TOKENS = frozenset({"", "NULL", "NA", "N/A", "NAN"})
NOT_APPLICABLE_COLUMNS = frozenset({"Agent", "Company"})
