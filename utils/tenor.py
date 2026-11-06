"""
utils/tenor.py
--------------
Trading days to expiry, for the Black-76 maths in analytics/black76.py.

That module's `T` is in TRADING days and it divides by 252 internally. Nothing
else in this project counted trading days — data/rates.py has days_to_expiry
(calendar) and time_to_expiry (calendar/365) — so feeding either of those to
black76 silently overstates the tenor by roughly 365/252 = 1.45x.

Counting convention: trading days in (today, expiry], i.e. today is excluded
and the expiry day is included. The option still has value through the expiry
session, so dropping it understates the tenor; the source project this maths
came from had exactly that off-by-one and it made every delta 34% steeper than
the broker's.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# NYSE full closures. Explicit rather than computed: the observance rules
# (Saturday -> preceding Friday, Sunday -> following Monday) and Good Friday's
# lunar date are not worth deriving, and a wrong holiday is worse than a
# missing year, which at least warns. EXTEND THIS as years are added.
US_MARKET_HOLIDAYS: tuple[str, ...] = (
    # 2026
    "2026-01-01",  # New Year's Day (Thu)
    "2026-01-19",  # Martin Luther King Jr. Day
    "2026-02-16",  # Washington's Birthday
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth (Fri)
    "2026-07-03",  # Independence Day observed (4th is a Saturday)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas (Fri)
    # 2027
    "2027-01-01",  # New Year's Day (Fri)
    "2027-01-18",  # Martin Luther King Jr. Day
    "2027-02-15",  # Washington's Birthday
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial Day
    "2027-06-18",  # Juneteenth observed (19th is a Saturday)
    "2027-07-05",  # Independence Day observed (4th is a Sunday)
    "2027-09-06",  # Labor Day
    "2027-11-25",  # Thanksgiving
    "2027-12-24",  # Christmas observed (25th is a Saturday)
)

_HOLIDAYS: frozenset[date] = frozenset(
    datetime.strptime(d, "%Y-%m-%d").date() for d in US_MARKET_HOLIDAYS
)
_COVERED_THROUGH: date = max(_HOLIDAYS)

# Half days (1pm close) are NOT listed: the market is open, so they count as
# trading days. They do carry less than a full session of decay, which this
# convention ignores.


def is_trading_day(d: date, holidays: Optional[Iterable[date]] = None) -> bool:
    """True when `d` is a weekday and not a listed full closure."""
    hol = _HOLIDAYS if holidays is None else frozenset(holidays)
    return d.weekday() < 5 and d not in hol


def count_trading_days(start: date, end: date,
                       holidays: Optional[Iterable[date]] = None) -> int:
    """Trading days in (start, end] — start excluded, end included.

    Returns 0 when end <= start.
    """
    if end <= start:
        return 0
    if end > _COVERED_THROUGH and holidays is None:
        logger.warning(
            "tenor: %s is past the last listed market holiday (%s); any closure "
            "after that is uncounted and the tenor will be slightly long. "
            "Extend US_MARKET_HOLIDAYS in utils/tenor.py.",
            end, _COVERED_THROUGH,
        )
    hol = _HOLIDAYS if holidays is None else frozenset(holidays)
    n = 0
    d = start
    while d < end:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            n += 1
    return n


def trading_days_to_expiry(expiration: str, today: Optional[date] = None,
                           holidays: Optional[Iterable[date]] = None) -> int:
    """Trading days from today to an IB expiry string (YYYYMMDD), inclusive.

    0 if the expiry is today or in the past — callers must treat 0 as "no
    tenor" rather than passing it to black76, which divides by sqrt(T).
    """
    exp = datetime.strptime(expiration, "%Y%m%d").date()
    return count_trading_days(today or date.today(), exp, holidays)
