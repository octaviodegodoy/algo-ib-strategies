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
from datetime import date, datetime, time, timedelta
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Same-day (0DTE) expiries get a FRACTION of a trading day, not 0. They are
# live and tradeable until the bell, with the largest gamma and theta on the
# board. Returning 0 made greeks_from_price reject them, which silently threw
# away 47% of a live SPY chain — half the contracts on a day with a 0DTE
# expiry. Caught only by running with the market open.
#
# Fraction is of the regular session, measured in US/Eastern. SPX closes 15:00
# Central, which is the same instant as 16:00 Eastern, so one clock covers both.
_MARKET_TZ = ZoneInfo("US/Eastern")
_SESSION_OPEN = time(9, 30)
_SESSION_CLOSE = time(16, 0)
SESSION_HOURS = 6.5

# Half days close at 13:00 and are not modelled: a 0DTE tenor on one of those
# will read long by up to three hours.

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


def session_fraction_remaining(now: Optional[datetime] = None) -> float:
    """How much of today's regular session is left, as a fraction of one day.

    1.0 before the open, 0.0 at or after the close, linear in between. Used for
    0DTE tenor, so an expiry today is a real fraction of a trading day rather
    than zero.
    """
    ref = (now or datetime.now(_MARKET_TZ)).astimezone(_MARKET_TZ)
    t = ref.time()
    if t >= _SESSION_CLOSE:
        return 0.0
    if t <= _SESSION_OPEN:
        return 1.0
    close = ref.replace(hour=_SESSION_CLOSE.hour, minute=_SESSION_CLOSE.minute,
                        second=0, microsecond=0)
    return max(0.0, (close - ref).total_seconds() / 3600.0 / SESSION_HOURS)


def trading_days_to_expiry(expiration: str, today: Optional[date] = None,
                           holidays: Optional[Iterable[date]] = None,
                           now: Optional[datetime] = None) -> float:
    """Trading days from now to an IB expiry string (YYYYMMDD), expiry included.

    Returns a FLOAT. An expiry later than today is a whole number of trading
    days. An expiry TODAY returns the fraction of the session still to run —
    0.4 at noon, 0.0 once the bell has gone — because a 0DTE option is live
    until the close and has the biggest greeks on the board. Returning 0 for it
    dropped half of a live chain.

    0.0 means genuinely no tenor left: expiry in the past, expiry today after
    the close, or expiry on a non-trading day. Callers must not hand 0.0 to
    black76, which divides by sqrt(T).

    `now` exists so the 0DTE branch can be tested deterministically.
    """
    exp = datetime.strptime(expiration, "%Y%m%d").date()
    ref_date = today or (now.astimezone(_MARKET_TZ).date() if now
                         else datetime.now(_MARKET_TZ).date())

    if exp == ref_date:
        if not is_trading_day(exp, holidays):
            return 0.0
        return session_fraction_remaining(now)
    return float(count_trading_days(ref_date, exp, holidays))
