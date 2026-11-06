"""
tests/test_local_greeks.py
--------------------------
Offline checks for utils/tenor.py and analytics/local_greeks.py — the path that
turns a quoted price into greeks without asking IBKR. No IB connection needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from analytics import black76 as b76
from analytics.local_greeks import (
    TRADING_TO_CALENDAR, greeks_from_price, mid_or_none,
)
from utils.tenor import (
    count_trading_days, is_trading_day, session_fraction_remaining,
    trading_days_to_expiry,
)

RATE = 0.045
SPOT = 7700.0


# ══ utils/tenor.py ═══════════════════════════════════════════════════════════

def test_weekends_are_not_trading_days():
    assert is_trading_day(date(2026, 11, 6))       # Friday
    assert not is_trading_day(date(2026, 11, 7))   # Saturday
    assert not is_trading_day(date(2026, 11, 8))   # Sunday
    assert is_trading_day(date(2026, 11, 9))       # Monday


def test_thanksgiving_is_closed():
    """2026-11-26 is Thanksgiving — a Thursday, and SPXW listed no expiry."""
    assert not is_trading_day(date(2026, 11, 26))
    assert is_trading_day(date(2026, 11, 25))
    assert is_trading_day(date(2026, 11, 27))


def test_expiry_day_is_included_start_day_is_not():
    """The convention, and the off-by-one that bit the source project."""
    # Wed 4 Nov -> Fri 6 Nov: Thu + Fri = 2
    assert count_trading_days(date(2026, 11, 4), date(2026, 11, 6)) == 2
    # one day forward is 1, not 0
    assert count_trading_days(date(2026, 11, 4), date(2026, 11, 5)) == 1
    # same day is 0
    assert count_trading_days(date(2026, 11, 4), date(2026, 11, 4)) == 0


def test_a_full_week_is_five_days():
    assert count_trading_days(date(2026, 11, 6), date(2026, 11, 13)) == 5


def test_thanksgiving_week_is_four_days():
    """Mon 23 -> Fri 27 Nov 2026 spans 5 weekdays but Thursday is closed."""
    assert count_trading_days(date(2026, 11, 20), date(2026, 11, 27)) == 4


def test_past_expiry_is_zero():
    assert trading_days_to_expiry("20261101", today=date(2026, 11, 4)) == 0


# ── 0DTE: a same-day expiry is a FRACTION of a trading day, never 0 ──────────
# This used to return 0, and greeks_from_price rejects T<=0, so on any day with
# a same-day expiry the whole front expiry vanished -- 47% of a live SPY chain
# (83 of 176 contracts). Only a run with the market open exposed it.

@pytest.mark.parametrize("hhmm,expected", [
    ((9, 0), 1.0),        # before the open: a full session ahead
    ((9, 30), 1.0),       # the bell
    ((12, 45), 0.5),      # half way
    ((16, 0), 0.0),       # the close
    ((20, 0), 0.0),       # after hours
])
def test_zero_dte_tracks_the_session(hhmm, expected):
    now = datetime(2026, 11, 6, hhmm[0], hhmm[1], tzinfo=ZoneInfo("US/Eastern"))
    t = trading_days_to_expiry("20261106", now=now)
    assert t == pytest.approx(expected, abs=1e-3)


def test_zero_dte_is_positive_during_the_session():
    """The property that actually matters: not 0, so the row is not dropped."""
    now = datetime(2026, 11, 6, 13, 52, tzinfo=ZoneInfo("US/Eastern"))
    t = trading_days_to_expiry("20261106", now=now)
    assert 0.0 < t < 1.0


def test_zero_dte_on_a_closed_day_is_zero():
    """A same-day 'expiry' on a Saturday has no session to run down."""
    now = datetime(2026, 11, 7, 12, 0, tzinfo=ZoneInfo("US/Eastern"))
    assert trading_days_to_expiry("20261107", now=now) == 0.0


def test_session_fraction_bounds():
    for h in range(0, 24):
        f = session_fraction_remaining(
            datetime(2026, 11, 6, h, 0, tzinfo=ZoneInfo("US/Eastern")))
        assert 0.0 <= f <= 1.0


def test_matches_the_observed_spx_tenor():
    """Pins against a figure measured live: 13 Nov 2026 was T=7 from 4 Nov."""
    assert trading_days_to_expiry("20261113", today=date(2026, 11, 4)) == 7


def test_trading_days_are_fewer_than_calendar_days():
    """The whole reason this module exists."""
    t = trading_days_to_expiry("20261204", today=date(2026, 11, 4))
    assert t == 21
    assert (date(2026, 12, 4) - date(2026, 11, 4)).days == 30


# ══ mid_or_none ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bid,ask,expected", [
    (10.0, 11.0, 10.5),
    (0.0, 11.0, None),
    (10.0, 0.0, None),
    (0.0, 0.0, None),
    (None, None, None),
    (-1.0, 11.0, None),
])
def test_mid_requires_a_two_sided_market(bid, ask, expected):
    assert mid_or_none(bid, ask) == expected


# ══ greeks_from_price ════════════════════════════════════════════════════════

def _price(strike, right, T=7, iv=0.13):
    """Build a self-consistent market price for a given IV."""
    factor = b76.discount_factor_from_rate(RATE, T)
    F = SPOT / factor
    fn = b76.fx_call if b76._resolve_right(right) == b76.CALL else b76.fx_put
    return fn(F, strike, T, iv, factor)


def test_recovers_the_iv_it_was_priced_with():
    for right in ("C", "P"):
        for strike in (7300.0, 7700.0, 8100.0):
            px = _price(strike, right)
            g = greeks_from_price(SPOT, strike, right, "20261113", px, RATE,
                                  today=date(2026, 11, 4))
            assert g is not None
            assert g.iv == pytest.approx(0.13, abs=1e-4)


def test_reports_the_tenor_and_forward_it_used():
    g = greeks_from_price(SPOT, 7700.0, "C", "20261113",
                          _price(7700.0, "C"), RATE, today=date(2026, 11, 4))
    assert g.trading_days == 7
    assert 0.0 < g.factor < 1.0
    assert g.forward > SPOT          # F = spot / factor, factor < 1


def test_greek_signs():
    for right, dsign in (("C", 1), ("P", -1)):
        g = greeks_from_price(SPOT, 7700.0, right, "20261113",
                              _price(7700.0, right), RATE,
                              today=date(2026, 11, 4))
        assert g.delta * dsign > 0
        assert g.gamma > 0
        assert g.vega > 0
        assert g.theta < 0
        assert g.theta_calendar < 0


def test_calendar_theta_is_smaller_in_magnitude_than_trading_theta():
    """252/365: the same decay spread over more days per year."""
    g = greeks_from_price(SPOT, 7700.0, "C", "20261113",
                          _price(7700.0, "C"), RATE, today=date(2026, 11, 4))
    assert abs(g.theta_calendar) < abs(g.theta)
    assert g.theta_calendar == pytest.approx(g.theta * TRADING_TO_CALENDAR)
    assert TRADING_TO_CALENDAR == pytest.approx(252 / 365)


def test_returns_none_rather_than_fabricating_greeks():
    px = _price(7700.0, "C")
    t = date(2026, 11, 4)
    assert greeks_from_price(SPOT, 7700.0, "C", "20261113", 0.0, RATE, today=t) is None
    assert greeks_from_price(0.0, 7700.0, "C", "20261113", px, RATE, today=t) is None
    assert greeks_from_price(SPOT, 0.0, "C", "20261113", px, RATE, today=t) is None
    # expiry already gone -> no tenor
    assert greeks_from_price(SPOT, 7700.0, "C", "20261101", px, RATE, today=t) is None
    # unusable right
    assert greeks_from_price(SPOT, 7700.0, "X", "20261113", px, RATE, today=t) is None


def test_absurd_price_yields_none_not_a_junk_iv():
    """A price above the forward cannot come from any sane vol."""
    g = greeks_from_price(SPOT, 7700.0, "C", "20261113", SPOT * 2, RATE,
                          today=date(2026, 11, 4))
    assert g is None


def test_accepts_both_right_conventions():
    px = _price(7700.0, "C")
    t = date(2026, 11, 4)
    a = greeks_from_price(SPOT, 7700.0, "C", "20261113", px, RATE, today=t)
    b = greeks_from_price(SPOT, 7700.0, b76.CALL, "20261113", px, RATE, today=t)
    assert a.delta == b.delta and a.iv == b.iv


def test_call_delta_exceeds_put_delta_magnitude_atm_ordering():
    """Sanity on the dispatch: a 7300 call is deeper ITM than a 7700 call."""
    t = date(2026, 11, 4)
    deep = greeks_from_price(SPOT, 7300.0, "C", "20261113",
                             _price(7300.0, "C"), RATE, today=t)
    atm = greeks_from_price(SPOT, 7700.0, "C", "20261113",
                            _price(7700.0, "C"), RATE, today=t)
    assert deep.delta > atm.delta
