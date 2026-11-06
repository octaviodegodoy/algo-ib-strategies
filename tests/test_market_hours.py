"""
tests/test_market_hours.py
--------------------------
Offline checks for utils/market_hours.py. No IB connection: the schedule
strings here are verbatim samples captured from IB Gateway, so the parser is
tested against the real format rather than an invented one.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from utils.market_hours import (
    MarketStatus, parse_sessions, status_from_hours,
)

# Captured from IB Gateway on 2026-11-05.
SPY_LIQUID = ("20261105:0930-20261105:1600;20261106:0930-20261106:1600;"
              "20261107:CLOSED;20261108:CLOSED;20261109:0930-20261109:1600")
SPY_TRADING = ("20261105:0400-20261105:2000;20261106:0400-20261106:2000;"
               "20261107:CLOSED;20261108:CLOSED;20261109:0400-20261109:2000")
# SPX reports CENTRAL time: 0830-1500 is the same session as SPY's 0930-1600.
SPX_LIQUID = ("20261105:0830-20261105:1500;20261106:0830-20261106:1500;"
              "20261107:CLOSED;20261108:CLOSED;20261109:0830-20261109:1500")

ET = ZoneInfo("US/Eastern")
CT = ZoneInfo("US/Central")


# ── parsing ──────────────────────────────────────────────────────────────────

def test_parses_sessions_and_skips_closed_days():
    s = parse_sessions(SPY_LIQUID, "US/Eastern")
    assert len(s) == 3                      # 3 open days, 2 CLOSED skipped
    assert s[0][0] == datetime(2026, 11, 5, 9, 30, tzinfo=ET)
    assert s[0][1] == datetime(2026, 11, 5, 16, 0, tzinfo=ET)


def test_empty_and_garbage_are_not_fatal():
    assert parse_sessions("", "US/Eastern") == []
    assert parse_sessions("nonsense", "US/Eastern") == []
    assert parse_sessions("20261105:CLOSED", "US/Eastern") == []
    # a bad timezone must not raise
    assert parse_sessions(SPY_LIQUID, "Not/AZone") == []


def test_short_end_form_is_accepted():
    """Some feeds give '20261105:0930-1600' without repeating the date."""
    s = parse_sessions("20261105:0930-1600", "US/Eastern")
    assert len(s) == 1
    assert s[0][1] == datetime(2026, 11, 5, 16, 0, tzinfo=ET)


# ── open / closed ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("when,expected", [
    (datetime(2026, 11, 5, 9, 29, tzinfo=ET), False),   # a minute early
    (datetime(2026, 11, 5, 9, 30, tzinfo=ET), True),    # the bell
    (datetime(2026, 11, 5, 13, 0, tzinfo=ET), True),
    (datetime(2026, 11, 5, 15, 59, tzinfo=ET), True),
    (datetime(2026, 11, 5, 16, 0, tzinfo=ET), False),   # close is exclusive
    (datetime(2026, 11, 7, 12, 0, tzinfo=ET), False),   # Saturday
])
def test_open_window_boundaries(when, expected):
    assert bool(status_from_hours(SPY_LIQUID, "US/Eastern", "SPY", now=when)) is expected


def test_per_contract_timezone_is_respected():
    """The trap: SPX's schedule is Central. 15:30 ET is 14:30 CT — open.

    Parsing SPX's '0830-1500' as Eastern would call this closed.
    """
    when = datetime(2026, 11, 5, 15, 30, tzinfo=ET)
    assert bool(status_from_hours(SPX_LIQUID, "US/Central", "SPX", now=when))
    # and both close at the same instant
    spy = status_from_hours(SPY_LIQUID, "US/Eastern", "SPY", now=when)
    spx = status_from_hours(SPX_LIQUID, "US/Central", "SPX", now=when)
    assert spy.next_close == spx.next_close


def test_liquid_and_trading_hours_differ():
    """06:00 ET is inside SPY's trading hours but outside liquid hours."""
    when = datetime(2026, 11, 5, 6, 0, tzinfo=ET)
    assert not status_from_hours(SPY_LIQUID, "US/Eastern", "SPY", now=when)
    assert status_from_hours(SPY_TRADING, "US/Eastern", "SPY", now=when)


def test_reason_names_the_schedule_used():
    when = datetime(2026, 11, 5, 6, 0, tzinfo=ET)
    st = status_from_hours(SPY_TRADING, "US/Eastern", "SPY", now=when,
                           label="trading hours")
    assert "trading hours" in st.reason and "liquid" not in st.reason


# ── next open / next close ───────────────────────────────────────────────────

def test_next_open_when_closed():
    st = status_from_hours(SPY_LIQUID, "US/Eastern", "SPY",
                           now=datetime(2026, 11, 5, 21, 30, tzinfo=ET))
    assert not st
    assert st.next_open == datetime(2026, 11, 6, 9, 30, tzinfo=ET)
    assert st.next_close is None


def test_next_close_when_open():
    st = status_from_hours(SPY_LIQUID, "US/Eastern", "SPY",
                           now=datetime(2026, 11, 5, 10, 0, tzinfo=ET))
    assert st
    assert st.next_close == datetime(2026, 11, 5, 16, 0, tzinfo=ET)
    assert st.next_open is None


def test_weekend_points_at_monday():
    st = status_from_hours(SPY_LIQUID, "US/Eastern", "SPY",
                           now=datetime(2026, 11, 7, 12, 0, tzinfo=ET))
    assert st.next_open == datetime(2026, 11, 9, 9, 30, tzinfo=ET)


def test_past_the_end_of_the_schedule():
    st = status_from_hours(SPY_LIQUID, "US/Eastern", "SPY",
                           now=datetime(2027, 1, 1, 12, 0, tzinfo=ET))
    assert not st
    assert st.next_open is None
    assert "no later session" in st.reason


# ── fails closed ─────────────────────────────────────────────────────────────

def test_unanswerable_is_closed_not_open():
    """A gate built on this must not treat 'I don't know' as tradeable."""
    assert not status_from_hours("", "US/Eastern", "SPY")
    assert not status_from_hours(SPY_LIQUID, "Not/AZone", "SPY")
    assert not MarketStatus(False, "reqContractDetails failed")


def test_truthiness_and_str():
    st = status_from_hours(SPY_LIQUID, "US/Eastern", "SPY",
                           now=datetime(2026, 11, 5, 10, 0, tzinfo=ET))
    assert bool(st) is True
    assert "OPEN" in str(st) and "SPY" in str(st)
    shut = status_from_hours(SPY_LIQUID, "US/Eastern", "SPY",
                             now=datetime(2026, 11, 5, 20, 0, tzinfo=ET))
    assert "CLOSED" in str(shut)
