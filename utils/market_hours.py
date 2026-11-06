"""
utils/market_hours.py
---------------------
Is the market open? Asked of IBKR, not of a hardcoded calendar.

reqContractDetails returns liquidHours and tradingHours per contract, so the
broker's own schedule answers the question — including half days, holidays and
unscheduled closures, none of which a local weekday/holiday table knows about.

Format, as observed from IB Gateway:

    liquidHours  "20261105:0930-20261105:1600;20261107:CLOSED;..."
    timeZoneId   "US/Eastern"

TWO TRAPS, both load-bearing:

  1. THE TIMEZONE IS PER CONTRACT. SPY reports US/Eastern and SPX reports
     US/Central — SPX's "0830-1500" is the same session as SPY's "0930-1600".
     Parsing everything as Eastern would place SPX's close an hour early.
  2. liquidHours is NOT tradingHours. For SPY, liquid is 0930-1600 while
     trading is 0400-2000 (pre/post market). Greeks computed from a two-sided
     market want liquid hours; tradingHours would call 05:00 "open" when the
     book is nearly empty.

Needs the tzdata package on Windows, which zoneinfo does not bundle. It is
already installed here, and is declared in requirements.txt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketStatus:
    is_open: bool
    reason: str
    symbol: str = ""
    now_local: Optional[datetime] = None
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None

    def __bool__(self) -> bool:
        return self.is_open

    def __str__(self) -> str:
        head = "OPEN" if self.is_open else "CLOSED"
        bits = ["%s: %s (%s)" % (self.symbol or "market", head, self.reason)]
        if self.now_local is not None:
            bits.append("local now %s" % self.now_local.strftime("%Y-%m-%d %H:%M %Z"))
        if self.is_open and self.next_close is not None:
            bits.append("closes %s" % self.next_close.strftime("%H:%M %Z"))
        if not self.is_open and self.next_open is not None:
            bits.append("next open %s" % self.next_open.strftime("%Y-%m-%d %H:%M %Z"))
        return "  |  ".join(bits)


def parse_sessions(hours: str, tz_id: str) -> List[Tuple[datetime, datetime]]:
    """Parse an IB hours string into tz-aware (start, end) pairs.

    Skips CLOSED days and anything unparseable rather than raising: a schedule
    string is data from outside this program and its shape can change.
    """
    if not hours:
        return []
    try:
        tz = ZoneInfo(tz_id)
    except Exception as exc:                       # bad or missing tzdata
        logger.warning("market_hours: unusable timeZoneId %r (%s)", tz_id, exc)
        return []

    out: List[Tuple[datetime, datetime]] = []
    for chunk in hours.split(";"):
        chunk = chunk.strip()
        if not chunk or chunk.endswith(":CLOSED"):
            continue
        try:
            start_s, end_s = chunk.split("-", 1)
            start = datetime.strptime(start_s, "%Y%m%d:%H%M").replace(tzinfo=tz)
            if ":" in end_s and len(end_s) > 5:
                end = datetime.strptime(end_s, "%Y%m%d:%H%M").replace(tzinfo=tz)
            else:                                   # "1600" form, same day
                end = datetime.strptime(
                    start_s.split(":")[0] + ":" + end_s, "%Y%m%d:%H%M"
                ).replace(tzinfo=tz)
        except ValueError:
            logger.debug("market_hours: skipping unparseable session %r", chunk)
            continue
        if end > start:
            out.append((start, end))
    return sorted(out)


def status_from_hours(hours: str, tz_id: str, symbol: str = "",
                      now: Optional[datetime] = None,
                      label: str = "liquid hours") -> MarketStatus:
    """Open/closed from an already-fetched hours string. Pure, for testing.

    `label` only names the schedule in the reason text, so a tradingHours
    answer does not claim to be about liquid hours.
    """
    sessions = parse_sessions(hours, tz_id)
    if not sessions:
        return MarketStatus(False, "no parseable sessions in schedule", symbol)

    tz = sessions[0][0].tzinfo
    ref = (now or datetime.now(tz)).astimezone(tz)

    for start, end in sessions:
        if start <= ref < end:
            return MarketStatus(True, "inside " + label, symbol, ref,
                                next_close=end)

    upcoming = [s for s, _ in sessions if s > ref]
    nxt = upcoming[0] if upcoming else None
    if nxt is None:
        reason = "outside %s; schedule has no later session" % label
    else:
        reason = "outside %s" % label
    return MarketStatus(False, reason, symbol, ref, next_open=nxt)


def market_status(ib, contract, use_liquid_hours: bool = True,
                  now: Optional[datetime] = None) -> MarketStatus:
    """Ask IBKR whether this contract is tradeable right now.

    `use_liquid_hours=True` (the default) uses regular-session hours. Set False
    for tradingHours, which for US equities includes pre/post market.

    Returns a falsy MarketStatus when the question cannot be answered, so a
    caller that treats it as a gate fails closed rather than assuming open.
    """
    try:
        details = ib.reqContractDetails(contract)
    except Exception as exc:
        return MarketStatus(False, "reqContractDetails failed: %s" % exc,
                            getattr(contract, "symbol", ""))
    if not details:
        return MarketStatus(False, "no contract details returned",
                            getattr(contract, "symbol", ""))

    d = details[0]
    hours = d.liquidHours if use_liquid_hours else d.tradingHours
    if not hours:
        # Some contracts publish only one of the two.
        hours = d.tradingHours or d.liquidHours
    return status_from_hours(hours or "", d.timeZoneId or "",
                             d.contract.symbol, now=now,
                             label="liquid hours" if use_liquid_hours
                             else "trading hours")


def is_market_open(ib, contract, **kwargs) -> bool:
    """Bare boolean, for an `if`. Use market_status() when the reason matters."""
    return bool(market_status(ib, contract, **kwargs))
