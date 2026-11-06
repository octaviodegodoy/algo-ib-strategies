"""
analytics/local_greeks.py
-------------------------
Greeks computed here, from a quoted price, instead of taken from IBKR.

One entry point, greeks_from_price(), so the plumbing between a quote and
analytics/black76.py lives in exactly one place. It does the three conversions
that are easy to get wrong:

  1. tenor      calendar expiry string -> TRADING days (utils/tenor.py)
  2. discount   annual rate            -> discount factor on that tenor
  3. forward    spot                   -> F = spot / factor

Then it solves IV from the price and derives delta, gamma, vega and theta from
that IV. Everything downstream is self-consistent, which the broker's figures
are not: IBKR stamps each greek with the underlying level it happened to see,
and reports theta per calendar day.

A two-sided market is required. With no mid there is no IV, and with no IV
there are no greeks — pricing off a stale last trade would manufacture all five
out of nothing, so this returns None instead.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

from analytics import black76 as b76
from utils.tenor import trading_days_to_expiry

logger = logging.getLogger(__name__)

# Theta convention bridge. black76 returns per TRADING day; IBKR reports per
# CALENDAR day. 252/365 converts ours to theirs for comparison.
TRADING_TO_CALENDAR = b76.TRADING_DAYS_PER_YEAR / 365.0


@dataclass(frozen=True)
class LocalGreeks:
    """Greeks and the inputs they came from, so a result can be audited."""
    iv: float                      # decimal, e.g. 0.1324
    delta: float
    gamma: float
    vega: float                    # per +1 percentage point of IV
    theta: float                   # per TRADING day
    theta_calendar: float          # per calendar day, for comparing with IBKR
    forward: float
    factor: float                  # discount factor used
    trading_days: float            # fractional for a 0DTE expiry
    price_used: float


def greeks_from_price(
    spot: float,
    strike: float,
    right,
    expiration: str,
    price: float,
    rate: float,
    today: Optional[date] = None,
) -> Optional[LocalGreeks]:
    """All five greeks from an option's market price. None if not computable.

    Parameters
    ----------
    spot       : underlying price
    strike     : option strike
    right      : 'C'/'P', 'CALL'/'PUT', or black76.CALL / black76.PUT
    expiration : IB expiry string, YYYYMMDD
    price      : the option's market price — a two-sided mid, not a last trade
    rate       : annualised continuous risk-free rate, decimal (e.g. 0.045)
    today      : override for testing

    Returns None when the inputs cannot support a solve: no price, no tenor,
    a non-positive spot or strike, or an IV solver that will not converge.
    That is a real answer, not a failure to report — a strike with no market
    has no greeks.
    """
    if spot <= 0 or strike <= 0 or price <= 0:
        return None

    try:
        option_type = b76._resolve_right(right)
    except ValueError:
        logger.warning("greeks_from_price: unusable right %r", right)
        return None

    T = trading_days_to_expiry(expiration, today=today)
    if T <= 0:
        return None

    factor = b76.discount_factor_from_rate(rate, T)
    F = spot / factor

    iv = b76.implied_vol(F, strike, T, price, factor, option_type)
    if iv is None or math.isnan(iv) or iv <= 0:
        # brentq could not bracket a root; the bisection always returns a
        # number, so try it before giving up.
        iv = b76.fx_vol(F, strike, T, price, factor, option_type)
    if iv is None or math.isnan(iv) or iv <= 1e-6 or iv >= 5.0:
        logger.debug(
            "greeks_from_price: no usable IV for K=%s exp=%s price=%s", strike,
            expiration, price,
        )
        return None

    # Residual check, not just a ceiling check. fx_vol bisects [0, 5] and
    # returns its upper bound for a price no volatility can produce -- and it
    # converges to 5 - 1e-9, which slips under any `iv >= 5` test. Re-pricing
    # at the solved IV is the only way to know the solve meant anything.
    repriced = (b76.fx_call if option_type == b76.CALL else b76.fx_put)(
        F, strike, T, iv, factor)
    if abs(repriced - price) > max(1e-4, 0.005 * price):
        logger.debug(
            "greeks_from_price: IV %.4f does not reproduce price %.4f for K=%s "
            "exp=%s (repriced %.4f) -- price is outside the attainable range",
            iv, price, strike, expiration, repriced,
        )
        return None

    delta = b76.fx_delta(F, strike, T, iv, factor, option_type, spot)
    vega, theta = b76.vega_theta(F, strike, T, iv, factor, option_type)

    # black76.fx_gamma is NOT per-share gamma: its formula carries an extra
    # factor of F in the numerator, so it returns gamma * F. Measured on a
    # 738 SPY 0DTE ATM: fx_gamma 44.433 against a textbook 0.060, ratio 738.00
    # exactly. That is the source project's definition and black76 keeps it, as
    # a faithful port whose numbers must match.
    #
    # But OptionRow.gamma means per-share gamma everywhere in THIS project --
    # gex.py's own bs_gamma returns it, and IBKR's modelGreeks.gamma is it --
    # and gex.py then computes dollar_gamma = gamma * spot**2 * multiplier.
    # Passing gamma*F through that inflates every GEX figure by a factor of
    # spot, about 738x here, which is three orders of magnitude.
    gamma = b76.fx_gamma(F, strike, spot, T, iv, factor) / F

    return LocalGreeks(
        iv=float(iv),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        theta_calendar=float(theta) * TRADING_TO_CALENDAR,
        forward=float(F),
        factor=float(factor),
        trading_days=float(T),
        price_used=float(price),
    )


def mid_or_none(bid: float, ask: float) -> Optional[float]:
    """Two-sided mid, or None. The gate in front of every local greek.

    Deliberately refuses to fall back to `last`: a stale print on an unquoted
    strike yields an IV and a full set of greeks that look entirely plausible
    and mean nothing.
    """
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return None
