"""
analytics/black76.py
--------------------
Forward-price (Black-76) option maths, ported from OptionsTradingIBKR's
functions/black_scholes.py and the two greek helpers in its strategies.py.

Ported as module-level functions rather than a class: the original was a
BlackScholesCalculator whose __init__ built a rate curve, and this project keeps
rate lookups in data/rates.py. Everything here is pure maths — no broker, no
network, no state.

===============================================================================
THREE CONVENTIONS DIFFER FROM analytics/gex.py. READ THIS BEFORE CALLING.
===============================================================================

                      this module (Black-76)        analytics/gex.py
  tenor               T in TRADING DAYS, /252       T in YEARS
  option type         CALL=0 / PUT=1 (ints)         right='C' / 'P' (str)
  the `r` argument    a DISCOUNT FACTOR (~0.998)    an annual RATE (~0.05)

Mixing these silently produces plausible-looking wrong numbers, so:

  * `T` is trading days here. 7 trading days, not 0.0278 years, not 9 calendar
    days. Use years_to_trading_days() / trading_days_to_years() to convert.
    This is deliberate and load-bearing: it is why the source project's deltas
    and thetas differ from IBKR's, which uses calendar time. Keeping the
    convention keeps the numbers comparable with that project.
  * `r` is a discount factor, exp(-rate*T), despite the name. It is kept as `r`
    so call sites copied from the source project keep working. Convert with
    discount_factor_from_rate() / rate_from_discount_factor().
  * the option type accepts CALL/PUT (0/1), 'C'/'P', or 'CALL'/'PUT' in any
    case, and raises ValueError on anything else. Accepting both styles is for
    porting; anything unrecognised fails loudly rather than defaulting to a
    call.

Function names match the source so ported call sites need no edits. `vega_theta`
and `price_black76` were `_bsm_vega_theta` and `_bsm_price_black76` there.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import brentq, newton
from scipy.stats import norm

__all__ = [
    "CALL", "PUT",
    "TRADING_DAYS_PER_YEAR",
    "years_to_trading_days", "trading_days_to_years",
    "discount_factor_from_rate", "rate_from_discount_factor",
    "d_1", "d_2",
    "black_scholes_call", "black_scholes_put",
    "fx_call", "fx_put",
    "fx_delta_call", "fx_delta_put", "fx_delta",
    "fx_call_vol", "fx_put_vol", "fx_vol",
    "implied_vol", "implied_volatility",
    "fx_gamma", "vega_theta", "price_black76",
]

CALL = 0
PUT = 1

TRADING_DAYS_PER_YEAR = 252


# ── unit conversion, so a mismatch is a deliberate act ───────────────────────

def years_to_trading_days(t_years: float) -> float:
    """Years -> trading days, the unit this module's `T` expects."""
    return t_years * TRADING_DAYS_PER_YEAR


def trading_days_to_years(T: float) -> float:
    """Trading days -> years, the unit analytics/gex.py expects."""
    return T / TRADING_DAYS_PER_YEAR


def discount_factor_from_rate(rate: float, T: float) -> float:
    """Annual continuous `rate` over `T` trading days -> discount factor."""
    return math.exp(-rate * trading_days_to_years(T))


def rate_from_discount_factor(factor: float, T: float) -> float:
    """Discount factor over `T` trading days -> annual continuous rate."""
    t = trading_days_to_years(T)
    if t <= 0:
        return 0.0
    return -math.log(max(factor, 1e-10)) / t


def _resolve_right(option_type) -> int:
    """Normalise an option type to CALL or PUT, or raise.

    Deliberately strict. A silent default would turn a typo into a put priced
    as a call, which is the kind of error that survives a code review.
    """
    if option_type in (CALL, PUT):
        return int(option_type)
    if isinstance(option_type, str):
        s = option_type.strip().upper()
        if s in ("C", "CALL"):
            return CALL
        if s in ("P", "PUT"):
            return PUT
    raise ValueError(
        "unrecognised option type {!r}; expected CALL/PUT (0/1), 'C'/'P' or "
        "'CALL'/'PUT'".format(option_type)
    )


# ── d1 / d2 ──────────────────────────────────────────────────────────────────

def d_1(F: float, K: float, T: float, sigma: float) -> float:
    """Black-76 d1. `T` in trading days."""
    t = T / TRADING_DAYS_PER_YEAR
    return (math.log(F / K) + sigma ** 2 * t / 2) / (sigma * t ** 0.5)


def d_2(F: float, K: float, T: float, sigma: float) -> float:
    """Black-76 d2. `T` in trading days."""
    t = T / TRADING_DAYS_PER_YEAR
    return d_1(F, K, T, sigma) - sigma * t ** 0.5


# ── prices ───────────────────────────────────────────────────────────────────
# Two pairs exist in the source and both are kept, because call sites use both:
# black_scholes_call/put guard on T<=0 and sigma<=0 and return 0.0, while
# fx_call/fx_put do not guard and instead floor a negative result at 0.0.

def black_scholes_call(F: float, K: float, T: float, sigma: float, r: float) -> float:
    """European call on the forward. `r` is a discount factor."""
    if T <= 0 or sigma <= 0:
        return 0.0
    t = T / TRADING_DAYS_PER_YEAR
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return max(r * (F * norm.cdf(d1) - K * norm.cdf(d2)), 0.0)


def black_scholes_put(F: float, K: float, T: float, sigma: float, r: float) -> float:
    """European put on the forward. `r` is a discount factor."""
    if T <= 0 or sigma <= 0:
        return 0.0
    t = T / TRADING_DAYS_PER_YEAR
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return max(r * (K * norm.cdf(-d2) - F * norm.cdf(-d1)), 0.0)


def fx_call(F: float, K: float, T: float, sigma: float, r: float) -> float:
    """Call price via d_1/d_2. Unguarded on T/sigma; floors at 0.0."""
    fx = (F * norm.cdf(d_1(F, K, T, sigma)) - K * norm.cdf(d_2(F, K, T, sigma))) * r
    return 0.0 if fx < 0 else fx


def fx_put(F: float, K: float, T: float, sigma: float, r: float) -> float:
    """Put price via d_1/d_2. Unguarded on T/sigma; floors at 0.0."""
    fx = (K * norm.cdf(-d_2(F, K, T, sigma)) - F * norm.cdf(-d_1(F, K, T, sigma))) * r
    return 0.0 if fx < 0 else fx


def price_black76(S_new: float, K: float, T_days: float, iv_pct: float,
                  factor: float, option_type) -> float:
    """Price at an arbitrary SPOT, for payoff curves. `iv_pct` is percent.

    Was _bsm_price_black76. Takes spot and derives F = S_new / factor, unlike
    the fx_* functions which take the forward directly.
    """
    sigma = iv_pct / 100.0
    if sigma <= 0 or S_new <= 0 or K <= 0 or factor <= 0:
        return 0.0
    right = _resolve_right(option_type)
    if T_days <= 0:
        return max(S_new - K, 0.0) if right == CALL else max(K - S_new, 0.0)
    t = T_days / TRADING_DAYS_PER_YEAR
    F = S_new / max(factor, 1e-12)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if right == CALL:
        return factor * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


# ── delta ────────────────────────────────────────────────────────────────────
# Note F/(spot*r): this converts the forward delta to a spot delta. With
# F = spot/r the factor is 1/r**2, not 1 -- that is the source's behaviour and
# it is preserved here rather than "corrected", so results stay comparable.
# Passing spot=0 falls back to the plain forward delta N(d1).

def fx_delta_call(F: float, K: float, T: float, sigma: float, r: float,
                  spot: float) -> float:
    x = d_1(F, K, T, sigma)
    if spot == 0:
        return norm.cdf(x)
    return F / (spot * r) * norm.cdf(x)


def fx_delta_put(F: float, K: float, T: float, sigma: float, r: float,
                 spot: float) -> float:
    x = d_1(F, K, T, sigma)
    if spot == 0:
        return norm.cdf(x) - 1
    return F / (spot * r) * (norm.cdf(x) - 1)


def fx_delta(F: float, K: float, T: float, sigma: float, r: float,
             ID, spot: float) -> float:
    """Delta for either right. `ID` is the option type (see _resolve_right)."""
    if _resolve_right(ID) == CALL:
        return fx_delta_call(F, K, T, sigma, r, spot)
    return fx_delta_put(F, K, T, sigma, r, spot)


# ── gamma ────────────────────────────────────────────────────────────────────

def fx_gamma(F: float, K: float, S: float, T: float, sigma: float,
             r: float) -> float:
    """Gamma. NOTE the argument order: F, K, S, T, sigma, r -- spot is third,
    matching the source. Easy to transpose with fx_delta's (F, K, T, ...).
    """
    x = d_1(F, K, T, sigma)
    a = (1 / (2 * math.pi)) ** 0.5 * np.exp(-x ** 2 / 2)
    return a * r * F / (S * sigma * (T / TRADING_DAYS_PER_YEAR) ** 0.5)


# ── vega and theta ───────────────────────────────────────────────────────────

def vega_theta(F: float, K: float, T_days: float, sigma: float, factor: float,
               option_type) -> Tuple[float, float]:
    """(vega per +1 percentage point of IV, theta per TRADING day) for a long.

    Was _bsm_vega_theta. Theta is per trading day; IBKR reports per calendar
    day, so scale by 252/365 before comparing with a broker figure.
    """
    if T_days <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return 0.0, 0.0
    right = _resolve_right(option_type)
    t = max(T_days / TRADING_DAYS_PER_YEAR, 1e-8)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    S = F * factor

    vega = S * norm.pdf(d1) * np.sqrt(t) * 0.01

    r_eff = -np.log(max(factor, 1e-10)) / t
    gamma_part = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(t))
    if right == CALL:
        r_part = -r_eff * K * factor * norm.cdf(d2)
    else:
        r_part = r_eff * K * factor * norm.cdf(-d2)
    daily_theta = (gamma_part + r_part) / TRADING_DAYS_PER_YEAR

    return vega, daily_theta


# ── implied volatility ───────────────────────────────────────────────────────
# Three solvers, all kept because they differ in failure mode:
#   fx_vol              bisection on [0, 5], always returns a number
#   implied_vol         brentq, returns np.nan on failure
#   implied_volatility  newton or brentq, returns None on failure

def fx_call_vol(F: float, K: float, T: float, price: float, r: float) -> float:
    """Call IV by bisection on [0, 5]. Returns a midpoint even if price is
    outside the attainable range -- check the result if that matters."""
    high, low = 5.0, 0.0
    while (high - low) > 0.00000001:
        mid = (high + low) / 2
        if fx_call(F, K, T, mid, r) > price:
            high = mid
        else:
            low = mid
    return (high + low) / 2


def fx_put_vol(F: float, K: float, T: float, price: float, r: float) -> float:
    """Put IV by bisection on [0, 5]. Same caveat as fx_call_vol."""
    high, low = 5.0, 0.0
    while (high - low) > 0.00000001:
        mid = (high + low) / 2
        if fx_put(F, K, T, mid, r) > price:
            high = mid
        else:
            low = mid
    return (high + low) / 2


def fx_vol(F: float, K: float, T: float, price: float, r: float, ID) -> float:
    """Bisection IV for either right."""
    if _resolve_right(ID) == CALL:
        return fx_call_vol(F, K, T, price, r)
    return fx_put_vol(F, K, T, price, r)


def implied_vol(F: float, K: float, T: float, price: float, r: float,
                option_type) -> float:
    """IV by Brent's method. Returns np.nan if it cannot bracket a root.

    T is trading days, r a discount factor, price the market price.
    """
    if price <= 0 or T <= 0:
        return np.nan
    right = _resolve_right(option_type)

    def objective(sigma):
        if sigma <= 0:
            return 1e10
        if right == CALL:
            return black_scholes_call(F, K, T, sigma, r) - price
        return black_scholes_put(F, K, T, sigma, r) - price

    try:
        return brentq(objective, 0.001, 5.0, maxiter=100)
    except (ValueError, RuntimeError):
        try:
            return brentq(objective, 0.0001, 10.0, maxiter=200)
        except (ValueError, RuntimeError):
            return np.nan


def implied_volatility(F: float, K: float, T: float, market_price: float,
                       r: float, option_type=CALL, method: str = "brentq",
                       initial_guess: float = 0.3, max_iter: int = 100,
                       tolerance: float = 1e-6) -> Optional[float]:
    """IV by Newton-Raphson or Brent. Returns None on failure.

    `method` is 'newton' or 'brentq'.
    """
    if market_price <= 0 or T <= 0:
        return None
    right = _resolve_right(option_type)

    def objective(sigma):
        if sigma <= 0:
            return 1e10
        if right == CALL:
            return black_scholes_call(F, K, T, sigma, r) - market_price
        return black_scholes_put(F, K, T, sigma, r) - market_price

    try:
        if method == "newton":
            iv = newton(objective, initial_guess, maxiter=max_iter, tol=tolerance)
        elif method == "brentq":
            iv = brentq(objective, 0.001, 5.0, maxiter=max_iter, xtol=tolerance)
        else:
            raise ValueError("method must be 'newton' or 'brentq'")
        if iv < 0 or iv > 5:
            return None
        return iv
    except (RuntimeError, ValueError):
        try:
            iv = brentq(objective, 0.0001, 10.0, maxiter=200)
            if 0 < iv <= 10:
                return iv
        except (RuntimeError, ValueError):
            pass
        return None
