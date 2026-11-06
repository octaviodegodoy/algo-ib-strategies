"""
tests/test_black76.py
---------------------
Offline checks for analytics/black76.py. No IB connection needed.

The port was verified numerically against OptionsTradingIBKR's
BlackScholesCalculator at the time it was written — 0.0 difference on all 21
functions across 180+ cases. That comparison cannot be a committed test, since
it needs the other project on disk. What is pinned here instead are the
properties that would catch a later edit breaking the maths, plus the three
convention traps the module exists to keep straight.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import numpy as np
import pytest

from analytics import black76 as b76
from analytics.black76 import CALL, PUT

# F, K, T(trading days), sigma, discount factor
F, K, T, SIGMA, R = 7700.0, 7700.0, 7.0, 0.13, 0.9988
SPOT = F * R

GRID = [
    (f, f * m, t, s)
    for f in (7700.0, 500.0, 100.0)
    for m in (0.90, 0.98, 1.00, 1.02, 1.10)
    for t in (1.0, 7.0, 21.0, 63.0)
    for s in (0.08, 0.13, 0.45)
]


# ── option type resolution ───────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (CALL, CALL), (PUT, PUT), (0, CALL), (1, PUT),
    ("C", CALL), ("P", PUT), ("c", CALL), ("p", PUT),
    ("CALL", CALL), ("put", PUT), (" Call ", CALL),
])
def test_resolve_right_accepts_both_conventions(value, expected):
    assert b76._resolve_right(value) == expected


@pytest.mark.parametrize("bad", ["X", "", "calls", None, 2, -1, 0.5, "CP"])
def test_resolve_right_raises_rather_than_defaulting(bad):
    """A silent default would price a put as a call. It must raise."""
    with pytest.raises(ValueError):
        b76._resolve_right(bad)


# ── unit conversion ──────────────────────────────────────────────────────────

def test_trading_day_round_trip():
    assert b76.years_to_trading_days(b76.trading_days_to_years(21.0)) == pytest.approx(21.0)
    assert b76.trading_days_to_years(252.0) == pytest.approx(1.0)


def test_discount_factor_round_trip():
    f = b76.discount_factor_from_rate(0.05, 63.0)
    assert 0.0 < f < 1.0
    assert b76.rate_from_discount_factor(f, 63.0) == pytest.approx(0.05)


def test_tenor_is_trading_days_not_years():
    """Guards the convention: T=252 must mean one year, not 252 years."""
    one_year = b76.black_scholes_call(100.0, 100.0, 252.0, 0.20, 1.0)
    # ATM call at 20% vol over 1y is ~7.97 undiscounted; 252y would be ~100.
    assert 7.0 < one_year < 9.0


# ── prices ───────────────────────────────────────────────────────────────────

def test_put_call_parity():
    """C - P == (F - K) * r for every point on the grid."""
    for f, k, t, s in GRID:
        c = b76.black_scholes_call(f, k, t, s, R)
        p = b76.black_scholes_put(f, k, t, s, R)
        if c > 1e-6 and p > 1e-6:          # parity is meaningless once floored
            assert c - p == pytest.approx((f - k) * R, abs=1e-6)


def test_fx_price_pair_matches_black_scholes_pair():
    """fx_call/fx_put and black_scholes_call/put must agree where both are
    defined; they differ only in guard behaviour."""
    for f, k, t, s in GRID:
        assert b76.fx_call(f, k, t, s, R) == pytest.approx(
            b76.black_scholes_call(f, k, t, s, R), abs=1e-9)
        assert b76.fx_put(f, k, t, s, R) == pytest.approx(
            b76.black_scholes_put(f, k, t, s, R), abs=1e-9)


def test_guarded_variants_return_zero():
    assert b76.black_scholes_call(F, K, 0.0, SIGMA, R) == 0.0
    assert b76.black_scholes_call(F, K, T, 0.0, R) == 0.0
    assert b76.black_scholes_put(F, K, -1.0, SIGMA, R) == 0.0


def test_price_monotonic_in_vol():
    prev = -1.0
    for s in (0.05, 0.10, 0.20, 0.40, 0.80):
        px = b76.black_scholes_call(F, K, T, s, R)
        assert px > prev
        prev = px


def test_price_black76_intrinsic_at_expiry():
    assert b76.price_black76(110.0, 100.0, 0.0, 20.0, 1.0, CALL) == pytest.approx(10.0)
    assert b76.price_black76(90.0, 100.0, 0.0, 20.0, 1.0, PUT) == pytest.approx(10.0)
    assert b76.price_black76(90.0, 100.0, 0.0, 20.0, 1.0, CALL) == 0.0


# ── delta ────────────────────────────────────────────────────────────────────

def test_delta_signs_and_dispatch():
    for f, k, t, s in GRID:
        spot = f * R
        dc = b76.fx_delta_call(f, k, t, s, R, spot)
        dp = b76.fx_delta_put(f, k, t, s, R, spot)
        # dp is <= 0, not < 0: for a deep-OTM put norm.cdf(d1) reaches exactly
        # 1.0 in float64, so (cdf - 1) is exactly 0.0.
        assert dc > 0 and dp <= 0
        assert b76.fx_delta(f, k, t, s, R, CALL, spot) == dc
        assert b76.fx_delta(f, k, t, s, R, PUT, spot) == dp
        assert b76.fx_delta(f, k, t, s, R, "C", spot) == dc
        assert b76.fx_delta(f, k, t, s, R, "P", spot) == dp


def test_delta_spot_zero_falls_back_to_forward_delta():
    from scipy.stats import norm
    x = b76.d_1(F, K, T, SIGMA)
    assert b76.fx_delta_call(F, K, T, SIGMA, R, 0) == pytest.approx(norm.cdf(x))
    assert b76.fx_delta_put(F, K, T, SIGMA, R, 0) == pytest.approx(norm.cdf(x) - 1)


def test_call_delta_rises_with_spot():
    prev = -1.0
    for k in (8400.0, 8000.0, 7700.0, 7400.0, 7000.0):   # falling strike
        d = b76.fx_delta_call(F, k, T, SIGMA, R, SPOT)
        assert d > prev
        prev = d


# ── gamma, vega, theta ───────────────────────────────────────────────────────

def test_gamma_positive_and_peaks_near_the_money():
    g_atm = b76.fx_gamma(F, F, SPOT, T, SIGMA, R)
    g_otm = b76.fx_gamma(F, F * 1.10, SPOT, T, SIGMA, R)
    assert g_atm > 0 and g_otm > 0
    assert g_atm > g_otm


def test_vega_positive_theta_negative_for_a_long():
    for right in (CALL, PUT):
        vega, theta = b76.vega_theta(F, K, T, SIGMA, R, right)
        assert vega > 0, "a long option gains from higher IV"
        assert theta < 0, "a long option loses to the passage of time"


def test_vega_identical_for_call_and_put():
    vc, _ = b76.vega_theta(F, K, T, SIGMA, R, CALL)
    vp, _ = b76.vega_theta(F, K, T, SIGMA, R, PUT)
    assert vc == pytest.approx(vp)


def test_vega_theta_guards():
    assert b76.vega_theta(F, K, 0.0, SIGMA, R, CALL) == (0.0, 0.0)
    assert b76.vega_theta(F, K, T, 0.0, R, CALL) == (0.0, 0.0)


def test_vega_is_per_one_percentage_point():
    """vega must be the price change for +1pp of IV, not +100pp."""
    vega, _ = b76.vega_theta(F, K, T, SIGMA, R, CALL)
    bumped = b76.black_scholes_call(F, K, T, SIGMA + 0.01, R)
    base = b76.black_scholes_call(F, K, T, SIGMA, R)
    assert vega == pytest.approx(bumped - base, rel=0.02)


def test_theta_is_per_trading_day_not_calendar():
    """Pins the convention that makes this module's theta differ from IBKR's.

    One trading day of decay must match the actual one-day price drop, so the
    divisor is 252. A calendar-day theta would be ~1.448x smaller.
    """
    _, theta = b76.vega_theta(F, K, 21.0, SIGMA, R, CALL)
    drop = (b76.black_scholes_call(F, K, 20.0, SIGMA, R)
            - b76.black_scholes_call(F, K, 21.0, SIGMA, R))
    # `drop` is already signed: C(20) - C(21) is negative, as is theta. Compare
    # them directly -- negating one was a sign error that made this fail.
    # rel=0.10 because this compares an analytic derivative at T=21 with a
    # one-day forward difference over 21->20; ~7% apart is expected. A
    # calendar-day convention would be 1.448x out, far outside this band.
    assert drop < 0 and theta < 0
    assert theta == pytest.approx(drop, rel=0.10)


# ── implied vol solvers ──────────────────────────────────────────────────────

def test_all_three_solvers_recover_the_input_vol():
    for f, k, t, s in GRID:
        for right, pricer in ((CALL, b76.fx_call), (PUT, b76.fx_put)):
            px = pricer(f, k, t, s, R)
            vega, _ = b76.vega_theta(f, k, t, s, R, right)
            # Vega IS the sensitivity the solvers invert. Where it vanishes the
            # root is not identifiable at any tolerance, so skip rather than
            # pretend. An absolute price floor does not work here: 0.01 is
            # meaningful on a 100 underlying and noise on a 7700 one.
            if px < 0.01 or vega < 1e-3:
                continue
            assert b76.fx_vol(f, k, t, px, R, right) == pytest.approx(s, abs=1e-4)
            assert b76.implied_vol(f, k, t, px, R, right) == pytest.approx(s, abs=1e-4)
            iv = b76.implied_volatility(f, k, t, px, R, right)
            assert iv is not None and iv == pytest.approx(s, abs=1e-4)


def test_newton_and_brentq_agree():
    px = b76.black_scholes_call(F, K, T, SIGMA, R)
    a = b76.implied_volatility(F, K, T, px, R, CALL, method="brentq")
    b = b76.implied_volatility(F, K, T, px, R, CALL, method="newton")
    assert a == pytest.approx(b, abs=1e-6)


def test_solver_failure_modes_differ_as_documented():
    """The three solvers are kept because they fail differently."""
    assert np.isnan(b76.implied_vol(F, K, T, 0.0, R, CALL))      # nan
    assert b76.implied_volatility(F, K, T, 0.0, R, CALL) is None  # None
    assert isinstance(b76.fx_vol(F, K, T, 0.0, R, CALL), float)   # always a number


def test_implied_volatility_swallows_a_bad_method_name():
    """Documents an inherited wart rather than asserting the sane behaviour.

    implied_volatility raises ValueError("method must be ...") for an unknown
    method -- but that raise sits inside its own try block, whose except clause
    catches ValueError. So the error is swallowed and the brentq fallback runs:
    a typo'd method name silently still returns an answer. Preserved because
    this is a faithful port; pinned here so the behaviour is visible and a
    future fix is a deliberate change, not a surprise.
    """
    px = b76.black_scholes_call(F, K, T, SIGMA, R)
    iv = b76.implied_volatility(F, K, T, px, R, CALL, method="secant")
    assert iv == pytest.approx(SIGMA, abs=1e-4)


def test_implied_vol_nan_on_nonpositive_inputs():
    assert np.isnan(b76.implied_vol(F, K, 0.0, 100.0, R, CALL))
    assert np.isnan(b76.implied_vol(F, K, T, -1.0, R, CALL))
