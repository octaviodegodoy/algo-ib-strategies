import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
tests/test_gex.py
------------------
Tests for GEX analytics.

Offline tests (no IB needed): TestBSFunctions, TestGEXCalcOffline
Live tests (IB required):     TestGEXLive

Run offline only:
    pytest tests/test_gex.py -v -s -k "not live"

Run all:
    pytest tests/test_gex.py -v -s
"""

import math
import pytest
import pandas as pd
from unittest.mock import MagicMock

from analytics.gex import (
    GEXAnalytics,
    GEXResult,
    bs_gamma,
    bs_price,
    implied_vol_newton,
    _norm_cdf,
    _norm_pdf,
)
from data.rates import RiskFreeRate
from tests.conftest import qualified_stock
from analytics.gex_plot import plot_gex, plot_gex_by_expiry


# ─────────────────────────────────────────────────────────────────────────────
# Offline: Black-Scholes primitives
# ─────────────────────────────────────────────────────────────────────────────

class TestBSFunctions:
    """Pure-math tests — no IB, no network."""

    def test_norm_cdf_symmetry(self):
        assert math.isclose(_norm_cdf(0), 0.5, rel_tol=1e-9)
        assert math.isclose(_norm_cdf(1.96), 0.975, abs_tol=0.001)
        assert math.isclose(_norm_cdf(-1.96), 1 - _norm_cdf(1.96), rel_tol=1e-9)

    def test_norm_pdf_peak(self):
        assert _norm_pdf(0) > _norm_pdf(1) > _norm_pdf(2)

    def test_bs_gamma_atm(self):
        """ATM gamma > OTM gamma for same expiry."""
        S, K_atm, K_otm = 200.0, 200.0, 220.0
        T, r, q, sigma = 0.25, 0.05, 0.01, 0.20
        g_atm = bs_gamma(S, K_atm, T, r, q, sigma)
        g_otm = bs_gamma(S, K_otm, T, r, q, sigma)
        assert g_atm > g_otm > 0

    def test_bs_gamma_zero_cases(self):
        """Gamma must be 0 for degenerate inputs."""
        assert bs_gamma(0, 200, 0.25, 0.05, 0, 0.20) == 0.0   # S=0
        assert bs_gamma(200, 200, 0, 0.05, 0, 0.20) == 0.0    # T=0
        assert bs_gamma(200, 200, 0.25, 0.05, 0, 0) == 0.0    # sigma=0

    def test_bs_gamma_longer_expiry_lower_atm(self):
        """For ATM options, gamma decreases as T increases (higher uncertainty)."""
        g_near = bs_gamma(200, 200, 0.05, 0.05, 0, 0.20)
        g_far  = bs_gamma(200, 200, 0.50, 0.05, 0, 0.20)
        assert g_near > g_far

    def test_bs_price_call_put_parity(self):
        """C - P = S*e^(-qT) - K*e^(-rT) (put-call parity)."""
        S, K, T, r, q, sigma = 200.0, 200.0, 0.25, 0.05, 0.01, 0.20
        call = bs_price(S, K, T, r, q, sigma, "C")
        put  = bs_price(S, K, T, r, q, sigma, "P")
        expected = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert math.isclose(call - put, expected, rel_tol=1e-6)

    def test_bs_price_call_positive(self):
        call = bs_price(200, 200, 0.25, 0.05, 0, 0.20, "C")
        assert call > 0

    def test_bs_price_deep_itm_call(self):
        """Deep ITM call ≈ intrinsic value."""
        call = bs_price(300, 100, 0.01, 0.05, 0, 0.01, "C")
        assert call > 190.0

    def test_implied_vol_roundtrip(self):
        """IV(BS_price(σ)) should recover σ."""
        S, K, T, r, q, sigma = 200.0, 205.0, 0.25, 0.05, 0.01, 0.22
        price = bs_price(S, K, T, r, q, sigma, "C")
        iv = implied_vol_newton(S, K, T, r, q, "C", price, sigma0=0.25)
        assert iv is not None
        assert math.isclose(iv, sigma, abs_tol=1e-5)

    def test_implied_vol_put_roundtrip(self):
        S, K, T, r, q, sigma = 200.0, 195.0, 0.25, 0.05, 0.01, 0.18
        price = bs_price(S, K, T, r, q, sigma, "P")
        iv = implied_vol_newton(S, K, T, r, q, "P", price, sigma0=0.25)
        assert iv is not None
        assert math.isclose(iv, sigma, abs_tol=1e-5)

    def test_implied_vol_zero_price_returns_none(self):
        assert implied_vol_newton(200, 200, 0.25, 0.05, 0, "C", 0.0) is None

    def test_implied_vol_zero_T_returns_none(self):
        assert implied_vol_newton(200, 200, 0.0, 0.05, 0, "C", 5.0) is None


# ─────────────────────────────────────────────────────────────────────────────
# Offline: GEX calculation using mocked chain data
# ─────────────────────────────────────────────────────────────────────────────

def _make_gex_result(spot: float = 200.0, rate: float = 0.05) -> GEXResult:
    """Build a synthetic GEXResult from fabricated chain data."""
    sigma = 0.20
    T = 30 / 365.0

    strikes = [180, 185, 190, 195, 200, 205, 210, 215, 220]
    rows = []
    for K in strikes:
        for right in ("C", "P"):
            g = bs_gamma(spot, K, T, rate, 0, sigma)
            rows.append({
                "symbol":           "TEST",
                "expiration":       "20260620",
                "strike":           float(K),
                "right":            right,
                "delta":            0.5 if K == spot else 0.3,
                "gamma":            g,
                "theta":            0.0,
                "vega":             0.0,
                "iv":               sigma,
                "bid":              5.0,
                "ask":              5.5,
                "last":             5.2,
                "mid":              5.25,
                "underlying_price": spot,
                "rate":             rate,
                "days_to_expiry":   30,
                "forward_price":    spot * math.exp(rate * T),
                "open_interest":    1000,
                "conId":            1,
            })

    df = pd.DataFrame(rows)
    sign = df["right"].map({"C": 1.0, "P": -1.0})
    df["gamma_used"]   = df["gamma"]
    df["dollar_gamma"] = df["gamma"] * spot**2 * 100
    df["gex_line"]     = df["dollar_gamma"] * df["open_interest"] * sign

    # Build profile manually
    ga = GEXAnalytics.__new__(GEXAnalytics)  # no __init__ needed
    profile = ga._build_strike_profile(df, spot)

    total_gex  = float(df["gex_line"].sum())
    dealer_gex = -total_gex
    zero_g = ga._zero_gamma_level(profile, spot)
    call_wall = ga._call_wall(profile)
    put_wall  = ga._put_wall(profile)

    return GEXResult(
        symbol="TEST",
        spot=spot,
        rate=rate,
        data=df,
        total_gex=total_gex,
        dealer_gex=dealer_gex,
        profile=profile,
        by_expiry=pd.DataFrame(),
        zero_gamma_level=zero_g,
        call_wall=call_wall,
        put_wall=put_wall,
        zero_oi_ratio=0.0,
    )


class TestGEXCalcOffline:
    """Tests on fabricated data — no IB needed."""

    def test_total_gex_is_float(self):
        res = _make_gex_result()
        assert isinstance(res.total_gex, float)

    def test_dealer_gex_is_negative_total(self):
        res = _make_gex_result()
        assert math.isclose(res.dealer_gex, -res.total_gex, rel_tol=1e-9)

    def test_profile_has_required_columns(self):
        res = _make_gex_result()
        for col in ("strike", "call_gex", "put_gex", "net_gex", "cumulative_gex"):
            assert col in res.profile.columns

    def test_profile_strike_sorted(self):
        res = _make_gex_result()
        strikes = res.profile["strike"].tolist()
        assert strikes == sorted(strikes)

    def test_call_gex_positive(self):
        """All call GEX contributions must be >= 0."""
        res = _make_gex_result()
        assert (res.profile["call_gex"] >= 0).all()

    def test_put_gex_negative(self):
        """All put GEX contributions must be <= 0."""
        res = _make_gex_result()
        assert (res.profile["put_gex"] <= 0).all()

    def test_zero_gamma_within_strike_range(self):
        res = _make_gex_result(spot=200.0)
        assert 170 <= res.zero_gamma_level <= 230

    def test_call_wall_is_valid_strike(self):
        res = _make_gex_result()
        valid_strikes = res.profile["strike"].tolist()
        assert res.call_wall in valid_strikes

    def test_put_wall_is_valid_strike(self):
        res = _make_gex_result()
        valid_strikes = res.profile["strike"].tolist()
        assert res.put_wall in valid_strikes

    def test_summary_contains_key_fields(self):
        res = _make_gex_result()
        s = res.summary()
        assert "Total GEX" in s
        assert "Dealer GEX" in s
        assert "Zero-gamma" in s
        assert "Call wall" in s
        assert "Put wall" in s

    def test_symmetric_chain_zero_gex(self):
        """
        Symmetric OI on calls and puts must produce total GEX ≈ 0
        at the ATM strike (calls cancel puts exactly at ATM).
        """
        res = _make_gex_result(spot=200.0)
        # Net GEX at ATM strike should be near zero (call and put gammas are equal)
        atm_row = res.profile[res.profile["strike"] == 200.0]
        if not atm_row.empty:
            net = float(atm_row["net_gex"].iloc[0])
            assert abs(net) < 1e-3, f"ATM net GEX not near 0: {net}"

    def test_print_gex_summary(self):
        res = _make_gex_result(spot=200.0)
        print("\n" + res.summary())
        print("\nStrike profile:")
        print(res.profile.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Live: IB-connected tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestGEXLive:
    """Requires IB Gateway running + options market data subscription."""

    @pytest.mark.parametrize("symbol", ["SPY", "AAPL"])
    def test_compute_returns_gex_result(self, ib, symbol):
        contract = qualified_stock(ib, symbol)
        ga = GEXAnalytics(ib)
        res = ga.compute(contract, expirations=2)
        if res.data.empty:
            pytest.skip(f"No options data for {symbol} — check options subscription")
        assert isinstance(res, GEXResult)
        assert res.spot > 0

    @pytest.mark.parametrize("symbol", ["SPY", "AAPL"])
    def test_profile_columns_present(self, ib, symbol):
        contract = qualified_stock(ib, symbol)
        ga = GEXAnalytics(ib)
        res = ga.compute(contract, expirations=2)
        if res.data.empty:
            pytest.skip(f"No options data for {symbol}")
        for col in ("strike", "call_gex", "put_gex", "net_gex"):
            assert col in res.profile.columns

    @pytest.mark.parametrize("symbol", ["SPY", "AAPL"])
    def test_zero_gamma_level_near_spot(self, ib, symbol):
        contract = qualified_stock(ib, symbol)
        ga = GEXAnalytics(ib)
        res = ga.compute(contract, expirations=2)
        if res.data.empty:
            pytest.skip(f"No options data for {symbol}")
        # Zero-gamma level should be within ±40% of spot
        assert 0.6 * res.spot <= res.zero_gamma_level <= 1.4 * res.spot

    def test_print_spy_gex_report(self, ib):
        """Print the full SPY GEX report (informational)."""
        contract = qualified_stock(ib, "SPY")
        ga = GEXAnalytics(ib)
        res = ga.compute(contract, expirations=3)
        if res.data.empty:
            pytest.skip("No SPY options data")
        print("\n" + res.summary())
        print("\nGEX profile (top 15 by |net_gex|):")
        top = res.profile.reindex(
            res.profile["net_gex"].abs().sort_values(ascending=False).index
        ).head(15)
        print(top.to_string(index=False))
        print("\nGEX by expiry:")
        print(res.by_expiry.to_string(index=False))

    def test_plot_gex(self, ib):
        """Plot the GEX profile interactively (Plotly opens in browser)."""
        contract = qualified_stock(ib, "SPY")
        ga = GEXAnalytics(ib)
        res = ga.compute(contract, expirations=3)
        print(f"\nSPY GEX rows: {len(res.data)}, spot={res.spot:.2f}")
        if res.data.empty:
            pytest.skip("No SPY options data \u2014 check market data subscription / market hours")
        fig1 = plot_gex(res, show=True)
        fig2 = plot_gex_by_expiry(res, show=True)
        assert fig1 is not None and fig2 is not None
        print("Opened GEX plots in browser")
