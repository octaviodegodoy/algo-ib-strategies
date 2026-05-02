import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
tests/test_rates.py
--------------------
Tests for data/rates.py — no IB connection required.

Run:
    pytest tests/test_rates.py -v -s
"""

import math
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from data.rates import RiskFreeRate


class TestFredFetch:

    def test_get_rate_returns_float(self):
        """get_rate() must return a float in a plausible range."""
        rfr = RiskFreeRate(series="DTB3")
        rate = rfr.get_rate()
        assert isinstance(rate, float)
        # Any non-negative rate up to 20% is considered plausible
        assert 0.0 <= rate <= 0.20, f"Rate out of expected range: {rate:.4f}"

    def test_get_rate_is_cached(self):
        """Second call must use the cache (no second HTTP request)."""
        rfr = RiskFreeRate(series="DTB3")
        with patch.object(rfr, "_fetch_fred", wraps=rfr._fetch_fred) as mock_fetch:
            rfr.get_rate()
            rfr.get_rate()
            assert mock_fetch.call_count == 1, "Expected only one HTTP fetch (cache miss)"

    def test_get_rate_falls_back_on_network_error(self):
        """If the HTTP fetch fails, DEFAULT_RATE must be returned without raising."""
        rfr = RiskFreeRate(series="DTB3")
        with patch.object(rfr, "_fetch_fred", side_effect=OSError("network error")):
            rate = rfr.get_rate()
        assert rate == RiskFreeRate.DEFAULT_RATE

    def test_get_rate_sofr(self):
        """SOFR series should also return a plausible rate."""
        rfr = RiskFreeRate(series="SOFR")
        rate = rfr.get_rate()
        assert 0.0 <= rate <= 0.20

    def test_get_rate_fedfunds(self):
        """FEDFUNDS series should return a plausible rate."""
        rfr = RiskFreeRate(series="FEDFUNDS")
        rate = rfr.get_rate()
        assert 0.0 <= rate <= 0.20

    def test_rate_prints_on_fetch(self, capsys):
        """Info log should be emitted on a fresh (non-cached) fetch."""
        import logging
        rfr = RiskFreeRate(series="DTB3")
        logging.basicConfig(level=logging.DEBUG)
        rfr.get_rate()   # first call — not cached


class TestForwardPrice:

    def test_forward_price_zero_rate(self):
        """With r=0 and q=0, forward == spot."""
        fwd = RiskFreeRate.forward_price(spot=200.0, rate=0.0, days_to_expiry=30)
        assert math.isclose(fwd, 200.0, rel_tol=1e-9)

    def test_forward_price_positive_rate(self):
        """Forward must be above spot when r > 0 and q=0."""
        fwd = RiskFreeRate.forward_price(spot=200.0, rate=0.05, days_to_expiry=30)
        assert fwd > 200.0

    def test_forward_price_dividend_reduces_forward(self):
        """Higher dividend yield must reduce the forward price."""
        fwd_no_div  = RiskFreeRate.forward_price(200.0, 0.05, 30, dividend_yield=0.0)
        fwd_with_div = RiskFreeRate.forward_price(200.0, 0.05, 30, dividend_yield=0.02)
        assert fwd_with_div < fwd_no_div

    def test_forward_price_longer_expiry_higher(self):
        """Longer time to expiry must yield a higher forward (r > 0)."""
        fwd_30  = RiskFreeRate.forward_price(200.0, 0.05, days_to_expiry=30)
        fwd_90  = RiskFreeRate.forward_price(200.0, 0.05, days_to_expiry=90)
        assert fwd_90 > fwd_30

    def test_forward_price_formula(self):
        """Verify F = S * exp((r-q)*T) exactly."""
        S, r, q, days = 150.0, 0.045, 0.01, 45
        T = days / 365.0
        expected = S * math.exp((r - q) * T)
        got = RiskFreeRate.forward_price(S, r, days, dividend_yield=q)
        assert math.isclose(got, expected, rel_tol=1e-9)

    def test_forward_price_zero_dte(self):
        """At expiry (0 days), forward must equal spot regardless of rate."""
        fwd = RiskFreeRate.forward_price(200.0, 0.05, days_to_expiry=0)
        assert math.isclose(fwd, 200.0, rel_tol=1e-9)


class TestDaysToExpiry:

    def test_future_expiry(self):
        """A future date must produce a positive DTE."""
        future = (datetime.today().date().replace(year=datetime.today().year + 1))
        exp_str = future.strftime("%Y%m%d")
        dte = RiskFreeRate.days_to_expiry(exp_str)
        assert dte > 0

    def test_past_expiry_returns_zero(self):
        """An already-expired date must return 0, not negative."""
        past_str = "20200101"
        dte = RiskFreeRate.days_to_expiry(past_str)
        assert dte == 0

    def test_time_to_expiry_consistent_with_days(self):
        """time_to_expiry must equal days_to_expiry / 365."""
        exp_str = "20261231"
        days = RiskFreeRate.days_to_expiry(exp_str)
        T    = RiskFreeRate.time_to_expiry(exp_str)
        assert math.isclose(T, days / 365.0, rel_tol=1e-9)

    def test_print_current_rates(self):
        """Informational: print all available series rates."""
        from data.rates import SERIES_DESCRIPTIONS
        print("\nCurrent FRED rates:")
        for series, desc in SERIES_DESCRIPTIONS.items():
            rfr = RiskFreeRate(series=series)
            rate = rfr.get_rate()
            print(f"  {series:10s}  {desc:50s}  {rate*100:6.3f} %")
