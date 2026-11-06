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
from datetime import datetime, timedelta
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


class TestNoCustomUserAgent:
    """Offline guard against the regression that broke every rate fetch.

    _fetch_fred used to send "User-Agent: algo-ib-strategies/1.0". FRED accepts
    the connection for it and then never sends a body, so the read timed out at
    any value and get_rate() silently returned DEFAULT_RATE (5.00%) while DTB3
    was 4.03%. Re-adding such a header looks like politeness and disables the
    whole rate layer, so this pins its absence without needing the network.
    """

    def test_fetch_sets_no_custom_user_agent(self):
        captured = {}

        class _Resp:
            def read(self):
                return b"DATE,DTB3\n2026-11-04,4.03\n"
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        def _fake_urlopen(req, timeout=None):
            captured["req"] = req
            return _Resp()

        with patch("urllib.request.urlopen", _fake_urlopen):
            rate = RiskFreeRate(series="DTB3")._fetch_fred("DTB3")

        assert rate == pytest.approx(0.0403)
        req = captured["req"]
        assert req.get_header("User-agent") is None, (
            "a custom User-Agent is set again — FRED black-holes non-default "
            "agents and every rate lookup will silently fall back to "
            "DEFAULT_RATE. See the comment in data/rates.py._fetch_fred."
        )


class TestFallbackCacheTtl:
    """The fallback must not be cached as if it were a rate.

    get_rate() caches whatever it returns so that a scan calling it per option
    does not hit FRED per option. But DEFAULT_RATE is not a rate, and giving it
    the same four-hour TTL meant one timeout pinned 5.00% for a whole session
    while DTB3 was 4.03% -- every IV and delta downstream computed off a number
    nobody measured. A failed lookup now expires in minutes.

    These age _cached_at directly rather than patching the clock: the TTL
    comparison is the thing under test, so moving the timestamp exercises it
    exactly as the passage of time would.
    """

    @staticmethod
    def _age(rfr, by):
        rfr._cached_at -= by

    def test_fallback_still_caches_within_its_own_ttl(self):
        """A failure must not mean a fetch attempt per call."""
        rfr = RiskFreeRate(series="DTB3")
        with patch.object(rfr, "_fetch_fred", side_effect=OSError("boom")) as m:
            assert rfr.get_rate() == RiskFreeRate.DEFAULT_RATE
            assert rfr.get_rate() == RiskFreeRate.DEFAULT_RATE
            assert rfr.get_rate() == RiskFreeRate.DEFAULT_RATE
            assert m.call_count == 1, "retried inside the fallback TTL"

    def test_fallback_is_retried_once_its_short_ttl_expires(self):
        rfr = RiskFreeRate(series="DTB3")
        with patch.object(rfr, "_fetch_fred", side_effect=OSError("boom")) as m:
            rfr.get_rate()
            self._age(rfr, RiskFreeRate._FALLBACK_CACHE_TTL + timedelta(seconds=1))
            rfr.get_rate()
            assert m.call_count == 2

    def test_one_outage_does_not_pin_the_wrong_rate_for_the_session(self):
        """The regression itself: an hour-old fallback must be re-fetched.

        One hour is past _FALLBACK_CACHE_TTL but well inside _CACHE_TTL, so
        under the old shared TTL this fetch would not happen and 5.00% would
        survive until the session ended.
        """
        rfr = RiskFreeRate(series="DTB3")
        with patch.object(rfr, "_fetch_fred", side_effect=OSError("boom")):
            assert rfr.get_rate() == RiskFreeRate.DEFAULT_RATE
        assert rfr.using_fallback

        self._age(rfr, timedelta(hours=1))
        assert timedelta(hours=1) < RiskFreeRate._CACHE_TTL, (
            "this test only means something while an hour is inside _CACHE_TTL"
        )

        with patch.object(rfr, "_fetch_fred", return_value=0.0403):
            assert rfr.get_rate() == pytest.approx(0.0403)
        assert not rfr.using_fallback

    def test_a_recovered_rate_then_gets_the_long_ttl(self):
        """Recovery must not leave it on the short TTL, re-fetching forever."""
        rfr = RiskFreeRate(series="DTB3")
        with patch.object(rfr, "_fetch_fred", side_effect=OSError("boom")):
            rfr.get_rate()
        self._age(rfr, RiskFreeRate._FALLBACK_CACHE_TTL + timedelta(seconds=1))
        with patch.object(rfr, "_fetch_fred", return_value=0.0403):
            rfr.get_rate()

        self._age(rfr, RiskFreeRate._FALLBACK_CACHE_TTL + timedelta(seconds=1))
        with patch.object(rfr, "_fetch_fred", side_effect=OSError("boom")) as m:
            assert rfr.get_rate() == pytest.approx(0.0403)
            assert m.call_count == 0, "a real rate was dropped on the short TTL"

    def test_a_fetched_rate_is_held_for_the_long_ttl(self):
        rfr = RiskFreeRate(series="DTB3")
        with patch.object(rfr, "_fetch_fred", return_value=0.0403) as m:
            rfr.get_rate()
            self._age(rfr, RiskFreeRate._FALLBACK_CACHE_TTL + timedelta(seconds=1))
            assert rfr.get_rate() == pytest.approx(0.0403)
            assert m.call_count == 1

    def test_the_two_ttls_are_not_the_same(self):
        """Pins the separation. Setting them equal reinstates the bug."""
        assert RiskFreeRate._FALLBACK_CACHE_TTL < RiskFreeRate._CACHE_TTL
        assert RiskFreeRate._FALLBACK_CACHE_TTL <= timedelta(minutes=15)
        assert RiskFreeRate._FALLBACK_CACHE_TTL >= timedelta(minutes=1), (
            "too short to be a cache -- a dead FRED would stall every call"
        )

    def test_using_fallback_starts_false_and_tracks_the_last_call(self):
        rfr = RiskFreeRate(series="DTB3")
        assert not rfr.using_fallback
        with patch.object(rfr, "_fetch_fred", return_value=0.0403):
            rfr.get_rate()
        assert not rfr.using_fallback
        self._age(rfr, RiskFreeRate._CACHE_TTL + timedelta(seconds=1))
        with patch.object(rfr, "_fetch_fred", side_effect=OSError("boom")):
            rfr.get_rate()
        assert rfr.using_fallback

@pytest.mark.live
class TestFredLive:
    """Needs network access. These are what would have caught the broken UA.

    Marked `live` so `-m "not live"` still runs offline; the marker covers
    network-dependent tests as well as ones needing IB.
    """

    def test_fetch_fred_actually_succeeds(self):
        """The authoritative guard: _fetch_fred raises on failure.

        Asserting get_rate() != DEFAULT_RATE would false-fail on the day DTB3
        is genuinely 5.00%. This cannot.
        """
        rate = RiskFreeRate(series="DTB3")._fetch_fred("DTB3")
        assert isinstance(rate, float)
        assert 0.0 < rate < 0.25, f"implausible rate {rate}"

    def test_get_rate_returns_the_fetched_value_not_the_fallback(self):
        rfr = RiskFreeRate(series="DTB3")
        fetched = rfr._fetch_fred("DTB3")
        assert rfr.get_rate() == pytest.approx(fetched), (
            "get_rate() did not return what _fetch_fred() returns — it fell "
            "back to DEFAULT_RATE despite the fetch working"
        )

    def test_all_series_fetch(self):
        from data.rates import SERIES_DESCRIPTIONS
        failed = []
        for series in SERIES_DESCRIPTIONS:
            try:
                r = RiskFreeRate(series=series)._fetch_fred(series)
                assert 0.0 < r < 0.25
            except Exception as exc:
                failed.append(f"{series}: {type(exc).__name__}: {exc}")
        assert not failed, "series that did not fetch: " + "; ".join(failed)

    def test_rate_is_decimal_not_percent(self):
        """4.03% must arrive as 0.0403. Forgetting the /100 would make every
        discount factor and forward price absurd."""
        rate = RiskFreeRate(series="DTB3")._fetch_fred("DTB3")
        assert rate < 0.25, f"{rate} looks like percent, not a decimal"
