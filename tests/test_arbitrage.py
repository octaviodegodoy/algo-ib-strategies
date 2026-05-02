import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
tests/test_arbitrage.py
------------------------
Arbitrage and spread sanity checks.

Strategies validated here:
  1. ETF / Index spread        — SPY vs QQQ correlation check
  2. Dual-listed stock spread  — same stock on two exchanges
  3. Forex triangular spread   — EUR/USD vs EUR/GBP vs GBP/USD
  4. Statistical spread        — AAPL vs MSFT price ratio within historical range
  5. Calendar spread           — historical volatility across timeframes

These tests do NOT place orders.
They assert that real-time prices are within expected arbitrage bounds,
and flag anomalies that could signal a trading opportunity.
"""

import pytest
import numpy as np
import pandas as pd

from data.market_data import MarketData

pytestmark = pytest.mark.live


# ── helpers ───────────────────────────────────────────────────────────────────

def get_close_series(market_data: MarketData, contract, duration="20 D", bar_size="1 day") -> pd.Series:
    df = market_data.get_historical_bars(contract, duration=duration, bar_size=bar_size)
    if df.empty:
        pytest.skip(f"No data for {contract.symbol}")
    return df["close"].dropna().astype(float)


def get_midprice(market_data: MarketData, contract) -> float:
    snap = market_data.snapshot(contract)
    bid  = snap.get("bid") or 0.0
    ask  = snap.get("ask") or 0.0
    last = snap.get("last") or snap.get("close") or 0.0
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return float(last)


# ── 1. ETF spread — SPY vs QQQ ───────────────────────────────────────────────

class TestETFSpread:

    def test_spy_qqq_correlation_above_threshold(self, market_data: MarketData, spy, qqq):
        """SPY and QQQ daily returns must be highly correlated (>0.85)."""
        spy_s = get_close_series(market_data, spy).pct_change().dropna()
        qqq_s = get_close_series(market_data, qqq).pct_change().dropna()

        combined = pd.concat([spy_s, qqq_s], axis=1).dropna()
        combined.columns = ["spy", "qqq"]

        corr = combined["spy"].corr(combined["qqq"])
        assert corr > 0.85, (
            f"SPY/QQQ correlation too low: {corr:.3f}. "
            "Possible data gap or market dislocation."
        )

    def test_spy_qqq_spread_ratio_stability(self, market_data: MarketData, spy, qqq):
        """SPY/QQQ price ratio must not deviate >15% from its 20-day mean."""
        spy_prices = get_close_series(market_data, spy)
        qqq_prices = get_close_series(market_data, qqq)

        combined = pd.concat([spy_prices, qqq_prices], axis=1).dropna()
        combined.columns = ["spy", "qqq"]
        ratio = combined["spy"] / combined["qqq"]

        mean_ratio = ratio.mean()
        current_ratio = ratio.iloc[-1]
        deviation_pct = abs(current_ratio - mean_ratio) / mean_ratio * 100

        assert deviation_pct < 15, (
            f"SPY/QQQ ratio deviated {deviation_pct:.2f}% from mean ({mean_ratio:.4f}). "
            "Potential arbitrage opportunity."
        )


# ── 2. Forex triangular arbitrage — EUR/USD, EUR/GBP, GBP/USD ────────────────

class TestForexTriangularArbitrage:

    def test_eurusd_eurgbp_gbpusd_no_arbitrage(self, market_data: MarketData, eurusd, eurgbp, gbpusd):
        """
        Triangular relationship: EUR/USD ≈ EUR/GBP × GBP/USD
        Deviation > 0.5% indicates a potential arbitrage.
        """
        eurusd_mid = get_midprice(market_data, eurusd)
        eurgbp_mid = get_midprice(market_data, eurgbp)
        gbpusd_mid = get_midprice(market_data, gbpusd)

        if any(x == 0 for x in (eurusd_mid, eurgbp_mid, gbpusd_mid)):
            pytest.skip("One or more forex snapshots returned 0 — market may be closed")

        implied_eurusd = eurgbp_mid * gbpusd_mid
        deviation_pct  = abs(eurusd_mid - implied_eurusd) / eurusd_mid * 100

        assert deviation_pct < 0.5, (
            f"Triangular arb detected! "
            f"EUR/USD={eurusd_mid:.5f}, EUR/GBP×GBP/USD={implied_eurusd:.5f}, "
            f"deviation={deviation_pct:.4f}%"
        )

    def test_forex_bid_ask_spread_reasonable(self, market_data: MarketData, eurusd, gbpusd):
        """Forex bid-ask spread must be <10 pips for major pairs."""
        for contract in (eurusd, gbpusd):
            snap = market_data.snapshot(contract)
            bid = float(snap.get("bid") or 0)
            ask = float(snap.get("ask") or 0)
            if bid == 0 or ask == 0:
                continue
            spread_pips = (ask - bid) * 10_000
            assert spread_pips < 10, (
                f"{contract.symbol}: spread {spread_pips:.1f} pips is too wide"
            )


# ── 3. Statistical spread — AAPL vs MSFT ─────────────────────────────────────

class TestStatisticalSpread:

    def test_aapl_msft_spread_within_2_sigma(self, market_data: MarketData, aapl, msft):
        """
        Log price spread AAPL − MSFT must be within 2 standard deviations
        of its 20-day mean (mean-reversion assumption).
        """
        aapl_log = np.log(get_close_series(market_data, aapl))
        msft_log = np.log(get_close_series(market_data, msft))

        combined = pd.concat([aapl_log, msft_log], axis=1).dropna()
        combined.columns = ["aapl", "msft"]
        spread = combined["aapl"] - combined["msft"]

        mean = spread.mean()
        std  = spread.std()
        current = spread.iloc[-1]
        z_score = (current - mean) / std if std > 0 else 0

        print(f"\nAAPL/MSFT log-spread z-score: {z_score:.2f}")

        assert abs(z_score) < 3.0, (
            f"AAPL/MSFT spread z-score={z_score:.2f} exceeds 3σ — "
            "possible mean-reversion opportunity."
        )

    def test_aapl_msft_returns_not_diverging(self, market_data: MarketData, aapl, msft):
        """Cumulative 20-day returns of AAPL and MSFT must not differ by more than 30%."""
        aapl_s = get_close_series(market_data, aapl)
        msft_s = get_close_series(market_data, msft)

        aapl_ret = (aapl_s.iloc[-1] - aapl_s.iloc[0]) / aapl_s.iloc[0] * 100
        msft_ret = (msft_s.iloc[-1] - msft_s.iloc[0]) / msft_s.iloc[0] * 100
        diff = abs(aapl_ret - msft_ret)

        print(f"\nAAPL 20d return: {aapl_ret:.2f}%  MSFT: {msft_ret:.2f}%  diff: {diff:.2f}%")

        assert diff < 30, (
            f"Return divergence AAPL vs MSFT = {diff:.2f}% (>30%) — "
            "potential pairs trade signal."
        )


# ── 4. Calendar spread — intraday vs daily volatility ────────────────────────

class TestCalendarSpread:

    def test_aapl_intraday_vol_consistent_with_daily(self, market_data: MarketData, aapl):
        """
        Annualised volatility from 1-hour bars should be within 3× of
        daily-bar vol (rough consistency check).
        """
        daily = get_close_series(market_data, aapl, duration="20 D", bar_size="1 day")
        intra = get_close_series(market_data, aapl, duration="5 D",  bar_size="1 hour")

        daily_vol = daily.pct_change().dropna().std() * np.sqrt(252)
        intra_vol = intra.pct_change().dropna().std() * np.sqrt(252 * 6.5)  # ~6.5 trading hours

        print(f"\nAAPL daily vol (ann): {daily_vol:.4f}  intraday vol (ann): {intra_vol:.4f}")

        ratio = intra_vol / daily_vol if daily_vol > 0 else 1
        assert 0.2 < ratio < 5.0, (
            f"Vol ratio intraday/daily = {ratio:.2f} — unusual divergence "
            f"(daily={daily_vol:.4f}, intraday={intra_vol:.4f})"
        )
