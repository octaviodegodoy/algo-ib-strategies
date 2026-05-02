import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
tests/test_market_data.py
--------------------------
Verify historical prices, snapshots, positions, orders and fills.
"""

import pytest
import pandas as pd
from ib_async import Stock

from data.market_data import MarketData
from tests.conftest import qualified_stock

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Parameterised: daily history by symbol + period
# ---------------------------------------------------------------------------
# Each entry: (symbol, duration, expected_min_bars)
# duration follows IB format: "N D" / "N W" / "N M" / "N Y"
DAILY_HISTORY_CASES = [
    ("AAPL",  "5 D",  3),
    ("AAPL",  "1 M",  15),
    ("AAPL",  "3 M",  50),
    ("AAPL",  "6 M",  100),
    ("AAPL",  "1 Y",  200),
    ("MSFT",  "5 D",  3),
    ("MSFT",  "1 M",  15),
    ("MSFT",  "3 M",  50),
    ("GOOGL", "1 M",  15),
    ("AMZN",  "1 M",  15),
    ("TSLA",  "1 M",  15),
    ("NVDA",  "1 M",  15),
    ("SPY",   "1 Y",  200),
    ("QQQ",   "6 M",  100),
]


class TestDailyHistory:
    """Fetch daily OHLCV bars for multiple symbols and periods."""

    @pytest.mark.parametrize("symbol,duration,min_bars", DAILY_HISTORY_CASES)
    def test_daily_bars_by_symbol_and_period(
        self, market_data: MarketData, ib, symbol: str, duration: str, min_bars: int
    ):
        """Daily bars must be non-empty, contain OHLCV columns and pass basic sanity checks."""
        contract = qualified_stock(ib, symbol)
        df = market_data.get_historical_bars(
            contract,
            duration=duration,
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=True,
        )

        # ── 1. non-empty ──────────────────────────────────────────────────────
        assert not df.empty, f"{symbol} {duration}: no data returned"

        # ── 2. enough bars ───────────────────────────────────────────────────
        assert len(df) >= min_bars, (
            f"{symbol} {duration}: expected >= {min_bars} bars, got {len(df)}"
        )

        # ── 3. required columns ──────────────────────────────────────────────
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns, f"{symbol}: missing column '{col}'"

        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        # ── 4. OHLC integrity ────────────────────────────────────────────────
        assert (df["high"] >= df["low"]).all(),   f"{symbol}: high < low"
        assert (df["high"] >= df["open"]).all(),  f"{symbol}: high < open"
        assert (df["high"] >= df["close"]).all(), f"{symbol}: high < close"
        assert (df["low"]  <= df["open"]).all(),  f"{symbol}: low > open"
        assert (df["low"]  <= df["close"]).all(), f"{symbol}: low > close"

        # ── 5. no negative prices ────────────────────────────────────────────
        assert (df[["open", "high", "low", "close"]] > 0).all().all(), (
            f"{symbol}: non-positive price found"
        )

        # ── 6. index is sorted ascending ─────────────────────────────────────
        assert df.index.is_monotonic_increasing, f"{symbol}: date index not sorted"

        # ── 7. print summary (visible with pytest -s) ─────────────────────────
        first = df.index[0]
        last  = df.index[-1]
        print(
            f"\n{symbol:6s} {duration:4s} | "
            f"{len(df):4d} bars | "
            f"{first} → {last} | "
            f"close={df['close'].iloc[-1]:.2f}"
        )


class TestHistoricalData:

    def test_aapl_daily_bars_returns_dataframe(self, market_data: MarketData, aapl):
        df = market_data.get_historical_bars(aapl, duration="10 D", bar_size="1 day")
        assert isinstance(df, pd.DataFrame), "Expected a DataFrame"
        assert not df.empty, "DataFrame is empty — no data returned"

    def test_aapl_bars_have_required_columns(self, market_data: MarketData, aapl):
        df = market_data.get_historical_bars(aapl, duration="5 D", bar_size="1 day")
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns, f"Missing column: {col}"

    def test_aapl_ohlc_validity(self, market_data: MarketData, aapl):
        """High >= Low, High >= Open, High >= Close for every bar."""
        df = market_data.get_historical_bars(aapl, duration="5 D", bar_size="1 day")
        assert (df["high"] >= df["low"]).all(),   "high < low found"
        assert (df["high"] >= df["open"]).all(),  "high < open found"
        assert (df["high"] >= df["close"]).all(), "high < close found"

    def test_aapl_intraday_bars(self, market_data: MarketData, aapl):
        df = market_data.get_historical_bars(aapl, duration="3 D", bar_size="1 hour")
        assert not df.empty, "No intraday bars returned"
        assert len(df) > 3, f"Too few intraday bars: {len(df)}"

    def test_msft_daily_bars(self, market_data: MarketData, msft):
        df = market_data.get_historical_bars(msft, duration="5 D", bar_size="1 day")
        assert not df.empty

    def test_forex_eurusd_bars(self, market_data: MarketData, eurusd):
        df = market_data.get_historical_bars(
            eurusd, duration="5 D", bar_size="1 hour", what_to_show="MIDPOINT"
        )
        assert not df.empty, "No EURUSD bars returned"


class TestSnapshot:

    def test_aapl_snapshot_has_fields(self, market_data: MarketData, aapl):
        snap = market_data.snapshot(aapl)
        assert "bid" in snap.index
        assert "ask" in snap.index
        assert "last" in snap.index

    def test_aapl_snapshot_bid_ask_spread(self, market_data: MarketData, aapl):
        snap = market_data.snapshot(aapl)
        bid = snap["bid"]
        ask = snap["ask"]
        if bid and ask and bid > 0 and ask > 0:
            assert ask >= bid, f"ask ({ask}) < bid ({bid}) — negative spread"
            spread_pct = (ask - bid) / bid * 100
            assert spread_pct < 5, f"Spread too wide: {spread_pct:.2f}%"


class TestAccountData:

    def test_account_summary_returns_dataframe(self, market_data: MarketData):
        df = market_data.get_account_summary()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_positions_returns_dataframe(self, market_data: MarketData):
        df = market_data.get_positions()
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            for col in ("symbol", "position", "avgCost"):
                assert col in df.columns

    def test_open_orders_returns_dataframe(self, market_data: MarketData):
        df = market_data.get_open_orders()
        assert isinstance(df, pd.DataFrame)

    def test_executions_returns_dataframe(self, market_data: MarketData):
        df = market_data.get_executions()
        assert isinstance(df, pd.DataFrame)
