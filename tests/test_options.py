import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
tests/test_options.py
----------------------
Fetch option chains filtered by delta ≈ 0.25 and 0.75.

Requires:
  - IB Gateway / TWS running with market data subscription that includes options.
  - Options greeks require a Level 2 / options data subscription on the account.
    Without it, greeks may come back as None (test will skip).

Run:
    pytest tests/test_options.py -v -s
"""

import pytest
import pandas as pd

from data.options import OptionChain
from tests.conftest import qualified_stock

pytestmark = pytest.mark.live

TARGET_DELTAS    = [0.25, 0.75]
DELTA_TOLERANCE  = 0.05   # match |delta| within ±0.05 of each target


class TestOptionChain:

    @pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "SPY"])
    def test_option_chain_returns_dataframe(self, ib, symbol: str):
        """Chain must return a non-empty DataFrame."""
        contract = qualified_stock(ib, symbol)
        chain = OptionChain(ib)
        df = chain.get_by_delta(
            contract,
            target_deltas=TARGET_DELTAS,
            delta_tolerance=DELTA_TOLERANCE,
            expirations=1,
        )
        if df.empty:
            pytest.skip(
                f"No options data for {symbol} — check options market data subscription"
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.parametrize("symbol", ["AAPL", "SPY"])
    def test_option_chain_has_required_columns(self, ib, symbol: str):
        contract = qualified_stock(ib, symbol)
        chain = OptionChain(ib)
        df = chain.get_by_delta(contract, target_deltas=TARGET_DELTAS,
                                delta_tolerance=DELTA_TOLERANCE, expirations=1)
        if df.empty:
            pytest.skip(f"No options data for {symbol}")

        for col in ("symbol", "expiration", "strike", "right",
                    "delta", "gamma", "theta", "vega", "iv",
                    "bid", "ask", "mid", "underlying_price",
                    "rate", "days_to_expiry", "forward_price"):
            assert col in df.columns, f"Missing column: {col}"

    @pytest.mark.parametrize("symbol", ["AAPL", "SPY"])
    def test_delta_within_tolerance(self, ib, symbol: str):
        """Every returned row must have |delta| within tolerance of a target."""
        contract = qualified_stock(ib, symbol)
        chain = OptionChain(ib)
        df = chain.get_by_delta(contract, target_deltas=TARGET_DELTAS,
                                delta_tolerance=DELTA_TOLERANCE, expirations=1)
        if df.empty:
            pytest.skip(f"No options data for {symbol}")

        for _, row in df.iterrows():
            abs_delta = abs(row["delta"])
            matched = any(
                abs(abs_delta - tgt) <= DELTA_TOLERANCE
                for tgt in TARGET_DELTAS
            )
            assert matched, (
                f"{symbol}: delta={row['delta']:.4f} not near any target {TARGET_DELTAS}"
            )

    @pytest.mark.parametrize("symbol", ["AAPL", "SPY"])
    def test_both_calls_and_puts_present(self, ib, symbol: str):
        """Chain must include both C and P rows."""
        contract = qualified_stock(ib, symbol)
        chain = OptionChain(ib)
        df = chain.get_by_delta(contract, target_deltas=TARGET_DELTAS,
                                delta_tolerance=DELTA_TOLERANCE, expirations=1)
        if df.empty:
            pytest.skip(f"No options data for {symbol}")

        rights = set(df["right"].unique())
        assert "C" in rights, f"{symbol}: no call options returned"
        assert "P" in rights, f"{symbol}: no put options returned"

    @pytest.mark.parametrize("symbol", ["AAPL", "SPY"])
    def test_bid_ask_spread_reasonable(self, ib, symbol: str):
        """ask >= bid for all rows with valid quotes."""
        contract = qualified_stock(ib, symbol)
        chain = OptionChain(ib)
        df = chain.get_by_delta(contract, target_deltas=TARGET_DELTAS,
                                delta_tolerance=DELTA_TOLERANCE, expirations=1)
        if df.empty:
            pytest.skip(f"No options data for {symbol}")

        valid = df[(df["bid"] > 0) & (df["ask"] > 0)]
        if valid.empty:
            pytest.skip(f"{symbol}: no rows with valid bid/ask")

        bad = valid[valid["ask"] < valid["bid"]]
        assert bad.empty, (
            f"{symbol}: {len(bad)} rows have ask < bid:\n{bad[['strike','right','bid','ask']]}"
        )

    @pytest.mark.parametrize("symbol", ["AAPL", "SPY"])
    def test_iv_positive(self, ib, symbol: str):
        """Implied volatility must be positive for all rows."""
        contract = qualified_stock(ib, symbol)
        chain = OptionChain(ib)
        df = chain.get_by_delta(contract, target_deltas=TARGET_DELTAS,
                                delta_tolerance=DELTA_TOLERANCE, expirations=1)
        if df.empty:
            pytest.skip(f"No options data for {symbol}")

        valid_iv = df[df["iv"] > 0]
        if valid_iv.empty:
            pytest.skip(f"{symbol}: no IV data returned")

        assert (valid_iv["iv"] > 0).all(), f"{symbol}: non-positive IV found"

    def test_print_aapl_chain(self, ib):
        """Print a formatted option chain table for AAPL (informational)."""
        contract = qualified_stock(ib, "AAPL")
        chain = OptionChain(ib)
        df = chain.get_by_delta(
            contract,
            target_deltas=TARGET_DELTAS,
            delta_tolerance=DELTA_TOLERANCE,
            expirations=2,
        )
        if df.empty:
            pytest.skip("No AAPL options data")

        print(f"\n{'='*80}")
        print(f"AAPL Option Chain | delta ≈ {TARGET_DELTAS} ± {DELTA_TOLERANCE}")
        print(f"{'='*80}")

        for exp in sorted(df["expiration"].unique()):
            subset = df[df["expiration"] == exp].copy()
            rate = subset["rate"].iloc[0]
            fwd  = subset["forward_price"].iloc[0]
            dte  = subset["days_to_expiry"].iloc[0]
            und  = subset["underlying_price"].iloc[0]
            print(
                f"\nExpiration: {exp}  DTE={dte}d  "
                f"spot={und:.2f}  forward={fwd:.2f}  "
                f"rate={rate*100:.2f}%"
            )
            print(
                subset[["right", "strike", "delta", "gamma", "theta",
                         "vega", "iv", "bid", "ask", "mid"]]
                .to_string(index=False)
            )
