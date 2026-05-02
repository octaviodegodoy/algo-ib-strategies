"""
scripts/collect_data.py
=======================
Standalone script to collect and display:
  - Historical OHLCV prices
  - Live price snapshot
  - Account summary
  - Open positions
  - Open orders
  - Trade executions (fills)

Usage:
    python scripts/collect_data.py

Symbols and parameters are configured in the SYMBOLS / CONFIG section below.
"""

import sys
import logging
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ib_async import Stock, Forex, Future
from core.connection import IBConnection
from data.market_data import MarketData
from utils.logger import setup_logging

setup_logging(level=logging.INFO, log_file="logs/collect_data.log")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION — edit here
# =============================================================================

# Contracts to collect historical + snapshot data for
CONTRACTS = [
    Stock("AAPL", "SMART", "USD"),
    Stock("MSFT", "SMART", "USD"),
    # Forex("EURUSD"),          # uncomment to add FX
    # Future("ES", "GLOBEX"),   # uncomment to add futures
]

# Historical bars settings
HIST_DURATION = "30 D"      # e.g. "1 D", "1 W", "1 M", "1 Y"
HIST_BAR_SIZE = "1 day"     # e.g. "1 min", "5 mins", "1 hour", "1 day"
HIST_WHAT     = "TRADES"    # TRADES | MIDPOINT | BID | ASK | BID_ASK
HIST_RTH      = True        # True = regular trading hours only

# =============================================================================


def section(title: str) -> None:
    bar = "=" * 60
    logger.info("\n%s\n  %s\n%s", bar, title, bar)


def main() -> None:
    with IBConnection() as ib:
        md = MarketData(ib)

        # ------------------------------------------------------------------
        # 1. Account summary
        # ------------------------------------------------------------------
        section("ACCOUNT SUMMARY")
        acc_df = md.get_account_summary()
        important_tags = [
            "NetLiquidation", "AvailableFunds", "BuyingPower",
            "TotalCashValue", "UnrealizedPnL", "RealizedPnL",
        ]
        subset = acc_df[acc_df.index.isin(important_tags)]
        logger.info("\n%s", subset.to_string())

        # ------------------------------------------------------------------
        # 2. Open positions
        # ------------------------------------------------------------------
        section("OPEN POSITIONS")
        pos_df = md.get_positions()
        if not pos_df.empty:
            logger.info("\n%s", pos_df.to_string(index=False))

        # ------------------------------------------------------------------
        # 3. Open orders
        # ------------------------------------------------------------------
        section("OPEN ORDERS")
        ord_df = md.get_open_orders()
        if not ord_df.empty:
            logger.info("\n%s", ord_df.to_string())

        # ------------------------------------------------------------------
        # 4. Executions / fills
        # ------------------------------------------------------------------
        section("EXECUTIONS (today)")
        exec_df = md.get_executions()
        if not exec_df.empty:
            logger.info("\n%s", exec_df.to_string())

        # ------------------------------------------------------------------
        # 5. Historical prices + live snapshot per contract
        # ------------------------------------------------------------------
        for contract in CONTRACTS:
            section(f"HISTORICAL BARS — {contract.symbol} ({HIST_DURATION} / {HIST_BAR_SIZE})")
            hist_df = md.get_historical_bars(
                contract,
                duration=HIST_DURATION,
                bar_size=HIST_BAR_SIZE,
                what_to_show=HIST_WHAT,
                use_rth=HIST_RTH,
            )
            if not hist_df.empty:
                logger.info("\n%s", hist_df[["open", "high", "low", "close", "volume"]].tail(10).to_string())

            section(f"LIVE SNAPSHOT — {contract.symbol}")
            snap = md.snapshot(contract)
            logger.info("\n%s", snap.to_string())

        logger.info("Collection complete.")


if __name__ == "__main__":
    main()
