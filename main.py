"""
main.py — entry point for algo-ib-strategies.

Usage:
    python main.py

Make sure TWS or IB Gateway is running with API access enabled,
and that your .env file is configured (copy from .env.example).
"""

import logging
from ib_async import Stock
from core.connection import IBConnection
from data.market_data import MarketData
from utils.logger import setup_logging

setup_logging(level=logging.INFO, log_file="logs/algo.log")
logger = logging.getLogger(__name__)


def main() -> None:
    with IBConnection() as ib:
        # ------------------------------------------------------------------
        # Quick sanity-check: print account summary
        # ------------------------------------------------------------------
        account = ib.managedAccounts()[0]
        summary = {s.tag: s.value for s in ib.accountSummary(account)}
        logger.info("Account: %s", account)
        logger.info("Net Liquidation: %s", summary.get("NetLiquidation", "N/A"))
        logger.info("Available Funds:  %s", summary.get("AvailableFunds", "N/A"))

        # ------------------------------------------------------------------
        # Fetch AAPL historical daily bars (last 30 days)
        # ------------------------------------------------------------------
        contract = Stock("AAPL", "SMART", "USD")
        md = MarketData(ib)
        df = md.get_historical_bars(
            contract,
            duration="30 D",
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=True,
        )

        if not df.empty:
            logger.info("AAPL — last 5 daily bars:\n%s", df[["open", "high", "low", "close", "volume"]].tail())
        else:
            logger.warning("No data returned for AAPL. Check your market data subscription.")

        logger.info("Done.")


if __name__ == "__main__":
    main()
