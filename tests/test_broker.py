import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
tests/test_broker.py
---------------------
Order placement and lifecycle tests.

WARNING: These tests place REAL orders on whatever account is connected.
         Use a PAPER TRADING account.
         Orders are submitted and then immediately cancelled.
"""

import pytest
from ib_async import IB

from core.broker import Broker

pytestmark = pytest.mark.live

# Safety guard: refuse to run against a live (non-paper) account.
# Paper accounts at IB typically start with "DU".
PAPER_ACCOUNT_PREFIX = "DU"


@pytest.fixture(autouse=True)
def require_paper_account(ib: IB):
    """Skip all broker tests when connected to a live account."""
    account = ib.managedAccounts()[0]
    if not account.startswith(PAPER_ACCOUNT_PREFIX):
        pytest.skip(
            f"Broker tests require a paper account (got '{account}'). "
            "Set PAPER_ACCOUNT_PREFIX in test_broker.py if your paper account differs."
        )


class TestLimitOrders:

    def test_place_and_cancel_limit_order(self, broker: Broker, ib: IB, aapl):
        """Place a far-away limit buy, verify it's open, then cancel."""
        ticker = ib.reqMktData(aapl, "", True, False)
        ib.sleep(2)
        last = ticker.last or ticker.close or 100.0
        limit_price = round(last * 0.80, 2)   # 20% below market → won't fill

        trade = broker.place_limit_order(aapl, "BUY", 1, limit_price)
        ib.sleep(1)

        assert trade is not None, "placeOrder returned None"
        assert trade.order.orderId > 0

        broker.cancel_order(trade)
        ib.sleep(1)

        assert trade.orderStatus.status in (
            "Cancelled", "Inactive", "ApiCancelled"
        ), f"Unexpected status after cancel: {trade.orderStatus.status}"

    def test_place_and_cancel_limit_sell(self, broker: Broker, ib: IB, aapl):
        """Place a far-away limit sell, verify it opens, then cancel."""
        ticker = ib.reqMktData(aapl, "", True, False)
        ib.sleep(2)
        last = ticker.last or ticker.close or 100.0
        limit_price = round(last * 1.20, 2)   # 20% above market → won't fill

        trade = broker.place_limit_order(aapl, "SELL", 1, limit_price)
        ib.sleep(1)

        assert trade is not None
        broker.cancel_order(trade)
        ib.sleep(1)

        assert trade.orderStatus.status in ("Cancelled", "Inactive", "ApiCancelled")


class TestStopOrders:

    def test_place_and_cancel_stop_order(self, broker: Broker, ib: IB):
        """Place a stop order well outside market, then cancel."""
        ticker = ib.reqMktData(MSFT, "", True, False)
        ib.sleep(2)
        last = ticker.last or ticker.close or 100.0
        stop_price = round(last * 0.70, 2)   # 30% below market

        trade = broker.place_stop_order(MSFT, "SELL", 1, stop_price)
        ib.sleep(1)

        assert trade is not None
        broker.cancel_order(trade)
        ib.sleep(1)

        assert trade.orderStatus.status in ("Cancelled", "Inactive", "ApiCancelled")


class TestOpenOrders:

    def test_open_trades_is_list(self, broker: Broker):
        trades = broker.open_trades()
        assert isinstance(trades, list)

    def test_positions_is_list(self, broker: Broker):
        positions = broker.positions()
        assert isinstance(positions, list)
