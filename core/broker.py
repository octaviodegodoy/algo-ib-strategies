import logging
from ib_async import IB, Contract, Order, Trade, LimitOrder, MarketOrder, StopOrder

logger = logging.getLogger(__name__)


class Broker:
    """Thin wrapper around IB order execution."""

    def __init__(self, ib: IB) -> None:
        self.ib = ib

    def place_market_order(self, contract: Contract, action: str, quantity: float) -> Trade:
        order = MarketOrder(action.upper(), quantity)
        trade = self.ib.placeOrder(contract, order)
        logger.info("Market order placed | %s %s x%.2f", action, contract.symbol, quantity)
        return trade

    def place_limit_order(
        self, contract: Contract, action: str, quantity: float, limit_price: float
    ) -> Trade:
        order = LimitOrder(action.upper(), quantity, limit_price)
        trade = self.ib.placeOrder(contract, order)
        logger.info(
            "Limit order placed | %s %s x%.2f @ %.4f",
            action,
            contract.symbol,
            quantity,
            limit_price,
        )
        return trade

    def place_stop_order(
        self, contract: Contract, action: str, quantity: float, stop_price: float
    ) -> Trade:
        order = StopOrder(action.upper(), quantity, stop_price)
        trade = self.ib.placeOrder(contract, order)
        logger.info(
            "Stop order placed | %s %s x%.2f stop=%.4f",
            action,
            contract.symbol,
            quantity,
            stop_price,
        )
        return trade

    def cancel_order(self, trade: Trade) -> None:
        self.ib.cancelOrder(trade.order)
        logger.info("Order cancelled | orderId=%d", trade.order.orderId)

    def open_trades(self) -> list[Trade]:
        return self.ib.openTrades()

    def positions(self):
        return self.ib.positions()
