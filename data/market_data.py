import logging
import pandas as pd
from ib_async import IB, Contract, BarDataList, util

logger = logging.getLogger(__name__)


class MarketData:
    """Historical prices, live quotes, executions and account data helpers."""

    def __init__(self, ib: IB) -> None:
        self.ib = ib

    # ------------------------------------------------------------------
    # Historical prices
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        contract: Contract,
        duration: str = "30 D",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """Return OHLCV bars as a DataFrame indexed by date."""
        bars: BarDataList = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
        )
        df = util.df(bars)
        if df is None or df.empty:
            logger.warning("No historical data returned for %s", contract.symbol)
            return pd.DataFrame()
        df = df.set_index("date")
        logger.info(
            "Loaded %d bars for %s (%s / %s)",
            len(df),
            contract.symbol,
            duration,
            bar_size,
        )
        return df

    # ------------------------------------------------------------------
    # Live quotes
    # ------------------------------------------------------------------

    def get_live_ticker(self, contract: Contract):
        """Subscribe to live/delayed market data and return the Ticker object."""
        ticker = self.ib.reqMktData(contract, "", False, False)
        logger.info("Subscribed to live data for %s", contract.symbol)
        return ticker

    def cancel_live_ticker(self, contract: Contract) -> None:
        self.ib.cancelMktData(contract)
        logger.info("Cancelled live data for %s", contract.symbol)

    def snapshot(self, contract: Contract) -> pd.Series:
        """Return a single price snapshot (bid/ask/last) as a pandas Series."""
        ticker = self.ib.reqMktData(contract, "", True, False)   # snapshot=True
        self.ib.sleep(2)
        data = {
            "symbol": contract.symbol,
            "bid": ticker.bid,
            "ask": ticker.ask,
            "last": ticker.last,
            "close": ticker.close,
            "volume": ticker.volume,
        }
        logger.info("Snapshot %s: bid=%.4s ask=%.4s last=%.4s", contract.symbol, ticker.bid, ticker.ask, ticker.last)
        return pd.Series(data)

    # ------------------------------------------------------------------
    # Executions / fills
    # ------------------------------------------------------------------

    def get_executions(self) -> pd.DataFrame:
        """Return today's fills/executions as a DataFrame."""
        fills = self.ib.fills()
        if not fills:
            logger.info("No executions found")
            return pd.DataFrame()
        rows = [
            {
                "time": f.time,
                "symbol": f.contract.symbol,
                "secType": f.contract.secType,
                "side": f.execution.side,
                "shares": f.execution.shares,
                "price": f.execution.price,
                "orderId": f.execution.orderId,
                "execId": f.execution.execId,
                "commission": f.commissionReport.commission if f.commissionReport else None,
                "realizedPnL": f.commissionReport.realizedPNL if f.commissionReport else None,
            }
            for f in fills
        ]
        df = pd.DataFrame(rows).set_index("time")
        logger.info("Loaded %d executions", len(df))
        return df

    # ------------------------------------------------------------------
    # Positions & account
    # ------------------------------------------------------------------

    def get_positions(self) -> pd.DataFrame:
        """Return current open positions as a DataFrame."""
        positions = self.ib.positions()
        if not positions:
            logger.info("No open positions")
            return pd.DataFrame()
        rows = [
            {
                "account": p.account,
                "symbol": p.contract.symbol,
                "secType": p.contract.secType,
                "currency": p.contract.currency,
                "position": p.position,
                "avgCost": p.avgCost,
            }
            for p in positions
        ]
        df = pd.DataFrame(rows)
        logger.info("Loaded %d positions", len(df))
        return df

    def get_account_summary(self, account: str = "") -> pd.DataFrame:
        """Return account summary tags as a DataFrame."""
        acc = account or self.ib.managedAccounts()[0]
        summary = self.ib.accountSummary(acc)
        df = pd.DataFrame(
            [{"tag": s.tag, "value": s.value, "currency": s.currency} for s in summary]
        ).set_index("tag")
        logger.info("Loaded account summary for %s (%d tags)", acc, len(df))
        return df

    # ------------------------------------------------------------------
    # Open orders
    # ------------------------------------------------------------------

    def get_open_orders(self) -> pd.DataFrame:
        """Return currently open orders as a DataFrame."""
        trades = self.ib.openTrades()
        if not trades:
            logger.info("No open orders")
            return pd.DataFrame()
        rows = [
            {
                "orderId": t.order.orderId,
                "symbol": t.contract.symbol,
                "secType": t.contract.secType,
                "action": t.order.action,
                "orderType": t.order.orderType,
                "totalQty": t.order.totalQuantity,
                "lmtPrice": t.order.lmtPrice,
                "auxPrice": t.order.auxPrice,
                "status": t.orderStatus.status,
                "filled": t.orderStatus.filled,
                "remaining": t.orderStatus.remaining,
                "avgFillPrice": t.orderStatus.avgFillPrice,
            }
            for t in trades
        ]
        df = pd.DataFrame(rows).set_index("orderId")
        logger.info("Loaded %d open orders", len(df))
        return df
