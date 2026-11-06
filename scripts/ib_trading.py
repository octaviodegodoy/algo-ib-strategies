"""
scripts/ib_trading.py
----------------------
Self-contained Interactive Brokers trading functions.

Covers:
  - Connection management
  - Contract helpers (STK, FUT, OPT, FOREX, CFD)
  - Order placement (Market, Limit, Stop, Stop-Limit, Bracket, OCA)
  - Order management (cancel, modify, list open orders)
  - Historical bars (OHLCV)
  - Live quotes / snapshots
  - Account summary & positions
  - Executions / fills
  - Portfolio PnL

Usage (standalone)
------------------
    python scripts/ib_trading.py

or import individual functions:

    from scripts.ib_trading import connect, get_historical_bars, place_limit_order
"""

from __future__ import annotations

import contextlib
import logging
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
from ib_async import (
    IB,
    BarDataList,
    BracketOrder,
    Contract,
    ContFuture,
    Forex,
    Future,
    Index,
    LimitOrder,
    MarketOrder,
    Option,
    Order,
    Stock,
    StopLimitOrder,
    StopOrder,
    Trade,
    util,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ib_trading")

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

_ib: Optional[IB] = None


def connect(
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 1,
    market_data_type: int = 3,
) -> IB:
    """
    Connect to TWS / IB Gateway and return the IB instance.

    port:
        7497 → TWS paper trading
        7496 → TWS live trading
        4002 → Gateway paper trading  (default)
        4001 → Gateway live trading

    market_data_type:
        1 = Live  (requires data subscription)
        2 = Frozen
        3 = Delayed free  (15-20 min)
        4 = Delayed frozen
    """
    global _ib
    if _ib is not None and _ib.isConnected():
        return _ib

    ib = IB()
    try:
        ib.connect(host=host, port=port, clientId=client_id)
    except ConnectionRefusedError:
        logger.error(
            "Cannot connect to IB at %s:%d — TWS/Gateway not running or API disabled.",
            host,
            port,
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected connection error: %s", exc)
        sys.exit(1)

    ib.reqMarketDataType(market_data_type)
    account = ib.managedAccounts()[0]
    logger.info(
        "Connected  host=%s  port=%d  clientId=%d  account=%s",
        host, port, client_id, account,
    )
    _ib = ib
    return ib


def disconnect(ib: Optional[IB] = None) -> None:
    """Disconnect from IB."""
    global _ib
    target = ib or _ib
    if target and target.isConnected():
        target.disconnect()
        logger.info("Disconnected from IB")
    _ib = None


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------

def make_stock(symbol: str, exchange: str = "CBOE", currency: str = "USD") -> Stock:
    """Create and qualify a stock contract."""
    return Stock(symbol, exchange, currency)


def make_future(
    symbol: str,
    last_trade_date: str,
    exchange: str,
    currency: str = "USD",
    multiplier: str = "",
) -> Future:
    """
    Create a futures contract.

    last_trade_date: 'YYYYMM' or 'YYYYMMDD'
    """
    return Future(
        symbol=symbol,
        lastTradeDateOrContractMonth=last_trade_date,
        exchange=exchange,
        currency=currency,
        multiplier=multiplier,
    )


def make_cont_future(symbol: str, exchange: str, currency: str = "USD") -> ContFuture:
    """Create a continuous futures contract (rolls automatically)."""
    return ContFuture(symbol=symbol, exchange=exchange, currency=currency)


def make_option(
    symbol: str,
    last_trade_date: str,
    strike: float,
    right: str,
    exchange: str = "SMART",
    currency: str = "USD",
    multiplier: str = "100",
) -> Option:
    """
    Create an equity option contract.

    right: 'C' (call) or 'P' (put)
    last_trade_date: 'YYYYMMDD'
    """
    return Option(
        symbol=symbol,
        lastTradeDateOrContractMonth=last_trade_date,
        strike=strike,
        right=right,
        exchange=exchange,
        currency=currency,
        multiplier=multiplier,
    )


def make_forex(pair: str) -> Forex:
    """
    Create a Forex contract.

    pair: e.g. 'EURUSD', 'USDBRL'
    """
    return Forex(pair)


def make_index(
    symbol: str,
    exchange: str = "CBOE",
    currency: str = "USD",
) -> Index:
    """
    Create an Index contract (secType='IND').

    Use this for cash-settled index underlyings whose options live on a single
    exchange — most importantly:

        SPX  → exchange='CBOE'  (S&P 500, multiplier 100, AM/PM settled)
        NDX  → exchange='NASDAQ'
        RUT  → exchange='RUSSELL'
        VIX  → exchange='CBOE'

    Index contracts must be used (not Stock) so that
    reqSecDefOptParams returns the correct chain with CBOE OI.
    """
    return Index(symbol, exchange, currency)


def qualify(ib: IB, *contracts: Contract) -> list[Contract]:
    """Qualify one or more contracts (resolves conId and trading class)."""
    qualified = ib.qualifyContracts(*contracts)
    for c in qualified:
        logger.info("Qualified: %s  conId=%s  exchange=%s", c.symbol, c.conId, c.exchange)
    return qualified


# ---------------------------------------------------------------------------
# Historical data
# ---------------------------------------------------------------------------

def get_historical_bars(
    ib: IB,
    contract: Contract,
    duration: str = "30 D",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    end_datetime: str = "",
) -> pd.DataFrame:
    """
    Fetch OHLCV bars and return a DataFrame indexed by date.

    duration examples : '1 D', '5 D', '1 W', '1 M', '1 Y'
    bar_size examples : '1 secs', '5 secs', '30 secs', '1 min', '5 mins',
                        '15 mins', '30 mins', '1 hour', '1 day', '1 week'
    what_to_show      : TRADES | MIDPOINT | BID | ASK | BID_ASK |
                        ADJUSTED_LAST | HISTORICAL_VOLATILITY | OPTION_IMPLIED_VOLATILITY
    """
    bars: BarDataList = ib.reqHistoricalData(
        contract,
        endDateTime=end_datetime,
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
    )
    df = util.df(bars)
    if df is None or df.empty:
        logger.warning("No historical data for %s", contract.symbol)
        return pd.DataFrame()
    df = df.set_index("date")
    logger.info(
        "Historical bars  %s  %s/%s  rows=%d",
        contract.symbol, duration, bar_size, len(df),
    )
    return df


def get_intraday_bars(
    ib: IB,
    contract: Contract,
    duration: str = "1 D",
    bar_size: str = "5 mins",
) -> pd.DataFrame:
    """Convenience wrapper for intraday OHLCV bars."""
    return get_historical_bars(
        ib, contract, duration=duration, bar_size=bar_size, use_rth=False
    )


# ---------------------------------------------------------------------------
# Live market data
# ---------------------------------------------------------------------------

def get_snapshot(ib: IB, contract: Contract) -> pd.Series:
    """
    Request a single delayed/live quote snapshot.
    Returns a Series with bid, ask, last, close, volume.
    """
    ticker = ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
    ib.sleep(2)
    data = {
        "symbol": contract.symbol,
        "bid":    ticker.bid,
        "ask":    ticker.ask,
        "last":   ticker.last,
        "close":  ticker.close,
        "volume": ticker.volume,
        "time":   datetime.now(),
    }
    logger.info(
        "Snapshot %s  bid=%.4s  ask=%.4s  last=%.4s",
        contract.symbol, ticker.bid, ticker.ask, ticker.last,
    )
    return pd.Series(data)


def subscribe_live(ib: IB, contract: Contract):
    """
    Subscribe to streaming market data.
    Returns the Ticker object — attach callbacks via ib.pendingTickersEvent.
    """
    ticker = ib.reqMktData(contract, genericTickList="", snapshot=False, regulatorySnapshot=False)
    logger.info("Subscribed to live data: %s", contract.symbol)
    return ticker


def unsubscribe_live(ib: IB, contract: Contract) -> None:
    """Cancel streaming market data subscription."""
    ib.cancelMktData(contract)
    logger.info("Unsubscribed from live data: %s", contract.symbol)


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------

def place_market_order(
    ib: IB,
    contract: Contract,
    action: str,
    quantity: float,
) -> Trade:
    """
    Place a market order.

    action: 'BUY' or 'SELL'
    """
    order = MarketOrder(action.upper(), quantity)
    trade = ib.placeOrder(contract, order)
    logger.info("Market order  %s %s x%.2f", action.upper(), contract.symbol, quantity)
    return trade


def place_limit_order(
    ib: IB,
    contract: Contract,
    action: str,
    quantity: float,
    limit_price: float,
    tif: str = "GTC",
) -> Trade:
    """
    Place a limit order.

    tif: 'DAY' | 'GTC' | 'IOC' | 'FOK'
    """
    order = LimitOrder(action.upper(), quantity, limit_price, tif=tif)
    trade = ib.placeOrder(contract, order)
    logger.info(
        "Limit order  %s %s x%.2f @ %.4f  tif=%s",
        action.upper(), contract.symbol, quantity, limit_price, tif,
    )
    return trade


def place_stop_order(
    ib: IB,
    contract: Contract,
    action: str,
    quantity: float,
    stop_price: float,
    tif: str = "GTC",
) -> Trade:
    """Place a stop (stop-market) order."""
    order = StopOrder(action.upper(), quantity, stop_price, tif=tif)
    trade = ib.placeOrder(contract, order)
    logger.info(
        "Stop order  %s %s x%.2f  stop=%.4f  tif=%s",
        action.upper(), contract.symbol, quantity, stop_price, tif,
    )
    return trade


def place_stop_limit_order(
    ib: IB,
    contract: Contract,
    action: str,
    quantity: float,
    stop_price: float,
    limit_price: float,
    tif: str = "GTC",
) -> Trade:
    """Place a stop-limit order."""
    order = StopLimitOrder(action.upper(), quantity, limit_price, stop_price, tif=tif)
    trade = ib.placeOrder(contract, order)
    logger.info(
        "StopLimit order  %s %s x%.2f  stop=%.4f  limit=%.4f  tif=%s",
        action.upper(), contract.symbol, quantity, stop_price, limit_price, tif,
    )
    return trade


def place_bracket_order(
    ib: IB,
    contract: Contract,
    action: str,
    quantity: float,
    entry_price: float,
    take_profit_price: float,
    stop_loss_price: float,
) -> list[Trade]:
    """
    Place a bracket order (entry limit + take profit + stop loss).
    Returns a list of three Trades: [parent, take_profit, stop_loss].
    """
    bracket: BracketOrder = ib.bracketOrder(
        action.upper(),
        quantity,
        entry_price,
        take_profit_price,
        stop_loss_price,
    )
    trades = [ib.placeOrder(contract, o) for o in bracket]
    logger.info(
        "Bracket order  %s %s x%.2f  entry=%.4f  tp=%.4f  sl=%.4f",
        action.upper(), contract.symbol, quantity,
        entry_price, take_profit_price, stop_loss_price,
    )
    return trades


def place_oca_orders(
    ib: IB,
    contract: Contract,
    orders: list[Order],
    oca_group: str,
    oca_type: int = 1,
) -> list[Trade]:
    """
    Attach OCA (One-Cancels-All) group to a list of orders and submit them.

    oca_type:
        1 = Cancel all remaining with block
        2 = Reduce on fill with block
        3 = Reduce on fill without block
    """
    ib.oneCancelsAll(orders, ocaGroup=oca_group, ocaType=oca_type)
    trades = [ib.placeOrder(contract, o) for o in orders]
    logger.info("OCA group '%s'  %d orders placed", oca_group, len(trades))
    return trades


# ---------------------------------------------------------------------------
# Order management
# ---------------------------------------------------------------------------

def cancel_order(ib: IB, trade: Trade) -> None:
    """Cancel a specific open order by its Trade object."""
    ib.cancelOrder(trade.order)
    logger.info("Cancelled order  orderId=%d", trade.order.orderId)


def cancel_all_orders(ib: IB) -> None:
    """Cancel ALL open orders for the connected account."""
    ib.reqGlobalCancel()
    logger.info("Global cancel sent — all open orders cancelled")


def modify_order(
    ib: IB,
    trade: Trade,
    new_limit_price: Optional[float] = None,
    new_quantity: Optional[float] = None,
    new_stop_price: Optional[float] = None,
) -> Trade:
    """
    Modify an existing open order.
    Only fields provided (not None) are updated.
    """
    order = trade.order
    if new_limit_price is not None:
        order.lmtPrice = new_limit_price
    if new_quantity is not None:
        order.totalQuantity = new_quantity
    if new_stop_price is not None:
        order.auxPrice = new_stop_price
    updated_trade = ib.placeOrder(trade.contract, order)
    logger.info("Modified order  orderId=%d", order.orderId)
    return updated_trade


def get_open_orders(ib: IB) -> pd.DataFrame:
    """Return all open orders as a DataFrame."""
    trades = ib.openTrades()
    if not trades:
        logger.info("No open orders")
        return pd.DataFrame()
    rows = [
        {
            "orderId":      t.order.orderId,
            "symbol":       t.contract.symbol,
            "secType":      t.contract.secType,
            "action":       t.order.action,
            "orderType":    t.order.orderType,
            "totalQty":     t.order.totalQuantity,
            "lmtPrice":     t.order.lmtPrice,
            "auxPrice":     t.order.auxPrice,
            "tif":          t.order.tif,
            "status":       t.orderStatus.status,
            "filled":       t.orderStatus.filled,
            "remaining":    t.orderStatus.remaining,
            "avgFillPrice": t.orderStatus.avgFillPrice,
        }
        for t in trades
    ]
    df = pd.DataFrame(rows).set_index("orderId")
    logger.info("Open orders: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Positions & account
# ---------------------------------------------------------------------------

def get_positions(ib: IB) -> pd.DataFrame:
    """Return all current open positions."""
    positions = ib.positions()
    if not positions:
        logger.info("No open positions")
        return pd.DataFrame()
    rows = [
        {
            "account":  p.account,
            "symbol":   p.contract.symbol,
            "secType":  p.contract.secType,
            "exchange":  p.contract.exchange,
            "currency": p.contract.currency,
            "position": p.position,
            "avgCost":  p.avgCost,
        }
        for p in positions
    ]
    df = pd.DataFrame(rows)
    logger.info("Positions: %d", len(df))
    return df


def get_account_summary(ib: IB, tags: str = "NetLiquidation,TotalCashValue,UnrealizedPnL,RealizedPnL,BuyingPower") -> pd.DataFrame:
    """
    Return key account metrics as a DataFrame.

    tags: comma-separated TWS account tag names.
    Common tags: NetLiquidation, TotalCashValue, AvailableFunds,
                 BuyingPower, UnrealizedPnL, RealizedPnL, GrossPositionValue.
    """
    summary = ib.accountSummary()
    if not summary:
        return pd.DataFrame()
    tag_list = [t.strip() for t in tags.split(",")]
    rows = [
        {"account": v.account, "tag": v.tag, "value": v.value, "currency": v.currency}
        for v in summary
        if v.tag in tag_list
    ]
    df = pd.DataFrame(rows)
    logger.info("Account summary: %d entries", len(df))
    return df


def get_portfolio_pnl(ib: IB) -> pd.DataFrame:
    """Return the portfolio entries with unrealized and realized PnL."""
    items = ib.portfolio()
    if not items:
        logger.info("Portfolio is empty")
        return pd.DataFrame()
    rows = [
        {
            "symbol":        i.contract.symbol,
            "secType":       i.contract.secType,
            "position":      i.position,
            "marketPrice":   i.marketPrice,
            "marketValue":   i.marketValue,
            "avgCost":       i.averageCost,
            "unrealizedPnL": i.unrealizedPNL,
            "realizedPnL":   i.realizedPNL,
        }
        for i in items
    ]
    df = pd.DataFrame(rows)
    logger.info("Portfolio PnL loaded: %d positions", len(df))
    return df


# ---------------------------------------------------------------------------
# Options — expiration dates & chain
# ---------------------------------------------------------------------------

def get_next_expirations(
    ib: IB,
    contract: Contract,
    n: int = 2,
    preferred_exchange: str = "",
) -> list[str]:
    """
    Return the next *n* expiration dates for the option chain of *contract*.

    The contract must already be qualified (has a valid conId).
    Dates are returned as strings in 'YYYYMMDD' format, sorted ascending.

    Parameters
    ----------
    contract           : qualified underlying (Stock, Index, Future, etc.)
    n                  : number of front expirations to return (default 2)
    preferred_exchange : pin to a specific exchange (e.g. 'CBOE').
                         If empty, picks the exchange with the most strikes.
    """
    chains = ib.reqSecDefOptParams(
        underlyingSymbol=underlying_symbol(contract),
        futFopExchange="",
        underlyingSecType=contract.secType,
        underlyingConId=contract.conId,
    )
    if not chains:
        logger.warning("No option chain params for %s", contract.symbol)
        return []

    chain = _pick_chain(chains, preferred_exchange)
    if chain is None:
        return []

    today = datetime.today().strftime("%Y%m%d")
    future_exps = sorted(e for e in chain.expirations if e >= today)
    selected = future_exps[:n]

    logger.info(
        "Next %d expirations for %s (exchange=%s): %s",
        n, contract.symbol, chain.exchange, selected,
    )
    return selected


def get_option_chain(
    ib: IB,
    contract: Contract,
    expirations: int = 2,
    strikes_around_atm: int = 10,
    rights: list[str] | None = None,
    preferred_exchange: str = "",
) -> pd.DataFrame:
    """
    Fetch the option chain for the next *expirations* expiry dates.

    Returns a DataFrame with columns:
        expiration, strike, right, bid, ask, mid, last,
        delta, gamma, theta, vega, iv, days_to_expiry, conId

    Parameters
    ----------
    contract           : qualified underlying
    expirations        : number of front expiry dates (default 2)
    strikes_around_atm : strikes on each side of ATM to include (default 10)
    rights             : ['C', 'P'] by default
    preferred_exchange : pin chain to a specific exchange
    """
    if rights is None:
        rights = ["C", "P"]

    # -- 1. underlying price --------------------------------------------------
    ticker = ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
    ib.sleep(2)
    und_price = ticker.last or ticker.close or 0.0
    if und_price <= 0:
        logger.error("Could not get underlying price for %s", contract.symbol)
        return pd.DataFrame()
    logger.info("%s price: %.2f", contract.symbol, und_price)

    # -- 2. chain params (strikes + expirations) ------------------------------
    chains = ib.reqSecDefOptParams(
        underlyingSymbol=underlying_symbol(contract),
        futFopExchange="",
        underlyingSecType=contract.secType,
        underlyingConId=contract.conId,
    )
    if not chains:
        logger.error("No option chain params for %s", contract.symbol)
        return pd.DataFrame()

    chain = _pick_chain(chains, preferred_exchange)
    if chain is None:
        logger.error("Could not select a chain for %s", contract.symbol)
        return pd.DataFrame()

    today = datetime.today().strftime("%Y%m%d")
    future_exps = sorted(e for e in chain.expirations if e >= today)
    selected_exps = future_exps[:expirations]
    if not selected_exps:
        logger.error("No future expirations found for %s", contract.symbol)
        return pd.DataFrame()

    # -- 3. strikes near ATM --------------------------------------------------
    sorted_strikes = sorted(chain.strikes)
    atm_idx = min(range(len(sorted_strikes)),
                  key=lambda i: abs(sorted_strikes[i] - und_price))
    lo = max(0, atm_idx - strikes_around_atm)
    hi = min(len(sorted_strikes), atm_idx + strikes_around_atm + 1)
    selected_strikes = sorted_strikes[lo:hi]

    logger.info(
        "Chain %s  exchange=%s  expirations=%s  strikes=%d–%.0f",
        contract.symbol, chain.exchange, selected_exps,
        len(selected_strikes), und_price,
    )

    # -- 4. build + qualify contracts -----------------------------------------
    option_contracts = [
        Option(underlying_symbol(contract), exp, strike, right, chain.exchange)
        for exp in selected_exps
        for strike in selected_strikes
        for right in rights
    ]
    with _suppress_ib_errors():
        qualified = ib.qualifyContracts(*option_contracts)
    qualified = [c for c in qualified if c is not None and c.conId]
    if not qualified:
        logger.error("No contracts qualified for %s", contract.symbol)
        return pd.DataFrame()

    # -- 5. request market data (delayed snapshot per contract) ---------------
    tickers = [
        ib.reqMktData(c, genericTickList="", snapshot=True, regulatorySnapshot=False)
        for c in qualified
    ]
    ib.sleep(3)

    today_dt = datetime.today()
    rows = []
    for c, t in zip(qualified, tickers):
        exp_dt = datetime.strptime(c.lastTradeDateOrContractMonth, "%Y%m%d")
        dte = max(0, (exp_dt - today_dt).days)
        bid = t.bid if t.bid and t.bid > 0 else float("nan")
        ask = t.ask if t.ask and t.ask > 0 else float("nan")
        mid = round((bid + ask) / 2, 4) if (bid == bid and ask == ask) else float("nan")
        greeks = t.modelGreeks or t.bidGreeks or None
        rows.append({
            "expiration":   c.lastTradeDateOrContractMonth,
            "strike":       c.strike,
            "right":        c.right,
            "bid":          bid,
            "ask":          ask,
            "mid":          mid,
            "last":         t.last if t.last and t.last > 0 else float("nan"),
            "delta":        greeks.delta if greeks else float("nan"),
            "gamma":        greeks.gamma if greeks else float("nan"),
            "theta":        greeks.theta if greeks else float("nan"),
            "vega":         greeks.vega  if greeks else float("nan"),
            "iv":           greeks.impliedVol if greeks else float("nan"),
            "days_to_expiry": dte,
            "conId":        c.conId,
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Option chain for %s: %d rows across %d expirations",
        contract.symbol, len(df), len(selected_exps),
    )
    return df


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def underlying_symbol(contract: Contract) -> str:
    """Return the symbol suitable for reqSecDefOptParams (strips spaces)."""
    return (contract.symbol or "").strip()


@contextlib.contextmanager
def _suppress_ib_errors():
    """Silence expected IB error-200 / error-10091 log noise during chain qualification."""
    wrapper_log = logging.getLogger("ib_async.wrapper")
    prev = wrapper_log.level
    wrapper_log.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        wrapper_log.setLevel(prev)


# Exchange preference order for option chain selection.
# Exchanges lower in this list (or absent) are preferred last.
# NASDAQOM / ISE / MIAX / BOX list many phantom strikes that fail qualification;
# SMART, CBOE and AMEX publish only contracts that actually trade.
_EXCHANGE_RANK: dict[str, int] = {
    "SMART":    0,
    "CBOE":     1,
    "CBOE2":    2,
    "AMEX":     3,
    "PHLX":     4,
    "ARCA":     5,
    "C2":       6,
    "GEMINI":   7,
    "ISE":      8,
    "MIAX":     9,
    "BOX":     10,
    "BATS":    11,
    "NASDAQOM": 12,   # often lists phantom strikes — use last
    "EDGX":    13,
}


def _pick_chain(chains: list, preferred_exchange: str = ""):
    """
    Select the best option-chain parameter set from the list returned by
    reqSecDefOptParams.

    Priority:
    1. If *preferred_exchange* is given, use that exchange (exact match).
    2. Otherwise rank by _EXCHANGE_RANK (SMART > CBOE > AMEX > … > NASDAQOM).
       Ties in rank are broken by the number of strikes (more = better).
    """
    if not chains:
        return None
    if preferred_exchange:
        matching = [c for c in chains if c.exchange == preferred_exchange]
        if matching:
            return max(matching, key=lambda c: len(c.strikes))
    # rank then fall back to strike count
    return min(
        chains,
        key=lambda c: (_EXCHANGE_RANK.get(c.exchange, 99), -len(c.strikes)),
    )


# ---------------------------------------------------------------------------
# Open interest
# ---------------------------------------------------------------------------

def get_open_interest(
    ib: IB,
    contract: Contract,
    expirations: int = 2,
    strikes_pct_range: float = 0.12,
    preferred_exchange: str = "",
) -> pd.DataFrame:
    """
    Fetch open interest (and daily volume) for the next *expirations* expiry
    dates, returning one row per (expiration, strike, right).

    IB sends OI via generic tick 101 on the streaming feed.
    With delayed data (MARKET_DATA_TYPE=3) OI is often zero; in that case
    daily volume (tick 100) is returned as a proxy in the ``volume`` column.

    Columns returned
    ----------------
    expiration, strike, right, open_interest, volume, bid, ask, mid,
    iv, delta, gamma, days_to_expiry, conId
    """
    # -- 1. underlying price --------------------------------------------------
    und_ticker = ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
    ib.sleep(2)
    und_price = float(und_ticker.last or und_ticker.close or 0.0)
    # IB auto-cancels snapshots server-side — no explicit cancelMktData needed
    if und_price <= 0:
        logger.error("Cannot get price for %s", contract.symbol)
        return pd.DataFrame()

    # -- 2. chain params ------------------------------------------------------
    chains = ib.reqSecDefOptParams(
        underlyingSymbol=underlying_symbol(contract),
        futFopExchange="",
        underlyingSecType=contract.secType,
        underlyingConId=contract.conId,
    )
    if not chains:
        logger.error("No chain params for %s", contract.symbol)
        return pd.DataFrame()

    chain = _pick_chain(chains, preferred_exchange)
    if chain is None:
        logger.error("Could not select a chain for %s", contract.symbol)
        return pd.DataFrame()

    today = datetime.today().strftime("%Y%m%d")
    future_exps = sorted(e for e in chain.expirations if e >= today)
    selected_exps = future_exps[:expirations]
    if not selected_exps:
        logger.error("No future expirations for %s", contract.symbol)
        return pd.DataFrame()

    lo = und_price * (1 - strikes_pct_range)
    hi = und_price * (1 + strikes_pct_range)
    selected_strikes = [s for s in sorted(chain.strikes) if lo <= s <= hi]

    logger.info(
        "%s OI request  exchange=%s  expirations=%s  strikes=%d (%.1f – %.1f)",
        contract.symbol, chain.exchange, selected_exps,
        len(selected_strikes), lo, hi,
    )

    # -- 3. build + qualify ---------------------------------------------------
    option_contracts = [
        Option(underlying_symbol(contract), exp, strike, right, chain.exchange)
        for exp in selected_exps
        for strike in selected_strikes
        for right in ["C", "P"]
    ]
    with _suppress_ib_errors():
        qualified = ib.qualifyContracts(*option_contracts)
    qualified = [c for c in qualified if c is not None and c.conId]
    if not qualified:
        logger.error("No contracts qualified for %s OI", contract.symbol)
        return pd.DataFrame()

    # -- 4. stream with OI + volume ticks (100=volume, 101=OI, 104=IV, 106=greeks) --
    BATCH = 90
    rows = []
    today_dt = datetime.today()

    with _suppress_ib_errors():
        for start in range(0, len(qualified), BATCH):
            batch = qualified[start:start + BATCH]
            tickers = [
                ib.reqMktData(c, "100,101,104,106", snapshot=False, regulatorySnapshot=False)
                for c in batch
            ]
            ib.sleep(4)
            for c, t in zip(batch, tickers):
                exp_dt = datetime.strptime(c.lastTradeDateOrContractMonth, "%Y%m%d")
                dte = max(0, (exp_dt - today_dt).days)
                greeks = t.modelGreeks or t.lastGreeks
                bid = t.bid if t.bid and t.bid > 0 else float("nan")
                ask = t.ask if t.ask and t.ask > 0 else float("nan")
                mid = round((bid + ask) / 2, 4) if bid == bid and ask == ask else float("nan")
                oi = int(t.openInterest) if t.openInterest and t.openInterest > 0 else 0
                vol = int(t.volume) if t.volume and t.volume > 0 else 0
                rows.append({
                    "expiration":    c.lastTradeDateOrContractMonth,
                    "strike":        c.strike,
                    "right":         c.right,
                    "open_interest": oi,
                    "volume":        vol,
                    "bid":           bid,
                    "ask":           ask,
                    "mid":           mid,
                    "iv":            round(greeks.impliedVol or 0, 4) if greeks else float("nan"),
                    "delta":         round(greeks.delta or 0, 4) if greeks else float("nan"),
                    "gamma":         round(greeks.gamma or 0, 6) if greeks else float("nan"),
                    "days_to_expiry": dte,
                    "conId":         c.conId,
                })
            with _suppress_ib_errors():
                for c in batch:
                    try:
                        ib.cancelMktData(c)
                    except Exception:
                        pass
                ib.sleep(0.5)  # absorb async error-300 responses while suppressed

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["expiration", "right", "strike"]).reset_index(drop=True)

    # summary per expiration
    summary = (
        df.groupby(["expiration", "right"])[["open_interest", "volume"]]
        .sum()
        .rename(columns={"open_interest": "total_oi", "volume": "total_volume"})
    )
    logger.info("OI summary for %s:\n%s", contract.symbol, summary.to_string())
    return df


# ---------------------------------------------------------------------------
# GEX for next N expirations
# ---------------------------------------------------------------------------

def get_gex_for_expirations(
    ib: IB,
    contract: Contract,
    expirations: int = 2,
    strikes_pct_range: float = 0.12,
    multiplier: int = 100,
    div_yield: float = 0.0,
) -> dict:
    """
    Compute Gamma Exposure (GEX) for the next *expirations* expiry dates.

    Uses IB's modelGreeks.gamma when available; falls back to Black-Scholes
    computed from the mid price.  When OI is zero (common with delayed data)
    daily volume is used as a proxy automatically.

    Parameters
    ----------
    contract          : qualified underlying (Stock, Index, …)
    expirations       : number of front expiry dates (default 2)
    strikes_pct_range : ±fraction of spot included (default 0.12 = ±12 %)
    multiplier        : option contract size (100 for US equity, 50 for SPX, etc.)
    div_yield         : continuous dividend yield for BS fallback (default 0.0)

    Returns
    -------
    dict with keys:
        symbol          : str
        spot            : float
        expirations     : list[str]   — YYYYMMDD dates used
        gex_by_expiry   : DataFrame   — expiration | net_gex | call_gex | put_gex
        oi_by_expiry    : DataFrame   — expiration | right | total_oi | total_volume
        strike_profile  : DataFrame   — strike | call_gex | put_gex | net_gex
        zero_gamma      : float       — zero-gamma price level
        call_wall       : float       — strike with highest call GEX
        put_wall        : float       — strike with highest abs put GEX
        total_gex       : float
        dealer_gex      : float       — −total_gex (dealers short calls / long puts)
        data            : DataFrame   — full row-level data with all GEX columns
    """
    import math

    # ── risk-free rate (FRED DTB3) ────────────────────────────────────────────
    try:
        from data.rates import RiskFreeRate
        rate = RiskFreeRate().get_rate()
    except Exception:
        rate = 0.05
        logger.warning("Could not fetch risk-free rate — using %.0f %%", rate * 100)

    # ── underlying price ──────────────────────────────────────────────────────
    und_ticker = ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
    ib.sleep(2)
    spot = float(und_ticker.last or und_ticker.close or 0.0)
    # IB auto-cancels snapshots server-side — no explicit cancelMktData needed
    if spot <= 0:
        logger.error("Cannot get spot price for %s", contract.symbol)
        return {}

    # ── chain params ──────────────────────────────────────────────────────────
    chains = ib.reqSecDefOptParams(
        underlyingSymbol=underlying_symbol(contract),
        futFopExchange="",
        underlyingSecType=contract.secType,
        underlyingConId=contract.conId,
    )
    if not chains:
        logger.error("No chain params for %s", contract.symbol)
        return {}

    chain = _pick_chain(chains)
    if chain is None:
        logger.error("Could not select a chain for %s", contract.symbol)
        return {}
    today = datetime.today().strftime("%Y%m%d")
    future_exps = sorted(e for e in chain.expirations if e >= today)
    selected_exps = future_exps[:expirations]
    if not selected_exps:
        logger.error("No future expirations for %s", contract.symbol)
        return {}

    lo = spot * (1 - strikes_pct_range)
    hi = spot * (1 + strikes_pct_range)
    selected_strikes = [s for s in sorted(chain.strikes) if lo <= s <= hi]

    logger.info(
        "GEX request  %s  exchange=%s  expirations=%s  strikes=%d  spot=%.2f",
        contract.symbol, chain.exchange, selected_exps, len(selected_strikes), spot,
    )

    # ── build + qualify ───────────────────────────────────────────────────────
    option_contracts = [
        Option(underlying_symbol(contract), exp, strike, right, chain.exchange)
        for exp in selected_exps
        for strike in selected_strikes
        for right in ["C", "P"]
    ]
    with _suppress_ib_errors():
        qualified = ib.qualifyContracts(*option_contracts)
    qualified = [c for c in qualified if c is not None and c.conId]
    if not qualified:
        logger.error("No contracts qualified for %s GEX", contract.symbol)
        return {}

    # ── stream market data (OI + greeks) ─────────────────────────────────────
    BATCH, SLEEP = 90, 4
    rows = []
    today_dt = datetime.today()

    def _bs_gamma(S, K, T, r, q, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        sqrt2pi = math.sqrt(2.0 * math.pi)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        return math.exp(-q * T) * math.exp(-0.5 * d1 * d1) / sqrt2pi / (S * sigma * math.sqrt(T))

    with _suppress_ib_errors():
        for start in range(0, len(qualified), BATCH):
            batch = qualified[start:start + BATCH]
            tickers = [
                ib.reqMktData(c, "100,101,104,106", snapshot=False, regulatorySnapshot=False)
                for c in batch
            ]
            ib.sleep(SLEEP)
            for c, t in zip(batch, tickers):
                exp_dt = datetime.strptime(c.lastTradeDateOrContractMonth, "%Y%m%d")
                dte = max(0, (exp_dt - today_dt).days)
                T = dte / 365.0
                greeks = t.modelGreeks or t.lastGreeks
                bid = t.bid if t.bid and t.bid > 0 else 0.0
                ask = t.ask if t.ask and t.ask > 0 else 0.0
                mid = (bid + ask) / 2 if bid and ask else 0.0
                ib_gamma = float(greeks.gamma or 0) if greeks and greeks.gamma else 0.0
                iv = float(greeks.impliedVol or 0) if greeks and greeks.impliedVol else 0.0
                # BS fallback when IB gamma is zero
                if ib_gamma <= 0:
                    ib_gamma = _bs_gamma(spot, c.strike, T, rate, div_yield, iv) if iv > 0 else 0.0
                oi = int(t.openInterest) if t.openInterest and t.openInterest > 0 else 0
                vol = int(t.volume) if t.volume and t.volume > 0 else 0
                rows.append({
                    "expiration":    c.lastTradeDateOrContractMonth,
                    "strike":        c.strike,
                    "right":         c.right,
                    "open_interest": oi,
                    "volume":        vol,
                    "bid":           bid,
                    "ask":           ask,
                    "mid":           mid,
                    "iv":            round(iv, 4),
                    "delta":         round(float(greeks.delta or 0), 4) if greeks and greeks.delta else float("nan"),
                    "gamma":         round(ib_gamma, 6),
                    "days_to_expiry": dte,
                    "conId":         c.conId,
                })
            with _suppress_ib_errors():
                for c in batch:
                    try:
                        ib.cancelMktData(c)
                    except Exception:
                        pass
                ib.sleep(0.5)  # absorb async error-300 responses while suppressed

    if not rows:
        logger.warning("No data rows collected for %s GEX", contract.symbol)
        return {}

    df = pd.DataFrame(rows)

    # ── OI vs volume proxy ────────────────────────────────────────────────────
    zero_oi_ratio = float((df["open_interest"] <= 0).mean())
    if zero_oi_ratio > 0.5:
        vol_total = df["volume"].sum()
        if vol_total > 0:
            logger.info("%s: OI unavailable — using volume as OI proxy", contract.symbol)
            df["_oi_used"] = df["volume"]
        else:
            logger.warning("%s: both OI and volume are zero — unit gamma profile", contract.symbol)
            df["_oi_used"] = 1
    else:
        df["_oi_used"] = df["open_interest"]

    # ── GEX per row ───────────────────────────────────────────────────────────
    sign = df["right"].map({"C": 1.0, "P": -1.0})
    df["dollar_gamma"] = df["gamma"] * (spot ** 2) * multiplier
    df["gex_line"] = df["dollar_gamma"] * df["_oi_used"] * sign

    # ── totals ────────────────────────────────────────────────────────────────
    total_gex = float(df["gex_line"].sum())
    dealer_gex = -total_gex

    # ── GEX by expiration ─────────────────────────────────────────────────────
    gex_by_expiry = (
        df.groupby("expiration")
        .apply(lambda g: pd.Series({
            "call_gex": g.loc[g["right"] == "C", "gex_line"].sum(),
            "put_gex":  g.loc[g["right"] == "P", "gex_line"].sum(),
            "net_gex":  g["gex_line"].sum(),
        }))
        .reset_index()
        .sort_values("expiration")
    )

    # ── OI by expiration ──────────────────────────────────────────────────────
    oi_by_expiry = (
        df.groupby(["expiration", "right"])[["open_interest", "volume"]]
        .sum()
        .rename(columns={"open_interest": "total_oi", "volume": "total_volume"})
        .reset_index()
    )

    # ── strike profile ────────────────────────────────────────────────────────
    calls = df[df["right"] == "C"].groupby("strike")["gex_line"].sum().rename("call_gex")
    puts  = df[df["right"] == "P"].groupby("strike")["gex_line"].sum().rename("put_gex")
    profile = pd.concat([calls, puts], axis=1).fillna(0.0)
    profile["net_gex"] = profile["call_gex"] + profile["put_gex"]
    profile = profile.sort_index().reset_index()

    # ── zero-gamma level (linear interpolation across strikes) ────────────────
    net = profile.set_index("strike")["net_gex"]
    zero_gamma = spot  # default
    for i in range(1, len(net)):
        s0, s1 = net.index[i - 1], net.index[i]
        v0, v1 = net.iloc[i - 1], net.iloc[i]
        if (v0 > 0) != (v1 > 0) and (v1 - v0) != 0:
            zero_gamma = s0 + (0 - v0) * (s1 - s0) / (v1 - v0)
            break

    call_wall = float(profile.loc[profile["call_gex"].idxmax(), "strike"]) if not profile.empty else 0.0
    put_wall  = float(profile.loc[profile["put_gex"].abs().idxmax(), "strike"]) if not profile.empty else 0.0

    # ── log summary ──────────────────────────────────────────────────────────
    logger.info(
        "\n%s\nGEX | %s  spot=%.2f  r=%.2f%%\n%s\n"
        "  Total GEX   : %+.0f\n"
        "  Dealer GEX  : %+.0f\n"
        "  Zero-gamma  : %.2f\n"
        "  Call wall   : %.2f\n"
        "  Put wall    : %.2f\n"
        "  OI=0 ratio  : %.1f%%\n%s\n"
        "GEX by expiration:\n%s",
        "=" * 60, contract.symbol, spot, rate * 100, "=" * 60,
        total_gex, dealer_gex, zero_gamma, call_wall, put_wall,
        zero_oi_ratio * 100, "=" * 60,
        gex_by_expiry.to_string(index=False),
    )

    return {
        "symbol":         contract.symbol,
        "spot":           spot,
        "rate":           rate,
        "expirations":    selected_exps,
        "total_gex":      total_gex,
        "dealer_gex":     dealer_gex,
        "zero_gamma":     zero_gamma,
        "call_wall":      call_wall,
        "put_wall":       put_wall,
        "zero_oi_ratio":  zero_oi_ratio,
        "gex_by_expiry":  gex_by_expiry,
        "oi_by_expiry":   oi_by_expiry,
        "strike_profile": profile,
        "data":           df,
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def underlying_symbol(contract: Contract) -> str:
    """Return the symbol suitable for reqSecDefOptParams (strips spaces)."""
    return (contract.symbol or "").strip()


# ---------------------------------------------------------------------------
# Executions / fills
# ---------------------------------------------------------------------------

def get_executions(ib: IB) -> pd.DataFrame:
    """Return today's fills/executions as a DataFrame."""
    fills = ib.fills()
    if not fills:
        logger.info("No executions found")
        return pd.DataFrame()
    rows = [
        {
            "time":        f.time,
            "symbol":      f.contract.symbol,
            "secType":     f.contract.secType,
            "side":        f.execution.side,
            "shares":      f.execution.shares,
            "price":       f.execution.price,
            "orderId":     f.execution.orderId,
            "execId":      f.execution.execId,
            "commission":  f.commissionReport.commission if f.commissionReport else None,
            "realizedPnL": f.commissionReport.realizedPNL if f.commissionReport else None,
        }
        for f in fills
    ]
    df = pd.DataFrame(rows).set_index("time")
    logger.info("Executions loaded: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# GEX plot
# ---------------------------------------------------------------------------

# Symbols that are cash-settled indices on a single exchange.
# These require secType="IND" for reqSecDefOptParams to return the correct chain.
_INDEX_SPECS: dict[str, tuple[str, str, int]] = {
    # symbol : (exchange, currency, multiplier)
    "SPX":   ("CBOE",    "USD", 100),
    "SPXW":  ("CBOE",    "USD", 100),
    "NDX":   ("NASDAQ",  "USD", 100),
    "NDXP":  ("NASDAQ",  "USD", 100),
    "RUT":   ("RUSSELL", "USD", 100),
    "VIX":   ("CBOE",    "USD", 100),
    "DJX":   ("CBOE",    "USD", 100),
    "XSP":   ("CBOE",    "USD", 100),
}


def _build_underlying(symbol: str) -> tuple[Contract, int, str]:
    """
    Return (contract, multiplier, preferred_exchange) for *symbol*.
    Indices (SPX, NDX …) use Index; everything else uses Stock.
    """
    sym = symbol.upper()
    if sym in _INDEX_SPECS:
        exch, ccy, mult = _INDEX_SPECS[sym]
        return Index(sym, exch, ccy), mult, exch
    return Stock(sym, "SMART", "USD"), 100, ""


# ── colour palette ───────────────────────────────────────────────────────────
_CALL_CLR = "#26a69a"
_PUT_CLR  = "#ef5350"
_NET_POS  = "#1976d2"
_NET_NEG  = "#e53935"
_SPOT_CLR = "#ff9800"
_ZERO_CLR = "#9c27b0"
_WALL_CLR = "#90a4ae"
_BG_DARK  = "#0d0d1a"
_PANEL    = "#1a1a2e"


def plot_gex_symbol(
    ib: IB,
    symbol: str,
    expirations: int = 2,
    strikes_pct_range: float = 0.12,
    show: bool = True,
) -> Optional[dict]:
    """
    Compute and plot GEX for *symbol* (e.g. 'SPX', 'SPY', 'QQQ').

    Automatically selects the correct contract type:
        SPX, NDX, RUT, VIX → Index  (CBOE / NASDAQ, real CBOE OI)
        Everything else     → Stock  (SMART routing)

    Opens two Plotly charts in the default browser:
        1. GEX Strike Profile  — call / put / net bars + key levels
        2. GEX by Expiration   — horizontal net-GEX bar per expiry

    Parameters
    ----------
    ib                : active IB connection
    symbol            : underlying symbol, e.g. 'SPX'
    expirations       : number of front expiry dates (default 2)
    strikes_pct_range : ±fraction of spot to include (default 0.12 = ±12 %)
    show              : open charts in browser (default True)

    Returns
    -------
    dict returned by get_gex_for_expirations(), or None on failure.
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        logger.error("plotly is required for GEX plots — pip install plotly")
        return None

    # -- build + qualify contract --------------------------------------------
    contract, multiplier, pref_exch = _build_underlying(symbol)
    qualify(ib, contract)

    # -- compute GEX ---------------------------------------------------------
    gex = get_gex_for_expirations(
        ib, contract,
        expirations=expirations,
        strikes_pct_range=strikes_pct_range,
        multiplier=multiplier,
    )
    if not gex:
        logger.error("No GEX data returned for %s", symbol)
        return None

    profile  = gex["strike_profile"]
    by_exp   = gex["gex_by_expiry"]
    spot     = gex["spot"]
    rate     = gex["rate"]
    zero_g   = gex["zero_gamma"]
    c_wall   = gex["call_wall"]
    p_wall   = gex["put_wall"]
    tot      = gex["total_gex"]
    exps     = gex["expirations"]
    B        = 1e9

    # ── Chart 1: strike profile ──────────────────────────────────────────────
    strikes   = profile["strike"].tolist()
    call_b    = (profile["call_gex"] / B).tolist()
    put_b     = (profile["put_gex"]  / B).tolist()
    net_b     = (profile["net_gex"]  / B).tolist()
    cum_b     = (profile["net_gex"].cumsum() / B).tolist()
    net_clrs  = [_NET_POS if v >= 0 else _NET_NEG for v in net_b]
    cum_clrs  = [_NET_POS if v >= 0 else _NET_NEG for v in cum_b]

    fig1 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=("", "Cumulative GEX ($B)"),
    )

    fig1.add_trace(go.Bar(
        x=strikes, y=call_b, name="Call GEX",
        marker_color=_CALL_CLR, opacity=0.75,
        hovertemplate="Strike %{x}<br>Call GEX %{y:.3f} B<extra></extra>",
    ), row=1, col=1)

    fig1.add_trace(go.Bar(
        x=strikes, y=put_b, name="Put GEX",
        marker_color=_PUT_CLR, opacity=0.75,
        hovertemplate="Strike %{x}<br>Put GEX %{y:.3f} B<extra></extra>",
    ), row=1, col=1)

    fig1.add_trace(go.Bar(
        x=strikes, y=net_b, name="Net GEX",
        marker_color=net_clrs, opacity=0.95, width=0.35,
        hovertemplate="Strike %{x}<br>Net GEX %{y:.3f} B<extra></extra>",
    ), row=1, col=1)

    fig1.add_trace(go.Bar(
        x=strikes, y=cum_b, name="Cumulative GEX",
        marker_color=cum_clrs, opacity=0.85, showlegend=False,
        hovertemplate="Strike %{x}<br>Cum GEX %{y:.3f} B<extra></extra>",
    ), row=2, col=1)

    for x_val, clr, dash, lw, label in [
        (spot,   _SPOT_CLR, "solid", 2.2, f"Spot {spot:.2f}"),
        (zero_g, _ZERO_CLR, "dash",  2.0, f"γ Flip {zero_g:.2f}"),
        (c_wall, _WALL_CLR, "dot",   1.8, f"Call Wall {c_wall:.2f}"),
        (p_wall, _PUT_CLR,  "dot",   1.8, f"Put Wall {p_wall:.2f}"),
    ]:
        if not x_val:
            continue
        for row in (1, 2):
            fig1.add_vline(
                x=x_val, line_color=clr, line_dash=dash, line_width=lw,
                annotation_text=label if row == 1 else "",
                annotation_position="top right",
                annotation_font=dict(color=clr, size=10),
                annotation_bgcolor="rgba(13,13,26,0.75)",
                row=row, col=1,
            )

    oi_note = "OI-weighted" if gex["zero_oi_ratio"] < 0.5 else "Volume-weighted (OI unavailable)"
    title1 = (
        f"{symbol} | GEX Strike Profile  "
        f"spot={spot:.2f}  r={rate*100:.2f}%  "
        f"total={tot/B:.2f}B  [{oi_note}]<br>"
        f"<sup>Expirations: {', '.join(exps)}</sup>"
    )
    fig1.update_layout(
        title=dict(text=title1, font=dict(color="white", size=14)),
        barmode="overlay",
        template="plotly_dark",
        paper_bgcolor=_BG_DARK,
        plot_bgcolor=_PANEL,
        bargap=0.05,
        legend=dict(
            orientation="h", y=1.07, x=1, xanchor="right",
            bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=10),
        ),
        margin=dict(l=65, r=30, t=90, b=50),
        hovermode="x unified",
    )
    fig1.update_yaxes(title_text="GEX ($B)", row=1, col=1, gridcolor="rgba(255,255,255,0.08)")
    fig1.update_yaxes(title_text="Cum ($B)", row=2, col=1, gridcolor="rgba(255,255,255,0.08)")
    fig1.update_xaxes(title_text="Strike",   row=2, col=1, gridcolor="rgba(255,255,255,0.08)")
    fig1.update_xaxes(showgrid=True,         row=1, col=1, gridcolor="rgba(255,255,255,0.08)")

    # ── Chart 2: GEX by expiration ────────────────────────────────────────────
    exp_labels = by_exp["expiration"].tolist()
    call_exp   = (by_exp["call_gex"] / B).tolist()
    put_exp    = (by_exp["put_gex"]  / B).tolist()
    net_exp    = (by_exp["net_gex"]  / B).tolist()
    net_e_clrs = [_NET_POS if v >= 0 else _NET_NEG for v in net_exp]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=exp_labels, x=call_exp, name="Call GEX",
        orientation="h", marker_color=_CALL_CLR, opacity=0.8,
        hovertemplate="%{y}<br>Call GEX %{x:.3f} B<extra></extra>",
    ))
    fig2.add_trace(go.Bar(
        y=exp_labels, x=put_exp, name="Put GEX",
        orientation="h", marker_color=_PUT_CLR, opacity=0.8,
        hovertemplate="%{y}<br>Put GEX %{x:.3f} B<extra></extra>",
    ))
    fig2.add_trace(go.Bar(
        y=exp_labels, x=net_exp, name="Net GEX",
        orientation="h", marker_color=net_e_clrs, opacity=0.95,
        hovertemplate="%{y}<br>Net GEX %{x:.3f} B<extra></extra>",
    ))
    fig2.add_vline(x=0, line_color="white", line_width=1)

    fig2.update_layout(
        title=dict(
            text=f"{symbol} | GEX by Expiration  spot={spot:.2f}",
            font=dict(color="white", size=14),
        ),
        barmode="overlay",
        template="plotly_dark",
        paper_bgcolor=_BG_DARK,
        plot_bgcolor=_PANEL,
        bargap=0.25,
        legend=dict(
            orientation="h", y=1.07, x=1, xanchor="right",
            bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=10),
        ),
        xaxis_title="Net GEX ($B)",
        yaxis_title="Expiration",
        margin=dict(l=90, r=30, t=80, b=50),
    )
    fig2.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig2.update_yaxes(gridcolor="rgba(255,255,255,0.08)", autorange="reversed")

    if show:
        fig1.show()
        fig2.show()

    logger.info("GEX plots rendered for %s", symbol)
    return gex


# ---------------------------------------------------------------------------
# Quick demo  (python scripts/ib_trading.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ib = connect(host="127.0.0.1", port=4002, client_id=10)

    # -- qualify a stock contract
    spy = make_stock("SPX", exchange="SMART", currency="USD")
    qualify(ib, spy)

    # -- historical daily bars (last 30 days)
    bars = get_historical_bars(ib, spy, duration="30 D", bar_size="1 day")
    print(bars.tail())

    # -- latest snapshot
    snap = get_snapshot(ib, spy)
    print(snap)

    # -- next 2 option expiration dates
    exps = get_next_expirations(ib, spy, n=2)
    print("Next 2 expirations:", exps)

    # -- open interest for next 2 expirations (±12 % of spot)
    oi_df = get_open_interest(ib, spy, expirations=2)
    print(oi_df.groupby(["expiration", "right"])[["open_interest", "volume"]].sum())

    # -- GEX for next 2 expirations
    gex = get_gex_for_expirations(ib, spy, expirations=2)
    if gex:
        print(f"\nspot={gex['spot']:.2f}  total_gex={gex['total_gex']:+,.0f}  "
              f"dealer_gex={gex['dealer_gex']:+,.0f}")
        print(f"zero_gamma={gex['zero_gamma']:.2f}  "
              f"call_wall={gex['call_wall']:.2f}  put_wall={gex['put_wall']:.2f}")
        print("\nGEX by expiration:")
        print(gex["gex_by_expiry"].to_string(index=False))
        print("\nOI by expiration / right:")
        print(gex["oi_by_expiry"].to_string(index=False))
        print("\nTop 10 strikes by |net_gex|:")
        top = gex["strike_profile"].copy()
        top["abs_net"] = top["net_gex"].abs()
        print(top.nlargest(10, "abs_net")[["strike", "call_gex", "put_gex", "net_gex"]].to_string(index=False))

    # ----------------------------------------------------------------
    # SPX Index — OI and GEX from CBOE
    # SPX must be created as Index (secType=IND), NOT Stock.
    # preferred_exchange='CBOE' pins the chain to CBOE where SPX
    # options exclusively trade and where real OI is published.
    # ----------------------------------------------------------------
    spx = make_index("SPX", exchange="CBOE", currency="USD")
    qualify(ib, spx)

    # next 2 SPX expirations (CBOE publishes Fri weeklies + monthlies)
    spx_exps = get_next_expirations(ib, spx, n=2, preferred_exchange="CBOE")
    print("SPX next 2 expirations:", spx_exps)

    # raw OI per (expiration, strike, right) — multiplier 100, ±12 % of spot
    spx_oi = get_open_interest(
        ib, spx,
        expirations=2,
        strikes_pct_range=0.12,
        preferred_exchange="CBOE",
    )
    print("\nSPX OI summary:")
    print(spx_oi.groupby(["expiration", "right"])[["open_interest", "volume"]].sum())

    # GEX — SPX options have multiplier 100
    spx_gex = get_gex_for_expirations(
        ib, spx,
        expirations=2,
        strikes_pct_range=0.12,
        multiplier=100,
    )
    if spx_gex:
        print(f"\nSPX spot={spx_gex['spot']:.2f}")
        print(f"total_gex={spx_gex['total_gex']:+,.0f}  "
              f"dealer_gex={spx_gex['dealer_gex']:+,.0f}")
        print(f"zero_gamma={spx_gex['zero_gamma']:.2f}  "
              f"call_wall={spx_gex['call_wall']:.2f}  "
              f"put_wall={spx_gex['put_wall']:.2f}")
        print("\nGEX by expiration:")
        print(spx_gex["gex_by_expiry"].to_string(index=False))

    # -- open positions & account
    print(get_positions(ib))
    print(get_account_summary(ib))

    # -- open orders
    print(get_open_orders(ib))

    # -- today's fills
    print(get_executions(ib))

    plot_gex_symbol(ib, "SPX", expirations=2)

    disconnect(ib)