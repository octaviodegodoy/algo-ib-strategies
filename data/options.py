import contextlib
import logging
import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from ib_async import IB, Contract, Option, Ticker

from analytics.local_greeks import greeks_from_price, mid_or_none
from data.rates import RiskFreeRate

logger = logging.getLogger(__name__)

# IB error 200 ("no security definition") and 10091 ("requires subscription —
# delayed data is available") are expected during options chain qualification
# and snapshot.  Suppress the noise at the ib_async.wrapper level.
_IB_WRAPPER_LOG = logging.getLogger("ib_async.wrapper")


@contextlib.contextmanager
def _suppress_error200():
    """Temporarily raise ib_async.wrapper to CRITICAL to hide expected error chatter."""
    prev = _IB_WRAPPER_LOG.level
    _IB_WRAPPER_LOG.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        _IB_WRAPPER_LOG.setLevel(prev)


def _safe_int(value) -> int:
    """Convert to int, treating None and NaN as 0."""
    if value is None:
        return 0
    try:
        if isinstance(value, float) and math.isnan(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass
class OptionRow:
    symbol: str
    expiration: str
    strike: float
    right: str          # "C" or "P"
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    bid: float
    ask: float
    last: float
    mid: float
    underlying_price: float
    rate: float           # annualised risk-free rate used (e.g. 0.045)
    days_to_expiry: int
    forward_price: float  # F = S * exp((r-q)*T)
    open_interest: int    # contract open interest (0 if broker doesn't publish)
    volume: int = 0       # daily volume (used as OI proxy when OI unavailable)
    conId: int = 0


class OptionChain:
    """
    Fetch and filter option chains from IB.

    Usage
    -----
    chain = OptionChain(ib)
    df = chain.get_by_delta(
        underlying=Stock("AAPL", "SMART", "USD"),
        target_deltas=[0.25, 0.75],
        delta_tolerance=0.05,
        expirations=2,          # number of front expirations to include
    )
    """

    # Number of strikes on each side of ATM to request greeks for
    STRIKES_AROUND_ATM = 15

    def __init__(self, ib: IB, rate_series: str = "DTB3",
                 use_local_greeks: bool = True) -> None:
        """
        use_local_greeks : compute delta/gamma/vega/theta/IV here from the
            quoted mid via analytics.local_greeks, instead of reading IBKR's
            modelGreeks. True by default.

            Local is self-consistent -- one spot, one tenor convention, IV
            solved from the price we can actually trade on -- and it works on
            an account with no live or index market-data subscription, where
            modelGreeks comes back empty. It also REQUIRES a two-sided market:
            rows without one are dropped rather than priced off a stale last.

            Set False to go back to IBKR's figures, remembering that its theta
            is per CALENDAR day and each greek is stamped at whatever
            underlying level it happened to see.
        """
        self.ib = ib
        self._rfr = RiskFreeRate(series=rate_series)
        self.use_local_greeks = use_local_greeks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_by_delta(
        self,
        underlying: Contract,
        target_deltas: list[float] | None = None,
        delta_tolerance: float = 0.05,
        expirations: int = 2,
        rights: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return option rows whose |delta| is within *delta_tolerance* of any
        value in *target_deltas*.

        Parameters
        ----------
        underlying       : qualified Stock / Index contract
        target_deltas    : list of target |delta| values, e.g. [0.25, 0.75]
        delta_tolerance  : ±band around each target, default 0.05
        expirations      : how many front expiry dates to include (default 2)
        rights           : ["C", "P"] by default (both calls and puts)
        """
        if target_deltas is None:
            target_deltas = [0.25, 0.75]
        if rights is None:
            rights = ["C", "P"]

        # ── 0. fetch risk-free rate once (cached) ──────────────────────────
        rate = self._rfr.get_rate()
        logger.info("Using risk-free rate: %.4f (%.2f %%)", rate, rate * 100)

        # ── 1. get underlying price ───────────────────────────────────────
        und_price = self._get_underlying_price(underlying)
        if und_price <= 0:
            logger.error("Could not get price for %s", underlying.symbol)
            return pd.DataFrame()

        logger.info("%s underlying price: %.2f", underlying.symbol, und_price)

        # ── 2. get available chain params (strikes + expirations) ─────────
        chains = self.ib.reqSecDefOptParams(
            underlyingSymbol=underlying.symbol,
            futFopExchange="",
            underlyingSecType=underlying.secType,
            underlyingConId=underlying.conId,
        )
        if not chains:
            logger.error("No option chain params returned for %s", underlying.symbol)
            return pd.DataFrame()

        # Pick the exchange with the most strikes (usually SMART or CBOE)
        chain = max(chains, key=lambda c: len(c.strikes))
        logger.info(
            "Chain: exchange=%s, expirations=%d, strikes=%d",
            chain.exchange,
            len(chain.expirations),
            len(chain.strikes),
        )

        # ── 3. select front N expirations ─────────────────────────────────
        today = datetime.today().strftime("%Y%m%d")
        future_exps = sorted(e for e in chain.expirations if e >= today)
        selected_exps = future_exps[:expirations]
        if not selected_exps:
            logger.error("No future expirations found for %s", underlying.symbol)
            return pd.DataFrame()

        # ── 4. select strikes near ATM ────────────────────────────────────
        sorted_strikes = sorted(chain.strikes)
        atm_idx = min(range(len(sorted_strikes)),
                      key=lambda i: abs(sorted_strikes[i] - und_price))
        lo = max(0, atm_idx - self.STRIKES_AROUND_ATM)
        hi = min(len(sorted_strikes), atm_idx + self.STRIKES_AROUND_ATM + 1)
        selected_strikes = sorted_strikes[lo:hi]

        logger.info(
            "Requesting greeks for %d expirations × %d strikes × %d rights = %d contracts",
            len(selected_exps),
            len(selected_strikes),
            len(rights),
            len(selected_exps) * len(selected_strikes) * len(rights),
        )

        # ── 5. build + qualify option contracts ───────────────────────────
        contracts = [
            Option(underlying.symbol, exp, strike, right, chain.exchange)
            for exp in selected_exps
            for strike in selected_strikes
            for right in rights
        ]
        # Error 200 ("no security definition") is expected for strike/expiry combos
        # that don't exist — qualifyContracts silently drops them.
        with _suppress_error200():
            qualified = self.ib.qualifyContracts(*contracts)
        # qualifyContracts can return None entries in some ib_async versions
        qualified = [c for c in qualified if c is not None and c.conId]
        logger.info(
            "Qualified %d / %d option contracts for %s",
            len(qualified), len(contracts), underlying.symbol,
        )
        if not qualified:
            logger.error("No contracts qualified for %s", underlying.symbol)
            return pd.DataFrame()

        # ── 6. request streaming mkt data (snapshot=False is required for greeks) ──
        # snapshot=True does NOT deliver modelGreeks; we must stream then cancel.
        with _suppress_error200():
            tickers: list[Ticker] = [
                self.ib.reqMktData(c, "100,101,104,106", snapshot=False, regulatorySnapshot=False)
                for c in qualified
            ]
            self.ib.sleep(3)   # allow greeks to populate

        # ── 7. collect into rows, filter by delta ─────────────────────────
        rows: list[OptionRow] = []
        for ticker in tickers:
            bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
            exp = ticker.contract.lastTradeDateOrContractMonth

            # Greeks first, then filter on OUR delta. Filtering on IBKR's and
            # then reporting ours would select rows by one model and describe
            # them with another.
            g = self._row_greeks(ticker, und_price, rate, bid, ask)
            if g is None:
                continue

            matched = any(
                abs(abs(g["delta"]) - tgt) <= delta_tolerance
                for tgt in target_deltas
            )
            if not matched:
                continue

            dte = RiskFreeRate.days_to_expiry(exp)
            fwd = g["forward"] or RiskFreeRate.forward_price(und_price, rate, dte)
            oi = _safe_int(ticker.openInterest)
            rows.append(OptionRow(
                symbol=ticker.contract.symbol,
                expiration=exp,
                strike=ticker.contract.strike,
                right=ticker.contract.right,
                delta=round(g["delta"], 4),
                gamma=round(g["gamma"], 6),
                theta=round(g["theta"], 4),
                vega=round(g["vega"], 4),
                iv=round(g["iv"], 4),
                bid=bid,
                ask=ask,
                last=ticker.last or 0.0,
                mid=round((bid + ask) / 2, 4) if bid and ask else 0.0,
                underlying_price=und_price,
                rate=round(rate, 6),
                days_to_expiry=dte,
                forward_price=round(fwd, 4),
                open_interest=oi,
                conId=ticker.contract.conId,
            ))

        # cancel all subscriptions (no-op for snapshot=True, safe to call)
        for c in qualified:
            try:
                self.ib.cancelMktData(c)
            except Exception:
                pass

        if not rows:
            logger.warning(
                "No options matched target deltas %s ± %.2f for %s",
                target_deltas, delta_tolerance, underlying.symbol
            )
            return pd.DataFrame()

        df = pd.DataFrame([r.__dict__ for r in rows])
        df = df.sort_values(["expiration", "right", "strike"]).reset_index(drop=True)
        logger.info(
            "Found %d option rows near delta %s for %s",
            len(df), target_deltas, underlying.symbol,
        )
        return df

    # ------------------------------------------------------------------
    # Full chain for GEX
    # ------------------------------------------------------------------

    def get_full_chain(
        self,
        underlying: Contract,
        expirations: int = 4,
        strikes_pct_range: float = 0.10,
        rights: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return ALL options within *strikes_pct_range* of spot for GEX analysis.
        Unlike get_by_delta(), no delta filter is applied — every strike is returned.

        Parameters
        ----------
        underlying         : qualified contract
        expirations        : number of front expirations to include (default 4)
        strikes_pct_range  : fraction of spot price to include on each side (default 0.10 = ±10%)
        rights             : ["C", "P"] by default
        """
        if rights is None:
            rights = ["C", "P"]

        rate = self._rfr.get_rate()
        und_price = self._get_underlying_price(underlying)
        if und_price <= 0:
            logger.error("Could not get price for %s", underlying.symbol)
            return pd.DataFrame()

        chains = self.ib.reqSecDefOptParams(
            underlyingSymbol=underlying.symbol,
            futFopExchange="",
            underlyingSecType=underlying.secType,
            underlyingConId=underlying.conId,
        )
        if not chains:
            return pd.DataFrame()

        chain = max(chains, key=lambda c: len(c.strikes))
        today = datetime.today().strftime("%Y%m%d")
        future_exps = sorted(e for e in chain.expirations if e >= today)
        selected_exps = future_exps[:expirations]
        if not selected_exps:
            return pd.DataFrame()

        lo_bound = und_price * (1 - strikes_pct_range)
        hi_bound = und_price * (1 + strikes_pct_range)
        selected_strikes = [s for s in sorted(chain.strikes) if lo_bound <= s <= hi_bound]

        contracts = [
            Option(underlying.symbol, exp, strike, right, chain.exchange)
            for exp in selected_exps
            for strike in selected_strikes
            for right in rights
        ]
        with _suppress_error200():
            qualified = self.ib.qualifyContracts(*contracts)
        # qualifyContracts can return None entries in some ib_async versions
        qualified = [c for c in qualified if c is not None and c.conId]
        logger.info(
            "Qualified %d / %d option contracts for %s (full chain)",
            len(qualified), len(contracts), underlying.symbol,
        )
        if not qualified:
            return pd.DataFrame()

        # IB caps simultaneous market-data subscriptions (default ~100 "ticker
        # lines").  Batch the requests, wait for greeks, snapshot the ticker
        # data, then cancel before issuing the next batch.
        BATCH_SIZE = 90
        BATCH_SLEEP = 4
        rows: list[OptionRow] = []

        with _suppress_error200():
            for batch_start in range(0, len(qualified), BATCH_SIZE):
                batch = qualified[batch_start:batch_start + BATCH_SIZE]
                logger.info(
                    "Requesting greeks batch %d-%d / %d",
                    batch_start + 1, batch_start + len(batch), len(qualified),
                )
                tickers: list[Ticker] = [
                    self.ib.reqMktData(c, "100,101,104,106", snapshot=False, regulatorySnapshot=False)
                    for c in batch
                ]
                self.ib.sleep(BATCH_SLEEP)
                rows.extend(self._collect_chain_rows(tickers, und_price, rate))
                for c in batch:
                    try:
                        self.ib.cancelMktData(c)
                    except Exception:
                        pass
                # brief pause to let IB release the ticker slots
                self.ib.sleep(0.5)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([r.__dict__ for r in rows])
        return df.sort_values(["expiration", "right", "strike"]).reset_index(drop=True)

    def _collect_chain_rows(
        self,
        tickers: list[Ticker],
        und_price: float,
        rate: float,
    ) -> list[OptionRow]:
        """Convert a batch of tickers into OptionRow records (full-chain version)."""
        rows: list[OptionRow] = []
        for ticker in tickers:
            bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
            exp = ticker.contract.lastTradeDateOrContractMonth
            g = self._row_greeks(ticker, und_price, rate, bid, ask)
            if g is None:
                continue
            dte = RiskFreeRate.days_to_expiry(exp)
            fwd = g["forward"] or RiskFreeRate.forward_price(und_price, rate, dte)
            oi = _safe_int(ticker.openInterest)
            vol = _safe_int(ticker.volume)
            rows.append(OptionRow(
                symbol=ticker.contract.symbol,
                expiration=exp,
                strike=ticker.contract.strike,
                right=ticker.contract.right,
                delta=round(g["delta"], 4),
                gamma=round(g["gamma"], 6),
                theta=round(g["theta"], 4),
                vega=round(g["vega"], 4),
                iv=round(g["iv"], 4),
                bid=bid,
                ask=ask,
                last=ticker.last or 0.0,
                mid=round((bid + ask) / 2, 4) if bid and ask else 0.0,
                underlying_price=und_price,
                rate=round(rate, 6),
                days_to_expiry=dte,
                forward_price=round(fwd, 4),
                open_interest=oi,
                volume=vol,
                conId=ticker.contract.conId,
            ))
        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_greeks(self, ticker: Ticker, und_price: float, rate: float,
                    bid: float, ask: float):
        """Greeks for one ticker, local or from IBKR. None means skip the row.

        Returns plain floats (never None inside the dict) so the row builders
        do not each need their own `or 0` dance.
        """
        if self.use_local_greeks:
            mid = mid_or_none(bid, ask)
            if mid is None:
                logger.debug("no two-sided market for %s - skipping",
                             ticker.contract.localSymbol)
                return None
            g = greeks_from_price(
                spot=und_price,
                strike=ticker.contract.strike,
                right=ticker.contract.right,
                expiration=ticker.contract.lastTradeDateOrContractMonth,
                price=mid,
                rate=rate,
            )
            if g is None:
                return None
            return {"delta": g.delta, "gamma": g.gamma, "theta": g.theta,
                    "vega": g.vega, "iv": g.iv, "forward": g.forward}

        greeks = ticker.modelGreeks or ticker.lastGreeks
        if greeks is None or greeks.delta is None:
            return None
        return {"delta": greeks.delta or 0.0, "gamma": greeks.gamma or 0.0,
                "theta": greeks.theta or 0.0, "vega": greeks.vega or 0.0,
                "iv": greeks.impliedVol or 0.0, "forward": 0.0}

    def _get_underlying_price(self, contract: Contract) -> float:
        ticker = self.ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
        self.ib.sleep(2)
        price = ticker.last or ticker.close or ticker.bid or 0.0
        self.ib.cancelMktData(contract)
        return float(price)
