"""
analytics/gex.py
-----------------
Gamma Exposure (GEX) engine for a single underlying.

Methodology ported from OptionsTradingVolatility / GEX_Analytics_options_futures
(gamma_exposure_calc.py + black_scholes.py), adapted for Interactive Brokers data.

GEX at each strike
------------------
    dollar_gamma  = gamma × S² × multiplier
    gex_line      = dollar_gamma × OI × sign     (call=+1, put=-1)
    dealer_gex    = −∑(gex_line)                 (dealers assumed short calls / long puts)

Zero-gamma level
----------------
The spot price where net dealer gamma exposure flips from positive (short gamma)
to negative (long gamma). Found by linear interpolation on the strike-GEX profile.

Usage
-----
    from analytics.gex import GEXAnalytics

    ga   = GEXAnalytics(ib)
    res  = ga.compute(aapl_contract)

    print(res.summary())
    print(res.profile.to_string())
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from data.options import OptionChain, OptionRow
from data.rates import RiskFreeRate
from ib_async import IB, Contract

logger = logging.getLogger(__name__)

# US equity options: 100 shares per contract
_DEFAULT_MULTIPLIER = 100


# ─────────────────────────────────────────────────────────────────────────────
# Pure Black-Scholes maths (no external dependencies)
# Ported from OptionsTradingVolatility/functions/black_scholes.py
#             + OptionsTradingVolatility/functions/gamma_exposure_calc.py
# ─────────────────────────────────────────────────────────────────────────────

_SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Black-Scholes gamma for European option (call = put).

    Parameters
    ----------
    S     : spot price
    K     : strike
    T     : time to expiry in years
    r     : risk-free rate (annualised, continuous)
    q     : dividend yield (annualised, continuous)
    sigma : implied volatility (annualised)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> float:
    """Black-Scholes price (right = 'C' or 'P')."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if right.upper() == "C":
        return df_q * S * _norm_cdf(d1) - df_r * K * _norm_cdf(d2)
    return df_r * K * _norm_cdf(-d2) - df_q * S * _norm_cdf(-d1)


def implied_vol_newton(
    S: float, K: float, T: float, r: float, q: float,
    right: str, price: float,
    sigma0: float = 0.25,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Safeguarded Newton-Raphson IV solver with bisection fallback.
    Returns None if it fails to converge.
    """
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    lo, hi = 5e-4, 5.0
    sigma = max(lo, min(hi, sigma0))
    for _ in range(max_iter):
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        df_r, df_q = math.exp(-r * T), math.exp(-q * T)
        if right.upper() == "C":
            model = df_q * S * _norm_cdf(d1) - df_r * K * _norm_cdf(d2)
        else:
            model = df_r * K * _norm_cdf(-d2) - df_q * S * _norm_cdf(-d1)
        vega = df_q * S * math.sqrt(T) * _norm_pdf(d1)
        diff = model - price
        if abs(diff) < tol:
            return max(lo, min(hi, sigma))
        if vega <= 1e-8:
            break
        step = diff / vega
        sigma -= step
        if sigma <= lo or sigma >= hi:
            sigma = 0.5 * (lo + hi)
        elif diff > 0:
            hi = sigma
        else:
            lo = sigma
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GEXResult:
    """All GEX outputs for one underlying snapshot."""
    symbol: str
    spot: float
    rate: float

    # Full option data with GEX columns added
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── Aggregate GEX ─────────────────────────────────────────────────────────
    # Sum of (call GEX − put GEX) across all strikes/expirations
    total_gex: float = 0.0
    # Flip the sign: represents the DEALER's net gamma position
    dealer_gex: float = 0.0

    # ── Strike profile ────────────────────────────────────────────────────────
    # DataFrame with columns: strike, call_gex, put_gex, net_gex, cumulative_gex
    profile: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── Expiration profile ────────────────────────────────────────────────────
    by_expiry: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── Key levels ────────────────────────────────────────────────────────────
    zero_gamma_level: float = 0.0    # spot where dealer flips long/short gamma
    call_wall: float = 0.0           # strike with highest net call GEX
    put_wall: float = 0.0            # strike with highest absolute put GEX

    # ── Data quality ─────────────────────────────────────────────────────────
    zero_oi_ratio: float = 0.0       # fraction of rows with OI = 0

    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"GEX Report | {self.symbol}  spot={self.spot:.2f}  r={self.rate*100:.2f}%",
            f"{'='*70}",
            f"  Total GEX (calls-puts):  {self.total_gex:>14,.0f}  $ gamma",
            f"  Dealer GEX (-total):     {self.dealer_gex:>14,.0f}  $ gamma",
            f"  Zero-gamma level:        {self.zero_gamma_level:>14.2f}",
            f"  Call wall (max GEX):     {self.call_wall:>14.2f}",
            f"  Put wall  (max |GEX|):   {self.put_wall:>14.2f}",
            f"  Rows with OI=0:          {self.zero_oi_ratio*100:>13.1f} %",
            f"{'='*70}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class GEXAnalytics:
    """
    Compute Gamma Exposure (GEX) for an IB-traded underlying.

    Two gamma sources are supported:
    - ``use_ib_greeks=True``  (default): use IB's ``modelGreeks.gamma`` directly.
    - ``use_ib_greeks=False``: recompute gamma from BS using the mid price as market price.
      Falls back automatically when IB provides zero gamma.

    Parameters
    ----------
    ib             : active IB connection
    multiplier     : option contract multiplier (100 for US equity options)
    rfr            : RiskFreeRate instance (created automatically if None)
    """

    def __init__(
        self,
        ib: IB,
        multiplier: int = _DEFAULT_MULTIPLIER,
        rfr: Optional[RiskFreeRate] = None,
    ) -> None:
        self.ib = ib
        self.multiplier = multiplier
        self._rfr = rfr or RiskFreeRate()
        self._chain = OptionChain(ib, rate_series=self._rfr.series if hasattr(self._rfr, "series") else "DTB3")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        underlying: Contract,
        expirations: int = 4,
        strikes_pct_range: float = 0.10,
        div_yield: float = 0.0,
        use_ib_greeks: bool = True,
    ) -> GEXResult:
        """
        Fetch option chain and compute full GEX analytics.

        Parameters
        ----------
        underlying        : qualified IB contract (Stock, Index …)
        expirations       : number of front expirations (default 4)
        strikes_pct_range : ±fraction of spot to include (default 0.10 = ±10%)
        div_yield         : continuous dividend yield for BS fallback
        use_ib_greeks     : prefer IB's model gamma; fall back to BS when zero
        """
        rate = self._rfr.get_rate()
        logger.info("Computing GEX for %s  r=%.4f", underlying.symbol, rate)

        df = self._chain.get_full_chain(
            underlying,
            expirations=expirations,
            strikes_pct_range=strikes_pct_range,
        )

        if df.empty:
            logger.warning("No chain data for %s — returning empty GEXResult", underlying.symbol)
            return GEXResult(symbol=underlying.symbol, spot=0.0, rate=rate)

        spot = float(df["underlying_price"].iloc[0])

        # ── resolve gamma ─────────────────────────────────────────────────────
        df = df.copy()
        if use_ib_greeks:
            df["gamma_used"] = df.apply(
                lambda row: row["gamma"] if row["gamma"] > 0
                else self._bs_gamma_fallback(spot, row, rate, div_yield),
                axis=1,
            )
        else:
            df["gamma_used"] = df.apply(
                lambda row: self._bs_gamma_fallback(spot, row, rate, div_yield),
                axis=1,
            )

        # ── GEX per row ───────────────────────────────────────────────────────
        sign = df["right"].map({"C": 1.0, "P": -1.0})
        df["dollar_gamma"] = df["gamma_used"] * (spot ** 2) * self.multiplier

        # ── OI / volume proxy ─────────────────────────────────────────────────
        # Delayed data (MARKET_DATA_TYPE=3) doesn't include open interest.
        # When OI is 0 for >50% of rows, fall back to daily volume, which IS
        # available with delayed data.  Result = "volume-weighted GEX" — same
        # key levels, slightly noisier than true OI-weighted GEX.
        zero_oi_ratio = float((df["open_interest"] <= 0).mean())
        if zero_oi_ratio > 0.5 and "volume" in df.columns:
            vol_sum = df["volume"].sum()
            if vol_sum > 0:
                logger.info(
                    "%s: OI unavailable (delayed data) — using volume as OI proxy "
                    "(volume-weighted GEX)",
                    underlying.symbol,
                )
                df["_oi_used"] = df["volume"]
            else:
                logger.warning(
                    "%s: OI and volume both zero — GEX profile will be flat",
                    underlying.symbol,
                )
                df["_oi_used"] = 1  # unit gamma profile (shape only)
        else:
            df["_oi_used"] = df["open_interest"]

        df["gex_line"] = df["dollar_gamma"] * df["_oi_used"] * sign

        # ── totals ────────────────────────────────────────────────────────────
        total_gex  = float(df["gex_line"].sum())
        dealer_gex = -total_gex

        # ── strike profile ────────────────────────────────────────────────────
        profile = self._build_strike_profile(df, spot)

        # ── expiry profile ────────────────────────────────────────────────────
        by_expiry = (
            df.groupby("expiration", as_index=False)["gex_line"]
            .sum()
            .rename(columns={"gex_line": "gex"})
            .sort_values("expiration")
        )

        # ── key levels ────────────────────────────────────────────────────────
        zero_gamma = self._zero_gamma_level(profile, spot)
        call_wall  = self._call_wall(profile)
        put_wall   = self._put_wall(profile)

        return GEXResult(
            symbol=underlying.symbol,
            spot=spot,
            rate=rate,
            data=df,
            total_gex=total_gex,
            dealer_gex=dealer_gex,
            profile=profile,
            by_expiry=by_expiry,
            zero_gamma_level=zero_gamma,
            call_wall=call_wall,
            put_wall=put_wall,
            zero_oi_ratio=zero_oi_ratio,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bs_gamma_fallback(
        self, spot: float, row: pd.Series, rate: float, div_yield: float
    ) -> float:
        """Compute BS gamma from mid price when IB gamma is unavailable."""
        T = row["days_to_expiry"] / 365.0
        mid = row["mid"]
        if mid <= 0:
            mid = row["bid"] if row["bid"] > 0 else row["ask"]
        sigma = row["iv"]
        if sigma <= 0 and mid > 0 and T > 0:
            sigma = implied_vol_newton(
                spot, row["strike"], T, rate, div_yield, row["right"], mid
            ) or 0.0
        return bs_gamma(spot, row["strike"], T, rate, div_yield, sigma) if sigma > 0 else 0.0

    def _build_strike_profile(self, df: pd.DataFrame, spot: float) -> pd.DataFrame:
        """Aggregate GEX by strike → call_gex, put_gex, net_gex, cumulative_gex."""
        calls = df[df["right"] == "C"].groupby("strike")["gex_line"].sum().rename("call_gex")
        puts  = df[df["right"] == "P"].groupby("strike")["gex_line"].sum().rename("put_gex")
        profile = pd.concat([calls, puts], axis=1).fillna(0.0)
        profile["net_gex"] = profile["call_gex"] + profile["put_gex"]
        profile = profile.sort_index()
        profile["cumulative_gex"] = profile["net_gex"].cumsum()
        return profile.reset_index()

    @staticmethod
    def _zero_gamma_level(profile: pd.DataFrame, spot: float) -> float:
        """
        Find the strike where cumulative net GEX crosses zero via linear interpolation.
        Falls back to the strike closest to spot if no crossing is found.
        """
        if profile.empty or "net_gex" not in profile.columns:
            return spot

        net = profile.set_index("strike")["net_gex"]
        signs = (net > 0).astype(int)
        # Find first crossing
        for i in range(1, len(signs)):
            s0, s1 = signs.index[i - 1], signs.index[i]
            v0, v1 = net.iloc[i - 1], net.iloc[i]
            if signs.iloc[i - 1] != signs.iloc[i] and (v1 - v0) != 0:
                # Linear interpolation
                return s0 + (0 - v0) * (s1 - s0) / (v1 - v0)
        # No crossing — return strike nearest spot
        return float(net.index[abs(net.index - spot).argmin()])

    @staticmethod
    def _call_wall(profile: pd.DataFrame) -> float:
        """Strike with the highest call GEX (largest positive gamma exposure)."""
        if profile.empty or "call_gex" not in profile.columns:
            return 0.0
        idx = profile["call_gex"].idxmax()
        return float(profile.loc[idx, "strike"])

    @staticmethod
    def _put_wall(profile: pd.DataFrame) -> float:
        """Strike with the highest absolute put GEX (most negative)."""
        if profile.empty or "put_gex" not in profile.columns:
            return 0.0
        idx = profile["put_gex"].abs().idxmax()
        return float(profile.loc[idx, "strike"])
