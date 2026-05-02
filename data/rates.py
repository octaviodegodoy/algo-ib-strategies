"""
data/rates.py
--------------
Fetch the current US risk-free interest rate from FRED (St. Louis Fed).

No API key required — uses the public CSV endpoint.

Default series: DTB3 (3-month Treasury Bill, secondary market rate).
Alternative: SOFR, FEDFUNDS.

Usage
-----
    from data.rates import RiskFreeRate

    rfr = RiskFreeRate()
    rate = rfr.get_rate()              # 0.0452  (= 4.52 %)
    fwd  = rfr.forward_price(spot=200.0, rate=rate, days_to_expiry=30)
"""

import csv
import io
import logging
import math
import urllib.request
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# FRED public CSV endpoint (no API key needed)
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

SERIES_DESCRIPTIONS: dict[str, str] = {
    "DTB3":     "3-Month Treasury Bill (secondary market)",
    "SOFR":     "Secured Overnight Financing Rate (SOFR)",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "DGS2":     "2-Year Treasury Constant Maturity Rate",
    "DGS10":    "10-Year Treasury Constant Maturity Rate",
}


class RiskFreeRate:
    """
    Fetch and cache the current US risk-free rate.

    Parameters
    ----------
    series  : FRED series ID (default 'DTB3' — 3-month T-bill)
    timeout : HTTP request timeout in seconds
    """

    # Hard fallback if the network request fails (e.g. running offline)
    DEFAULT_RATE = 0.05        # 5 %

    # Cache TTL — avoid hammering FRED on every option request
    _CACHE_TTL = timedelta(hours=4)

    def __init__(self, series: str = "DTB3", timeout: int = 4) -> None:
        self.series = series
        self.timeout = timeout
        self._cached_rate: float | None = None
        self._cached_at: datetime | None = None

    # ------------------------------------------------------------------
    # Rate retrieval
    # ------------------------------------------------------------------

    def get_rate(self) -> float:
        """
        Return the current annualised risk-free rate as a decimal.
        e.g. 4.52 % → 0.0452

        Caches the result for _CACHE_TTL hours.
        Falls back to DEFAULT_RATE if the HTTP request fails.
        """
        now = datetime.now()
        if (
            self._cached_rate is not None
            and self._cached_at is not None
            and (now - self._cached_at) < self._CACHE_TTL
        ):
            return self._cached_rate

        try:
            rate = self._fetch_fred(self.series)
            desc = SERIES_DESCRIPTIONS.get(self.series, self.series)
            logger.info(
                "Risk-free rate  [%s | %s]: %.4f  (%.2f %%)",
                self.series, desc, rate, rate * 100,
            )
        except Exception as exc:
            logger.warning(
                "Could not fetch rate from FRED series '%s': %s  —  using default %.2f %%",
                self.series, exc, self.DEFAULT_RATE * 100,
            )
            rate = self.DEFAULT_RATE

        # Cache the result (including the fallback) so we don't retry on every call
        self._cached_rate = rate
        self._cached_at = now
        return rate

    def _fetch_fred(self, series: str) -> float:
        """Download the FRED CSV and return the latest non-null value / 100."""
        url = _FRED_CSV_URL.format(series=series)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "algo-ib-strategies/1.0"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            content = resp.read().decode("utf-8")

        reader = csv.reader(io.StringIO(content))
        next(reader)  # skip header row ("DATE", "VALUE")

        last_value: float | None = None
        for row in reader:
            if len(row) >= 2 and row[1].strip() not in (".", ""):
                try:
                    last_value = float(row[1])
                except ValueError:
                    continue

        if last_value is None:
            raise ValueError(f"No valid data rows in FRED series '{series}'")

        # FRED stores T-bill / treasury rates as percent (e.g. 4.52 → 0.0452)
        return last_value / 100.0

    # ------------------------------------------------------------------
    # Forward price helpers
    # ------------------------------------------------------------------

    @staticmethod
    def forward_price(
        spot: float,
        rate: float,
        days_to_expiry: int,
        dividend_yield: float = 0.0,
    ) -> float:
        """
        Continuous-compounding forward price.

            F = S × exp( (r - q) × T )

        Parameters
        ----------
        spot            : current underlying spot price
        rate            : annualised risk-free rate, decimal  (e.g. 0.045)
        days_to_expiry  : calendar days until option expiration
        dividend_yield  : annualised continuous dividend yield, decimal
        """
        T = days_to_expiry / 365.0
        return spot * math.exp((rate - dividend_yield) * T)

    @staticmethod
    def time_to_expiry(expiration: str) -> float:
        """
        Fraction of a year from today to *expiration* (YYYYMMDD string).
        Returns 0.0 if already expired.
        """
        exp_date = datetime.strptime(expiration, "%Y%m%d").date()
        today = datetime.today().date()
        return max((exp_date - today).days, 0) / 365.0

    @staticmethod
    def days_to_expiry(expiration: str) -> int:
        """Calendar days from today to *expiration* (YYYYMMDD string)."""
        exp_date = datetime.strptime(expiration, "%Y%m%d").date()
        today = datetime.today().date()
        return max((exp_date - today).days, 0)
