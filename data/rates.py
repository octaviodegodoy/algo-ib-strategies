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

    # Cache TTL for a rate that was actually fetched. A T-bill rate moves by
    # basis points in a day, so four hours of staleness costs nothing and we
    # don't hammer FRED on every option request.
    _CACHE_TTL = timedelta(hours=4)

    # Cache TTL for DEFAULT_RATE after a FAILED fetch. This is a different
    # thing from the TTL above and must not share it: the fallback is not a
    # rate, it is an admission that we don't have one. Caching it for four
    # hours lets a single timeout pin 5.00% in place for a whole session and
    # misprice every option behind it -- measured against a real DTB3 of 4.03%,
    # that understates IV by 0.36 vol points and overstates delta by 0.011.
    # Five minutes is short enough that a transient outage self-heals within a
    # chain refresh or two, and long enough that a genuine outage costs one
    # timeout every five minutes rather than one per option.
    _FALLBACK_CACHE_TTL = timedelta(minutes=5)

    def __init__(self, series: str = "DTB3", timeout: int = 4) -> None:
        self.series = series
        self.timeout = timeout
        self._cached_rate: float | None = None
        self._cached_at: datetime | None = None
        # True when _cached_rate is DEFAULT_RATE because the fetch failed, so
        # get_rate() knows which of the two TTLs applies to it.
        self._cached_is_fallback: bool = False

    # ------------------------------------------------------------------
    # Rate retrieval
    # ------------------------------------------------------------------

    def get_rate(self) -> float:
        """
        Return the current annualised risk-free rate as a decimal.
        e.g. 4.52 % -> 0.0452

        A fetched rate is cached for _CACHE_TTL. The DEFAULT_RATE fallback is
        cached only for _FALLBACK_CACHE_TTL, so a transient FRED failure cannot
        hold the wrong rate for the rest of the session.
        """
        now = datetime.now()
        ttl = (
            self._FALLBACK_CACHE_TTL if self._cached_is_fallback
            else self._CACHE_TTL
        )
        if (
            self._cached_rate is not None
            and self._cached_at is not None
            and (now - self._cached_at) < ttl
        ):
            return self._cached_rate

        was_fallback = self._cached_is_fallback
        try:
            rate = self._fetch_fred(self.series)
            is_fallback = False
            desc = SERIES_DESCRIPTIONS.get(self.series, self.series)
            logger.info(
                "Risk-free rate  [%s | %s]: %.4f  (%.2f %%)",
                self.series, desc, rate, rate * 100,
            )
            if was_fallback:
                logger.info(
                    "FRED recovered for '%s' — %.2f %% replaces the %.2f %% "
                    "fallback", self.series, rate * 100, self.DEFAULT_RATE * 100,
                )
        except Exception as exc:
            logger.warning(
                "Could not fetch rate from FRED series '%s': %s  —  using "
                "default %.2f %%, retrying in %.0f min",
                self.series, exc, self.DEFAULT_RATE * 100,
                self._FALLBACK_CACHE_TTL.total_seconds() / 60.0,
            )
            rate = self.DEFAULT_RATE
            is_fallback = True

        # Cache the result, including the fallback, so we don't retry on every
        # call -- but remember WHICH it is, because the fallback gets the short
        # TTL and must be retried long before a real rate would be.
        self._cached_rate = rate
        self._cached_at = now
        self._cached_is_fallback = is_fallback
        return rate

    @property
    def using_fallback(self) -> bool:
        """True when the last get_rate() returned DEFAULT_RATE, not a real one.

        Lets a caller say so in a log line or a report instead of presenting a
        guessed 5 % as a measured rate.
        """
        return self._cached_is_fallback

    def _fetch_fred(self, series: str) -> float:
        """Download the FRED CSV and return the latest non-null value / 100."""
        url = _FRED_CSV_URL.format(series=series)
        # DO NOT set a custom User-Agent here. FRED accepts the connection for
        # "algo-ib-strategies/1.0" and then never sends the body, so the read
        # times out at whatever value you give it -- 4s and 20s both fail, while
        # urllib's default UA returns 301 KB in 0.1s every time. That one header
        # was silently pinning every rate lookup to DEFAULT_RATE (5.00%) when
        # DTB3 was really 4.03%, and the fallback is cached for 4 hours, so a
        # single failure fixed the wrong rate in place for a whole session.
        req = urllib.request.Request(url)
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
