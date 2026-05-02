"""
tests/conftest.py
-----------------
Shared pytest fixtures for all test modules.

All tests marked with @pytest.mark.live require IB Gateway / TWS running.
Run only live tests:    pytest -m live
Skip live tests:        pytest -m "not live"
"""

import sys
import logging
from pathlib import Path

import pytest
from ib_async import Stock, Forex

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.connection import IBConnection
from data.market_data import MarketData
from data.options import OptionChain
from core.broker import Broker
from risk.risk_manager import RiskManager
from utils.logger import setup_logging

setup_logging(level=logging.INFO)


# ── shared IB session (module-scoped = one connection per test file) ──────────

@pytest.fixture(scope="module")
def ib():
    """Live IB connection — requires TWS/Gateway running."""
    conn = IBConnection()
    ib_instance = conn.connect()
    yield ib_instance
    conn.disconnect()


# ── qualified contracts (conId resolved from IB) ──────────────────────────────
# IB requires contracts to be qualified before hashing/market data subscription.
# These fixtures resolve conId once per test session.

@pytest.fixture(scope="module")
def aapl(ib):
    c = Stock("AAPL", "SMART", "USD")
    ib.qualifyContracts(c)
    return c

@pytest.fixture(scope="module")
def msft(ib):
    c = Stock("MSFT", "SMART", "USD")
    ib.qualifyContracts(c)
    return c

@pytest.fixture(scope="module")
def spy(ib):
    c = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(c)
    return c

@pytest.fixture(scope="module")
def qqq(ib):
    c = Stock("QQQ", "SMART", "USD")
    ib.qualifyContracts(c)
    return c

@pytest.fixture(scope="module")
def eurusd(ib):
    c = Forex("EURUSD")
    ib.qualifyContracts(c)
    return c

@pytest.fixture(scope="module")
def eurgbp(ib):
    c = Forex("EURGBP")
    ib.qualifyContracts(c)
    return c

@pytest.fixture(scope="module")
def gbpusd(ib):
    c = Forex("GBPUSD")
    ib.qualifyContracts(c)
    return c


def qualified_stock(ib, symbol: str, exchange: str = "SMART", currency: str = "USD"):
    """Helper — returns a qualified Stock contract for any symbol."""
    c = Stock(symbol, exchange, currency)
    ib.qualifyContracts(c)
    return c


@pytest.fixture(scope="module")
def market_data(ib):
    return MarketData(ib)


@pytest.fixture(scope="module")
def option_chain(ib):
    return OptionChain(ib)


@pytest.fixture(scope="module")
def broker(ib):
    return Broker(ib)


@pytest.fixture(scope="module")
def risk(ib):
    return RiskManager(ib)
