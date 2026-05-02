import logging
from abc import ABC, abstractmethod
from ib_async import IB, Contract
from core.broker import Broker
from data.market_data import MarketData
from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base for all strategies.

    Subclass this and implement:
      - on_bar()   — called on each new price bar
      - on_start() — called once after connection (optional setup)
      - on_stop()  — called before shutdown (optional cleanup)
    """

    def __init__(self, ib: IB, contract: Contract) -> None:
        self.ib = ib
        self.contract = contract
        self.broker = Broker(ib)
        self.market_data = MarketData(ib)
        self.risk = RiskManager(ib)
        self.name = self.__class__.__name__

    def run(self) -> None:
        logger.info("Strategy '%s' starting on %s", self.name, self.contract.symbol)
        self.on_start()
        try:
            self.ib.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.on_stop()
            logger.info("Strategy '%s' stopped", self.name)

    def on_start(self) -> None:
        """Override to add setup logic (e.g. subscribe to data, load state)."""

    def on_stop(self) -> None:
        """Override to add teardown logic (e.g. cancel orders, save state)."""

    @abstractmethod
    def on_bar(self, bars) -> None:
        """Called on each new bar. Implement your signal logic here."""
