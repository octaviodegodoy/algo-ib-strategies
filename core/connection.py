import logging
import sys
from ib_async import IB
from config import settings

logger = logging.getLogger(__name__)


class IBConnection:
    """Manages a single IB connection (TWS or Gateway)."""

    def __init__(self) -> None:
        self.ib = IB()
        self._connected = False

    def connect(self) -> IB:
        if self._connected:
            return self.ib

        try:
            self.ib.connect(
                host=settings.IB_HOST,
                port=settings.IB_PORT,
                clientId=settings.IB_CLIENT_ID,
            )
        except ConnectionRefusedError:
            logger.error(
                "Cannot connect to IB at %s:%d — TWS/Gateway not running or API not enabled.\n"
                "  1. Start TWS or IB Gateway\n"
                "  2. Enable API: Edit → Global Configuration → API → Settings\n"
                "     → check 'Enable ActiveX and Socket Clients'\n"
                "  3. Confirm socket port matches IB_PORT=%d in your .env",
                settings.IB_HOST,
                settings.IB_PORT,
                settings.IB_PORT,
            )
            sys.exit(1)
        except Exception as exc:
            logger.error("Unexpected connection error: %s", exc)
            sys.exit(1)

        self._connected = True

        self.ib.reqMarketDataType(settings.MARKET_DATA_TYPE)

        account = settings.IB_ACCOUNT or self.ib.managedAccounts()[0]
        logger.info(
            "Connected to IB | host=%s port=%d clientId=%d account=%s",
            settings.IB_HOST,
            settings.IB_PORT,
            settings.IB_CLIENT_ID,
            account,
        )
        return self.ib

    def disconnect(self) -> None:
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    def __enter__(self) -> IB:
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
