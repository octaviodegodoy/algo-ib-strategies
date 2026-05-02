import logging
from ib_async import IB
from config import settings

logger = logging.getLogger(__name__)


class RiskManager:
    """Pre-trade risk checks and position sizing."""

    def __init__(self, ib: IB) -> None:
        self.ib = ib

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def size_by_fixed_usd(self, price: float, risk_usd: float | None = None) -> float:
        """Return a share/contract quantity based on a fixed USD risk amount."""
        budget = risk_usd if risk_usd is not None else settings.MAX_POSITION_USD
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        quantity = budget / price
        quantity = min(quantity, settings.MAX_ORDER_SIZE)
        return round(quantity, 2)

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------

    def check_order(self, action: str, quantity: float, price: float) -> bool:
        """Return True if the order passes all risk checks."""
        if quantity <= 0:
            logger.warning("Risk rejected: quantity=%s is not positive", quantity)
            return False

        if quantity > settings.MAX_ORDER_SIZE:
            logger.warning(
                "Risk rejected: quantity=%.2f exceeds MAX_ORDER_SIZE=%d",
                quantity,
                settings.MAX_ORDER_SIZE,
            )
            return False

        notional = quantity * price
        if notional > settings.MAX_POSITION_USD:
            logger.warning(
                "Risk rejected: notional=%.2f exceeds MAX_POSITION_USD=%.2f",
                notional,
                settings.MAX_POSITION_USD,
            )
            return False

        if self._daily_loss_breached():
            logger.warning("Risk rejected: daily loss limit reached")
            return False

        return True

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def _daily_loss_breached(self) -> bool:
        """Check realised + unrealised P&L against the daily loss limit."""
        account = settings.IB_ACCOUNT or self.ib.managedAccounts()[0]
        summary = {s.tag: s.value for s in self.ib.accountSummary(account)}
        unrealized = float(summary.get("UnrealizedPnL", 0))
        realized = float(summary.get("RealizedPnL", 0))
        total_pnl = unrealized + realized
        return total_pnl < -abs(settings.MAX_DAILY_LOSS_USD)
