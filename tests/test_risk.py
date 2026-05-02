"""
tests/test_risk.py
-------------------
Unit tests for RiskManager — position sizing and pre-trade checks.
Most tests here are offline (no IB connection required).
"""

import pytest
from unittest.mock import MagicMock, patch
from risk.risk_manager import RiskManager
from config import settings


# ── helpers ──────────────────────────────────────────────────────────────────

def make_risk(unrealized_pnl: float = 0.0, realized_pnl: float = 0.0) -> RiskManager:
    """Build a RiskManager with a mocked IB that returns specific P&L values."""
    mock_ib = MagicMock()
    mock_ib.managedAccounts.return_value = ["DU123456"]
    mock_ib.accountSummary.return_value = [
        MagicMock(tag="UnrealizedPnL", value=str(unrealized_pnl)),
        MagicMock(tag="RealizedPnL", value=str(realized_pnl)),
    ]
    return RiskManager(mock_ib)


# ── position sizing ───────────────────────────────────────────────────────────

class TestPositionSizing:

    def test_size_by_fixed_usd_basic(self):
        risk = make_risk()
        qty = risk.size_by_fixed_usd(price=100.0, risk_usd=1000.0)
        assert qty == pytest.approx(10.0, rel=1e-3)

    def test_size_by_fixed_usd_uses_settings_default(self):
        risk = make_risk()
        qty = risk.size_by_fixed_usd(price=50.0)
        expected = min(settings.MAX_POSITION_USD / 50.0, settings.MAX_ORDER_SIZE)
        assert qty == pytest.approx(expected, rel=1e-3)

    def test_size_capped_by_max_order_size(self):
        risk = make_risk()
        # Very low price → raw quantity would exceed MAX_ORDER_SIZE
        qty = risk.size_by_fixed_usd(price=0.01, risk_usd=9_999_999)
        assert qty <= settings.MAX_ORDER_SIZE

    def test_size_raises_on_zero_price(self):
        risk = make_risk()
        with pytest.raises(ValueError):
            risk.size_by_fixed_usd(price=0.0)

    def test_size_raises_on_negative_price(self):
        risk = make_risk()
        with pytest.raises(ValueError):
            risk.size_by_fixed_usd(price=-10.0)


# ── pre-trade checks ──────────────────────────────────────────────────────────

class TestPreTradeChecks:

    def test_valid_order_passes(self):
        risk = make_risk()
        assert risk.check_order("BUY", quantity=10, price=100.0) is True

    def test_zero_quantity_fails(self):
        risk = make_risk()
        assert risk.check_order("BUY", quantity=0, price=100.0) is False

    def test_negative_quantity_fails(self):
        risk = make_risk()
        assert risk.check_order("BUY", quantity=-5, price=100.0) is False

    def test_exceeds_max_order_size_fails(self):
        risk = make_risk()
        assert risk.check_order("BUY", quantity=settings.MAX_ORDER_SIZE + 1, price=1.0) is False

    def test_notional_exceeds_max_position_fails(self):
        risk = make_risk()
        # quantity * price > MAX_POSITION_USD
        qty = settings.MAX_ORDER_SIZE
        price = (settings.MAX_POSITION_USD / qty) * 2
        assert risk.check_order("BUY", quantity=qty, price=price) is False

    def test_daily_loss_limit_breached_fails(self):
        """When total P&L is below -MAX_DAILY_LOSS_USD, order must be rejected."""
        loss = -(settings.MAX_DAILY_LOSS_USD + 1)
        risk = make_risk(unrealized_pnl=loss)
        assert risk.check_order("BUY", quantity=1, price=10.0) is False

    def test_daily_loss_not_breached_passes(self):
        """When P&L is within limit, order must pass."""
        risk = make_risk(unrealized_pnl=-1.0, realized_pnl=0.0)
        assert risk.check_order("BUY", quantity=1, price=10.0) is True
