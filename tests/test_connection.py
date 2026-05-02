import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
tests/test_connection.py
------------------------
Verify IB Gateway / TWS connectivity and basic account access.
"""

import pytest
from ib_async import IB

pytestmark = pytest.mark.live


class TestConnection:

    def test_ib_is_connected(self, ib: IB):
        """IB object must report connected."""
        assert ib.isConnected(), "IB is not connected — check TWS/Gateway"

    def test_managed_accounts_not_empty(self, ib: IB):
        """At least one account must be returned."""
        accounts = ib.managedAccounts()
        assert len(accounts) > 0, "No managed accounts returned"

    def test_account_summary_has_net_liquidation(self, ib: IB):
        """NetLiquidation must be present and numeric."""
        account = ib.managedAccounts()[0]
        summary = {s.tag: s.value for s in ib.accountSummary(account)}
        assert "NetLiquidation" in summary, "NetLiquidation tag missing from account summary"
        assert float(summary["NetLiquidation"]) >= 0

    def test_account_summary_has_available_funds(self, ib: IB):
        """AvailableFunds must be present and numeric."""
        account = ib.managedAccounts()[0]
        summary = {s.tag: s.value for s in ib.accountSummary(account)}
        assert "AvailableFunds" in summary
        assert float(summary["AvailableFunds"]) >= 0

    def test_server_time(self, ib: IB):
        """Server time must be accessible and recent."""
        import datetime
        t = ib.reqCurrentTime()
        assert t is not None
        # Should be within 5 minutes of local time
        diff = abs((datetime.datetime.now(datetime.timezone.utc) - t).total_seconds())
        assert diff < 300, f"Server time drift too large: {diff:.0f}s"
