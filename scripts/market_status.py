"""
scripts/market_status.py
------------------------
Is the market open? Prints the answer and exits 0 (open) or 1 (closed), so it
can gate a test run from the shell:

    py -3 scripts/market_status.py && py -3 -m pytest -m live

Asks IBKR for the contract's own schedule rather than consulting a calendar,
so half days, holidays and unscheduled closures are all accounted for.

    py -3 scripts/market_status.py              # SPY
    py -3 scripts/market_status.py SPX AAPL     # several
    py -3 scripts/market_status.py --trading    # include pre/post market
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ib_async import Stock

from core.connection import IBConnection
from scripts.ib_trading import make_index
from utils.market_hours import market_status

# Index underlyings must be Index (secType IND), not Stock, or the schedule
# comes back for the wrong instrument.
_INDEX = {"SPX", "NDX", "RUT", "VIX"}
_INDEX_EXCHANGE = {"SPX": "CBOE", "VIX": "CBOE", "NDX": "NASDAQ", "RUT": "RUSSELL"}


def contract_for(symbol: str):
    if symbol.upper() in _INDEX:
        return make_index(symbol.upper(), _INDEX_EXCHANGE[symbol.upper()], "USD")
    return Stock(symbol.upper(), "SMART", "USD")


def main(argv) -> int:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("ib_async").setLevel(logging.CRITICAL)

    liquid = "--trading" not in argv
    symbols = [a for a in argv if not a.startswith("--")] or ["SPY"]

    with IBConnection() as ib:
        statuses = [market_status(ib, contract_for(s), use_liquid_hours=liquid)
                    for s in symbols]

    for st in statuses:
        print("  " + str(st))

    all_open = all(bool(st) for st in statuses)
    print()
    print("  => %s" % ("OPEN — live tests will see two-sided quotes"
                       if all_open else
                       "CLOSED — live tests that need a two-sided market will "
                       "skip or fail for environmental reasons"))
    return 0 if all_open else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
