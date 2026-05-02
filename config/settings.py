import os
from dotenv import load_dotenv

load_dotenv()

# --- IB Connection ---
IB_HOST: str = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT: int = int(os.getenv("IB_PORT", "7497"))   # 7497=TWS paper, 7496=TWS live, 4001=Gateway live
IB_CLIENT_ID: int = int(os.getenv("IB_CLIENT_ID", "1"))

# --- Account ---
IB_ACCOUNT: str = os.getenv("IB_ACCOUNT", "")      # Leave empty to use default managed account

# --- Risk ---
MAX_POSITION_USD: float = float(os.getenv("MAX_POSITION_USD", "10000"))
MAX_DAILY_LOSS_USD: float = float(os.getenv("MAX_DAILY_LOSS_USD", "500"))
MAX_ORDER_SIZE: int = int(os.getenv("MAX_ORDER_SIZE", "1000"))

# --- Market data ---
MARKET_DATA_TYPE: int = int(os.getenv("MARKET_DATA_TYPE", "3"))
# 1 = Live (requires subscription)
# 2 = Frozen
# 3 = Delayed (free, 15-20 min delay)
# 4 = Delayed frozen
