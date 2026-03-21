# config.py

# === Data parameters ===
SYMBOL      = "BTC/USDT"
TIMEFRAME   = "4h"
START_DATE  = "2020-01-01"
END_DATE    = "2024-01-01"  # holdout starts here
HOLDOUT_END = "2024-07-01"

# === Paths ===
DATA_RAW       = "data/raw/"
DATA_PROCESSED = "data/processed/"
MODELS_DIR     = "models/"

# === Binance ===
# no API key needed for public market data