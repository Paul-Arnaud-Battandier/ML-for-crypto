import pandas as pd
import numpy as np
import time
import yfinance as yf
import ccxt
import requests
from config import SYMBOL, SYMBOL_FUTURES, TIMEFRAME, START_DATE, END_DATE, DATA_RAW, BINANCE_FUTURES_URL


def fetch_ohlcv_full(symbol, timeframe, start_date, end_date):

    # 1. exchange instance
    exchange = ccxt.binance()

    # 2. timestamps in milliseconds
    since         = pd.Timestamp(start_date).value // 10**6
    end_timestamp = pd.Timestamp(end_date).value   // 10**6

    # 3. collector
    candle_batches = []

    # 4. loop
    while True:
        # 4a. fetch
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

        # 4b. empty batch = no more data
        if not candles:
            break

        # 4c. collect
        candle_batches.append(candles)

        # 4d. advance window
        since = candles[-1][0] + 1

        # 4e. past end date
        if candles[-1][0] >= end_timestamp:
            break

        # 4f. rate limit
        time.sleep(exchange.rateLimit / 1000)

    # 5. flatten into DataFrame
    df = pd.DataFrame(
        [candle for batch in candle_batches for candle in batch],
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # 6. convert timestamp to UTC datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # 7. set as index
    df.set_index("timestamp", inplace=True)

    # 8. filter to exact date range
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts   = pd.Timestamp(end_date,   tz="UTC")
    df = df[(df.index >= start_ts) & (df.index < end_ts)]

    # 9. return
    return df


def fetch_funding_rates(symbol, start_date, end_date):

    # 1. convert dates to milliseconds
    since         = pd.Timestamp(start_date).value // 10**6
    end_timestamp = pd.Timestamp(end_date).value   // 10**6

    # 2. empty list to collect batches
    batches = []

    # 3. loop
    while True:

        # 3a. build params dict
        #     keys: symbol, startTime, limit (1000)
        params = {
            "symbol": symbol,
            "startTime": since,
            "limit": 1000
        }

        # 3b. make the GET request to BINANCE_FUTURES_URL
        response = requests.get(BINANCE_FUTURES_URL, params=params)

        # 3c. raise if HTTP error
        response.raise_for_status()

        # 3d. parse JSON
        data = response.json()

        # 3e. if empty → break
        if not data:
            break

        # 3f. append to batches
        batches.append(data)

        # 3g. advance since to last record's fundingTime + 1
        since = data[-1]["fundingTime"] + 1

        # 3h. break if past end_date
        # 3h. break if past end_date
        if since > end_timestamp:
            break

        # 3i. rate limit
        time.sleep(0.5)

    # 4. concatenate all batches into one DataFrame
    df = pd.concat([pd.DataFrame(batch) for batch in batches])

    # 4b. filter to exact date range
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts   = pd.Timestamp(end_date,   tz="UTC")
    df = df[(df.index >= start_ts) & (df.index < end_ts)]

    # 5. convert fundingTime to UTC datetime and set as index
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df.set_index("fundingTime", inplace=True)

    # 6. cast fundingRate to float
    df["fundingRate"] = df["fundingRate"].astype(float)

    # 7. return only the fundingRate column
    df.index.name = "timestamp"
    return df[["fundingRate"]].rename(columns={"fundingRate": "funding_rate"})


def fetch_fear_greed(start_date, end_date):

    response = requests.get("https://api.alternative.me/fng/?limit=3000")
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["data"])  # ← extract the list

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.set_index("timestamp", inplace=True)
    df["value"] = df["value"].astype(int)

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts   = pd.Timestamp(end_date,   tz="UTC")
    df = df[(df.index >= start_ts) & (df.index < end_ts)]

    return df[["value"]].rename(columns={"value": "fear_greed"})


def fetch_macro(start_date, end_date):

    dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date, auto_adjust=True)
    dxy.columns = dxy.columns.get_level_values(0)  # ← flatten MultiIndex if present
    dxy = dxy[["Close"]].rename(columns={"Close": "dxy"})

    qqq = yf.download("QQQ", start=start_date, end=end_date, auto_adjust=True)
    qqq.columns = qqq.columns.get_level_values(0)  # ← same
    qqq = qqq[["Close"]].rename(columns={"Close": "nasdaq"})

    df = pd.concat([dxy, qqq], axis=1)
    df.index = df.index.tz_localize("UTC")
    df.index.name = "timestamp"

    return df


