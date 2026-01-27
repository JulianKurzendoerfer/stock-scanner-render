import os, time, json
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import pandas_ta as ta
import pandas_market_calendars as mcal
import redis
from fastapi import FastAPI, Query

EODHD_KEY = os.environ["EODHD_API_KEY"]
REDIS_URL = os.environ["REDIS_URL"]
WATCHLIST_PATH = "watchlist.txt"

rdb = redis.from_url(REDIS_URL, decode_responses=True)
app = FastAPI()

EODHD_BASE = "https://eodhd.com/api"

def load_watchlist():
    with open(WATCHLIST_PATH) as f:
        return [l.strip() for l in f if l.strip()]

def nyse_times():
    cal = mcal.get_calendar("XNYS")
    today = datetime.now(timezone.utc).date()
    sched = cal.schedule(start_date=today, end_date=today)
    if sched.empty:
        return {"isTradingDay": False}

    open_ = sched.iloc[0]["market_open"].to_pydatetime()
    t1 = open_ + timedelta(minutes=30)
    t2 = open_ + timedelta(hours=3, minutes=30)

    return {
        "isTradingDay": True,
        "t1_utc": t1.isoformat(),
        "t2_utc": t2.isoformat()
    }

def intraday(symbol):
    now = int(time.time())
    key = f"{symbol}:{now//300}"
    cached = rdb.get(key)
    if cached:
        data = json.loads(cached)
    else:
        url = f"{EODHD_BASE}/intraday/{symbol}"
        r = requests.get(url, params={
            "api_token": EODHD_KEY,
            "interval": "5m",
            "fmt": "json"
        })
        data = r.json()
        rdb.setex(key, 240, json.dumps(data))

    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.tail(200)

def signal(df):
    close = df["close"]
    rsi = ta.rsi(close, 14)
    macd = ta.macd(close)

    if rsi.iloc[-1] < 30 and macd["MACDh_12_26_9"].iloc[-1] > macd["MACDh_12_26_9"].iloc[-2]:
        return "BUY"
    if rsi.iloc[-1] > 70 and macd["MACDh_12_26_9"].iloc[-1] < macd["MACDh_12_26_9"].iloc[-2]:
        return "SELL"
    return None

@app.get("/times")
def times():
    return nyse_times()

@app.post("/scan")
def scan(run: int = Query(...)):
    buys, sells = [], []
    for s in load_watchlist():
        try:
            df = intraday(s)
            sig = signal(df)
            if sig == "BUY":
                buys.append(s)
            elif sig == "SELL":
                sells.append(s)
        except:
            pass

    text = f"RUN {run}\nBUY: {buys}\nSELL: {sells}"
    return {"text": text}
