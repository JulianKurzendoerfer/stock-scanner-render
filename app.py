import os
import math
import requests
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

API_TOKEN = os.getenv("EODHD_API_TOKEN", "").strip()
if not API_TOKEN:
    raise RuntimeError("Missing env var EODHD_API_TOKEN")

BASE = "https://eodhd.com/api"
WATCHLIST_PATH = os.getenv("WATCHLIST_PATH", "watchlist.txt")
PERIOD_RSI = int(os.getenv("RSI_PERIOD", "14"))
BB_LEN = int(os.getenv("BB_LEN", "20"))
BB_STD = float(os.getenv("BB_STD", "2"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
STOCH_K = int(os.getenv("STOCH_K", "14"))
STOCH_D = int(os.getenv("STOCH_D", "3"))
STOCH_SMOOTH = int(os.getenv("STOCH_SMOOTH", "3"))
BB_MAX_DISTANCE_PCT = float(os.getenv("BB_MAX_DISTANCE_PCT", "35"))

app = FastAPI()

def _read_watchlist():
    if not os.path.exists(WATCHLIST_PATH):
        return []
    out = []
    with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out

def _get_json(url, params):
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def _today_utc_date():
    return datetime.now(timezone.utc).date()

def fetch_eod_daily(symbol, start_date, end_date):
    url = f"{BASE}/eod/{symbol}"
    params = {
        "api_token": API_TOKEN,
        "fmt": "json",
        "from": start_date,
        "to": end_date,
        "order": "a",
        "period": "d",
    }
    data = _get_json(url, params)
    if not isinstance(data, list) or len(data) == 0:
        return None
    df = pd.DataFrame(data)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.date
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA
    df = df.dropna(subset=["close"]).copy()
    if df.empty:
        return None
    df = df.sort_values("date")
    return df

def fetch_realtime_price(symbol):
    url = f"{BASE}/real-time/{symbol}"
    params = {"api_token": API_TOKEN, "fmt": "json"}
    data = _get_json(url, params)
    price = None
    if isinstance(data, dict):
        for k in ["close", "price", "last", "last_close"]:
            if k in data and data[k] not in (None, "", "null"):
                try:
                    price = float(data[k])
                    break
                except:
                    pass
    return price

def build_df_with_current(symbol):
    today = _today_utc_date()
    end = today.isoformat()
    start = (today - relativedelta(years=1, months=4)).isoformat()
    df = fetch_eod_daily(symbol, start, end)
    if df is None or df.empty:
        return None, None, "no_eod_data"
    current_price = fetch_realtime_price(symbol)
    if current_price is None or math.isnan(current_price) or current_price <= 0:
        return df, None, "no_realtime_price"
    last_date = df["date"].iloc[-1]
    if last_date != today:
        prev_close = float(df["close"].iloc[-1])
        o = prev_close
        h = max(prev_close, current_price)
        l = min(prev_close, current_price)
        new_row = {
            "date": today,
            "open": o,
            "high": h,
            "low": l,
            "close": current_price,
            "volume": 0.0,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df.loc[df.index[-1], "close"] = current_price
        df.loc[df.index[-1], "high"] = max(float(df["high"].iloc[-1] or current_price), current_price)
        df.loc[df.index[-1], "low"] = min(float(df["low"].iloc[-1] or current_price), current_price)
    return df, current_price, None

def compute_indicators(df):
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    df["rsi"] = ta.rsi(close, length=PERIOD_RSI)

    bb = ta.bbands(close, length=BB_LEN, std=BB_STD)
    if bb is not None and not bb.empty:
        df["bb_low"] = bb.iloc[:, 0]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_high"] = bb.iloc[:, 2]
    else:
        df["bb_low"] = pd.NA
        df["bb_mid"] = pd.NA
        df["bb_high"] = pd.NA

    macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd is not None and not macd.empty:
        df["macd"] = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"] = macd.iloc[:, 2]
    else:
        df["macd"] = pd.NA
        df["macd_signal"] = pd.NA
        df["macd_hist"] = pd.NA

    st = ta.stoch(high, low, close, k=STOCH_K, d=STOCH_D, smooth_k=STOCH_SMOOTH)
    if st is not None and not st.empty:
        df["stoch_k"] = st.iloc[:, 0]
        df["stoch_d"] = st.iloc[:, 1]
    else:
        df["stoch_k"] = pd.NA
        df["stoch_d"] = pd.NA

    return df

def classify_signal(df):
    if len(df) < 3:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi_now = last.get("rsi")
    rsi_prev = prev.get("rsi")
    if pd.isna(rsi_now) or pd.isna(rsi_prev):
        return None

    rsi_cross = (float(rsi_prev) < 30.0) and (float(rsi_now) >= 30.0)
    if not rsi_cross:
        return None

    price = float(last["close"])

    bb_ok = False
    bb_low = last.get("bb_low")
    bb_dist = None
    if bb_low is not None and not pd.isna(bb_low) and float(bb_low) > 0:
        bb_dist = (price - float(bb_low)) / float(bb_low) * 100.0
        bb_ok = bb_dist <= BB_MAX_DISTANCE_PCT

    macd_ok = False
    h0 = df["macd_hist"].iloc[-1]
    h1 = df["macd_hist"].iloc[-2]
    if not pd.isna(h0) and not pd.isna(h1):
        macd_ok = float(h0) > float(h1)

    stoch_ok = False
    k0 = df["stoch_k"].iloc[-1]
    k1 = df["stoch_k"].iloc[-2]
    if not pd.isna(k0) and not pd.isna(k1):
        stoch_ok = float(k0) > float(k1)

    strong_score = sum([bb_ok, macd_ok, stoch_ok])
    sig_type = "STRONG_BUY" if strong_score >= 2 else "WEAK_BUY"

    return {
        "price": round(price, 4),
        "rsi": round(float(rsi_now), 2),
        "rsi_prev": round(float(rsi_prev), 2),
        "bb_distance_pct": None if bb_dist is None else round(float(bb_dist), 2),
        "bb_ok": bool(bb_ok),
        "macd_ok": bool(macd_ok),
        "stoch_ok": bool(stoch_ok),
        "type": sig_type,
    }

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/scan")
def scan():
    tickers = _read_watchlist()
    signals = []
    errors = []

    for sym in tickers:
        try:
            df, current_price, err = build_df_with_current(sym)
            if df is None:
                errors.append({"symbol": sym, "error": err or "no_data"})
                continue
            df = compute_indicators(df)
            s = classify_signal(df)
            if s is None:
                continue
            s["symbol"] = sym
            s["source"] = "eodhd"
            signals.append(s)
        except Exception as e:
            errors.append({"symbol": sym, "error": str(e)[:200]})

    return {
        "signals": signals,
        "meta": {
            "total": len(tickers),
            "signals_count": len(signals),
            "errors_count": len(errors),
            "errors_sample": errors[:5],
        },
    }
