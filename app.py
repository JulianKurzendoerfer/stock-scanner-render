import os
import math
import datetime as dt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI
import pandas_market_calendars as mcal

APP_TZ = dt.timezone.utc
NYSE_TZ = dt.timezone(dt.timedelta(hours=-5))

RSI_LEN = int(os.getenv("RSI_LEN", "14"))
BB_LEN = int(os.getenv("BB_LEN", "20"))
BB_STD = float(os.getenv("BB_STD", "2"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
STOCH_K = int(os.getenv("STOCH_K", "14"))
STOCH_D = int(os.getenv("STOCH_D", "3"))
RSI_TRIGGER = float(os.getenv("RSI_TRIGGER", "30"))
INTERVAL = os.getenv("INTERVAL", "1d")
PERIOD = os.getenv("PERIOD", "6mo")

EODHD_TOKEN = os.getenv("EODHD_TOKEN", "").strip()

app = FastAPI()

def _read_watchlist():
    path = os.path.join(os.path.dirname(__file__), "watchlist.txt")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out

def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            if math.isnan(float(x)):
                return None
            return float(x)
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    denom = (hh - ll).replace(0, np.nan)
    k_line = 100 * (close - ll) / denom
    d_line = k_line.rolling(d, min_periods=d).mean()
    return k_line, d_line

def _fetch_yfinance(symbol: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        interval=INTERVAL,
        period=PERIOD,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=lambda c: str(c).strip().title())
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            return pd.DataFrame()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df

def _fetch_eodhd(symbol: str) -> pd.DataFrame:
    if not EODHD_TOKEN:
        return pd.DataFrame()
    s = symbol
    if "." not in s:
        s = f"{s}.US"
    url = f"https://eodhd.com/api/eod/{s}"
    params = {"api_token": EODHD_TOKEN, "fmt": "json", "period": "d"}
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        for col in ["open", "high", "low", "close", "date"]:
            if col not in df.columns:
                return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date").sort_index()
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        df = df[["Open", "High", "Low", "Close"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()

def _compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    rsi = _rsi(close, RSI_LEN)
    bb_mid, bb_up, bb_low = _bbands(close, BB_LEN, BB_STD)
    macd, macd_sig, macd_hist = _macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    stoch_k, stoch_d = _stoch(high, low, close, STOCH_K, STOCH_D)

    last = df.iloc[-1]
    out = {
        "price": _safe_float(last["Close"]),
        "rsi": _safe_float(rsi.iloc[-1]) if len(rsi) else None,
        "bb_mid": _safe_float(bb_mid.iloc[-1]) if len(bb_mid) else None,
        "bb_upper": _safe_float(bb_up.iloc[-1]) if len(bb_up) else None,
        "bb_lower": _safe_float(bb_low.iloc[-1]) if len(bb_low) else None,
        "macd": _safe_float(macd.iloc[-1]) if len(macd) else None,
        "macd_signal": _safe_float(macd_sig.iloc[-1]) if len(macd_sig) else None,
        "macd_hist": _safe_float(macd_hist.iloc[-1]) if len(macd_hist) else None,
        "stoch_k": _safe_float(stoch_k.iloc[-1]) if len(stoch_k) else None,
        "stoch_d": _safe_float(stoch_d.iloc[-1]) if len(stoch_d) else None,
    }
    if out["bb_lower"] is not None and out["price"] is not None and out["bb_lower"] != 0:
        out["bb_dist_pct"] = _safe_float((out["price"] - out["bb_lower"]) / out["bb_lower"] * 100.0)
    else:
        out["bb_dist_pct"] = None
    return out

def _rsi_signal(ind: dict) -> str | None:
    rsi = ind.get("rsi")
    if rsi is None:
        return None
    if rsi <= RSI_TRIGGER:
        return "WEAK_BUY"
    return None

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/times", "/scan", "/scan_debug"]}

@app.get("/times")
def times():
    nyse = mcal.get_calendar("NYSE")
    now_utc = dt.datetime.now(tz=APP_TZ)
    today = now_utc.date()
    schedule = nyse.schedule(start_date=today, end_date=today)
    if schedule.empty:
        return {"isTradingDay": False, "open": None, "run1": None, "run2": None}
    open_ts = schedule.iloc[0]["market_open"].to_pydatetime().replace(tzinfo=APP_TZ)
    run1 = open_ts + dt.timedelta(minutes=30)
    run2 = open_ts + dt.timedelta(minutes=210)
    return {
        "isTradingDay": True,
        "open": open_ts.isoformat(),
        "run1": run1.isoformat(),
        "run2": run2.isoformat(),
    }

@app.get("/scan")
def scan():
    watch = _read_watchlist()
    signals = []
    errors = []
    for sym in watch:
        df = _fetch_yfinance(sym)
        source = "yfinance"
        if df.empty:
            df = _fetch_eodhd(sym)
            source = "eodhd" if not df.empty else "none"
        if df.empty or len(df) < max(RSI_LEN + 2, BB_LEN + 2, STOCH_K + 2, MACD_SLOW + 2):
            errors.append({"symbol": sym, "source": source, "error": "no_data_or_too_short"})
            continue
        ind = _compute_indicators(df)
        sig_type = _rsi_signal(ind)
        if sig_type:
            payload = {
                "symbol": sym,
                "source": source,
                "type": sig_type,
                "price": None if ind["price"] is None else round(ind["price"], 2),
                "rsi": None if ind["rsi"] is None else round(ind["rsi"], 2),
                "bb_dist_pct": None if ind["bb_dist_pct"] is None else round(ind["bb_dist_pct"], 2),
                "bb_lower": None if ind["bb_lower"] is None else round(ind["bb_lower"], 2),
                "macd_hist": None if ind["macd_hist"] is None else round(ind["macd_hist"], 6),
                "stoch_k": None if ind["stoch_k"] is None else round(ind["stoch_k"], 2),
            }
            signals.append(payload)
    return {"signals": signals}

@app.get("/scan_debug")
def scan_debug():
    watch = _read_watchlist()
    meta = {
        "total": len(watch),
        "interval": INTERVAL,
        "period": PERIOD,
        "rsi_len": RSI_LEN,
        "bb_len": BB_LEN,
        "stoch_k": STOCH_K,
        "macd_slow": MACD_SLOW,
    }
    ok_yf = 0
    ok_eod = 0
    too_short = 0
    empty = 0
    errors_sample = []
    signals_preview = []
    for sym in watch[:200]:
        df = _fetch_yfinance(sym)
        source = "yfinance"
        if df.empty:
            df = _fetch_eodhd(sym)
            source = "eodhd" if not df.empty else "none"
        if df.empty:
            empty += 1
            if len(errors_sample) < 5:
                errors_sample.append({"symbol": sym, "source": source, "error": "no_data"})
            continue
        if source == "yfinance":
            ok_yf += 1
        elif source == "eodhd":
            ok_eod += 1
        if len(df) < max(RSI_LEN + 2, BB_LEN + 2, STOCH_K + 2, MACD_SLOW + 2):
            too_short += 1
            if len(errors_sample) < 5:
                errors_sample.append({"symbol": sym, "source": source, "error": f"too_short_len_{len(df)}"})
            continue
        ind = _compute_indicators(df)
        sig_type = _rsi_signal(ind)
        if sig_type and len(signals_preview) < 10:
            signals_preview.append({
                "symbol": sym,
                "source": source,
                "type": sig_type,
                "price": None if ind["price"] is None else round(ind["price"], 2),
                "rsi": None if ind["rsi"] is None else round(ind["rsi"], 2),
                "bb_dist_pct": None if ind["bb_dist_pct"] is None else round(ind["bb_dist_pct"], 2),
                "macd_hist": None if ind["macd_hist"] is None else round(ind["macd_hist"], 6),
                "stoch_k": None if ind["stoch_k"] is None else round(ind["stoch_k"], 2),
            })
    meta.update({
        "ok_yfinance": ok_yf,
        "ok_eodhd": ok_eod,
        "too_short": too_short,
        "empty_or_err": empty,
        "signals_preview": signals_preview,
        "errors_sample": errors_sample,
    })
    return meta
