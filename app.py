import os
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pandas_ta as ta
import pandas_market_calendars as mcal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

APP_TZ = "Europe/Berlin"
CAL = mcal.get_calendar("XNYS")

EODHD_API_KEY = os.getenv("EODHD_API_KEY", "")

WATCHLIST_PATH = os.getenv("WATCHLIST_PATH", "watchlist.txt")
TIMEFRAME = os.getenv("TIMEFRAME", "1d")
HISTORY_PERIOD = os.getenv("HISTORY_PERIOD", "6mo")

RSI_LEN = int(os.getenv("RSI_LEN", "14"))
BB_LEN = int(os.getenv("BB_LEN", "20"))
BB_STD = float(os.getenv("BB_STD", "2"))
STOCH_K = int(os.getenv("STOCH_K", "14"))
STOCH_D = int(os.getenv("STOCH_D", "3"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))

RSI_WEAK_MAX = float(os.getenv("RSI_WEAK_MAX", "30"))
RSI_WEAK_NEAR = float(os.getenv("RSI_WEAK_NEAR", "29"))
RSI_DELTA_MIN = float(os.getenv("RSI_DELTA_MIN", "0.6"))

BB_MAX_PCT_FROM_LOWER = float(os.getenv("BB_MAX_PCT_FROM_LOWER", "35"))

STOCH_RISE_MIN = float(os.getenv("STOCH_RISE_MIN", "8"))
STOCH_LOW_TH = float(os.getenv("STOCH_LOW_TH", "20"))

MACD_HIST_NEED_BARS = int(os.getenv("MACD_HIST_NEED_BARS", "2"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_watchlist() -> List[str]:
    if not os.path.exists(WATCHLIST_PATH):
        return []
    out: List[str] = []
    with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out

def now_utc() -> pd.Timestamp:
    return pd.Timestamp.utcnow()

def nyse_next_times() -> Dict[str, Any]:
    now = pd.Timestamp.now(tz=APP_TZ)
    start = (now - pd.Timedelta(days=2)).date()
    end = (now + pd.Timedelta(days=7)).date()
    sched = CAL.schedule(start_date=start, end_date=end)
    if sched.empty:
        return {"isTradingDay": False, "open": None, "run1": None, "run2": None}
    local = sched.tz_convert(APP_TZ)
    today = now.date()
    day = local[local.index.date == today]
    if day.empty:
        nxt = local.iloc[0]
        market_open = nxt["market_open"]
    else:
        market_open = day.iloc[0]["market_open"]
    open_time = market_open.tz_convert("UTC")
    run1 = (market_open + pd.Timedelta(minutes=30)).tz_convert("UTC")
    run2 = (market_open + pd.Timedelta(hours=3, minutes=30)).tz_convert("UTC")
    return {
        "isTradingDay": True,
        "open": open_time.isoformat(),
        "run1": run1.isoformat(),
        "run2": run2.isoformat(),
    }

def _yf_download(symbol: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            symbol,
            period=HISTORY_PERIOD,
            interval=TIMEFRAME,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            return None
        df = df.rename(columns={c: c.lower() for c in df.columns})
        if "close" not in df.columns:
            return None
        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "datetime"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "datetime"})
        if "datetime" not in df.columns:
            return None
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception:
        return None

def _eodhd_download(symbol: str) -> Optional[pd.DataFrame]:
    if not EODHD_API_KEY:
        return None
    try:
        sym = symbol
        if "." not in sym:
            sym = f"{sym}.US"
        url = f"https://eodhd.com/api/eod/{sym}"
        params = {
            "api_token": EODHD_API_KEY,
            "fmt": "json",
            "period": "d",
            "from": (pd.Timestamp.utcnow() - pd.Timedelta(days=365)).date().isoformat(),
            "to": pd.Timestamp.utcnow().date().isoformat(),
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        df = pd.DataFrame(data)
        need = {"date", "close", "open", "high", "low", "volume"}
        if not need.issubset(set(df.columns)):
            return None
        df = df.rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"])
        return df
    except Exception:
        return None

def get_data(symbol: str) -> Tuple[Optional[pd.DataFrame], str, Optional[str]]:
    df = _yf_download(symbol)
    if df is not None and len(df) >= 60:
        return df, "yfinance", None
    df2 = _eodhd_download(symbol)
    if df2 is not None and len(df2) >= 60:
        return df2, "eodhd", None
    if df is not None and not df.empty:
        return df, "yfinance", "short_history"
    return None, "none", "no_data"

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    o["rsi"] = ta.rsi(o["close"], length=RSI_LEN)
    bb = ta.bbands(o["close"], length=BB_LEN, std=BB_STD)
    if bb is not None and not bb.empty:
        cols = {c.lower(): c for c in bb.columns}
        lcol = cols.get(f"bbl_{BB_LEN}_{BB_STD}".lower())
        mcol = cols.get(f"bbm_{BB_LEN}_{BB_STD}".lower())
        ucol = cols.get(f"bbu_{BB_LEN}_{BB_STD}".lower())
        if lcol:
            o["bb_lower"] = bb[lcol].values
        if mcol:
            o["bb_mid"] = bb[mcol].values
        if ucol:
            o["bb_upper"] = bb[ucol].values
    macd = ta.macd(o["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd is not None and not macd.empty:
        cols = {c.lower(): c for c in macd.columns}
        hcol = cols.get(f"macdh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}".lower())
        if hcol:
            o["macd_hist"] = macd[hcol].values
    st = ta.stoch(o["high"], o["low"], o["close"], k=STOCH_K, d=STOCH_D)
    if st is not None and not st.empty:
        cols = {c.lower(): c for c in st.columns}
        kcol = cols.get(f"stochk_{STOCH_K}_{STOCH_D}_{STOCH_D}".lower())
        dcol = cols.get(f"stochd_{STOCH_K}_{STOCH_D}_{STOCH_D}".lower())
        if kcol:
            o["stoch_k"] = st[kcol].values
        if dcol:
            o["stoch_d"] = st[dcol].values
    return o

def pct_from_lower_bb(close: float, bb_lower: float) -> Optional[float]:
    if bb_lower is None or not np.isfinite(bb_lower) or bb_lower == 0:
        return None
    return float((close - bb_lower) / abs(bb_lower) * 100.0)

def macd_improving(hist: pd.Series, n: int) -> bool:
    h = hist.dropna()
    if len(h) < n + 1:
        return False
    last = h.iloc[-(n+1):].values
    diffs = np.diff(last)
    return bool(np.all(diffs > 0))

def stoch_impulse(k: pd.Series) -> Tuple[bool, Optional[float]]:
    s = k.dropna()
    if len(s) < 3:
        return (False, None)
    k0 = float(s.iloc[-1])
    k1 = float(s.iloc[-2])
    delta = k0 - k1
    cond1 = (k1 <= STOCH_LOW_TH and delta >= STOCH_RISE_MIN)
    cond2 = (delta >= STOCH_RISE_MIN * 1.2)
    return (bool(cond1 or cond2), float(delta))

def rsi_weak(rsi: pd.Series) -> Tuple[bool, Optional[float], Optional[float]]:
    s = rsi.dropna()
    if len(s) < 3:
        return (False, None, None)
    r0 = float(s.iloc[-1])
    r1 = float(s.iloc[-2])
    delta = r0 - r1
    near = (r0 <= RSI_WEAK_MAX) or (r1 <= RSI_WEAK_NEAR and delta >= RSI_DELTA_MIN)
    return (bool(near), r0, float(delta))

def analyze_symbol(symbol: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    df, source, err = get_data(symbol)
    meta = {"symbol": symbol, "source": source, "error": err}
    if df is None or df.empty:
        return [], meta
    if len(df) < 60:
        meta["error"] = "too_short"
        return [], meta

    dfi = compute_indicators(df)
    last = dfi.iloc[-1]
    close = float(last["close"]) if np.isfinite(last.get("close", np.nan)) else None
    if close is None:
        meta["error"] = "no_close"
        return [], meta

    weak_ok, rsi_val, rsi_delta = rsi_weak(dfi["rsi"])
    if not weak_ok or rsi_val is None:
        return [], meta

    strong_parts: List[str] = []
    bb_dist = None
    if "bb_lower" in dfi.columns:
        bb_dist = pct_from_lower_bb(close, float(last.get("bb_lower", np.nan)))
        if bb_dist is not None and bb_dist <= BB_MAX_PCT_FROM_LOWER:
            strong_parts.append("BB_OK")

    macd_ok = False
    if "macd_hist" in dfi.columns:
        macd_ok = macd_improving(dfi["macd_hist"], MACD_HIST_NEED_BARS)
        if macd_ok:
            strong_parts.append("MACD_OK")

    stoch_ok, stoch_delta = (False, None)
    if "stoch_k" in dfi.columns:
        stoch_ok, stoch_delta = stoch_impulse(dfi["stoch_k"])
        if stoch_ok:
            strong_parts.append("STOCH_OK")

    strong_ok = (len(strong_parts) >= 2)

    sig_type = "STRONG_BUY" if strong_ok else "WEAK_BUY"
    reason = []
    reason.append(f"RSI_OK rsi={round(rsi_val,2)} delta={round(rsi_delta or 0.0,2)}")
    if bb_dist is not None:
        reason.append(f"BB_DIST {round(bb_dist,2)}%")
    if stoch_delta is not None:
        reason.append(f"STOCH_DELTA {round(stoch_delta,2)}")
    if strong_parts:
        reason.append("STRONG_PARTS " + ",".join(strong_parts))
    else:
        reason.append("STRONG_PARTS none")

    out = [{
        "symbol": symbol,
        "source": source,
        "price": round(close, 4),
        "rsi": round(float(rsi_val), 2),
        "rsi_delta": round(float(rsi_delta or 0.0), 2),
        "bb_distance": None if bb_dist is None else round(float(bb_dist), 2),
        "type": sig_type,
        "reason": " | ".join(reason),
    }]
    return out, meta

@app.get("/")
def root():
    return {"ok": True}

@app.get("/times")
def times():
    return nyse_next_times()

@app.get("/scan")
def scan():
    wl = read_watchlist()
    signals: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    ok_yf = 0
    ok_eod = 0
    too_short = 0
    empty_or_err = 0

    for sym in wl:
        sigs, meta = analyze_symbol(sym)
        if meta.get("error") == "too_short" or meta.get("error") == "short_history":
            too_short += 1
        if meta.get("source") == "yfinance" and meta.get("error") is None:
            ok_yf += 1
        if meta.get("source") == "eodhd" and meta.get("error") is None:
            ok_eod += 1
        if meta.get("error") is not None and meta.get("error") not in ["too_short", "short_history"]:
            empty_or_err += 1
            errors.append(meta)
        if sigs:
            signals.extend(sigs)

    return {
        "signals": signals,
        "meta": {
            "total": len(wl),
            "ok_yfinance": ok_yf,
            "ok_eodhd": ok_eod,
            "too_short": too_short,
            "empty_or_err": empty_or_err,
            "signals_count": len(signals),
            "errors_sample": errors[:3],
        },
    }
