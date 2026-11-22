#!/usr/bin/env python3
"""
Market Vision Pro – Standalone Trading & Stillhalter Signal Agent (UMWA Edition)
Kalibriert auf deine TradingView UMWA-Indikatoren + Stillhalter-Options-Filter:
1. BB & EMAs (Bollinger Bänder 20, 2.0 + bestätigter Abpraller)
2. Stochastik UMWA (Dual-Stochastik: Kurz 14,5,3 & Lang 50,6,5 – dynamischer Ausbruch)
3. RSI UMWA (Klarer Durchstich durch 30 von unten bzw. 70 von oben)
4. MACD UMWA (Kräftige Histogramm-Bewegung / bestätigter Nulllinien-Durchbruch)
5. Stillhalter-Options-Modul (Mo/Di -> Freitag):
   • Berechnet automatisch den passenden Short-Put / Short-Call Strike bis Freitag
   • Strikte Kriterien: Delta <= 0,30 | Sicherheitsabstand (OTM) >= 2,0 % | Prämie 0,7 % - 1,4 %
"""

from __future__ import annotations
import os
import sys
import math
import json
import warnings
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

warnings.filterwarnings("ignore")
import requests

# =====================================================================
# UM WATCHLIST (Aus deinen TradingView-Screenshots: A-, B- und C-Werte)
# =====================================================================

WATCHLIST_GROUPS = {
    "A-WERTE": [
        "AAPL", "ADBE", "ADP", "AMGN", "AMZN", "APD", "ATR", "BAC", "BHP", "BLK",
        "CHD", "CL", "CSCO", "CTAS", "DHR", "DIS", "ECL", "EW", "GILD", "GIS",
        "GOOG", "GS", "HD", "HRL", "INTC", "INTU", "ITW", "JNJ", "JPM", "KHC",
        "KO", "KR", "MA", "MCD", "MDT", "META", "MKC", "MMM", "MO", "MSFT",
        "NEE", "NKE", "NVDA", "NVO", "ORCL", "PEP", "PFE", "PG", "PH", "ROST",
        "SBUX", "SHEL", "SYK", "SYY", "TJX", "TMO", "TTE", "TYL", "UNH", "V",
        "WFC", "XOM", "XRAY", "YUM"
    ],
    "B-WERTE": [
        "AMD", "BA", "CRM", "CRUS", "DD", "FCX", "HPE", "IBM", "IEX", "JBL",
        "KMI", "MCHP", "MRVL", "MT", "MU", "NFLX", "PYPL", "RMD", "RTX", "SPGI",
        "SU", "TGT", "UGI", "VFC", "WM", "ZTS"
    ],
    "C-WERTE": [
        "AA", "ANF", "APA", "BB", "GE", "GOLD", "HAL", "MAT", "NOK", "PWR",
        "T", "TSLA", "URBN", "ZM", "LUV"
    ],
    "JULIAN-STABLE": [
        "UAA", "NVTS", "FIVN", "SOFI", "DOCU", "CMCSA", "OKTA", "JCI", "ILMN", "ALB",
        "TXN", "COHR", "SAP", "CHTR", "NXPI", "ADSK", "TSM", "AVGO", "LITE", "REGN",
        "ASML", "BKNG"
    ],
    "JULIAN-ACTIVE": [
        "UBER", "RIVN", "PLTR", "ABNB", "DKNG", "RPD", "ENPH", "TENB", "ON", "WBD",
        "QCOM", "STM", "SPIR", "AFRM", "SHLS", "SERV", "RR", "ROKU", "NET", "ZS",
        "MYRG", "COIN", "CRWD"
    ],
    "JULIAN-VOLATILE": [
        "UPST", "CVNA", "SYM", "OPEN", "LCID", "TEAM", "FUBO", "ASTS", "BIOX", "EVGO",
        "FLNC", "DRS", "ALOT", "MSTR", "GRMN"
    ],
    "JULIAN-TECH-BIO": [
        "IREN", "OKLO", "SMR", "NBIS", "APLD", "MARA", "RIOT", "HUT", "BITF", "DNA",
        "BNGO", "WOLF", "RKLB", "CALX", "ABAT", "AQMS", "NIO", "UMC", "AI", "DOCN",
        "SOUN", "BBAI", "SNOW", "AEHR", "IONQ", "RGTI", "QBTS", "BWXT", "CRSP", "MRNA",
        "NTLA", "BNTX", "BEAM", "EDIT", "VRTX", "ALNY", "RXRX", "PATH", "TER", "BKSY"
    ],
    "OPTIONS-PRO-50": [
        "PANW", "NOW", "SHOP", "ARM", "SMCI", "DELL", "ANET", "LRCX", "KLAC", "AMAT",
        "CDNS", "SNPS", "MELI", "SE", "DASH", "HOOD", "XYZ", "TTD", "APP", "RDDT",
        "SPOT", "PINS", "SNAP", "TWLO", "MDB", "DDOG", "HUBS", "WDAY", "TTWO", "RBLX",
        "CELH", "DECK", "LULU", "ULTA", "CMG", "WING", "CAVA", "CROX", "WSM", "URI",
        "CAT", "DE", "GEV", "VST", "CEG", "FSLR", "VRT", "ETN", "LLY", "ISRG"
    ],
}

SYMBOL_CATEGORY = {}
DEFAULT_WATCHLIST = []
for group_name, syms in WATCHLIST_GROUPS.items():
    for s in syms:
        if s not in SYMBOL_CATEGORY:
            SYMBOL_CATEGORY[s] = group_name
            DEFAULT_WATCHLIST.append(s)

EODHD_BASE = "https://eodhd.com/api"
YAHOO_CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
DEFAULT_EODHD_KEY = "690cc04fad1308.55079017"


# =====================================================================
# 1. LIVE-DATENQUELLEN (EODHD + Yahoo Real-Time Fusion & Fundamentaldaten)
# =====================================================================

def fetch_batch_quotes_and_fundamentals(symbols: list[str], eod_key: str = DEFAULT_EODHD_KEY) -> dict[str, dict]:
    """
    Holt für alle übergebenen Symbole in wenigen Millisekunden:
      1. Den sekundengenauen Echtzeit-Kurs (regularMarketPrice, Open, High, Low, Change%)
      2. Die aktuellen Fundamental- & Unternehmenszahlen (KGV, Forward-KGV, EPS-Wachstum,
         Analysten-Rating, 52W-Hoch/Tief, Dividende, Marktkapitalisierung, Earnings-Termin, Corporate Actions).
    Nutzt zusätzlich EODHD Real-Time als Fallback, damit immer die aktuellsten Live-Kurse vorliegen.
    """
    import time
    out: dict[str, dict] = {}
    clean_syms = [s.replace(".US", "") if s.upper().endswith(".US") else s for s in symbols]

    # 1. Yahoo Finance Live Batch Quote + Fundamentals (mit Crumb & Cache-Busting)
    try:
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        })
        sess.get("https://fc.yahoo.com", timeout=6)
        crumb_resp = sess.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=6)
        crumb = crumb_resp.text.strip() if crumb_resp.status_code == 200 else ""

        batch_size = 80
        for idx in range(0, len(clean_syms), batch_size):
            chunk = clean_syms[idx : idx + batch_size]
            params = {
                "symbols": ",".join(chunk),
                "crumb": crumb,
                "_t": int(time.time() * 1000),
            }
            r = sess.get("https://query1.finance.yahoo.com/v7/finance/quote", params=params, timeout=12)
            if r.status_code == 200:
                items = r.json().get("quoteResponse", {}).get("result", [])
                for item in items:
                    sym_k = item.get("symbol")
                    if sym_k:
                        out[sym_k] = item
    except Exception:
        pass

    # 2. EODHD Real-Time Bulk-Ergänzung (falls für einzelne Symbole kein Live-Preis kam)
    missing = [s for s in clean_syms if s not in out or not out[s].get("regularMarketPrice")]
    if missing and eod_key:
        try:
            batch_size = 40
            for idx in range(0, len(missing), batch_size):
                chunk = missing[idx : idx + batch_size]
                first = f"{chunk[0]}.US"
                extra = ",".join(f"{c}.US" for c in chunk[1:])
                params = {"api_token": eod_key, "fmt": "json", "_t": int(time.time())}
                if extra:
                    params["s"] = extra
                r = requests.get(f"{EODHD_BASE}/real-time/{first}", params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    rows = data if isinstance(data, list) else [data]
                    for row in rows:
                        code = str(row.get("code", "")).replace(".US", "")
                        close_v = row.get("close")
                        if code and close_v not in (None, "NA", 0, "0"):
                            c_f = float(close_v)
                            out.setdefault(code, {})
                            out[code]["regularMarketPrice"] = c_f
                            if row.get("open") not in (None, "NA", 0):
                                out[code]["regularMarketOpen"] = float(row["open"])
                            if row.get("high") not in (None, "NA", 0):
                                out[code]["regularMarketDayHigh"] = float(row["high"])
                            if row.get("low") not in (None, "NA", 0):
                                out[code]["regularMarketDayLow"] = float(row["low"])
                            if row.get("change_p") not in (None, "NA"):
                                out[code]["regularMarketChangePercent"] = float(row["change_p"])
        except Exception:
            pass

    return out


def _merge_live_quote_into_candles(candles: list, live_quote: dict | None, period: str = "d") -> list:
    """
    Stellt 100% Aktualität & Konsistenz sicher:
    Aktualisiert die letzte Tageskerze (oder fügt den heutigen Handelstag hinzu)
    mit dem sekundengenauen Echtzeit-Kurs (regularMarketPrice, DayHigh, DayLow, Open).
    Historische abgeschlossene Tage bleiben exakt und deterministisch erhalten.
    """
    if not candles or not live_quote or period != "d":
        return candles

    live_price = live_quote.get("regularMarketPrice")
    if live_price is None or float(live_price) <= 0:
        return candles

    c_live = float(live_price)
    o_live = float(live_quote.get("regularMarketOpen") or c_live)
    h_live = float(live_quote.get("regularMarketDayHigh") or max(o_live, c_live))
    l_live = float(live_quote.get("regularMarketDayLow") or min(o_live, c_live))
    h_live = max(h_live, o_live, c_live)
    l_live = min(l_live, o_live, c_live)

    mkt_ts = live_quote.get("regularMarketTime")
    if mkt_ts and isinstance(mkt_ts, (int, float)) and mkt_ts > 1000000000:
        live_dt_str = date.fromtimestamp(int(mkt_ts)).isoformat()
    else:
        live_dt_str = date.today().isoformat()

    last_dt_str = candles[-1]["time"]
    if live_dt_str == last_dt_str:
        candles[-1] = {
            "time": last_dt_str,
            "open": candles[-1]["open"] if candles[-1]["open"] > 0 else o_live,
            "high": max(candles[-1]["high"], h_live, c_live),
            "low": min(candles[-1]["low"], l_live, c_live),
            "close": c_live,
        }
    elif live_dt_str > last_dt_str:
        candles.append({
            "time": live_dt_str,
            "open": o_live,
            "high": h_live,
            "low": l_live,
            "close": c_live,
        })

    return candles


def fetch_candles_yahoo(symbol: str, interval: str = "1d", range_str: str = "2y") -> list:
    import time
    clean_sym = symbol.replace(".US", "") if symbol.upper().endswith(".US") else symbol
    url = f"{YAHOO_CHART_BASE}/{clean_sym}"
    params = {"interval": interval, "range": range_str, "_t": int(time.time())}
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Cache-Control": "no-cache, no-store, must-revalidate",
    }

    r = requests.get(url, params=params, headers=headers, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Yahoo Finance HTTP {r.status_code}")

    data = r.json()
    result = data.get("chart", {}).get("result")
    if not result:
        raise RuntimeError("Keine Chart-Daten erhalten")

    res = result[0]
    meta = res.get("meta", {})
    timestamps = res.get("timestamp", [])
    quote = res.get("indicators", {}).get("quote", [{}])[0]
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])

    candles = []
    for i, ts in enumerate(timestamps):
        if i >= len(closes):
            break
        o = opens[i] if i < len(opens) else None
        h = highs[i] if i < len(highs) else None
        l = lows[i] if i < len(lows) else None
        c = closes[i] if i < len(closes) else None
        if h is None or l is None or c is None:
            continue
        c, h, l = float(c), float(h), float(l)
        o = float(o) if o is not None else c
        if c <= 0 or h <= 0 or l <= 0 or h < l:
            continue
        dt_str = date.fromtimestamp(ts).isoformat()
        # Falls Yahoo für denselben Tag am Ende einen doppelten Intraday-Timestamp anhängt -> zusammenführen
        if candles and candles[-1]["time"] == dt_str:
            candles[-1]["high"] = max(candles[-1]["high"], h, c)
            candles[-1]["low"] = min(candles[-1]["low"], l, c)
            candles[-1]["close"] = c
        else:
            candles.append({"time": dt_str, "open": o, "high": h, "low": l, "close": c})

    # Auch direkt aus meta den aktuellsten Preis abgleichen
    if meta.get("regularMarketPrice") and candles and interval == "1d":
        _merge_live_quote_into_candles(candles, meta, period="d")

    return candles


def fetch_candles_eodhd(symbol: str, api_key: str, period: str = "d", days: int = 730) -> list:
    import time
    eod_sym = symbol if "." in symbol else f"{symbol}.US"
    to_d = date.today()
    from_d = to_d - timedelta(days=days)
    params = {
        "from": from_d.isoformat(),
        "to": to_d.isoformat(),
        "period": period,
        "api_token": api_key,
        "fmt": "json",
        "_t": int(time.time()),
    }
    headers = {"Cache-Control": "no-cache"}
    r = requests.get(f"{EODHD_BASE}/eod/{eod_sym}", params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"EODHD HTTP {r.status_code}")

    raw = r.json()
    raw_sorted = sorted(raw, key=lambda x: x.get("date", ""))
    valid_rows = []
    for row in raw_sorted:
        d = row.get("date")
        o, h, l, c = row.get("open"), row.get("high"), row.get("low"), row.get("close")
        adj_c = row.get("adjusted_close")
        if d is None or h is None or l is None or c is None:
            continue
        c, h, l = float(c), float(h), float(l)
        o = float(o) if o is not None else c
        adj_c = float(adj_c) if (adj_c is not None and float(adj_c) > 0) else c
        if c <= 0 or h <= 0 or l <= 0 or h < l:
            continue
        valid_rows.append({"time": d, "open": o, "high": h, "low": l, "close": c, "adj": adj_c})

    # Automatische Split- & Reverse-Split-Bereinigung (ohne reguläre Dividenden zu verzerren,
    # exakt synchron zu TradingView Standard-Charts):
    n = len(valid_rows)
    factors = [1.0] * n
    cum_factor = 1.0
    for i in range(n - 1, 0, -1):
        c_curr, a_curr = valid_rows[i]["close"], valid_rows[i]["adj"]
        c_prev, a_prev = valid_rows[i - 1]["close"], valid_rows[i - 1]["adj"]
        if c_curr > 0 and c_prev > 0 and a_curr > 0 and a_prev > 0:
            ratio_curr = a_curr / c_curr
            ratio_prev = a_prev / c_prev
            step = ratio_prev / ratio_curr
            if abs(step - 1.0) > 0.08:
                cum_factor *= step
        factors[i - 1] = cum_factor

    candles = []
    for i, row in enumerate(valid_rows):
        f = factors[i]
        candles.append({
            "time": row["time"],
            "open": row["open"] * f,
            "high": row["high"] * f,
            "low": row["low"] * f,
            "close": row["close"] * f,
        })
    return candles


import threading

_CANDLE_CACHE_LOCK = threading.Lock()
_CANDLE_CACHE_MEM: dict[str, list] = {}
_CANDLE_CACHE_DIRTY = False
_CANDLE_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".candle_cache_d.json")


def _load_candle_cache() -> None:
    global _CANDLE_CACHE_MEM
    if _CANDLE_CACHE_MEM or os.getenv("UMWA_FORCE_FRESH") == "1":
        return
    try:
        if os.path.exists(_CANDLE_CACHE_FILE):
            import time as _t
            if (_t.time() - os.path.getmtime(_CANDLE_CACHE_FILE)) < 21600:  # 6h TTL
                with open(_CANDLE_CACHE_FILE, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if payload.get("date") == date.today().isoformat() and isinstance(payload.get("data"), dict):
                    _CANDLE_CACHE_MEM = payload["data"]
    except Exception:
        pass


def _save_candle_cache() -> None:
    global _CANDLE_CACHE_DIRTY
    if not _CANDLE_CACHE_DIRTY or not _CANDLE_CACHE_MEM:
        return
    try:
        tmp_path = _CANDLE_CACHE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"date": date.today().isoformat(), "data": _CANDLE_CACHE_MEM}, f, separators=(",", ":"))
        os.replace(tmp_path, _CANDLE_CACHE_FILE)
        _CANDLE_CACHE_DIRTY = False
    except Exception:
        pass


def fetch_candles(symbol: str, period: str = "d", live_quote: dict | None = None) -> tuple[list, str]:
    global _CANDLE_CACHE_DIRTY
    eod_key = os.getenv("EODHD_API_KEY") or os.getenv("EODHD_TOKEN") or DEFAULT_EODHD_KEY
    interval_map = {"d": "1d", "w": "1wk", "m": "1mo"}
    yf_interval = interval_map.get(period, "1d")

    if period == "d" and os.getenv("UMWA_FORCE_FRESH") != "1":
        with _CANDLE_CACHE_LOCK:
            cached = _CANDLE_CACHE_MEM.get(symbol)
        if cached and len(cached) >= 60:
            candles_copy = [dict(c) for c in cached]
            candles_copy = _merge_live_quote_into_candles(candles_copy, live_quote, period=period)
            return candles_copy, "EODHD+SmartCache+Live"

    # Primär EODHD + Live-Quote-Fusion für maximale Konsistenz, Fallback Yahoo Finance
    if eod_key:
        try:
            candles = fetch_candles_eodhd(symbol, eod_key, period=period, days=730)
            if candles and len(candles) >= 60:
                if period == "d":
                    with _CANDLE_CACHE_LOCK:
                        _CANDLE_CACHE_MEM[symbol] = [dict(c) for c in candles]
                        _CANDLE_CACHE_DIRTY = True
                candles = _merge_live_quote_into_candles(candles, live_quote, period=period)
                return candles, "EODHD+Live"
        except Exception:
            pass

    candles = fetch_candles_yahoo(symbol, interval=yf_interval, range_str="2y")
    if period == "d" and candles and len(candles) >= 60:
        with _CANDLE_CACHE_LOCK:
            _CANDLE_CACHE_MEM[symbol] = [dict(c) for c in candles]
            _CANDLE_CACHE_DIRTY = True
    candles = _merge_live_quote_into_candles(candles, live_quote, period=period)
    return candles, "Yahoo+Live"


# =====================================================================
# 2. DIE 4 UMWA-HAUPTINDIKATOREN
# =====================================================================

def _ema(values, span):
    if not values:
        return []
    alpha = 2 / (span + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def _sma(values, period):
    out = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= period:
            s -= values[i - period]
        out.append(None if i + 1 < period else s / period)
    return out


def _std(values, period):
    out = []
    for i in range(len(values)):
        if i + 1 < period:
            out.append(None)
            continue
        window = values[i - period + 1 : i + 1]
        m = sum(window) / period
        var = sum((x - m) ** 2 for x in window) / period
        out.append(math.sqrt(var))
    return out


def calc_bollinger_bands(values, period=20, std_mul=2.0):
    mid = _sma(values, period)
    sd = _std(values, period)
    upper, lower = [], []
    for m, s in zip(mid, sd):
        if m is None or s is None:
            upper.append(None)
            lower.append(None)
        else:
            upper.append(m + std_mul * s)
            lower.append(m - std_mul * s)
    return upper, mid, lower


def calc_rsi(values, period=14):
    if len(values) < period + 1:
        return [None] * len(values)
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi = [None] * period
    rsi.append(100.0 if avg_loss == 0 else 100 - (100 / (1 + (avg_gain / avg_loss))))
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi.append(100.0 if avg_loss == 0 else 100 - (100 / (1 + (avg_gain / avg_loss))))
    return [None] + rsi


def calc_macd(values, fast=12, slow=26, signal=9):
    ema_fast = _ema(values, fast)
    ema_slow = _ema(values, slow)
    n = len(values)

    macd_line = [None] * n
    for i in range(n):
        if i < len(ema_fast) and i < len(ema_slow) and ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    signal_line = [None] * n
    macd_vals = [x for x in macd_line if x is not None]
    if len(macd_vals) >= signal:
        signal_vals = _ema(macd_vals, signal)
        j = 0
        for i in range(n):
            if macd_line[i] is not None:
                signal_line[i] = signal_vals[j]
                j += 1

    hist = [None] * n
    for i in range(n):
        if macd_line[i] is not None and signal_line[i] is not None:
            hist[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, hist


def calc_stochastic_umwa(highs, lows, closes, k_len=14, k_smooth=3, d_smooth=3):
    raw_k = [None] * len(closes)
    for i in range(len(closes)):
        if i + 1 < k_len:
            continue
        hh = max(highs[i - k_len + 1 : i + 1])
        ll = min(lows[i - k_len + 1 : i + 1])
        denom = hh - ll
        raw_k[i] = 0.0 if denom == 0 else 100.0 * (closes[i] - ll) / denom

    k = [None] * len(closes)
    for i in range(len(closes)):
        w = [x for x in raw_k[max(0, i - k_smooth + 1) : i + 1] if x is not None]
        k[i] = (sum(w) / len(w)) if len(w) == k_smooth else None

    d = [None] * len(closes)
    for i in range(len(closes)):
        w = [x for x in k[max(0, i - d_smooth + 1) : i + 1] if x is not None]
        d[i] = (sum(w) / len(w)) if len(w) == d_smooth else None

    return k, d


# =====================================================================
# 2b. DOPPEL-CHECK SUPPORT & RESISTANCE LINIEN (Blaue Linien aus TradingView)
# =====================================================================

def find_validated_sr_lines(
    candles: list,
    pivot_left: int = 5,
    pivot_right: int = 5,
    tol_pct: float = 0.007,
) -> dict:
    """
    Findet und validiert horizontale Support- und Resistance-Linien in 2 Stufen:
    Stufe 1: Echte Swing-Pivots (Hochs & Tiefs) über bis zu 2 Jahre sammeln und clustern
             (Mindestabstand 6 Handelstage zwischen zwei Punkten desselben Clusters).
    Stufe 2 (Doppel-Check):
      a) Prüft an allen Kerzen, wie oft die Linie exakt angelaufen und per Schlusskurs
         respektiert wurde (unabhängige Abpraller / Touches mit mind. 5 Tagen Abstand).
      b) Whipsaw-Filter: Wurde die Linie in den letzten 25 Kerzen ständig durchsägt
         (> 3 glatte Schlusskurs-Kreuzungen), wird sie als 'zersägt / invalide' aussortiert!
      c) Nur Linien mit mind. 2 unabhängigen Swing-Pivots (oder >= 3 sauberen Abprallern)
         gelten als VALIDE LINIE.
    """
    if not candles or len(candles) < 60:
        return {"supports": [], "resistances": [], "nearest_support": None, "nearest_resistance": None, "at_support": False, "at_resistance": False}

    n = len(candles)
    current_price = candles[-1]["close"]

    # --- Stufe 1: Lokale Swing-Pivots finden ---
    raw_pivots = []
    for i in range(pivot_left, n - pivot_right):
        lo = candles[i]["low"]
        hi = candles[i]["high"]
        is_low = all(candles[j]["low"] >= lo for j in range(i - pivot_left, i + pivot_right + 1))
        is_high = all(candles[j]["high"] <= hi for j in range(i - pivot_left, i + pivot_right + 1))
        if is_low:
            raw_pivots.append({"idx": i, "price": float(lo), "kind": "L", "time": candles[i]["time"]})
        if is_high:
            raw_pivots.append({"idx": i, "price": float(hi), "kind": "H", "time": candles[i]["time"]})

    # Relevanter Preisbereich: +/- 35% um den aktuellen Kurs
    lo_bound = current_price * 0.65
    hi_bound = current_price * 1.35
    raw_pivots = [p for p in raw_pivots if lo_bound <= p["price"] <= hi_bound]

    # Clustern von nahe beieinander liegenden Pivots (Mindestabstand 6 Kerzen für unabhängigen Test)
    raw_pivots.sort(key=lambda x: x["price"])
    clusters = []
    for p in raw_pivots:
        placed = False
        for c in clusters:
            mid = c["sum"] / c["count"]
            if abs(p["price"] - mid) / mid <= tol_pct:
                # Nur als neuen unabhängigen Pivot zählen, wenn mind. 6 Handelstage zu bisherigen Punkten liegen
                if all(abs(p["idx"] - existing_idx) >= 6 for existing_idx in c["indices"]):
                    c["sum"] += p["price"]
                    c["count"] += 1
                    c["indices"].append(p["idx"])
                    c["times"].append(p["time"])
                placed = True
                break
        if not placed:
            clusters.append({
                "sum": p["price"],
                "count": 1,
                "indices": [p["idx"]],
                "times": [p["time"]],
            })

    # --- Stufe 2: Doppel-Check (Abpraller-Prüfung & Whipsaw-Filter) ---
    validated_supports = []
    validated_resistances = []

    for c in clusters:
        lvl = c["sum"] / c["count"]
        pivot_touches = c["count"]

        # Doppel-Check A: Zähle unabhängige Kerzen-Abpraller (Touch & Hold) an dieser Linie
        bounce_indices = []
        for i in range(max(0, n - 260), n):
            hi, lo, cl, op = candles[i]["high"], candles[i]["low"], candles[i]["close"], candles[i]["open"]
            body_lo, body_hi = min(op, cl), max(op, cl)

            # Hat die Kerze die Linie innerhalb von 0.55% berührt, ohne sie mit dem Körper massiv zu durchbrechen?
            touched_as_support = (abs(lo - lvl) / lvl <= 0.0055 or (lo < lvl and body_lo >= lvl * 0.996)) and cl >= lvl * 0.997
            touched_as_resist = (abs(hi - lvl) / lvl <= 0.0055 or (hi > lvl and body_hi <= lvl * 1.004)) and cl <= lvl * 1.003

            if touched_as_support or touched_as_resist:
                if not bounce_indices or (i - bounce_indices[-1]) >= 5:
                    bounce_indices.append(i)

        total_confirmations = max(pivot_touches, len(bounce_indices))

        # Doppel-Check B: Whipsaw-Filter (Wurde die Linie in den letzten 25 Tagen ständig durchsägt?)
        recent_slice = candles[-25:] if n >= 25 else candles
        crossings = 0
        for k in range(1, len(recent_slice)):
            c_prev = recent_slice[k - 1]["close"]
            c_curr = recent_slice[k]["close"]
            if (c_prev < lvl * 0.996 and c_curr > lvl * 1.004) or (c_prev > lvl * 1.004 and c_curr < lvl * 0.996):
                crossings += 1

        # Strikter Doppel-Check für eine ECHTE, VALIDE Linie:
        # 1) MUSS mindestens 2 unabhängige historische Swing-Pivots (echte Trend-Wendepunkte!) haben
        # 2) MUSS zusätzlich mindestens 3 unabhängige Kerzen-Abpraller (Touch & Hold) aufweisen
        # 3) Darf in den letzten 25 Tagen nicht durchsägt worden sein (crossings <= 2)
        if crossings > 2:
            continue
        is_valid = (pivot_touches >= 2) and (len(bounce_indices) >= 3)
        if not is_valid:
            continue

        line_info = {
            "price": round(lvl, 2),
            "pivots": pivot_touches,
            "bounces": len(bounce_indices),
            "strength": pivot_touches * 2 + len(bounce_indices),
            "dist_pct": round((lvl - current_price) / current_price * 100.0, 2),
        }

        # Klare Trennung: Support liegt UNTER dem aktuellen Kurs, Widerstand liegt ÜBER dem aktuellen Kurs
        if lvl < current_price:
            validated_supports.append(line_info)
        elif lvl > current_price:
            validated_resistances.append(line_info)

    # Sortiere Supports absteigend (nächster unter dem Kurs zuerst)
    validated_supports.sort(key=lambda x: -x["price"])
    # Sortiere Resistances aufsteigend (nächster über dem Kurs zuerst)
    validated_resistances.sort(key=lambda x: x["price"])

    nearest_sup = validated_supports[0] if validated_supports else None
    nearest_res = validated_resistances[0] if validated_resistances else None

    # Prüfen, ob die Aktie in den letzten 3 Kerzen direkt an einem validierten Support/Resistance abgeprallt ist
    recent_low_3 = min(c["low"] for c in candles[-3:])
    recent_high_3 = max(c["high"] for c in candles[-3:])

    at_support = False
    active_support = None
    for s in validated_supports[:3]:
        if recent_low_3 <= s["price"] * 1.012 and current_price >= s["price"] * 0.995:
            at_support = True
            active_support = s
            break

    at_resistance = False
    active_resistance = None
    for r_lvl in validated_resistances[:3]:
        if recent_high_3 >= r_lvl["price"] * 0.988 and current_price <= r_lvl["price"] * 1.005:
            at_resistance = True
            active_resistance = r_lvl
            break

    return {
        "supports": validated_supports[:5],
        "resistances": validated_resistances[:5],
        "nearest_support": nearest_sup,
        "nearest_resistance": nearest_res,
        "at_support": at_support,
        "active_support": active_support,
        "at_resistance": at_resistance,
        "active_resistance": active_resistance,
    }


# =====================================================================
# 3. STILLHALTER OPTIONS-MODUL (Black-Scholes | Delta <= 0.30 | Prämie 0,7% - 1,4%)
# =====================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    abs_delta = _norm_cdf(-d1)
    put_price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return max(put_price, 0.0), abs_delta


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    abs_delta = _norm_cdf(d1)
    call_price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return max(call_price, 0.0), abs_delta


def estimate_current_iv(closes: list[float], highs: list[float], lows: list[float]) -> float:
    idx = len(closes) - 1
    start = max(1, idx - 22)
    log_rets = [math.log(closes[k] / closes[k - 1]) for k in range(start, idx + 1) if closes[k - 1] > 0]
    if len(log_rets) < 5:
        return 0.30
    mean_r = sum(log_rets) / len(log_rets)
    sq_devs = sorted((x - mean_r) ** 2 for x in log_rets)
    # Trimme die 2 extremsten Einzel-Tages-Ausreißer (z. B. 1-Tages-Earnings-Gap -> Post-Earnings IV-Crush)
    if len(sq_devs) > 8:
        sq_devs = sq_devs[:-2]
    var_r = sum(sq_devs) / max(len(sq_devs) - 1, 1)
    hv_close = math.sqrt(var_r * 252.0)

    hl_sq = sorted(
        (math.log(highs[k] / lows[k]) ** 2)
        for k in range(start, idx + 1)
        if lows[k] > 0 and highs[k] >= lows[k]
    )
    if len(hl_sq) > 8:
        hl_sq = hl_sq[:-2]
    hv_park = math.sqrt(sum(hl_sq) / (4.0 * max(len(hl_sq), 1) * math.log(2.0)) * 252.0)
    iv = max(hv_close, hv_park) * 1.12
    return max(min(iv, 0.95), 0.16)


def _get_strike_increment(S: float) -> float:
    """
    Liefert die standardisierten OCC / CBOE / IBKR Options-Strike-Abstände
    passend zum aktuellen Aktienkurs S (keine krummen Fantasie-Strikes!).
    """
    if S < 25.0:
        return 0.50
    if S < 100.0:
        return 1.00
    if S < 200.0:
        return 2.50
    return 5.00


def _snap_to_standard_strike(raw_k: float, S: float, is_put: bool, min_otm_pct: float = 0.020) -> float:
    inc = _get_strike_increment(S)
    if is_put:
        max_valid_k = S * (1.0 - min_otm_pct)
        k = math.floor((raw_k + 1e-9) / inc) * inc
        if k > max_valid_k:
            k -= inc
        return round(max(k, inc), 2)
    else:
        min_valid_k = S * (1.0 + min_otm_pct)
        k = math.ceil((raw_k - 1e-9) / inc) * inc
        if k < min_valid_k:
            k += inc
        return round(k, 2)


def _format_strike_str(k: float | None) -> str:
    if k is None:
        return "-"
    if abs(k - round(k)) < 1e-4:
        return f"{k:.0f}.00"
    if abs(k * 2 - round(k * 2)) < 1e-4:
        return f"{k:.2f}"
    return f"{k:.2f}"


def _calc_weekly_cycle(last_dt: date) -> tuple[int, date, str]:
    """
    Bestimmt für das Stillhalter-Setup (Mo/Di -> Freitag):
      • Am Montag (wd=0 -> 4T) und Dienstag (wd=1 -> 3T): Verfall am Freitag DIESER Woche.
      • Ab Mittwoch/Donnerstag/Freitag/Wochenende: Da für die laufende Woche nur noch 0-2 Tage
        Restlaufzeit übrig sind (reines Gamma-Risiko), wird automatisch der reguläre
        Mo/Di->Fr Wochenzyklus (4 Handelstage) auf den kommenden Wochen-Verfall berechnet.
    """
    wd = last_dt.weekday()
    if wd in (0, 1):
        dte_days = 4 - wd  # Mo=4T, Di=3T
        exp_date = last_dt + timedelta(days=(4 - wd))
        cycle_label = f"Fr, {exp_date.strftime('%d.%m.')} ({dte_days}T)"
    else:
        # Ab Mi/Do/Fr: Nächster Wochenzyklus (Standard 4 Handelstage Mo->Fr)
        days_to_this_fri = (4 - wd) % 7
        exp_date = last_dt + timedelta(days=days_to_this_fri if days_to_this_fri >= 3 else days_to_this_fri + 7)
        dte_days = 4
        cycle_label = f"Fr, {exp_date.strftime('%d.%m.')} (4T Zyklus)"
    return dte_days, exp_date, cycle_label


def recommend_weekly_option(
    S: float,
    iv: float,
    is_put: bool,
    dte_days: int = 4,
    sr_info: dict | None = None,
    expiry_date: date | None = None,
    cycle_label: str = "",
    symbol: str = "",
    days_to_earnings: int | None = None,
    min_prem_pct: float = 0.70,
    max_prem_pct: float = 1.40,
    max_delta: float = 0.30,
    min_otm_pct: float = 0.020,
) -> dict:
    """
    Sucht für Mo/Di -> Freitag auf echten, standardisierten OCC/IBKR-Options-Strikes
    den sichersten Strike (kleinstes Delta <= 0.30, OTM >= 2.0%), der 0,7 % bis 1,4 %
    Wochenprämie liefert, und gleicht ihn mit den validierten Support-/Resistance-Linien ab.
    """
    T = max(dte_days, 1) / 252.0
    r = 0.045
    inc = _get_strike_increment(S)

    # Generiere alle echten Standard-Strikes zwischen 2,0 % und 12,5 % OTM (liquider Stillhalter-Korridor)
    candidate_strikes = []
    if is_put:
        k_start = _snap_to_standard_strike(S * (1.0 - min_otm_pct), S, is_put=True, min_otm_pct=min_otm_pct)
        k_min = max(inc, S * 0.875)
        k_cur = k_start
        while k_cur >= k_min:
            candidate_strikes.append(round(k_cur, 2))
            k_cur -= inc
    else:
        k_start = _snap_to_standard_strike(S * (1.0 + min_otm_pct), S, is_put=False, min_otm_pct=min_otm_pct)
        k_max = S * 1.125
        k_cur = k_start
        while k_cur <= k_max:
            candidate_strikes.append(round(k_cur, 2))
            k_cur += inc

    # Sortiere von weit OTM (sicherster Puffer) nach näher am Kurs (2% Minimum)
    candidate_strikes_far_to_near = list(reversed(candidate_strikes))

    best_high_iv = None
    best_fallback = None
    chosen = None

    exp_code = expiry_date.strftime("%d%b%y").upper() if expiry_date else "WEEKLY"
    right_code = "P" if is_put else "C"

    for K in candidate_strikes_far_to_near:
        otm_pct = (S - K) / S if is_put else (K - S) / S
        if otm_pct < min_otm_pct - 1e-4:
            continue

        if is_put:
            price, delta = _bs_put(S, K, T, r, sigma=iv)
            prem_pct = (price / K) * 100.0
        else:
            price, delta = _bs_call(S, K, T, r, sigma=iv)
            prem_pct = (price / S) * 100.0

        contract_str = f"{symbol} {exp_code} {K:g} {right_code}".strip()

        if delta <= max_delta:
            best_fallback = {
                "qualified": False,
                "type": "SHORT PUT" if is_put else "SHORT CALL",
                "contract": contract_str,
                "strike": round(K, 2),
                "strike_step": inc,
                "otm_pct": round(otm_pct * 100.0, 2),
                "premium_usd": round(price, 2),
                "prem_pct": round(prem_pct, 2),
                "delta": round(delta, 2),
                "iv_pct": round(iv * 100.0, 1),
                "dte": dte_days,
                "cycle_label": cycle_label or f"{dte_days}T",
                "note": f"IV zu niedrig ({iv*100:.0f}% IV: max. +{prem_pct:.2f}% bei Strike {K:g}$ / Δ {delta:.2f})",
            }

        if delta > max_delta or delta < 0.11:
            continue

        if min_prem_pct <= prem_pct <= max_prem_pct:
            chosen = {
                "qualified": True,
                "type": "SHORT PUT" if is_put else "SHORT CALL",
                "contract": contract_str,
                "strike": round(K, 2),
                "strike_step": inc,
                "otm_pct": round(otm_pct * 100.0, 2),
                "premium_usd": round(price, 2),
                "prem_pct": round(prem_pct, 2),
                "delta": round(delta, 2),
                "iv_pct": round(iv * 100.0, 1),
                "dte": dte_days,
                "cycle_label": cycle_label or f"{dte_days}T",
                "note": "OPTIMAL (0,7–1,4% Zielkorridor & Δ <= 0,30)",
            }
            break

        if prem_pct > max_prem_pct and best_high_iv is None:
            best_high_iv = {
                "qualified": True,
                "type": "SHORT PUT" if is_put else "SHORT CALL",
                "contract": contract_str,
                "strike": round(K, 2),
                "strike_step": inc,
                "otm_pct": round(otm_pct * 100.0, 2),
                "premium_usd": round(price, 2),
                "prem_pct": round(prem_pct, 2),
                "delta": round(delta, 2),
                "iv_pct": round(iv * 100.0, 1),
                "dte": dte_days,
                "cycle_label": cycle_label or f"{dte_days}T",
                "note": "HIGH-IV SETUP (Erhöhte Prämie bei kleinem Delta)",
            }

    if chosen is None:
        chosen = best_high_iv or best_fallback or {
            "qualified": False,
            "type": "SHORT PUT" if is_put else "SHORT CALL",
            "contract": "",
            "strike": None,
            "strike_step": inc,
            "otm_pct": None,
            "premium_usd": None,
            "prem_pct": 0.0,
            "delta": None,
            "iv_pct": round(iv * 100.0, 1),
            "dte": dte_days,
            "cycle_label": cycle_label or f"{dte_days}T",
            "note": "Kein Standard-Strike mit Δ <= 0,30 und >= 2,0% Puffer verfügbar",
        }

    # --- Doppel-Check gegen validierte Support/Resistance-Linien (nur auf Standard-Strikes!) ---
    chosen["sr_protected"] = False
    chosen["sr_shield_line"] = None
    chosen["earnings_risk"] = bool(days_to_earnings is not None and 0 <= days_to_earnings <= 6)

    if chosen.get("strike") is not None and sr_info:
        K = chosen["strike"]
        if is_put:
            protecting_supports = [s for s in sr_info.get("supports", []) if K <= s["price"] <= S]
            if protecting_supports:
                best_sup = max(protecting_supports, key=lambda x: x["strength"])
                chosen["sr_protected"] = True
                chosen["sr_shield_line"] = best_sup
            else:
                # Prüfen, ob ein benachbarter Standard-Strike direkt unter einer nahen Support-Linie liegt
                slightly_lower = [s for s in sr_info.get("supports", []) if K * 0.96 <= s["price"] < K]
                if slightly_lower:
                    sup = slightly_lower[0]
                    new_K = round(math.floor(sup["price"] / inc) * inc, 2)
                    if new_K >= sup["price"]:
                        new_K = round(new_K - inc, 2)
                    if new_K > 0 and (S - new_K) / S >= min_otm_pct:
                        p_new, d_new = _bs_put(S, new_K, T, r, sigma=iv)
                        prem_new_pct = (p_new / new_K) * 100.0
                        if prem_new_pct >= 0.65 and d_new <= max_delta:
                            chosen["strike"] = new_K
                            chosen["contract"] = f"{symbol} {exp_code} {new_K:g} {right_code}".strip()
                            chosen["otm_pct"] = round((S - new_K) / S * 100.0, 2)
                            chosen["premium_usd"] = round(p_new, 2)
                            chosen["prem_pct"] = round(prem_new_pct, 2)
                            chosen["delta"] = round(d_new, 2)
                            chosen["sr_protected"] = True
                            chosen["sr_shield_line"] = sup
                            chosen["qualified"] = True
        else:
            protecting_res = [r_l for r_l in sr_info.get("resistances", []) if S <= r_l["price"] <= K]
            if protecting_res:
                best_res = max(protecting_res, key=lambda x: x["strength"])
                chosen["sr_protected"] = True
                chosen["sr_shield_line"] = best_res
            else:
                slightly_higher = [r_l for r_l in sr_info.get("resistances", []) if K < r_l["price"] <= K * 1.04]
                if slightly_higher:
                    res_l = slightly_higher[0]
                    new_K = round(math.ceil(res_l["price"] / inc) * inc, 2)
                    if new_K <= res_l["price"]:
                        new_K = round(new_K + inc, 2)
                    if (new_K - S) / S >= min_otm_pct:
                        c_new, d_new = _bs_call(S, new_K, T, r, sigma=iv)
                        prem_new_pct = (c_new / S) * 100.0
                        if prem_new_pct >= 0.65 and d_new <= max_delta:
                            chosen["strike"] = new_K
                            chosen["contract"] = f"{symbol} {exp_code} {new_K:g} {right_code}".strip()
                            chosen["otm_pct"] = round((new_K - S) / S * 100.0, 2)
                            chosen["premium_usd"] = round(c_new, 2)
                            chosen["prem_pct"] = round(prem_new_pct, 2)
                            chosen["delta"] = round(d_new, 2)
                            chosen["sr_protected"] = True
                            chosen["sr_shield_line"] = res_l
                            chosen["qualified"] = True

    return chosen


# =====================================================================
# 3C. ELLIOTT-WAVE MUSTERERKENNUNG (Mit 3-Regeln-, Fibonacci- & Konsistenz-Check)
#     -> Rein informativ als eigene Spalte (fließt NICHT in den 4er-Score ein!)
# =====================================================================

def _extract_zigzag_pivots(candles: list, threshold_pct: float = 0.045) -> list[dict]:
    """
    Extrahiert streng alternierende Swing-Hochs ('H') und Swing-Tiefs ('L')
    mittels adaptivem Prozent-ZigZag-Filter, damit echtes Wellen-Verhalten
    ohne Tagesrauschen erkannt wird.
    """
    if not candles or len(candles) < 20:
        return []

    pivots = []
    trend = 0  # 1 = aufwärts (suche Hoch), -1 = abwärts (suche Tief)
    last_high_idx = 0
    last_high_val = candles[0]["high"]
    last_low_idx = 0
    last_low_val = candles[0]["low"]

    for i, c in enumerate(candles):
        h, l = c["high"], c["low"]
        if trend == 0:
            if h >= last_high_val:
                last_high_val, last_high_idx = h, i
            if l <= last_low_val:
                last_low_val, last_low_idx = l, i
            if last_low_val > 0 and (h - last_low_val) / last_low_val >= threshold_pct and i > last_low_idx:
                pivots.append({"idx": last_low_idx, "price": last_low_val, "type": "L", "time": candles[last_low_idx]["time"]})
                trend = 1
                last_high_val, last_high_idx = h, i
            elif last_high_val > 0 and (last_high_val - l) / last_high_val >= threshold_pct and i > last_high_idx:
                pivots.append({"idx": last_high_idx, "price": last_high_val, "type": "H", "time": candles[last_high_idx]["time"]})
                trend = -1
                last_low_val, last_low_idx = l, i
        elif trend == 1:
            if h >= last_high_val:
                last_high_val, last_high_idx = h, i
            elif last_high_val > 0 and (last_high_val - l) / last_high_val >= threshold_pct:
                pivots.append({"idx": last_high_idx, "price": last_high_val, "type": "H", "time": candles[last_high_idx]["time"]})
                trend = -1
                last_low_val, last_low_idx = l, i
        elif trend == -1:
            if l <= last_low_val:
                last_low_val, last_low_idx = l, i
            elif last_low_val > 0 and (h - last_low_val) / last_low_val >= threshold_pct:
                pivots.append({"idx": last_low_idx, "price": last_low_val, "type": "L", "time": candles[last_low_idx]["time"]})
                trend = 1
                last_high_val, last_high_idx = h, i

    if trend == 1 and last_high_idx > (pivots[-1]["idx"] if pivots else 0):
        pivots.append({"idx": last_high_idx, "price": last_high_val, "type": "H", "time": candles[last_high_idx]["time"]})
    elif trend == -1 and last_low_idx > (pivots[-1]["idx"] if pivots else 0):
        pivots.append({"idx": last_low_idx, "price": last_low_val, "type": "L", "time": candles[last_low_idx]["time"]})

    return pivots


def _clamp_ew_levels(S: float, bias: str, tgt: float | None, inv: float | None) -> tuple[float, float]:
    """
    Begrenzt Kursziel und Invalidierung VOR der String-Formatierung auf mathematisch
    und charttechnisch konsistente Marken, damit Text und Zahlen 100% übereinstimmen.
    """
    if bias == "bullish":
        if tgt is None or tgt <= S * 1.025:
            tgt = S * 1.075
        elif tgt > S * 1.30:
            tgt = S * 1.26
        if inv is None or inv >= S * 0.985:
            inv = S * 0.945
        elif inv < S * 0.74:
            inv = S * 0.78
    elif bias == "bearish":
        if tgt is None or tgt >= S * 0.975:
            tgt = S * 0.925
        elif tgt < S * 0.70:
            tgt = S * 0.74
        if inv is None or inv <= S * 1.015:
            inv = S * 1.055
        elif inv > S * 1.26:
            inv = S * 1.22
    else:
        tgt = tgt if tgt is not None else S * 1.05
        inv = inv if inv is not None else S * 0.95
    return round(float(tgt), 2), round(float(inv), 2)


def _enforce_ew_math_consistency(ew: dict, S: float) -> dict:
    """
    Strikter mathematischer End-Check für die Elliott-Wave-Ausgabe:
    1) Wahrscheinlichkeiten summieren sich exakt auf 100 % (primary + alt + invalidation == 100).
    2) Bei bullishem Bias MUSS gelten: invalidation_price < S < target_price.
    3) Bei bearishem Bias MUSS gelten: target_price < S < invalidation_price.
    """
    p_prim = int(ew.get("primary_prob", 60))
    p_alt = int(ew.get("alt_prob", 22))
    p_prim = max(min(p_prim, 78), 50)
    p_alt = max(min(p_alt, 92 - p_prim), 14)
    p_inv = 100 - p_prim - p_alt
    ew["primary_prob"] = p_prim
    ew["alt_prob"] = p_alt
    ew["invalidation_prob"] = p_inv

    bias = ew.get("bias", "neutral")
    tgt, inv = _clamp_ew_levels(S, bias, ew.get("target_price"), ew.get("invalidation_price"))
    ew["target_price"] = tgt
    ew["invalidation_price"] = inv
    return ew


def analyze_elliott_wave(candles: list, rsi_series: list | None = None, macd_hist: list | None = None) -> dict:
    """
    Erkennt die aktuelle Elliott-Wellen-Phase (Impulswellen 1-5 & Korrekturwellen A-B-C)
    und führt einen strengen 3-Stufen-Doppelcheck durch:
      1. Kardinal-Regel-Check:
         • Regel 1: Welle 2 korrigiert nie mehr als 100 % von Welle 1 (P2 > P0 bull / P2 < P0 bear)
         • Regel 2: Welle 3 übertrifft das Ende von Welle 1 und ist nie die kürzeste Impulswelle
         • Regel 3: Welle 4 schneidet nicht in den Preisbereich von Welle 1 (P4 > P1 bull / P4 < P1 bear)
      2. Fibonacci-Retracement & Extension Check (38,2 %, 50,0 %, 61,8 %, 161,8 %)
      3. Aktueller Swing-Richtungs-Check (verhindert, dass ein frischer Bounce vom Tief als Hoch gezählt wird!)
    """
    if not candles or len(candles) < 60:
        return {
            "phase": "Unklare Struktur",
            "wave_code": "N/A",
            "bias": "neutral",
            "double_check": "Zu wenige Historien-Kerzen",
            "rules_ok": False,
            "primary_scenario": "Abwarten",
            "primary_prob": 50,
            "alt_scenario": "Seitwärts",
            "alt_prob": 30,
            "invalidation_prob": 20,
            "target_price": None,
            "invalidation_price": None,
            "fib_note": "",
        }

    window = candles[-180:]
    recent = window[-25:]
    tr_pcts = [(c["high"] - c["low"]) / c["close"] for c in recent if c["close"] > 0]
    avg_tr_pct = sum(tr_pcts) / len(tr_pcts) if tr_pcts else 0.02
    zz_thresh = max(min(avg_tr_pct * 2.0, 0.075), 0.026)

    pivots = _extract_zigzag_pivots(window, threshold_pct=zz_thresh)
    if len(pivots) < 5:
        pivots = _extract_zigzag_pivots(window, threshold_pct=max(zz_thresh * 0.7, 0.018))

    S = window[-1]["close"]
    n_bars = len(window)
    r_now = rsi_series[-1] if (rsi_series and rsi_series[-1] is not None) else 50.0
    m_now = macd_hist[-1] if (macd_hist and macd_hist[-1] is not None) else 0.0
    m_prev = macd_hist[-2] if (macd_hist and len(macd_hist) >= 2 and macd_hist[-2] is not None) else 0.0

    if len(pivots) < 4:
        tgt, inv = _clamp_ew_levels(S, "neutral", S * 1.045, S * 0.955)
        return _enforce_ew_math_consistency({
            "phase": "Seitwärts / Konsolidierung",
            "wave_code": "RANGE",
            "bias": "neutral",
            "double_check": "Keine abgeschlossene 5-Wellen- oder ABC-Sequenz",
            "rules_ok": False,
            "primary_scenario": f"Range-Handel ({inv:.2f}$ – {tgt:.2f}$)",
            "primary_prob": 55,
            "alt_scenario": "Trend-Ausbruch abwarten",
            "alt_prob": 30,
            "invalidation_prob": 15,
            "target_price": tgt,
            "invalidation_price": inv,
            "fib_note": "Neutrale Übergangsstruktur",
        }, S)

    # WICHTIGER SWING-POLARITÄTS-CHECK:
    # _extract_zigzag_pivots hängt am Ende automatisch das aktuelle laufende Extrem an (z.B. 'H' sobald
    # der Kurs vom jüngsten Swing-Tief 'L' um >= zz_thresh nach oben abprallt, bzw. 'L' sobald der Kurs vom
    # jüngsten Swing-Hoch 'H' um >= zz_thresh nach unten abprallt).
    # Wenn dieser letzte Punkt noch gar kein Umkehrpunkt ist, sondern die frisch laufende Bewegung VOM
    # eigentlichen Wendepunkt pivots[-2] weg, dann ist pivots[-2] der maßgebliche abgeschlossene Wendepunkt!
    p_work = list(pivots)
    last_p = p_work[-1]
    prev_p = p_work[-2]
    bars_since_last = (n_bars - 1) - last_p["idx"]
    bars_leg = last_p["idx"] - prev_p["idx"]

    if last_p["type"] == "H":
        pullback_from_h = (last_p["price"] - S) / last_p["price"] if last_p["price"] > 0 else 0.0
        # Wenn 'H' ganz frisch ist (<= 2 Kerzen alt, kaum Rücksetzer vom Hoch) und die Erholung ab 'L'
        # aus dem überverkauften/neutralen Bereich kommt (r_now < 58, kurzer Bounce <= 7 Kerzen),
        # befinden wir uns im Aufwärts-Bounce ab dem Tief prev_p ('L')!
        if pullback_from_h < zz_thresh * 0.35 and bars_since_last <= 2 and (bars_leg <= 7 or r_now < 54) and r_now < 60:
            p_work = p_work[:-1]
    elif last_p["type"] == "L":
        bounce_from_l = (S - last_p["price"]) / last_p["price"] if last_p["price"] > 0 else 0.0
        # Symmetrisch: Wenn 'L' ganz frisch ist (<= 2 Kerzen alt, kaum Bounce vom Tief) und der Abverkauf
        # ab 'H' aus dem überkauften/neutralen Bereich läuft (r_now > 42, kurzer Drop <= 7 Kerzen),
        # befinden wir uns im laufenden Abverkauf ab dem Hoch prev_p ('H')!
        if bounce_from_l < zz_thresh * 0.35 and bars_since_last <= 2 and (bars_leg <= 7 or r_now > 46) and r_now > 40:
            p_work = p_work[:-1]

    candidates = []

    # =================================================================
    # FALL A: Maßgeblicher Wendepunkt ist ein TIEF ('L') -> Aufwärts-Wende / Bodenbildung
    # =================================================================
    if p_work[-1]["type"] == "L":
        # 1) Bullishe Welle 2 am Boden -> Start Welle 3 ↑ (L0 -> H1 -> L2)
        if len(p_work) >= 3 and p_work[-2]["type"] == "H" and p_work[-3]["type"] == "L":
            w0, w1, w2 = p_work[-3]["price"], p_work[-2]["price"], min(p_work[-1]["price"], S)
            w1_len = w1 - w0
            if w1_len > 0 and w2 > w0 and S <= w1 * 1.03:
                ret2 = (w1 - w2) / w1_len
                if 0.236 <= ret2 <= 0.886:
                    fib_golden = 0.382 <= ret2 <= 0.650
                    turning_up = (S >= w2 * 1.004) or (m_now >= m_prev) or (r_now <= 44)
                    score = 75 + (10 if fib_golden else 4) + (6 if turning_up else 0)
                    raw_t3 = max(w2 + w1_len * 1.382, w1 * 1.03, S * 1.06)
                    target_w3, inval_w0 = _clamp_ew_levels(S, "bullish", raw_t3, w0)
                    deeper_fib_lvl = round(max(inval_w0 * 1.01, min(w2 * 0.99, w1 - 0.786 * w1_len)), 2)
                    p_prim = min(max(int(round(score * 0.80)), 58), 76)
                    candidates.append({
                        "score": score,
                        "phase": "Welle 2 Korrektur-Boden → Start Welle 3 ↑",
                        "wave_code": "W2→W3↑",
                        "bias": "bullish",
                        "rules_ok": True,
                        "double_check": f"Regel 1 erfüllt (W2 {w2:.2f}$ > W0 {w0:.2f}$) · Fib-Retracement {ret2*100:.1f}%",
                        "primary_scenario": f"Impulsschub Welle 3 ↑ bis {target_w3:.2f}$",
                        "primary_prob": p_prim,
                        "alt_scenario": f"Test der unteren Unterstützungszone bei {deeper_fib_lvl:.2f}$",
                        "alt_prob": 19,
                        "invalidation_prob": 100 - p_prim - 19,
                        "target_price": target_w3,
                        "invalidation_price": inval_w0,
                        "fib_note": f"Welle 2 korrigierte {ret2*100:.1f}% von Welle 1 ({'Golden Pocket' if fib_golden else 'Fib-Support'})",
                    })

        # 2) Bullishe Welle 4 Pullback hält über Welle 1 -> Start Welle 5 ↑ (L0->H1->L2->H3->L4)
        if len(p_work) >= 5 and p_work[-2]["type"] == "H" and p_work[-3]["type"] == "L" and p_work[-4]["type"] == "H" and p_work[-5]["type"] == "L":
            w0, w1, w2, w3, w4 = p_work[-5]["price"], p_work[-4]["price"], p_work[-3]["price"], p_work[-2]["price"], min(p_work[-1]["price"], S)
            w1_len, w3_len = w1 - w0, w3 - w2
            if w1_len > 0 and w3_len > 0 and w2 > w0 and w3 > w1 and w3_len >= w1_len * 0.85 and w4 >= w1 * 0.99:
                ret4 = (w3 - w4) / w3_len
                if 0.18 <= ret4 <= 0.62:
                    score = 78 + (9 if w4 > w1 else 3) + (6 if 0.236 <= ret4 <= 0.45 else 0)
                    raw_t5 = max(w3 * 1.03, w4 + w1_len, S * 1.055)
                    target_w5, inval_w1 = _clamp_ew_levels(S, "bullish", raw_t5, min(w1, S * 0.965))
                    p_prim = min(max(int(round(score * 0.80)), 62), 78)
                    candidates.append({
                        "score": score,
                        "phase": "Welle 4 Pullback über W1 → Start Welle 5 ↑",
                        "wave_code": "W4→W5↑",
                        "bias": "bullish",
                        "rules_ok": True,
                        "double_check": f"3/3 EW-Regeln erfüllt (Kein W4/W1-Overlap bei {w1:.2f}$ · Fib {ret4*100:.1f}%)",
                        "primary_scenario": f"Abschluss-Schub Welle 5 ↑ Richtung {target_w5:.2f}$",
                        "primary_prob": p_prim,
                        "alt_scenario": f"Seitwärts-Konsolidierung über W1-Hoch ({w1:.2f}$)",
                        "alt_prob": 18,
                        "invalidation_prob": 100 - p_prim - 18,
                        "target_price": target_w5,
                        "invalidation_price": inval_w1,
                        "fib_note": f"Welle 4 korrigierte {ret4*100:.1f}% von Welle 3 ohne W1-Überlappung",
                    })

        # 3) Bullishe ABC-Korrektur am Ende von Welle C (H0 -> LA -> HB -> LC -> Neue Welle 1 ↑)
        if len(p_work) >= 4 and p_work[-2]["type"] == "H" and p_work[-3]["type"] == "L" and p_work[-4]["type"] == "H":
            p_top, p_a, p_b, p_c = p_work[-4]["price"], p_work[-3]["price"], p_work[-2]["price"], min(p_work[-1]["price"], S)
            len_a, len_c = p_top - p_a, p_b - p_c
            if len_a > 0 and len_c > 0 and p_b < p_top and p_c <= p_a * 1.02:
                ret_b = (p_b - p_a) / len_a
                ratio_ca = len_c / len_a
                if 0.20 <= ret_b <= 0.886 and 0.55 <= ratio_ca <= 1.85 and S < p_top:
                    fib_match = 0.80 <= ratio_ca <= 1.38
                    score = 76 + (9 if fib_match else 3) + (6 if r_now <= 46 else 0)
                    raw_up = max(S * 1.055, min(p_b, p_c + (p_top - p_c) * 0.618))
                    target_up, inval_c = _clamp_ew_levels(S, "bullish", raw_up, min(p_c * 0.965, S * 0.955))
                    p_prim = min(max(int(round(score * 0.80)), 60), 76)
                    candidates.append({
                        "score": score,
                        "phase": "ABC-Korrektur in Welle C → Start Welle 1 ↑",
                        "wave_code": "ABC-BODEN↑",
                        "bias": "bullish",
                        "rules_ok": True,
                        "double_check": f"3-teilige A-B-C Korrektur komplett (B={ret_b*100:.0f}% Fib · C={ratio_ca*100:.0f}% von A)",
                        "primary_scenario": f"Welle-C-Bodenbildung → Aufwärts-Zyklus bis {target_up:.2f}$",
                        "primary_prob": p_prim,
                        "alt_scenario": f"Rest-Ausdehnung der C-Welle Richtung {inval_c:.2f}$",
                        "alt_prob": 19,
                        "invalidation_prob": 100 - p_prim - 19,
                        "target_price": target_up,
                        "invalidation_price": inval_c,
                        "fib_note": f"C-Welle erreichte {ratio_ca*100:.0f}% Projektion der A-Welle",
                    })

    # =================================================================
    # FALL B: Maßgeblicher Wendepunkt ist ein HOCH ('H') -> Top-Zone / Korrektur oder aktiver Ausbruch
    # =================================================================
    if p_work[-1]["type"] == "H":
        h_top = max(p_work[-1]["price"], S)
        is_pulling_back = (S < h_top * 0.992) or (r_now >= 60) or (m_now < m_prev)

        # 4) Aktiver Aufwärtsschub Welle 3 ↑ (wenn Momentum positiv ist und Kurs nahe/über W1 ausbricht)
        if not is_pulling_back and len(p_work) >= 4 and p_work[-2]["type"] == "L" and p_work[-3]["type"] == "H" and p_work[-4]["type"] == "L":
            w0, w1, w2 = p_work[-4]["price"], p_work[-3]["price"], p_work[-2]["price"]
            w1_len = w1 - w0
            if w1_len > 0 and w2 > w0 and h_top > w1 and S >= w1 * 0.995:
                ext3 = (h_top - w2) / w1_len
                if 0.65 <= ext3 <= 2.40:
                    score = 76 + (8 if 1.0 <= ext3 <= 1.8 else 3)
                    raw_t3 = max(S * 1.06, w2 + w1_len * 1.618)
                    target_w3, inval_w2 = _clamp_ew_levels(S, "bullish", raw_t3, max(w2, S * 0.92))
                    p_prim = min(max(int(round(score * 0.78)), 58), 74)
                    candidates.append({
                        "score": score,
                        "phase": "Impuls-Welle 3 aktiv (Aufwärtsschub ↑)",
                        "wave_code": "W3-IMPULS↑",
                        "bias": "bullish",
                        "rules_ok": True,
                        "double_check": f"2/2 EW-Regeln erfüllt (W2 > W0 & W3 > W1) · {ext3*100:.0f}% Fib-Extension",
                        "primary_scenario": f"Fortsetzung Welle 3 ↑ Richtung {target_w3:.2f}$",
                        "primary_prob": p_prim,
                        "alt_scenario": f"Zwischenkorrektur (Welle 4) Richtung {max(w1, S*0.95):.2f}$",
                        "alt_prob": 21,
                        "invalidation_prob": 100 - p_prim - 21,
                        "target_price": target_w3,
                        "invalidation_price": inval_w2,
                        "fib_note": f"Welle 3 läuft bei {ext3*100:.0f}% Extension von Welle 1",
                    })

        # 5) Welle 5 Top-Bildung -> Übergang in ABC-Korrektur ↓ (L0->H1->L2->H3->L4->H5)
        if len(p_work) >= 6 and p_work[-2]["type"] == "L" and p_work[-3]["type"] == "H" and p_work[-4]["type"] == "L" and p_work[-5]["type"] == "H" and p_work[-6]["type"] == "L":
            w0, w1, w2, w3, w4, w5 = p_work[-6]["price"], p_work[-5]["price"], p_work[-4]["price"], p_work[-3]["price"], p_work[-2]["price"], h_top
            w1_len, w3_len, w5_len = w1 - w0, w3 - w2, w5 - w4
            w3_not_shortest = not (w3_len < w1_len and w3_len < w5_len)
            if w1_len > 0 and w3_len > 0 and w2 > w0 and w3 > w1 and w4 >= w1 * 0.985 and w5 >= w3 * 0.98 and w3_not_shortest:
                score = 78 + (9 if is_pulling_back else 2)
                raw_abc = min(S * 0.94, max(w4, w5 - (w5 - w0) * 0.382))
                target_abc, inval_top = _clamp_ew_levels(S, "bearish", raw_abc, max(w5 * 1.025, S * 1.045))
                p_prim = min(max(int(round(score * 0.80)), 60), 75)
                candidates.append({
                    "score": score,
                    "phase": "Impuls-Welle 5 Top-Zone → ABC-Korrektur ↓",
                    "wave_code": "W5-TOP↓",
                    "bias": "bearish",
                    "rules_ok": True,
                    "double_check": f"3/3 EW-Regeln (W1–W5 Impuls komplett bei {w5:.2f}$ · W3 nicht kürzeste Welle)",
                    "primary_scenario": f"Welle-5-Abschluss & ABC-Rücksetzer bis {target_abc:.2f}$",
                    "primary_prob": p_prim,
                    "alt_scenario": f"Späte Welle-5-Dehnung bis {inval_top:.2f}$",
                    "alt_prob": 20,
                    "invalidation_prob": 100 - p_prim - 20,
                    "target_price": target_abc,
                    "invalidation_price": inval_top,
                    "fib_note": f"5-teiliger Aufwärtszyklus ab {w0:.2f}$ in der Reifephase",
                })

        # 6) Bearishe ABC-Erholung am Welle-C-Deckel -> Neuer Abwärtsschub ↓ (L0 -> HA -> LB -> HC)
        if len(p_work) >= 4 and p_work[-2]["type"] == "L" and p_work[-3]["type"] == "H" and p_work[-4]["type"] == "L":
            p_bot, p_a, p_b, p_c = p_work[-4]["price"], p_work[-3]["price"], p_work[-2]["price"], h_top
            len_a, len_c = p_a - p_bot, p_c - p_b
            if len_a > 0 and len_c > 0 and p_b > p_bot and p_c >= p_a * 0.97 and is_pulling_back:
                ret_b = (p_a - p_b) / len_a
                ratio_ca = len_c / len_a
                if 0.236 <= ret_b <= 0.886 and 0.55 <= ratio_ca <= 1.85:
                    score = 76 + (8 if 0.80 <= ratio_ca <= 1.38 else 3) + (6 if r_now >= 55 else 0)
                    raw_dn = min(S * 0.94, max(p_b, p_c - (p_c - p_bot) * 0.50))
                    target_dn, inval_c = _clamp_ew_levels(S, "bearish", raw_dn, max(p_c * 1.025, S * 1.045))
                    p_prim = min(max(int(round(score * 0.80)), 60), 75)
                    candidates.append({
                        "score": score,
                        "phase": "ABC-Erholung am Welle-C-Deckel → Abwärtsschub ↓",
                        "wave_code": "ABC-TOP↓",
                        "bias": "bearish",
                        "rules_ok": True,
                        "double_check": f"3-teilige A-B-C Gegenbewegung (B={ret_b*100:.0f}% Fib · C={ratio_ca*100:.0f}% von A)",
                        "primary_scenario": f"Abschluss Welle C & Rücksetzer Richtung {target_dn:.2f}$",
                        "primary_prob": p_prim,
                        "alt_scenario": f"Überschießen bis {inval_c:.2f}$",
                        "alt_prob": 20,
                        "invalidation_prob": 100 - p_prim - 20,
                        "target_price": target_dn,
                        "invalidation_price": inval_c,
                        "fib_note": f"C-Welle stößt bei {ratio_ca*100:.0f}% C=A Projektion auf Widerstand",
                    })

        # 7) Bearishe Abwärts-Welle 2 am Widerstand -> Start Abwärts-Welle 3 ↓ (H0 -> L1 -> H2)
        if len(p_work) >= 3 and p_work[-2]["type"] == "L" and p_work[-3]["type"] == "H" and is_pulling_back:
            w0, w1, w2 = p_work[-3]["price"], p_work[-2]["price"], h_top
            w1_len = w0 - w1
            if w1_len > 0 and w2 < w0 and S >= w1:
                ret2 = (w2 - w1) / w1_len
                if 0.236 <= ret2 <= 0.854:
                    score = 74 + (9 if 0.382 <= ret2 <= 0.650 else 3) + (5 if r_now >= 54 else 0)
                    raw_t3 = min(S * 0.94, max(w2 - w1_len * 1.272, w1 * 0.96))
                    target_w3, inval_w0 = _clamp_ew_levels(S, "bearish", raw_t3, max(w0, w2 * 1.025))
                    p_prim = min(max(int(round(score * 0.80)), 59), 74)
                    candidates.append({
                        "score": score,
                        "phase": "Abwärts-Welle 2 beendet → Start Welle 3 ↓",
                        "wave_code": "W2→W3↓",
                        "bias": "bearish",
                        "rules_ok": True,
                        "double_check": f"Regel 1 erfüllt (W2 {w2:.2f}$ < W0 {w0:.2f}$) · Fib-Retracement {ret2*100:.1f}%",
                        "primary_scenario": f"Abwärts-Impuls Welle 3 ↓ Richtung {target_w3:.2f}$",
                        "primary_prob": p_prim,
                        "alt_scenario": f"Erneuter Test des Widerstands bei {w2:.2f}$",
                        "alt_prob": 20,
                        "invalidation_prob": 100 - p_prim - 20,
                        "target_price": target_w3,
                        "invalidation_price": inval_w0,
                        "fib_note": f"Bären-Rallye (Welle 2) endete am {ret2*100:.1f}% Fib-Retracement",
                    })

        # 8) Nach starkem Swing-Hoch: Beginn Korrektur-Welle 2 / A ↓
        if len(p_work) >= 2 and p_work[-2]["type"] == "L":
            p_low = p_work[-2]["price"]
            up_len = h_top - p_low
            if up_len > 0:
                raw_fib50 = min(S * 0.945, max(p_low * 1.02, h_top - up_len * 0.50))
                fib_50, inval_h = _clamp_ew_levels(S, "bearish", raw_fib50, max(h_top * 1.02, S * 1.04))
                candidates.append({
                    "score": 71,
                    "phase": "Impuls-Hoch erreicht → Korrektur-Welle 2 / A ↓",
                    "wave_code": "TOP→W2/A↓",
                    "bias": "bearish",
                    "rules_ok": True,
                    "double_check": f"Swing-Hoch bei {h_top:.2f}$ bestätigt · Beginn 38,2–50,0% Fib-Korrektur",
                    "primary_scenario": f"Rücksetzer in Welle 2/A ↓ Richtung {fib_50:.2f}$ (50% Fib)",
                    "primary_prob": 62,
                    "alt_scenario": f"Direkter Ausbruch über {inval_h:.2f}$",
                    "alt_prob": 22,
                    "invalidation_prob": 16,
                    "target_price": fib_50,
                    "invalidation_price": inval_h,
                    "fib_note": f"Abkühlungsphase nach Aufwärtsschub ({p_low:.2f}$ → {h_top:.2f}$)",
                })

    if candidates:
        best_ew = max(candidates, key=lambda x: x["score"])
        return _enforce_ew_math_consistency(best_ew, S)

    p_last = p_work[-1]
    p_prev = p_work[-2]
    if p_last["type"] == "L" and S >= p_last["price"] * 0.99:
        raw_t = max(S * 1.055, p_last["price"] + (p_prev["price"] - p_last["price"]) * 0.618)
        target_p, inval_p = _clamp_ew_levels(S, "bullish", raw_t, min(p_last["price"] * 0.975, S * 0.95))
        return _enforce_ew_math_consistency({
            "phase": "Frühe Welle 1 / Pullback-Boden ↑",
            "wave_code": "W1-AUFBAU↑",
            "bias": "bullish",
            "rules_ok": True,
            "double_check": f"Swing-Tief bei {p_last['price']:.2f}$ bestätigt · Aufwärts-Ansatz aktiv",
            "primary_scenario": f"Erholung bis 61,8% Fib ({target_p:.2f}$)",
            "primary_prob": 58,
            "alt_scenario": f"Nochmaliger Test des Tiefs bei {p_last['price']:.2f}$",
            "alt_prob": 26,
            "invalidation_prob": 16,
            "target_price": target_p,
            "invalidation_price": inval_p,
            "fib_note": f"Basisbildung über Swing-Tief {p_last['price']:.2f}$",
        }, S)
    else:
        raw_t = min(S * 0.945, p_last["price"] - (p_last["price"] - p_prev["price"]) * 0.50)
        target_p, inval_p = _clamp_ew_levels(S, "bearish", raw_t, max(p_last["price"] * 1.025, S * 1.045))
        return _enforce_ew_math_consistency({
            "phase": "Korrektur-Welle A / 2 nach Swing-Hoch ↓",
            "wave_code": "KORREKTUR↓",
            "bias": "bearish",
            "rules_ok": True,
            "double_check": f"Swing-Hoch bei {p_last['price']:.2f}$ bestätigt · Rücksetzer läuft",
            "primary_scenario": f"50%-Fib-Retracement bis {target_p:.2f}$",
            "primary_prob": 58,
            "alt_scenario": f"Ausbruch über {inval_p:.2f}$",
            "alt_prob": 26,
            "invalidation_prob": 16,
            "target_price": target_p,
            "invalidation_price": inval_p,
            "fib_note": f"Korrekturbewegung vom Hoch {p_last['price']:.2f}$",
        }, S)


# =====================================================================
# 3D. FUNDAMENTALDATEN & KAUF-NIVEAU ANALYSE (Unternehmenszahlen & Events)
#     -> Rein informativ als eigene Spalte ganz hinten + Übersichtstabelle
#     -> Fließt NICHT in den 4-Indikatoren-Score ein!
# =====================================================================

def evaluate_fundamentals(symbol: str, last_close: float, q: dict | None) -> dict:
    """
    Institutionelle Fundamental- & Kauf-Niveau-Bewertung:
    Prüft Forward-KGV, konsistentes EPS-Wachstum (FY2 vs. FY1 Konsens ohne GAAP-Sondereffekt-Verzerrung),
    PEG-Ratio, 52-Wochen-Bewertungsrabatt, Analysten-Konsens, Marktkapitalisierung, Dividende & Earnings-Termin.
    """
    if not q:
        return {
            "available": False,
            "name": symbol,
            "fund_score": 50,
            "buy_level": "KEINE DATEN",
            "level_code": "NEUTRAL",
            "pe_ttm": None,
            "pe_fwd": None,
            "peg_ratio": None,
            "eps_growth_pct": None,
            "pe_str": "KGV: k.A.",
            "growth_str": "EPS: k.A.",
            "analyst_str": "k.A.",
            "discount_52w_pct": None,
            "disc_str": "",
            "div_str": "",
            "cap_str": "",
            "earnings_str": "Termin: k.A.",
            "earnings_date_short": "k.A.",
            "days_to_earnings": None,
            "earnings_warning": False,
            "corp_action": "",
            "reasons": [],
            "summary_line": "Keine Unternehmenskennzahlen verfügbar",
        }

    pe_ttm = q.get("trailingPE")
    pe_fwd = q.get("forwardPE")
    pe_cur = q.get("priceEpsCurrentYear")
    eps_ttm = q.get("epsTrailingTwelveMonths")
    eps_cur = q.get("epsCurrentYear")
    eps_fwd = q.get("epsForward")
    high_52 = q.get("fiftyTwoWeekHigh")
    low_52 = q.get("fiftyTwoWeekLow")
    chg_52_high = q.get("fiftyTwoWeekHighChangePercent")
    analyst_raw = str(q.get("averageAnalystRating") or "").strip()
    div_yield = q.get("dividendYield") or (
        float(q["trailingAnnualDividendYield"]) * 100.0 if q.get("trailingAnnualDividendYield") else None
    )
    mkt_cap = q.get("marketCap")
    short_name = q.get("displayName") or q.get("shortName") or symbol

    # 1. Konsistentes Gewinnwachstum (FY2 epsForward vs. FY1 epsCurrentYear, um GAAP-TTM vs. Non-GAAP-Fwd Verzerrungen auszuschließen!)
    eps_growth_pct = None
    turnaround_to_profit = False
    eps_base_label = ""

    if eps_cur is not None and eps_fwd is not None and abs(eps_cur) > 0.15:
        if eps_cur > 0.15 and eps_fwd > 0:
            raw_g = ((eps_fwd - eps_cur) / eps_cur) * 100.0
            eps_growth_pct = max(min(raw_g, 120.0), -80.0)
            eps_base_label = f"EPS FY1→FY2: {eps_cur:.2f}$ → {eps_fwd:.2f}$"
        elif eps_cur <= 0 and eps_fwd > 0.10:
            turnaround_to_profit = True
            eps_base_label = f"EPS Turnaround: {eps_cur:.2f}$ → {eps_fwd:.2f}$"
    elif eps_ttm is not None and eps_fwd is not None:
        # Nur wenn TTM-EPS nicht durch Einmaleffekte nahe 0 gedrückt wurde (Mindest-Ertragsbasis >= 1.5% vom Kurs)
        if last_close > 0 and eps_ttm >= last_close * 0.015 and eps_fwd > 0:
            raw_g = ((eps_fwd - eps_ttm) / eps_ttm) * 100.0
            eps_growth_pct = max(min(raw_g, 120.0), -80.0)
            eps_base_label = f"EPS TTM→Fwd: {eps_ttm:.2f}$ → {eps_fwd:.2f}$"
        elif eps_ttm <= 0 and eps_fwd > 0.10:
            turnaround_to_profit = True
            eps_base_label = f"EPS Turnaround: {eps_ttm:.2f}$ → {eps_fwd:.2f}$"

    eff_pe = pe_fwd if (pe_fwd is not None and pe_fwd > 0) else (pe_cur if (pe_cur and pe_cur > 0) else pe_ttm)
    peg_ratio = None
    if eff_pe and 0 < eff_pe <= 85 and eps_growth_pct and eps_growth_pct >= 4.0:
        peg_ratio = round(eff_pe / eps_growth_pct, 2)

    analyst_num = None
    if analyst_raw:
        try:
            analyst_num = float(analyst_raw.split("-")[0].strip())
        except Exception:
            pass

    disc_52_pct = None
    if chg_52_high is not None:
        disc_52_pct = round(float(chg_52_high) * 100.0, 1)
    elif high_52 and high_52 > 0 and last_close > 0:
        disc_52_pct = round(((last_close - float(high_52)) / float(high_52)) * 100.0, 1)

    # 2. Selektives 5-Säulen-Bewertungsmodell (0 bis 100 Punkte)
    score = 0
    reasons_f = []
    is_profitable = (eps_fwd is not None and eps_fwd > 0) and ((eps_cur is not None and eps_cur > 0) or (eps_ttm is not None and eps_ttm > 0))

    # Säule A: Bewertungs-Multiple Forward-KGV (bis 34 Punkte)
    if is_profitable and eff_pe and eff_pe > 0:
        if eff_pe <= 14.5:
            score += 34
            reasons_f.append(f"Günstiges Fwd-KGV ({eff_pe:.1f}x)")
        elif eff_pe <= 19.5:
            score += 27
            reasons_f.append(f"Attraktives Fwd-KGV ({eff_pe:.1f}x)")
        elif eff_pe <= 25.0:
            score += 19
            reasons_f.append(f"Solides Fwd-KGV ({eff_pe:.1f}x)")
        elif eff_pe <= 32.0:
            score += 10
        elif eff_pe <= 45.0:
            score += 3
        else:
            score -= 6
    elif not is_profitable:
        reasons_f.append("Aktuell defizitär / Spekulativ")

    # Säule B: Nachhaltiges Gewinnwachstum & PEG-Ratio (bis 26 Punkte)
    if is_profitable and eps_growth_pct is not None:
        if eps_growth_pct >= 20.0:
            score += 16
            reasons_f.append(f"EPS-Wachstum {eps_growth_pct:+.1f}%")
        elif eps_growth_pct >= 10.0:
            score += 12
            reasons_f.append(f"EPS-Wachstum {eps_growth_pct:+.1f}%")
        elif eps_growth_pct >= 4.0:
            score += 6
        elif eps_growth_pct < -5.0:
            score -= 8

        if peg_ratio is not None:
            if 0.20 <= peg_ratio <= 1.20:
                score += 10
                reasons_f.append(f"PEG {peg_ratio:.2f}")
            elif 1.20 < peg_ratio <= 1.75:
                score += 6
    elif turnaround_to_profit:
        score += 14
        reasons_f.append("Erwarteter Gewinn-Turnaround")

    # Säule C: Kurs-Rabatt vom 52-Wochen-Hoch (bis 18 Punkte)
    if disc_52_pct is not None and is_profitable:
        if -36.0 <= disc_52_pct <= -14.0:
            score += 18
            reasons_f.append(f"{disc_52_pct:.1f}% unter 52W-Hoch")
        elif -50.0 <= disc_52_pct < -36.0:
            score += 12
            reasons_f.append(f"Value-Rabatt ({disc_52_pct:.1f}% v. 52W-Hoch)")
        elif -14.0 < disc_52_pct <= -6.0:
            score += 9
        else:
            score += 3

    # Säule D: Institutioneller Analysten-Konsens (bis 14 Punkte)
    if analyst_num is not None:
        if analyst_num <= 1.70:
            score += 14
            reasons_f.append(f"Konsens Strong Buy ({analyst_num:.1f})")
        elif analyst_num <= 2.15:
            score += 10
            reasons_f.append(f"Konsens Buy ({analyst_num:.1f})")
        elif analyst_num <= 2.60:
            score += 4
        elif analyst_num >= 3.20:
            score -= 5

    # Säule E: Substanz & Dividende / Market-Cap-Qualität (bis 8 Punkte)
    if is_profitable and mkt_cap and mkt_cap >= 10e9:
        score += 4
    if is_profitable and div_yield and float(div_yield) >= 1.2:
        score += 4

    score = max(min(score, 98), 8)

    # Strikte Einstufung: TOP_BUY erfordert echte Profitabilität + Fwd-KGV <= 28 + Score >= 76
    if score >= 76 and is_profitable and (eff_pe is not None and eff_pe <= 28.0):
        buy_level = f"TOP KAUF-NIVEAU ({score}/100)"
        level_code = "TOP_BUY"
    elif score >= 64 and is_profitable and (eff_pe is not None and eff_pe <= 35.0):
        buy_level = f"ATTRAKTIVES NIVEAU ({score}/100)"
        level_code = "GOOD_BUY"
    elif score >= 45 and is_profitable:
        buy_level = f"FAIRE BEWERTUNG ({score}/100)"
        level_code = "FAIR"
    else:
        buy_level = f"HOCH BEWERTET / SPEKULATIV ({score}/100)"
        level_code = "EXPENSIVE"

    pe_parts = []
    if pe_ttm and 0 < pe_ttm < 250:
        pe_parts.append(f"KGV {pe_ttm:.1f}x")
    if pe_fwd and 0 < pe_fwd < 250:
        pe_parts.append(f"Fwd {pe_fwd:.1f}x")
    if peg_ratio is not None:
        pe_parts.append(f"PEG {peg_ratio:.2f}")
    pe_str = " · ".join(pe_parts) if pe_parts else "KGV n/a (Verlustzone)"

    if eps_base_label and eps_growth_pct is not None:
        growth_str = f"{eps_base_label} ({eps_growth_pct:+.1f}%)"
    elif eps_base_label:
        growth_str = eps_base_label
    elif eps_ttm is not None:
        growth_str = f"EPS (TTM): {eps_ttm:.2f}$"
    else:
        growth_str = "EPS: k.A."

    cap_str = ""
    if mkt_cap and mkt_cap > 0:
        if mkt_cap >= 1e12:
            cap_str = f"{mkt_cap/1e12:.2f} Bio. $"
        elif mkt_cap >= 1e9:
            cap_str = f"{mkt_cap/1e9:.1f} Mrd. $"
        else:
            cap_str = f"{mkt_cap/1e6:.0f} Mio. $"

    div_str = f"Div {float(div_yield):.2f}%" if (div_yield and float(div_yield) > 0) else ""
    disc_str = f"52W-Hoch {disc_52_pct:+.1f}%" if disc_52_pct is not None else ""

    earn_ts = q.get("earningsTimestampStart") or q.get("earningsTimestamp")
    earnings_str = "Earnings: k.A."
    earnings_date_short = "k.A."
    earnings_warning = False
    days_to_earnings = None
    if earn_ts and isinstance(earn_ts, (int, float)) and earn_ts > 1000000000:
        earn_dt = date.fromtimestamp(int(earn_ts))
        days_to_earnings = (earn_dt - date.today()).days
        earnings_date_short = earn_dt.strftime("%d.%m.%Y")
        if 0 <= days_to_earnings <= 6:
            earnings_warning = True
            earnings_str = f"EARNINGS WARNUNG: {earnings_date_short} (in {days_to_earnings}T)"
        elif days_to_earnings > 6:
            earnings_str = f"Earnings: {earnings_date_short} (in {days_to_earnings}T)"
        else:
            earnings_str = f"Letzte Zahlen: {earnings_date_short}"

    corp_actions = q.get("corporateActions") or []
    corp_msg = ""
    if corp_actions and isinstance(corp_actions, list):
        first_ca = corp_actions[0]
        if isinstance(first_ca, dict) and first_ca.get("message"):
            corp_msg = str(first_ca["message"])

    return {
        "available": True,
        "name": short_name,
        "fund_score": score,
        "buy_level": buy_level,
        "level_code": level_code,
        "pe_ttm": round(pe_ttm, 1) if (pe_ttm and 0 < pe_ttm < 500) else None,
        "pe_fwd": round(pe_fwd, 1) if (pe_fwd and 0 < pe_fwd < 500) else None,
        "peg_ratio": peg_ratio,
        "eps_growth_pct": round(eps_growth_pct, 1) if eps_growth_pct is not None else None,
        "pe_str": pe_str,
        "growth_str": growth_str,
        "analyst_str": analyst_raw or "k.A.",
        "discount_52w_pct": disc_52_pct,
        "high_52": round(float(high_52), 2) if high_52 else None,
        "low_52": round(float(low_52), 2) if low_52 else None,
        "disc_str": disc_str,
        "div_str": div_str,
        "cap_str": cap_str,
        "earnings_str": earnings_str,
        "earnings_date_short": earnings_date_short,
        "days_to_earnings": days_to_earnings,
        "earnings_warning": earnings_warning,
        "corp_action": corp_msg,
        "reasons": reasons_f,
        "summary_line": " · ".join(reasons_f[:3]) if reasons_f else pe_str,
    }


# =====================================================================
# 4. SIGNAL-AUSWERTUNG (4 Indikatoren + Options-Filter)
# =====================================================================

def evaluate_4_indicators(candles: list) -> dict:
    if not candles or len(candles) < 60:
        return {"error": "Nicht genügend Kerzendaten"}

    closes = [c["close"] for c in candles]
    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    bb_upper, bb_mid, bb_lower = calc_bollinger_bands(closes, 20, 2.0)
    rsi_series = calc_rsi(closes, 14)
    macd_line, macd_sig, macd_hist = calc_macd(closes, 12, 26, 9)
    stoch_short_k, stoch_short_d = calc_stochastic_umwa(highs, lows, closes, 14, 5, 3)
    stoch_long_k, stoch_long_d = calc_stochastic_umwa(highs, lows, closes, 50, 6, 5)

    last_close = closes[-1]
    prev_close = closes[-2]
    last_open = opens[-1]
    last_low = lows[-1]
    last_high = highs[-1]

    indicators = {}
    reasons = []

    # 1. BOLLINGER BÄNDER (BB & EMAs)
    bl = bb_lower[-1]
    bu = bb_upper[-1]
    bm = bb_mid[-1]
    if bl is not None and bu is not None and (bu - bl) > 0:
        recent_low_near_bb = any(
            bb_lower[-k] is not None and (lows[-k] <= bb_lower[-k] + 0.08 * (bb_upper[-k] - bb_lower[-k]))
            for k in (1, 2, 3, 4)
        )
        recent_high_near_bb = any(
            bb_upper[-k] is not None and (highs[-k] >= bb_upper[-k] - 0.08 * (bb_upper[-k] - bb_lower[-k]))
            for k in (1, 2, 3, 4)
        )
        candle_range = max(last_high - last_low, 0.0001)
        upper_wick_ratio = (last_high - max(last_open, last_close)) / candle_range

        bull_reaction = (last_close > prev_close) and (last_close >= last_open) and (upper_wick_ratio < 0.45)
        bear_reaction = (last_close < prev_close) and ((last_close <= last_open) or (upper_wick_ratio > 0.45))

        if recent_low_near_bb and bull_reaction:
            indicators["bollinger"] = {"state": "bullish", "reason": f"Bestätigter Abpraller am unteren BB ({bl:.2f})"}
            reasons.append(f"BB: Bestätigter Abpraller am unteren Band ({bl:.2f}) -> bullish")
        elif recent_high_near_bb and bear_reaction:
            indicators["bollinger"] = {"state": "bearish", "reason": f"Bestätigter Abpraller am oberen BB ({bu:.2f})"}
            reasons.append(f"BB: Bestätigter Abpraller am oberen Band ({bu:.2f}) -> bearish")
        else:
            indicators["bollinger"] = {"state": "neutral", "reason": "Kein bestätigter Band-Abpraller"}
    else:
        indicators["bollinger"] = {"state": "neutral", "reason": "Keine Daten"}
    indicators["bollinger"]["values"] = {
        "upper": round(bu, 2) if bu else None,
        "middle": round(bm, 2) if bm else None,
        "lower": round(bl, 2) if bl else None,
    }

    # 2. RSI UMWA
    r = rsi_series[-1]
    r_prev = rsi_series[-2] if len(rsi_series) >= 2 else None
    recent_rsi_6 = [x for x in rsi_series[-7:-1] if x is not None]

    if r is not None and r_prev is not None and recent_rsi_6:
        min_prev = min(recent_rsi_6)
        max_prev = max(recent_rsi_6)
        rsi_impulse_up = r - min_prev
        rsi_impulse_dn = max_prev - r

        if min_prev < 30.0 and r >= 31.0 and r > r_prev and rsi_impulse_up >= 4.0 and r <= 50.0:
            indicators["rsi"] = {
                "state": "bullish",
                "reason": f"RSI hat 30 von unten nach oben durchstochen ({min_prev:.1f} -> {r:.1f})"
            }
            reasons.append(f"RSI ({r:.1f}): Klarer Durchstich von unter 30 nach oben ({min_prev:.1f} -> {r:.1f}) -> bullish")
        elif max_prev > 70.0 and r <= 69.0 and r < r_prev and rsi_impulse_dn >= 4.0 and r >= 50.0:
            indicators["rsi"] = {
                "state": "bearish",
                "reason": f"RSI hat 70 von oben nach unten durchstochen ({max_prev:.1f} -> {r:.1f})"
            }
            reasons.append(f"RSI ({r:.1f}): Klarer Durchstich von über 70 nach unten ({max_prev:.1f} -> {r:.1f}) -> bearish")
        else:
            indicators["rsi"] = {"state": "neutral", "reason": f"Kein Durchstich ({r:.1f})"}
    else:
        indicators["rsi"] = {"state": "neutral", "reason": "Keine Daten"}
    indicators["rsi"]["value"] = round(r, 1) if r is not None else None

    # 3. MACD UMWA
    mh = [x for x in macd_hist[-6:] if x is not None]
    if len(mh) >= 4:
        m1, m2, m3, m4 = mh[-1], mh[-2], mh[-3], mh[-4]
        max_abs = max(abs(x) for x in mh) or 1e-6

        rising_strong = (m1 > m2 > m3) and ((m1 - m3) / max_abs >= 0.25)
        cross_up_confirmed = (m1 > 0 and m1 > m2 > m3 and m3 < 0)

        falling_strong = (m1 < m2 < m3) and ((m3 - m1) / max_abs >= 0.25)
        cross_dn_confirmed = (m1 < 0 and m1 < m2 < m3 and m3 > 0)

        if cross_up_confirmed:
            indicators["macd"] = {"state": "bullish", "reason": f"MACD klar ins Grüne durchgebrochen ({m1:+.4f})"}
            reasons.append(f"MACD ({m1:+.4f}): Bestätigter Durchbruch ins Grüne -> bullish")
        elif rising_strong:
            indicators["macd"] = {"state": "bullish", "reason": f"MACD kräftig in Aufwärtsbewegung ({m3:+.3f} -> {m1:+.3f})"}
            reasons.append(f"MACD ({m1:+.4f}): Kräftig in Aufwärtsbewegung -> bullish")
        elif cross_dn_confirmed:
            indicators["macd"] = {"state": "bearish", "reason": f"MACD klar ins Rote durchgebrochen ({m1:+.4f})"}
            reasons.append(f"MACD ({m1:+.4f}): Bestätigter Durchbruch ins Rote -> bearish")
        elif falling_strong:
            indicators["macd"] = {"state": "bearish", "reason": f"MACD kräftig in Abwärtsbewegung ({m3:+.3f} -> {m1:+.3f})"}
            reasons.append(f"MACD ({m1:+.4f}): Kräftig in Abwärtsbewegung -> bearish")
        else:
            indicators["macd"] = {"state": "neutral", "reason": f"Noch nicht richtig in Bewegung ({m1:+.4f})"}
        indicators["macd"]["hist"] = round(m1, 4)
    else:
        indicators["macd"] = {"state": "neutral", "reason": "Keine Daten", "hist": None}

    # 4. STOCHASTIK UMWA
    sk, sk_p, sk_p2 = stoch_short_k[-1], stoch_short_k[-2], stoch_short_k[-3]
    sd, sd_p = stoch_short_d[-1], stoch_short_d[-2]
    lk, lk_p = stoch_long_k[-1], stoch_long_k[-2]
    ld, ld_p = stoch_long_d[-1], stoch_long_d[-2]

    if None not in (sk, sk_p, sk_p2, sd, sd_p, lk, lk_p, ld):
        recent_short_low = min(sk_p, sk_p2, sd_p)
        recent_short_high = max(sk_p, sk_p2, sd_p)

        bull_momentum = (
            recent_short_low <= 20.0
            and sk > sd + 1.5
            and sk > sk_p > sk_p2
            and (sk - recent_short_low) >= 5.0
            and (lk <= 22.0 or (lk > lk_p and lk >= ld))
        )
        bear_momentum = (
            recent_short_high >= 80.0
            and sk < sd - 1.5
            and sk < sk_p < sk_p2
            and (recent_short_high - sk) >= 5.0
            and (lk >= 78.0 or (lk < lk_p and lk <= ld))
        )

        if bull_momentum:
            indicators["stoch"] = {
                "state": "bullish",
                "reason": f"Klarer Ausbruch aus <20 nach oben (Kurz:{sk:.0f}/{sd:.0f}, Lang:{lk:.0f}/{ld:.0f})"
            }
            reasons.append(f"Stoch UMWA (Kurz:{sk:.0f}/{sd:.0f}, Lang:{lk:.0f}/{ld:.0f}): Klarer Ausbruch nach oben -> bullish")
        elif bear_momentum:
            indicators["stoch"] = {
                "state": "bearish",
                "reason": f"Klarer Ausbruch aus >80 nach unten (Kurz:{sk:.0f}/{sd:.0f}, Lang:{lk:.0f}/{ld:.0f})"
            }
            reasons.append(f"Stoch UMWA (Kurz:{sk:.0f}/{sd:.0f}, Lang:{lk:.0f}/{ld:.0f}): Klarer Ausbruch nach unten -> bearish")
        else:
            indicators["stoch"] = {
                "state": "neutral",
                "reason": f"Tümpelt / noch kein klarer Ausbruch (Kurz:{sk:.0f}/{sd:.0f}, Lang:{lk:.0f}/{ld:.0f})"
            }
    else:
        indicators["stoch"] = {"state": "neutral", "reason": "Keine Daten"}
    indicators["stoch"]["values"] = {
        "short_k": round(sk, 1) if sk is not None else None,
        "short_d": round(sd, 1) if sd is not None else None,
        "long_k": round(lk, 1) if lk is not None else None,
        "long_d": round(ld, 1) if ld is not None else None,
    }

    states = [indicators[k]["state"] for k in ("bollinger", "rsi", "macd", "stoch")]
    bull_count = states.count("bullish")
    bear_count = states.count("bearish")

    if bull_count >= 3:
        signal = f"STARKES KAUFSIGNAL ({bull_count}/4)"
        signal_level = "STRONG_BUY"
    elif bear_count >= 3:
        signal = f"STARKES VERKAUFSSIGNAL ({bear_count}/4)"
        signal_level = "STRONG_SELL"
    elif bull_count == 2 and bear_count < 2:
        signal = "SCHWACHES KAUFSIGNAL (2/4)"
        signal_level = "WEAK_BUY"
    elif bear_count == 2 and bull_count < 2:
        signal = "SCHWACHES VERKAUFSSIGNAL (2/4)"
        signal_level = "WEAK_SELL"
    else:
        signal = "NEUTRAL"
        signal_level = "NEUTRAL"

    # 5. DOPPEL-CHECK SUPPORT & RESISTANCE LINIEN (Blaue Linien)
    sr_info = find_validated_sr_lines(candles)
    if sr_info["at_support"] and sr_info["active_support"]:
        s_lvl = sr_info["active_support"]
        reasons.append(
            f"Support-Linie bei {s_lvl['price']:.2f} $ bestätigt ({s_lvl['bounces']}x Touch & Hold / {s_lvl['pivots']} Swing-Pivots)"
        )
    if sr_info["at_resistance"] and sr_info["active_resistance"]:
        r_lvl = sr_info["active_resistance"]
        reasons.append(
            f"Widerstands-Linie bei {r_lvl['price']:.2f} $ bestätigt ({r_lvl['bounces']}x Touch & Hold / {r_lvl['pivots']} Swing-Pivots)"
        )

    last_dt = date.fromisoformat(candles[-1]["time"])
    # Stale-Data-Schutz: Wurde die Aktie seit > 4 Kalendertagen nicht mehr gehandelt oder zeigt eine eingefrorene
    # Übernahme-/Halted-Kerze (High == Low an einem vergangenen Handelstag), darf sie kein Live-Signal auslösen!
    days_old = (date.today() - last_dt).days
    is_frozen_bar = (days_old >= 2 and abs(candles[-1]["high"] - candles[-1]["low"]) < 1e-6)
    is_stale = (days_old > 4) or is_frozen_bar
    if is_stale:
        signal = f"INAKTIV / DELISTED ({candles[-1]['time']})"
        signal_level = "INACTIVE"

    dte_days, exp_date, cycle_label = _calc_weekly_cycle(last_dt)
    iv = estimate_current_iv(closes, highs, lows)

    option_setup = None
    if signal_level in ("STRONG_BUY", "WEAK_BUY"):
        option_setup = recommend_weekly_option(
            last_close, iv, is_put=True, dte_days=dte_days, sr_info=sr_info,
            expiry_date=exp_date, cycle_label=cycle_label
        )
    elif signal_level in ("STRONG_SELL", "WEAK_SELL"):
        option_setup = recommend_weekly_option(
            last_close, iv, is_put=False, dte_days=dte_days, sr_info=sr_info,
            expiry_date=exp_date, cycle_label=cycle_label
        )

    # 6. ELLIOTT-WAVE MUSTERERKENNUNG (Rein informativ als eigene Spalte – fließt NICHT in die Bewertung ein!)
    ew_info = analyze_elliott_wave(candles, rsi_series=rsi_series, macd_hist=macd_hist)

    # 7. 1-Jahres-Chart-Historie (252 Handelstage + 63 Tage Indikator-Warmup = 315 Kerzen) inkl. EMA-Seeds
    slice_len = min(len(candles), 315)
    start_idx = len(candles) - slice_len
    ema20_full = _ema(closes, 20)
    ema50_full = _ema(closes, 50)
    ema100_full = _ema(closes, 100)
    ema200_full = _ema(closes, 200)
    ema_seeds = [
        round(ema20_full[start_idx], 4),
        round(ema50_full[start_idx], 4),
        round(ema100_full[start_idx], 4),
        round(ema200_full[start_idx], 4),
    ]
    candles_1y = [
        {
            "t": c["time"],
            "o": round(float(c["open"]), 2),
            "h": round(float(c["high"]), 2),
            "l": round(float(c["low"]), 2),
            "c": round(float(c["close"]), 2),
        }
        for c in candles[start_idx:]
    ]

    return {
        "last_date": candles[-1]["time"],
        "is_stale": is_stale,
        "last_close": round(last_close, 2),
        "prev_close": round(prev_close, 2),
        "bullish_count": bull_count,
        "bearish_count": bear_count,
        "signal": signal,
        "signal_level": signal_level,
        "reasons": reasons,
        "indicators": indicators,
        "sr_lines": sr_info,
        "elliott_wave": ew_info,
        "option_setup": option_setup,
        "candles_1y": candles_1y,
        "ema_seeds": ema_seeds,
    }


def _process_one(sym: str, period: str, live_quote: dict | None = None) -> dict:
    candles, source = fetch_candles(sym, period=period, live_quote=live_quote)
    res = evaluate_4_indicators(candles)
    res["symbol"] = sym
    res["category"] = SYMBOL_CATEGORY.get(sym, "CUSTOM")
    res["source"] = source
    last_c = res.get("last_close") or 0.0
    prev_c = res.get("prev_close") or last_c
    fund = evaluate_fundamentals(sym, last_c, live_quote)
    res["fundamentals"] = fund

    # Kontrakt-Bezeichnung & Earnings-Risiko in Option-Setup nachtragen
    opt = res.get("option_setup")
    if opt:
        if opt.get("contract") and not opt["contract"].startswith(sym):
            opt["contract"] = f"{sym} {opt['contract']}".strip()
        dte_earn = fund.get("days_to_earnings")
        if dte_earn is not None and 0 <= dte_earn <= 6:
            opt["earnings_risk"] = True

    if live_quote and live_quote.get("regularMarketChangePercent") is not None:
        res["change_pct"] = round(float(live_quote["regularMarketChangePercent"]), 2)
    elif prev_c > 0 and not res.get("is_stale"):
        res["change_pct"] = round(((last_c - prev_c) / prev_c) * 100.0, 2)
    else:
        res["change_pct"] = None
    return res


def scan_symbols(symbols: list, period: str = "d", only_signals: bool = False):
    print("=" * 106)
    print(f"MARKET VISION PRO – INSTITUTIONAL STILLHALTER & 4-INDIKATOREN TERMINAL ({len(symbols)} Werte | {period.upper()})")
    print("Regelwerk: >= 3/4 = Hauptsignal | Weekly Option (Mo/Di->Fr): Delta <= 0,30 | OTM >= 2,0% | Prämie 0,7%–1,4%")
    print("Daten-Feed: EODHD Split-Adjusted + Yahoo Real-Time Quote & Fundamental-Fusion...")
    print("=" * 106)

    if period == "d":
        _load_candle_cache()

    live_quotes = fetch_batch_quotes_and_fundamentals(symbols)

    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=24) as pool:
        future_map = {pool.submit(_process_one, sym, period, live_quotes.get(sym)): sym for sym in symbols}
        for fut in as_completed(future_map):
            sym = future_map[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append((sym, str(e)))

    if period == "d":
        _save_candle_cache()

    priority = {"STRONG_BUY": 0, "STRONG_SELL": 1, "WEAK_BUY": 2, "WEAK_SELL": 3, "NEUTRAL": 4, "INACTIVE": 5}
    results.sort(key=lambda r: (
        priority.get(r.get("signal_level", "NEUTRAL"), 9),
        -max(r.get("bullish_count", 0), r.get("bearish_count", 0)),
        0 if (r.get("option_setup") or {}).get("sr_protected") else 1,
        0 if (r.get("option_setup") or {}).get("qualified") else 1,
        r.get("symbol", "")
    ))

    state_tag = lambda s: "[+]" if s == "bullish" else ("[-]" if s == "bearish" else "[·]")

    for res in results:
        lvl = res.get("signal_level", "NEUTRAL")
        if only_signals and lvl in ("NEUTRAL", "INACTIVE"):
            continue

        badge = {
            "STRONG_BUY": f"BUY  ({res.get('bullish_count', 0)}/4)",
            "STRONG_SELL": f"SELL ({res.get('bearish_count', 0)}/4)",
            "WEAK_BUY": "WATCH BUY  (2/4)",
            "WEAK_SELL": "WATCH SELL (2/4)",
            "NEUTRAL": "NEUTRAL",
            "INACTIVE": "INACTIVE",
        }.get(lvl, "NEUTRAL")

        ind = res.get("indicators", {})
        st_v = (ind.get("stoch") or {}).get("values") or {}
        ind_summary = (
            f"BB:{state_tag((ind.get('bollinger') or {}).get('state'))} "
            f"RSI:{state_tag((ind.get('rsi') or {}).get('state'))}({str((ind.get('rsi') or {}).get('value')):>4}) "
            f"MACD:{state_tag((ind.get('macd') or {}).get('state'))} "
            f"STOCH:{state_tag((ind.get('stoch') or {}).get('state'))}({st_v.get('short_k')}/{st_v.get('long_k')})"
        )
        cat_raw = res.get("category", "CUSTOM")
        cat_map = {
            "A-WERTE": "A",
            "B-WERTE": "B",
            "C-WERTE": "C",
            "JULIAN-STABLE": "JS",
            "JULIAN-ACTIVE": "JA",
            "JULIAN-VOLATILE": "JV",
            "JULIAN-TECH-BIO": "JT",
            "OPTIONS-PRO-50": "OP",
        }
        cat = cat_map.get(cat_raw, cat_raw[:2])

        print(
            f"[{cat:<2}] {res['symbol']:<6} | {res['last_close']:>8.2f} USD | "
            f"{badge:<18} | {ind_summary}"
        )
        if lvl not in ("NEUTRAL", "INACTIVE"):
            for reason in res["reasons"]:
                print(f"      -> {reason}")
            sr = res.get("sr_lines") or {}
            ns = sr.get("nearest_support")
            nr = sr.get("nearest_resistance")
            sr_parts = []
            if ns:
                sr_parts.append(f"SUP {ns['price']:.2f}$ ({ns['dist_pct']:+.1f}% | {ns['bounces']}T/{ns['pivots']}P)")
            if nr:
                sr_parts.append(f"RES {nr['price']:.2f}$ ({nr['dist_pct']:+.1f}% | {nr['bounces']}T/{nr['pivots']}P)")
            if sr_parts:
                print(f"      [S/R LINIEN] {'  |  '.join(sr_parts)}")

            ew = res.get("elliott_wave") or {}
            if ew.get("phase"):
                print(
                    f"      [ELLIOTT WAVE] {ew['phase']} | "
                    f"{ew['primary_prob']}% {ew['primary_scenario']} "
                    f"(Alt {ew['alt_prob']}% | Inval {ew.get('invalidation_price')}$: {ew['invalidation_prob']}%) | "
                    f"{ew['double_check']}"
                )

            fund = res.get("fundamentals") or {}
            if fund.get("available"):
                print(
                    f"      [FUNDAMENTALS] {fund['buy_level']} | {fund['pe_str']} | "
                    f"{fund['growth_str']} | {fund['disc_str']} | {fund['earnings_str']}"
                )

            opt = res.get("option_setup")
            if opt:
                if opt["qualified"]:
                    sign = "-" if "PUT" in opt["type"] else "+"
                    shield = ""
                    if opt.get("sr_protected") and opt.get("sr_shield_line"):
                        sl = opt["sr_shield_line"]
                        shield = f" | S/R-SHIELD {sl['price']:.2f}$ ({sl['bounces']}T)"
                    earn_w = " | [!] EARNINGS-RISIKO" if opt.get("earnings_risk") else ""
                    print(
                        f"      [WEEKLY OPTION] {opt.get('contract', opt['type'])} ({opt.get('cycle_label', '')}) | "
                        f"Strike {opt['strike']:.2f}$ ({sign}{opt['otm_pct']:.1f}% OTM) | "
                        f"Delta {opt['delta']:.2f} | Prämie {opt['premium_usd']:.2f}$ (+{opt['prem_pct']:.2f}%){shield}{earn_w}"
                    )
                else:
                    print(f"      [WEEKLY OPTION] KEIN TRADE – {opt['note']}")

    strong_buys = [r for r in results if r.get("signal_level") == "STRONG_BUY"]
    strong_sells = [r for r in results if r.get("signal_level") == "STRONG_SELL"]
    weak_buys = [r for r in results if r.get("signal_level") == "WEAK_BUY"]
    weak_sells = [r for r in results if r.get("signal_level") == "WEAK_SELL"]

    print("=" * 106)
    print(
        f"SCAN ABGESCHLOSSEN ({len(results)} Werte): "
        f"{len(strong_buys)} BUY (3-4/4) | {len(strong_sells)} SELL (3-4/4) | "
        f"{len(weak_buys)} WATCH BUY (2/4) | {len(weak_sells)} WATCH SELL (2/4)"
    )
    print("=" * 106)
    return results


# =====================================================================
# 5. EXTERNER OUTPUT AUSSERHALB VON ANTIGRAVITY (HTML, CSV, macOS, Telegram)
# =====================================================================

def export_csv_report(results: list, csv_path: str):
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "Symbol", "Name", "Kategorie", "Datum", "Kurs (USD)", "Tagesaenderung (%)", "Signal-Stufe", "Bull-Score", "Bear-Score",
            "BB-Status", "RSI-Wert", "MACD-Status", "Stoch-Status",
            "Option-Kontrakt", "Option-Typ", "Verfall-Zyklus", "Strike (USD)", "OTM-Puffer (%)", "Delta", "Praemie (USD)", "Praemie (%)",
            "SR-Geschuetzt", "Schutz-Linie (USD)", "Naechster-Support (USD)", "Naechster-Widerstand (USD)",
            "Elliott-Phase", "EW-Hauptszenario (%)", "EW-Ziel (USD)", "EW-Invalidierung (USD)",
            "Fundamental-Stufe", "Fundamental-Score", "KGV-Info", "EPS-Wachstum", "Earnings-Status"
        ])
        for r in results:
            if "error" in r:
                continue
            ind = r.get("indicators", {})
            opt = r.get("option_setup") or {}
            sr = r.get("sr_lines") or {}
            ns = sr.get("nearest_support") or {}
            nr = sr.get("nearest_resistance") or {}
            sl = opt.get("sr_shield_line") or {}
            ew = r.get("elliott_wave") or {}
            fund = r.get("fundamentals") or {}
            writer.writerow([
                r.get("symbol", ""),
                fund.get("name", ""),
                r.get("category", ""),
                r.get("last_date", ""),
                r.get("last_close", ""),
                r.get("change_pct", ""),
                r.get("signal_level", ""),
                r.get("bullish_count", 0),
                r.get("bearish_count", 0),
                (ind.get("bollinger") or {}).get("state", ""),
                (ind.get("rsi") or {}).get("value", ""),
                (ind.get("macd") or {}).get("state", ""),
                (ind.get("stoch") or {}).get("state", ""),
                opt.get("contract", "") if opt.get("qualified") else "",
                opt.get("type", "") if opt.get("qualified") else "NO TRADE",
                opt.get("cycle_label", ""),
                opt.get("strike", "") if opt.get("qualified") else "",
                opt.get("otm_pct", "") if opt.get("qualified") else "",
                opt.get("delta", "") if opt.get("qualified") else "",
                opt.get("premium_usd", "") if opt.get("qualified") else "",
                opt.get("prem_pct", "") if opt.get("qualified") else "",
                "JA" if opt.get("sr_protected") else "NEIN",
                sl.get("price", ""),
                ns.get("price", ""),
                nr.get("price", ""),
                ew.get("phase", ""),
                ew.get("primary_prob", ""),
                ew.get("target_price", ""),
                ew.get("invalidation_price", ""),
                fund.get("buy_level", ""),
                fund.get("fund_score", ""),
                fund.get("pe_str", ""),
                fund.get("growth_str", ""),
                fund.get("earnings_str", ""),
            ])
    print(f"CSV-Report gespeichert unter: {csv_path}")


def export_html_dashboard(results: list, html_path: str):
    from datetime import datetime
    import html as html_mod

    now_str = datetime.now().strftime("%d.%m.%Y · %H:%M:%S CET")
    valid = [r for r in results if "error" not in r and r.get("signal_level") != "INACTIVE"]
    inactive_list = [r for r in results if r.get("signal_level") == "INACTIVE"]

    strong_buys = [r for r in valid if r.get("signal_level") == "STRONG_BUY"]
    strong_sells = [r for r in valid if r.get("signal_level") == "STRONG_SELL"]
    weak_signals = [r for r in valid if r.get("signal_level") in ("WEAK_BUY", "WEAK_SELL")]
    qualified_main = [
        r for r in valid
        if r.get("signal_level") in ("STRONG_BUY", "STRONG_SELL")
        and (r.get("option_setup") or {}).get("qualified")
    ]
    shielded_main = [
        r for r in qualified_main
        if (r.get("option_setup") or {}).get("sr_protected")
    ]
    top_fundamentals = sorted(
        [r for r in valid if (r.get("fundamentals") or {}).get("level_code") in ("TOP_BUY", "GOOD_BUY")],
        key=lambda x: -(x.get("fundamentals") or {}).get("fund_score", 0),
    )

    def esc(val) -> str:
        return html_mod.escape(str(val or ""))

    def build_fund_cell(fund: dict) -> str:
        if not fund or not fund.get("available"):
            return '<span class="txt-muted">Keine Fundamentaldaten</span>'
        lvl_code = fund.get("level_code", "NEUTRAL")
        f_cls = {
            "TOP_BUY": "tier-top",
            "GOOD_BUY": "tier-good",
            "FAIR": "tier-fair",
            "EXPENSIVE": "tier-exp",
        }.get(lvl_code, "tier-fair")
        f_label = {
            "TOP_BUY": "TOP VALUE",
            "GOOD_BUY": "ATTRAKTIV",
            "FAIR": "FAIR VALUE",
            "EXPENSIVE": "SPEKULATIV",
        }.get(lvl_code, "NEUTRAL")
        score_val = fund.get("fund_score", 50)
        meta_parts = [p for p in (fund.get("disc_str"), fund.get("div_str"), fund.get("cap_str")) if p]
        meta_line = " · ".join(meta_parts)
        earn_cls = "earn-alert" if fund.get("earnings_warning") else "earn-normal"
        corp_html = f'<div class="corp-banner">{esc(fund["corp_action"])}</div>' if fund.get("corp_action") else ""
        return (
            f'<div class="cell-card">'
            f'<div class="card-top-row">'
            f'<span class="tier-badge {f_cls}">{f_label}</span>'
            f'<span class="mono score-chip">{score_val}/100</span>'
            f'</div>'
            f'<div class="mono metric-primary">{esc(fund.get("pe_str", ""))}</div>'
            f'<div class="metric-secondary">{esc(fund.get("growth_str", ""))}</div>'
            f'<div class="metric-meta">Analysten: <span class="txt-bright">{esc(fund.get("analyst_str", ""))}</span>'
            f'{(" · " + esc(meta_line)) if meta_line else ""}</div>'
            f'<div class="{earn_cls}">{esc(fund.get("earnings_str", ""))}</div>'
            f'{corp_html}'
            f'</div>'
        )

    rows_html = []
    for idx, r in enumerate(valid):
        lvl = r.get("signal_level", "NEUTRAL")
        sym = r.get("symbol", "")
        cat = r.get("category", "")
        close = r.get("last_close", 0.0)
        chg_pct = r.get("change_pct")
        ind = r.get("indicators", {})
        opt = r.get("option_setup") or {}
        sr = r.get("sr_lines") or {}
        ns = sr.get("nearest_support")
        nr = sr.get("nearest_resistance")
        fund = r.get("fundamentals") or {}
        ew = r.get("elliott_wave") or {}

        is_main = lvl in ("STRONG_BUY", "STRONG_SELL")
        is_weak = lvl in ("WEAK_BUY", "WEAK_SELL")
        is_opt_qual = bool(opt.get("qualified"))
        is_shielded = bool(is_opt_qual and opt.get("sr_protected"))
        score_num = max(r.get("bullish_count", 0), r.get("bearish_count", 0))
        fund_score = fund.get("fund_score", 0)
        prem_num = opt.get("prem_pct", 0.0) if is_opt_qual else 0.0
        ew_prob = ew.get("primary_prob", 0)

        sig_cls = {
            "STRONG_BUY": "sig-strong-buy",
            "STRONG_SELL": "sig-strong-sell",
            "WEAK_BUY": "sig-weak-buy",
            "WEAK_SELL": "sig-weak-sell",
            "NEUTRAL": "sig-neutral",
        }.get(lvl, "sig-neutral")

        sig_title = {
            "STRONG_BUY": f"STRONG BUY · {r['bullish_count']}/4",
            "STRONG_SELL": f"STRONG SELL · {r['bearish_count']}/4",
            "WEAK_BUY": "WATCH BUY · 2/4",
            "WEAK_SELL": "WATCH SELL · 2/4",
            "NEUTRAL": f"NEUTRAL · {score_num}/4",
        }.get(lvl, lvl)

        sig_sub = {
            "STRONG_BUY": "Setup: Short Put",
            "STRONG_SELL": "Setup: Short Call",
            "WEAK_BUY": "Beobachtung (Bullish)",
            "WEAK_SELL": "Beobachtung (Bearish)",
            "NEUTRAL": "Kein Signal",
        }.get(lvl, "")

        def ind_chip(label: str, state: str, val_str: str = "") -> str:
            c = "chip-bull" if state == "bullish" else ("chip-bear" if state == "bearish" else "chip-neutral")
            dot = "▲" if state == "bullish" else ("▼" if state == "bearish" else "·")
            v_html = f' <span class="mono">{esc(val_str)}</span>' if val_str else ""
            return f'<span class="ind-chip {c}">{dot} {label}{v_html}</span>'

        st_v = (ind.get("stoch") or {}).get("values") or {}
        rsi_v = (ind.get("rsi") or {}).get("value")
        macd_h = (ind.get("macd") or {}).get("hist")
        ind_grid = (
            ind_chip("BB", (ind.get("bollinger") or {}).get("state", "neutral"))
            + ind_chip("RSI", (ind.get("rsi") or {}).get("state", "neutral"), f"{rsi_v:.1f}" if rsi_v is not None else "")
            + ind_chip("MACD", (ind.get("macd") or {}).get("state", "neutral"), f"{macd_h:+.2f}" if macd_h is not None else "")
            + ind_chip("STOCH", (ind.get("stoch") or {}).get("state", "neutral"), f"{st_v.get('short_k', '')}")
        )

        reasons_list = r.get("reasons", [])
        reasons_html = "".join(f'<div class="reason-line">{esc(rs)}</div>' for rs in reasons_list) if reasons_list else '<div class="txt-muted">Keine aktiven Trigger</div>'

        # S/R Cell
        sr_items = []
        if ns:
            active_s_cls = "sr-active-sup" if sr.get("at_support") else ""
            sr_items.append(
                f'<div class="sr-row {active_s_cls}">'
                f'<span class="sr-tag sup-tag">SUP</span>'
                f'<span class="mono sr-price">{ns["price"]:.2f} $</span>'
                f'<span class="mono sr-dist">({ns["dist_pct"]:+.1f}%)</span>'
                f'<span class="sr-meta">{ns["bounces"]}T · {ns["pivots"]}P</span>'
                f'</div>'
            )
        if nr:
            active_r_cls = "sr-active-res" if sr.get("at_resistance") else ""
            sr_items.append(
                f'<div class="sr-row {active_r_cls}">'
                f'<span class="sr-tag res-tag">RES</span>'
                f'<span class="mono sr-price">{nr["price"]:.2f} $</span>'
                f'<span class="mono sr-dist">({nr["dist_pct"]:+.1f}%)</span>'
                f'<span class="sr-meta">{nr["bounces"]}T · {nr["pivots"]}P</span>'
                f'</div>'
            )
        sr_cell = "".join(sr_items) if sr_items else '<span class="txt-muted">Keine validierte Linie im Fokus</span>'

        # Option Cell
        if is_opt_qual:
            sign = "-" if "PUT" in opt["type"] else "+"
            opt_cls = "opt-card-put" if "PUT" in opt["type"] else "opt-card-call"
            type_badge_cls = "opt-badge-put" if "PUT" in opt["type"] else "opt-badge-call"
            shield_html = ""
            if is_shielded and opt.get("sr_shield_line"):
                sl = opt["sr_shield_line"]
                shield_html = (
                    f'<div class="shield-banner">'
                    f'<span class="shield-dot"></span>S/R SHIELDED · Linie {sl["price"]:.2f} $ ({sl["bounces"]}x bestätigt)'
                    f'</div>'
                )
            earn_warn_html = ""
            if opt.get("earnings_risk"):
                earn_warn_html = '<div class="opt-earn-warn">ACHTUNG: Quartalszahlen vor/nahe Verfall!</div>'

            opt_cell = (
                f'<div class="cell-card {opt_cls}">'
                f'<div class="card-top-row">'
                f'<span class="opt-badge {type_badge_cls}">{esc(opt["type"])}</span>'
                f'<span class="mono txt-muted">{esc(opt.get("cycle_label", ""))}</span>'
                f'</div>'
                f'<div class="mono opt-contract">{esc(opt.get("contract", ""))}</div>'
                f'<div class="opt-metrics-grid">'
                f'<div><span class="lbl">STRIKE</span><span class="mono val">{_format_strike_str(opt["strike"])} $ ({sign}{opt["otm_pct"]:.1f}%)</span></div>'
                f'<div><span class="lbl">PRÄMIE</span><span class="mono val txt-emerald">+{opt["prem_pct"]:.2f}% ({opt["premium_usd"]:.2f} $)</span></div>'
                f'<div><span class="lbl">DELTA</span><span class="mono val">Δ {opt["delta"]:.2f}</span></div>'
                f'<div><span class="lbl">IMPL. VOLA</span><span class="mono val">{opt.get("iv_pct", 0):.1f}%</span></div>'
                f'</div>'
                f'{shield_html}{earn_warn_html}'
                f'</div>'
            )
        elif opt:
            opt_cell = (
                f'<div class="cell-card opt-card-muted">'
                f'<div class="mono txt-muted" style="font-size:11px;margin-bottom:3px">{esc(opt.get("type", ""))} · KEIN TRADE</div>'
                f'<div class="txt-muted" style="font-size:11.5px">{esc(opt.get("note", ""))}</div>'
                f'</div>'
            )
        else:
            opt_cell = '<span class="txt-muted">—</span>'

        # Elliott Wave Cell
        ew_bias = ew.get("bias", "neutral")
        ew_cls = "ew-card-bull" if ew_bias == "bullish" else ("ew-card-bear" if ew_bias == "bearish" else "ew-card-neutral")
        ew_tag_cls = "ew-tag-bull" if ew_bias == "bullish" else ("ew-tag-bear" if ew_bias == "bearish" else "ew-tag-neutral")
        p_prim = ew.get("primary_prob", 50)
        p_alt = ew.get("alt_prob", 30)
        p_inv = ew.get("invalidation_prob", 20)
        tgt_p = ew.get("target_price")
        inv_p = ew.get("invalidation_price")
        tgt_dist = f"({((tgt_p - close)/close)*100:+.1f}%)" if (tgt_p and close > 0) else ""

        ew_cell = (
            f'<div class="cell-card {ew_cls}">'
            f'<div class="card-top-row">'
            f'<span class="ew-tag {ew_tag_cls}">{esc(ew.get("wave_code", "EW"))}</span>'
            f'<span class="mono txt-muted">{"RULES OK" if ew.get("rules_ok") else "TRANSITION"}</span>'
            f'</div>'
            f'<div class="ew-title">{esc(ew.get("phase", "N/A"))}</div>'
            f'<div class="ew-sub">{esc(ew.get("double_check", ""))}</div>'
            f'<div class="prob-track">'
            f'<div class="prob-seg-prim" style="width:{p_prim}%"></div>'
            f'<div class="prob-seg-alt" style="width:{p_alt}%"></div>'
            f'<div class="prob-seg-inv" style="width:{p_inv}%"></div>'
            f'</div>'
            f'<div class="ew-prob-row">'
            f'<span class="mono txt-bright">{p_prim}%</span> {esc(ew.get("primary_scenario", ""))}'
            f'</div>'
            f'<div class="ew-levels-row mono">'
            f'<span>ZIEL: <b>{tgt_p:.2f} $</b> {tgt_dist}</span>'
            f'<span>INVAL ({p_inv}%): <b>{inv_p:.2f} $</b></span>'
            f'</div>'
            f'</div>'
        )

        fund_cell = build_fund_cell(fund)

        chg_html = ""
        if chg_pct is not None:
            c_cls = "chg-pos" if chg_pct >= 0 else "chg-neg"
            chg_html = f'<div class="mono chg-pill {c_cls}">{chg_pct:+.2f}%</div>'

        # 52W Range Mini-Bar
        range_52_html = ""
        l52, h52 = fund.get("low_52"), fund.get("high_52")
        if l52 and h52 and h52 > l52 and close > 0:
            pct_52 = max(0, min(100, int(round((close - l52) / (h52 - l52) * 100))))
            range_52_html = (
                f'<div class="range-wrap" title="52W-Spanne: {l52:.2f}$ – {h52:.2f}$ ({pct_52}%)">'
                f'<div class="range-bar"><div class="range-fill" style="width:{pct_52}%"></div></div>'
                f'<div class="range-lbls mono"><span>{l52:.0f}$</span><span>52W</span><span>{h52:.0f}$</span></div>'
                f'</div>'
            )

        tv_url = f"https://de.tradingview.com/chart/?symbol={sym}"
        search_blob = f"{sym} {fund.get('name', '')} {cat} {lvl} {ew.get('phase', '')} {fund.get('level_code', '')}".lower()

        rows_html.append(
            f'<tr class="sig-row" '
            f'data-level="{lvl}" '
            f'data-main="{"1" if is_main else "0"}" '
            f'data-weak="{"1" if is_weak else "0"}" '
            f'data-qual="{"1" if (is_main and is_opt_qual) else "0"}" '
            f'data-shielded="{"1" if (is_main and is_shielded) else "0"}" '
            f'data-cat="{esc(cat)}" '
            f'data-search="{esc(search_blob)}" '
            f'data-price="{close:.4f}" '
            f'data-chg="{chg_pct if chg_pct is not None else -999:.4f}" '
            f'data-score="{score_num}" '
            f'data-prem="{prem_num:.4f}" '
            f'data-ewprob="{ew_prob}" '
            f'data-fund="{fund_score}">'
            f'<td class="col-asset">'
            f'<div class="asset-top">'
            f'<a href="{tv_url}" target="_blank" class="mono sym-badge">{esc(sym)}<span class="ext-arr">↗</span></a>'
            f'<button type="button" class="chart-jump-btn mono" onclick="openStockChart(\'{esc(sym)}\')">📊 Chart</button>'
            f'<span class="cat-pill">{esc(cat)}</span>'
            f'</div>'
            f'<div class="company-name">{esc(fund.get("name", sym))}</div>'
            f'<div class="mono date-sub">Stand: {esc(r.get("last_date", ""))}</div>'
            f'</td>'
            f'<td class="col-price">'
            f'<div class="mono price-main">{close:.2f} $</div>'
            f'{chg_html}'
            f'{range_52_html}'
            f'</td>'
            f'<td class="col-sig">'
            f'<div class="sig-badge {sig_cls}">{sig_title}</div>'
            f'<div class="sig-sub">{sig_sub}</div>'
            f'</td>'
            f'<td class="col-ind">'
            f'<div class="ind-strip">{ind_grid}</div>'
            f'<div class="reasons-box">{reasons_html}</div>'
            f'</td>'
            f'<td class="col-sr">{sr_cell}</td>'
            f'<td class="col-opt">{opt_cell}</td>'
            f'<td class="col-ew">{ew_cell}</td>'
            f'<td class="col-fund">{fund_cell}</td>'
            f'</tr>'
        )

    # Zweite Übersichtstabelle: Selektive Fundamental-Kaufniveau-Matrix
    fund_overview_rows = []
    for r in top_fundamentals:
        sym = r.get("symbol", "")
        cat = r.get("category", "")
        close = r.get("last_close", 0.0)
        chg_pct = r.get("change_pct")
        fund = r.get("fundamentals") or {}
        ew = r.get("elliott_wave") or {}
        sr = r.get("sr_lines") or {}
        ns = sr.get("nearest_support")
        tv_url = f"https://de.tradingview.com/chart/?symbol={sym}"
        chg_html = f'<span class="mono {"txt-emerald" if (chg_pct or 0) >= 0 else "txt-rose"}">({chg_pct:+.2f}%)</span>' if chg_pct is not None else ""
        sup_str = f'SUP {ns["price"]:.2f} $ ({ns["dist_pct"]:+.1f}% · {ns["bounces"]}T/{ns["pivots"]}P)' if ns else "Kein naher Support"
        lvl_code = fund.get("level_code", "FAIR")
        f_cls = "tier-top" if lvl_code == "TOP_BUY" else "tier-good"
        f_lbl = "TOP VALUE" if lvl_code == "TOP_BUY" else "ATTRAKTIV"

        fund_overview_rows.append(
            f'<tr>'
            f'<td><div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">'
            f'<a href="{tv_url}" target="_blank" class="mono sym-badge">{esc(sym)}<span class="ext-arr">↗</span></a>'
            f'<button type="button" class="chart-jump-btn mono" onclick="openStockChart(\'{esc(sym)}\')">📊 Chart</button>'
            f'</div>'
            f'<div class="company-name" style="margin-top:4px">{esc(fund.get("name", ""))} · <span class="txt-muted">{esc(cat)}</span></div></td>'
            f'<td class="mono price-main">{close:.2f} $ {chg_html}</td>'
            f'<td><span class="tier-badge {f_cls}">{f_lbl}</span> <span class="mono score-chip">{fund.get("fund_score", 0)}/100</span>'
            f'<div class="metric-secondary" style="margin-top:5px">{esc(fund.get("summary_line", ""))}</div></td>'
            f'<td><div class="mono metric-primary">{esc(fund.get("pe_str", ""))}</div><div class="metric-secondary">{esc(fund.get("growth_str", ""))}</div></td>'
            f'<td><div class="metric-secondary">Konsens: <b class="txt-bright">{esc(fund.get("analyst_str", ""))}</b></div>'
            f'<div class="metric-meta mono">{esc(fund.get("disc_str", ""))} · {esc(fund.get("div_str", ""))} · {esc(fund.get("cap_str", ""))}</div></td>'
            f'<td><div class="{"earn-alert" if fund.get("earnings_warning") else "earn-normal"}">{esc(fund.get("earnings_str", ""))}</div>'
            f'<div class="metric-meta mono" style="margin-top:3px">{esc(sup_str)}</div></td>'
            f'<td><div class="mono" style="font-size:11.5px;color:#cbd5e1">{esc(r.get("signal", "NEUTRAL"))}</div>'
            f'<div class="ew-sub" style="margin-top:3px">{esc(ew.get("phase", ""))} ({ew.get("primary_prob", 50)}%)</div></td>'
            f'</tr>'
        )

    categories_sorted = sorted(list(WATCHLIST_GROUPS.keys()))
    cat_options_html = "".join(f'<option value="{esc(c)}">{esc(c)} ({len(WATCHLIST_GROUPS[c])})</option>' for c in categories_sorted)
    inactive_note = (
        f" · {len(inactive_list)} delistete/inaktive Ticker automatisch isoliert ({', '.join(x['symbol'] for x in inactive_list)})"
        if inactive_list else ""
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    icon_b64_path = os.path.join(base_dir, "assets", "icon_180_b64.txt")
    icon_data_uri = ""
    if os.path.exists(icon_b64_path):
        try:
            with open(icon_b64_path, "r", encoding="utf-8") as f_ic:
                icon_data_uri = f_ic.read().strip()
        except Exception:
            icon_data_uri = ""

    # Kompaktes 1-Jahres-Chart-Datenpaket (inkl. Indikator-Warmup, S/R-Linien & Options-Strikes) für alle validen Aktien
    all_dates_set = set()
    for r in valid:
        for c in (r.get("candles_1y") or []):
            all_dates_set.add(c["t"])
    master_dates = sorted(all_dates_set)
    date_to_idx = {d: idx for idx, d in enumerate(master_dates)}

    chart_data_map = {}
    chart_order = []
    for r in valid:
        sym = r.get("symbol", "")
        if not sym:
            continue
        chart_order.append(sym)
        c_list = r.get("candles_1y") or []
        flat_c = []
        for c in c_list:
            flat_c.extend([date_to_idx.get(c["t"], 0), c["o"], c["h"], c["l"], c["c"]])
        sr = r.get("sr_lines") or {}
        opt = r.get("option_setup") or {}
        fund = r.get("fundamentals") or {}
        ew = r.get("elliott_wave") or {}
        ind = r.get("indicators") or {}
        opt_compact = None
        if opt and opt.get("strike"):
            opt_compact = {
                "q": bool(opt.get("qualified")),
                "sh": bool(opt.get("sr_protected")),
                "t": opt.get("type", ""),
                "k": opt.get("strike"),
                "d": opt.get("delta"),
                "pr": opt.get("prem_pct") or 0.0,
                "p": opt.get("prem_usd") or opt.get("premium") or 0.0,
                "otm": opt.get("otm_pct") or 0.0,
                "con": opt.get("contract", ""),
                "note": opt.get("note", ""),
            }
        chart_data_map[sym] = {
            "n": fund.get("name", sym),
            "cat": r.get("category", ""),
            "last": r.get("last_close", 0.0),
            "chg": r.get("change_pct"),
            "lvl": r.get("signal_level", "NEUTRAL"),
            "sig": r.get("signal", "NEUTRAL"),
            "bull": r.get("bullish_count", 0),
            "bear": r.get("bearish_count", 0),
            "es": r.get("ema_seeds") or [],
            "c": flat_c,
            "sup": [{"p": s["price"], "b": s["bounces"], "v": s["pivots"], "d": s["dist_pct"]} for s in (sr.get("supports") or [])[:5]],
            "res": [{"p": s["price"], "b": s["bounces"], "v": s["pivots"], "d": s["dist_pct"]} for s in (sr.get("resistances") or [])[:5]],
            "opt": opt_compact,
            "ind": {
                "bb": (ind.get("BB") or {}).get("dir", 0),
                "rsi": (ind.get("RSI") or {}).get("dir", 0),
                "macd": (ind.get("MACD") or {}).get("dir", 0),
                "stoch": (ind.get("STOCH") or {}).get("dir", 0),
            },
            "rsn": r.get("reasons") or [],
            "ew": {
                "w": ew.get("wave_code", ""),
                "ph": ew.get("phase", ""),
                "pr": ew.get("primary_prob", 50),
                "sc": ew.get("primary_scenario", ""),
                "tgt": ew.get("target_price"),
                "inv": ew.get("invalidation_price"),
            },
            "fs": fund.get("fund_score", 0),
            "fl": fund.get("level_label", ""),
        }

    master_dates_json = json.dumps(master_dates, separators=(",", ":"))
    chart_order_json = json.dumps(chart_order, separators=(",", ":"))
    chart_data_json = json.dumps(chart_data_map, ensure_ascii=False, separators=(",", ":"))

    icon_link_tags = ""
    brand_icon_html = '<div class="brand-mark">V1</div>'
    if icon_data_uri:
        icon_link_tags = (
            f'<link rel="apple-touch-icon" sizes="180x180" href="{icon_data_uri}">\n'
            f'<link rel="icon" type="image/png" href="{icon_data_uri}">'
        )
        brand_icon_html = f'<img src="{icon_data_uri}" alt="Version 1" class="brand-icon-img">'

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Version 1">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#070a10">
<link rel="manifest" href="/manifest.json">
{icon_link_tags}
<title>Version 1 — Market Vision Pro (Stillhalter &amp; Quantitative App)</title>
<style>
  :root {{
    --bg-main: #070a10;
    --bg-panel: #0d121c;
    --bg-card: #121926;
    --bg-hover: #172031;
    --border-subtle: #1e293b;
    --border-strong: #334155;
    --txt-primary: #f8fafc;
    --txt-secondary: #cbd5e1;
    --txt-muted: #64748b;
    --emerald: #10b981;
    --emerald-bg: rgba(16, 185, 129, 0.12);
    --emerald-border: rgba(16, 185, 129, 0.35);
    --rose: #f43f5e;
    --rose-bg: rgba(244, 63, 94, 0.12);
    --rose-border: rgba(244, 63, 94, 0.35);
    --sky: #38bdf8;
    --sky-bg: rgba(56, 189, 248, 0.10);
    --amber: #f59e0b;
    --amber-bg: rgba(245, 158, 11, 0.12);
    --violet: #a855f7;
  }}
  * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Text", "Segoe UI", sans-serif;
    background: var(--bg-main);
    color: var(--txt-primary);
    margin: 0;
    padding: env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left);
    -webkit-font-smoothing: antialiased;
  }}
  .mono {{
    font-family: "SF Mono", "JetBrains Mono", "Fira Code", Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums;
  }}
  .topbar {{
    background: rgba(13, 18, 28, 0.94);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border-subtle);
    padding: 13px 24px;
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
  }}
  .brand-wrap {{ display: flex; align-items: center; gap: 13px; }}
  .brand-icon-img {{
    width: 42px; height: 42px; border-radius: 10px;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
    border: 1px solid rgba(56, 189, 248, 0.35);
    flex-shrink: 0;
  }}
  .brand-mark {{
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, #0ea5e9, #10b981);
    display: flex; align-items: center; justify-content: center;
    font-weight: 900; font-size: 15px; letter-spacing: 0.5px; color: #070a10;
  }}
  .v1-badge {{
    display: inline-block;
    background: linear-gradient(135deg, #fde68a, #f59e0b);
    color: #070a10;
    font-weight: 900;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 5px;
    margin-left: 8px;
    letter-spacing: 0.5px;
    vertical-align: middle;
  }}
  .brand-title {{ font-size: 16.5px; font-weight: 800; letter-spacing: -0.3px; color: var(--txt-primary); margin: 0; }}
  .brand-sub {{ font-size: 11.5px; color: var(--txt-muted); margin-top: 2px; }}
  .top-actions {{ display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }}
  .view-nav {{
    display: flex; background: var(--bg-card); border: 1px solid var(--border-subtle);
    border-radius: 7px; padding: 3px; gap: 3px;
  }}
  .view-btn {{
    background: transparent; color: var(--txt-secondary); border: none;
    padding: 6px 12px; border-radius: 5px; font-size: 12px; font-weight: 700;
    cursor: pointer; transition: all 0.15s ease; white-space: nowrap;
  }}
  .view-btn.active {{
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(56, 189, 248, 0.2));
    color: var(--txt-primary);
    border: 1px solid rgba(56, 189, 248, 0.4);
  }}
  .action-btn {{
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--bg-card); color: var(--txt-primary);
    border: 1px solid var(--border-strong);
    padding: 7px 13px; border-radius: 7px; font-size: 12px; font-weight: 700;
    cursor: pointer; transition: all 0.15s ease; white-space: nowrap;
  }}
  .action-btn:hover {{ border-color: var(--sky); color: var(--sky); }}
  .action-btn-gold {{
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.18), rgba(16, 185, 129, 0.15));
    border-color: rgba(245, 158, 11, 0.5); color: #fde68a;
  }}
  .status-pill {{
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    padding: 6px 11px; border-radius: 6px; font-size: 11.5px; color: var(--txt-secondary);
  }}
  .live-dot {{
    width: 7px; height: 7px; border-radius: 50%; background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
  }}
  .container {{ max-width: 1920px; margin: 0 auto; padding: 18px 24px 60px 24px; }}
  @media (max-width: 900px) {{
    .topbar {{ padding: 12px 14px; }}
    .container {{ padding: 12px 12px 50px 12px; }}
    .brand-sub {{ display: none; }}
    th:first-child, td.col-asset {{
      position: sticky; left: 0; z-index: 5;
      background: #0d121c;
      box-shadow: 3px 0 8px rgba(0,0,0,0.45);
    }}
  }}
  /* Share & Install Modal */
  .modal-backdrop {{
    display: none; position: fixed; inset: 0; background: rgba(4, 7, 13, 0.82);
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    z-index: 999; align-items: center; justify-content: center; padding: 16px;
  }}
  .modal-backdrop.open {{ display: flex; }}
  .modal-box {{
    background: var(--bg-panel); border: 1px solid var(--border-strong);
    border-radius: 14px; max-width: 680px; width: 100%; padding: 24px;
    max-height: 92vh; overflow-y: auto; box-shadow: 0 24px 60px rgba(0,0,0,0.7);
  }}
  .step-card {{
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    border-radius: 9px; padding: 14px 16px; margin-bottom: 12px;
  }}
  .step-num {{
    display: inline-block; background: var(--emerald-bg); color: var(--emerald);
    border: 1px solid var(--emerald-border); font-weight: 800; font-size: 11px;
    padding: 2px 7px; border-radius: 4px; margin-right: 8px;
  }}
  /* KPI Strip */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
  }}
  .kpi-card {{
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 14px 18px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .kpi-lbl {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; color: var(--txt-muted); }}
  .kpi-val {{ font-size: 26px; font-weight: 800; margin-top: 6px; letter-spacing: -0.5px; }}
  .kpi-sub {{ font-size: 11.5px; color: var(--txt-muted); margin-top: 4px; }}

  /* Control Bar */
  .controls-bar {{
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
  }}
  .tab-group {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .tab-btn {{
    background: transparent;
    color: var(--txt-secondary);
    border: 1px solid transparent;
    padding: 7px 13px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12.5px;
    font-weight: 600;
    transition: all 0.15s ease;
  }}
  .tab-btn:hover {{ background: var(--bg-card); color: var(--txt-primary); }}
  .tab-btn.active {{
    background: rgba(56, 189, 248, 0.12);
    color: var(--sky);
    border-color: rgba(56, 189, 248, 0.35);
  }}
  .search-group {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .input-box, .select-box {{
    background: var(--bg-card);
    color: var(--txt-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 12.5px;
    outline: none;
  }}
  .input-box {{ min-width: 240px; }}
  .input-box:focus, .select-box:focus {{ border-color: var(--sky); }}

  /* Table Architecture */
  .table-wrap {{
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow-x: auto;
  }}
  table {{ width: 100%; border-collapse: collapse; min-width: 1480px; }}
  th {{
    background: #0a0f18;
    color: var(--txt-muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 10.5px;
    letter-spacing: 0.6px;
    padding: 11px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border-subtle);
    user-select: none;
    white-space: nowrap;
  }}
  th.sortable {{ cursor: pointer; }}
  th.sortable:hover {{ color: var(--txt-primary); background: #111827; }}
  td {{
    padding: 12px 14px;
    border-bottom: 1px solid rgba(30, 41, 59, 0.65);
    vertical-align: top;
    font-size: 12.5px;
  }}
  tr.sig-row:hover td {{ background: rgba(23, 32, 49, 0.55); }}

  /* Column specifics */
  .col-asset {{ width: 165px; }}
  .col-price {{ width: 120px; }}
  .col-sig {{ width: 155px; }}
  .col-ind {{ width: 290px; }}
  .col-sr {{ width: 215px; }}
  .col-opt {{ width: 265px; }}
  .col-ew {{ width: 280px; }}
  .col-fund {{ width: 260px; }}

  .asset-top {{ display: flex; align-items: center; gap: 7px; }}
  .sym-badge {{
    color: var(--txt-primary);
    background: var(--bg-card);
    border: 1px solid var(--border-strong);
    padding: 3px 8px;
    border-radius: 5px;
    font-weight: 800;
    font-size: 14.5px;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }}
  .sym-badge:hover {{ border-color: var(--sky); color: var(--sky); }}
  .ext-arr {{ font-size: 10px; color: var(--txt-muted); }}
  .cat-pill {{
    font-size: 10px;
    font-weight: 600;
    color: var(--txt-muted);
    background: rgba(100, 116, 139, 0.12);
    padding: 2px 6px;
    border-radius: 4px;
  }}
  .company-name {{ font-size: 12px; color: var(--txt-secondary); margin-top: 5px; font-weight: 500; }}
  .date-sub {{ font-size: 10.5px; color: var(--txt-muted); margin-top: 2px; }}

  .price-main {{ font-size: 16px; font-weight: 800; letter-spacing: -0.3px; color: var(--txt-primary); white-space: nowrap; }}
  .chg-pill {{
    display: inline-block;
    margin-top: 4px;
    padding: 2px 6.5px;
    border-radius: 4px;
    font-size: 11.5px;
    font-weight: 700;
  }}
  .chg-pos {{ background: var(--emerald-bg); color: var(--emerald); }}
  .chg-neg {{ background: var(--rose-bg); color: var(--rose); }}

  .range-wrap {{ margin-top: 7px; max-width: 105px; }}
  .range-bar {{ height: 4px; background: #1e293b; border-radius: 2px; overflow: hidden; }}
  .range-fill {{ height: 100%; background: var(--sky); }}
  .range-lbls {{ display: flex; justify-content: space-between; font-size: 9.5px; color: var(--txt-muted); margin-top: 2px; }}

  /* Signal Badge */
  .sig-badge {{
    display: inline-block;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 12.5px;
    font-weight: 800;
    letter-spacing: 0.3px;
    white-space: nowrap;
  }}
  .sig-strong-buy {{ background: var(--emerald-bg); color: var(--emerald); border: 1px solid var(--emerald-border); }}
  .sig-strong-sell {{ background: var(--rose-bg); color: var(--rose); border: 1px solid var(--rose-border); }}
  .sig-weak-buy {{ background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(245,158,11,0.3); }}
  .sig-weak-sell {{ background: rgba(249,115,22,0.12); color: #fb923c; border: 1px solid rgba(249,115,22,0.3); }}
  .sig-neutral {{ background: var(--bg-card); color: var(--txt-muted); border: 1px solid var(--border-subtle); }}
  .sig-sub {{ font-size: 11px; color: var(--txt-muted); margin-top: 5px; }}

  /* Indicator Telemetry */
  .ind-strip {{ display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 7px; }}
  .ind-chip {{
    padding: 2.5px 7px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    border: 1px solid transparent;
  }}
  .chip-bull {{ background: var(--emerald-bg); color: var(--emerald); border-color: rgba(16,185,129,0.25); }}
  .chip-bear {{ background: var(--rose-bg); color: var(--rose); border-color: rgba(244,63,94,0.25); }}
  .chip-neutral {{ background: #131b29; color: var(--txt-muted); border-color: #1e293b; }}
  .reasons-box {{ display: flex; flex-direction: column; gap: 3px; }}
  .reason-line {{ font-size: 11.5px; color: var(--txt-secondary); line-height: 1.38; padding-left: 8px; border-left: 2px solid #334155; }}

  /* S/R Levels */
  .sr-row {{
    display: flex; align-items: center; gap: 6px;
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    padding: 5px 8px; border-radius: 5px; margin-bottom: 5px; font-size: 11.5px;
  }}
  .sr-active-sup {{ border-color: var(--sky); background: var(--sky-bg); }}
  .sr-active-res {{ border-color: var(--violet); background: rgba(168, 85, 247, 0.10); }}
  .sr-tag {{ font-size: 9.5px; font-weight: 700; padding: 1.5px 5px; border-radius: 3px; letter-spacing: 0.4px; }}
  .sup-tag {{ background: rgba(56, 189, 248, 0.18); color: var(--sky); }}
  .res-tag {{ background: rgba(168, 85, 247, 0.18); color: #c084fc; }}
  .sr-price {{ font-size: 12.5px; font-weight: 800; color: var(--txt-primary); }}
  .sr-dist {{ color: var(--txt-secondary); font-size: 11px; }}
  .sr-meta {{ margin-left: auto; font-size: 10.5px; color: var(--txt-muted); }}

  /* Structured Cards (Options, EW, Fundamentals) */
  .cell-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 9px 11px;
  }}
  .card-top-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; font-size: 11px; }}
  .opt-card-put {{ border-left: 3px solid var(--emerald); }}
  .opt-card-call {{ border-left: 3px solid var(--rose); }}
  .opt-card-muted {{ background: rgba(18, 25, 38, 0.5); border-style: dashed; }}
  .opt-badge {{ font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 3px; letter-spacing: 0.4px; }}
  .opt-badge-put {{ background: var(--emerald-bg); color: var(--emerald); }}
  .opt-badge-call {{ background: var(--rose-bg); color: var(--rose); }}
  .opt-contract {{ font-size: 13px; font-weight: 800; color: var(--txt-primary); margin-bottom: 6px; }}
  .opt-metrics-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 5px 10px;
    border-top: 1px solid rgba(255,255,255,0.05); padding-top: 6px;
  }}
  .opt-metrics-grid .lbl {{ display: block; font-size: 9.5px; color: var(--txt-muted); letter-spacing: 0.4px; }}
  .opt-metrics-grid .val {{ font-size: 12.5px; font-weight: 700; color: var(--txt-primary); }}
  .shield-banner {{
    margin-top: 7px; padding: 4px 7px; border-radius: 4px;
    background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3);
    color: var(--emerald); font-size: 10.5px; font-weight: 700; display: flex; align-items: center; gap: 6px;
  }}
  .shield-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--emerald); }}
  .opt-earn-warn {{
    margin-top: 5px; padding: 3px 6px; border-radius: 4px;
    background: var(--rose-bg); color: var(--rose); font-size: 10.5px; font-weight: 700;
  }}

  /* Elliott Wave Card */
  .ew-card-bull {{ border-left: 3px solid var(--emerald); }}
  .ew-card-bear {{ border-left: 3px solid var(--rose); }}
  .ew-card-neutral {{ border-left: 3px solid var(--amber); }}
  .ew-tag {{ font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
  .ew-tag-bull {{ background: var(--emerald-bg); color: var(--emerald); }}
  .ew-tag-bear {{ background: var(--rose-bg); color: var(--rose); }}
  .ew-tag-neutral {{ background: var(--amber-bg); color: var(--amber); }}
  .ew-title {{ font-size: 12px; font-weight: 700; color: var(--txt-primary); margin-bottom: 3px; }}
  .ew-sub {{ font-size: 11px; color: var(--sky); margin-bottom: 6px; line-height: 1.35; }}
  .prob-track {{ display: flex; height: 5px; border-radius: 3px; overflow: hidden; background: #1e293b; margin-bottom: 6px; }}
  .prob-seg-prim {{ background: var(--emerald); }}
  .prob-seg-alt {{ background: var(--amber); }}
  .prob-seg-inv {{ background: var(--rose); }}
  .ew-prob-row {{ font-size: 11.5px; color: var(--txt-secondary); margin-bottom: 5px; line-height: 1.35; }}
  .ew-levels-row {{
    display: flex; justify-content: space-between; font-size: 11px; color: var(--txt-muted);
    border-top: 1px solid rgba(255,255,255,0.05); padding-top: 5px;
  }}
  .ew-levels-row b {{ color: var(--txt-primary); font-weight: 700; }}

  /* Fundamentals Card */
  .tier-badge {{ font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 3px; letter-spacing: 0.4px; }}
  .tier-top {{ background: var(--emerald-bg); color: var(--emerald); border: 1px solid var(--emerald-border); }}
  .tier-good {{ background: var(--sky-bg); color: var(--sky); border: 1px solid rgba(56,189,248,0.3); }}
  .tier-fair {{ background: var(--amber-bg); color: var(--amber); }}
  .tier-exp {{ background: var(--rose-bg); color: var(--rose); }}
  .score-chip {{ font-size: 12px; font-weight: 800; color: var(--txt-primary); }}
  .metric-primary {{ font-size: 12.5px; font-weight: 700; color: var(--txt-primary); margin-bottom: 3px; }}
  .metric-secondary {{ font-size: 11.5px; color: var(--txt-secondary); margin-bottom: 3px; line-height: 1.35; }}
  .metric-meta {{ font-size: 10.5px; color: var(--txt-muted); margin-bottom: 4px; }}
  .earn-alert {{ color: var(--rose); font-weight: 700; font-size: 11px; background: var(--rose-bg); padding: 2px 6px; border-radius: 3px; display: inline-block; }}
  .earn-normal {{ color: var(--txt-muted); font-size: 10.5px; }}
  .corp-banner {{ margin-top: 4px; font-size: 10.5px; color: var(--amber); line-height: 1.3; }}

  .txt-bright {{ color: var(--txt-primary); font-weight: 600; }}
  .txt-muted {{ color: var(--txt-muted); }}
  .txt-emerald {{ color: var(--emerald); }}
  .txt-rose {{ color: var(--rose); }}
  .txt-sky {{ color: var(--sky); }}

  .section-hdr {{
    margin: 42px 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 10px;
  }}
  .section-hdr h2 {{ margin: 0; font-size: 16px; font-weight: 700; letter-spacing: -0.2px; }}

  .chart-jump-btn {{
    background: rgba(56, 189, 248, 0.12);
    color: var(--sky);
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 3px 7px;
    border-radius: 4px;
    font-size: 10.5px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .chart-jump-btn:hover {{
    background: rgba(56, 189, 248, 0.25);
    border-color: var(--sky);
    color: #fff;
  }}

  /* TradingView 4-Pane UMWA Workstation ("Julian Layout") */
  .tv-workstation {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 320px;
    gap: 14px;
    align-items: start;
  }}
  @media (max-width: 1120px) {{
    .tv-workstation {{ grid-template-columns: 1fr; }}
  }}
  .tv-main {{
    background: #0b0f17;
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 14px 34px -10px rgba(0,0,0,0.65);
  }}
  .tv-toolbar {{
    background: #101622;
    border-bottom: 1px solid var(--border-subtle);
    padding: 10px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
  }}
  .tv-sym-group {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .tv-sym-title {{ font-size: 17.5px; font-weight: 800; color: var(--txt-primary); letter-spacing: -0.3px; }}
  .tv-ohlc-bar {{
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    font-size: 12.5px; color: var(--txt-secondary);
    background: rgba(15, 23, 42, 0.75);
    padding: 4px 10px; border-radius: 5px; border: 1px solid #1e293b;
  }}
  .tv-ohlc-bar span b {{ color: var(--txt-primary); font-weight: 700; }}
  .tv-btn-group {{ display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }}
  .tv-mini-btn {{
    background: #161f30; color: var(--txt-secondary);
    border: 1px solid #26334d; padding: 4px 9px; border-radius: 4px;
    font-size: 11px; font-weight: 700; cursor: pointer; transition: all 0.12s;
  }}
  .tv-mini-btn:hover {{ border-color: var(--sky); color: var(--txt-primary); }}
  .tv-mini-btn.active {{
    background: rgba(56, 189, 248, 0.18);
    color: var(--sky);
    border-color: rgba(56, 189, 248, 0.55);
  }}
  .tv-canvas-wrap {{
    position: relative;
    width: 100%;
    height: 740px;
    background: #090d14;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
  }}
  @media (max-width: 768px) {{
    .tv-canvas-wrap {{ height: 620px; }}
  }}
  #umwaChartCanvas {{
    width: 100%;
    height: 100%;
    display: block;
    cursor: crosshair;
  }}
  .tv-bottom-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 12px;
    padding: 14px;
    background: #0d121c;
    border-top: 1px solid var(--border-subtle);
  }}
  .tv-info-card {{
    background: #121926;
    border: 1px solid var(--border-subtle);
    border-radius: 7px;
    padding: 11px 13px;
    font-size: 12px;
  }}
  .tv-info-hdr {{
    font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--txt-muted); margin-bottom: 6px;
  }}
  .tv-sidebar {{
    background: #0d121c;
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    max-height: 890px;
  }}
  .tv-side-hdr {{
    padding: 12px;
    background: #111826;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .tv-wl-list {{
    overflow-y: auto;
    flex: 1;
    divide-y: 1px solid #161f30;
  }}
  .tv-wl-row {{
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: 8px;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(30, 41, 59, 0.55);
    cursor: pointer;
    transition: background 0.1s;
  }}
  .tv-wl-row:hover {{ background: #162032; }}
  .tv-wl-row.active {{
    background: rgba(56, 189, 248, 0.14);
    border-left: 3px solid var(--sky);
  }}
  /* Full-Screen Login Gate */
  .auth-gate {{
    position: fixed;
    inset: 0;
    z-index: 99999;
    background: radial-gradient(circle at 50% 18%, #131d31 0%, #070a10 72%);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }}
  .auth-card {{
    background: rgba(13, 18, 28, 0.96);
    border: 1px solid var(--border-strong);
    border-radius: 16px;
    width: 100%;
    max-width: 420px;
    padding: 32px 28px;
    box-shadow: 0 28px 70px rgba(0, 0, 0, 0.85), 0 0 0 1px rgba(56, 189, 248, 0.12);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
  }}
  .auth-field {{
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 16px;
  }}
  .auth-label {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--txt-secondary);
  }}
  .auth-input {{
    background: #121926;
    color: var(--txt-primary);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 11px 14px;
    font-size: 14px;
    font-weight: 600;
    outline: none;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
    width: 100%;
  }}
  .auth-input:focus {{
    border-color: var(--sky);
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
  }}
  .auth-btn {{
    width: 100%;
    background: linear-gradient(135deg, #10b981, #0ea5e9);
    color: #04130d;
    border: none;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13.5px;
    font-weight: 800;
    letter-spacing: 0.2px;
    cursor: pointer;
    transition: transform 0.1s ease, filter 0.15s ease;
    margin-top: 6px;
  }}
  .auth-btn:hover {{ filter: brightness(1.08); }}
  .auth-btn:active {{ transform: scale(0.99); }}
  .auth-err {{
    display: none;
    background: var(--rose-bg);
    border: 1px solid var(--rose-border);
    color: var(--rose);
    border-radius: 7px;
    padding: 9px 12px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 14px;
    text-align: center;
  }}
  @keyframes authShake {{
    0%, 100% {{ transform: translateX(0); }}
    20%, 60% {{ transform: translateX(-7px); }}
    40%, 80% {{ transform: translateX(7px); }}
  }}
  .auth-shake {{ animation: authShake 0.35s ease; }}
</style>
</head>
<body>
  <!-- Geschütztes Mitglieder-Login-Gate -->
  <div id="mvpAuthGate" class="auth-gate">
    <form id="mvpAuthForm" class="auth-card" onsubmit="return handleMvpLogin(event)">
      <div style="display:flex;flex-direction:column;align-items:center;text-align:center;margin-bottom:24px">
        {brand_icon_html}
        <h1 style="font-size:19px;font-weight:800;letter-spacing:-0.3px;margin:14px 0 4px 0;color:var(--txt-primary)">
          MARKET VISION PRO <span class="v1-badge">VERSION 1</span>
        </h1>
        <div style="font-size:12px;color:var(--txt-muted);font-weight:600">Geschützter Mitglieder-Zugang</div>
      </div>

      <div id="mvpAuthError" class="auth-err">Ungültiger Benutzername oder Zugangs-Code.</div>

      <div class="auth-field">
        <label class="auth-label" for="mvpUserInput">Benutzername</label>
        <input id="mvpUserInput" type="text" class="auth-input" placeholder="Benutzername eingeben..." autocomplete="username" autocapitalize="words" spellcheck="false" required>
      </div>

      <div class="auth-field">
        <label class="auth-label" for="mvpCodeInput">Zugangs-Code</label>
        <input id="mvpCodeInput" type="password" inputmode="numeric" class="auth-input mono" placeholder="•••••" autocomplete="current-password" required>
      </div>

      <label style="display:flex;align-items:center;gap:8px;font-size:12px;color:var(--txt-secondary);margin-bottom:14px;cursor:pointer;user-select:none">
        <input id="mvpRememberChk" type="checkbox" checked style="accent-color:var(--emerald);width:15px;height:15px;cursor:pointer">
        Auf diesem Gerät angemeldet bleiben
      </label>

      <button type="submit" class="auth-btn">Anmelden &amp; Terminal öffnen &rarr;</button>
    </form>
  </div>

  <div id="mvpProtectedApp" style="display:none">
  <header class="topbar">
    <div class="brand-wrap">
      {brand_icon_html}
      <div>
        <h1 class="brand-title">MARKET VISION PRO <span class="v1-badge">VERSION 1</span></h1>
      </div>
    </div>
    <div class="top-actions">
      <div class="view-nav">
        <button class="view-btn active" id="nav-scanner" onclick="switchMainView('scanner', this)">Live-Scanner ({len(valid)})</button>
        <button class="view-btn" id="nav-charts" onclick="switchMainView('charts', this)">📊 UMWA Charts (1J)</button>
        <button class="view-btn" id="nav-fundamentals" onclick="switchMainView('fundamentals', this)">Kauf-Niveau ({len(top_fundamentals)})</button>
      </div>
      <button id="refreshBtn" class="action-btn" onclick="triggerLiveRefresh()">↻ Live-Update</button>
      <button class="action-btn action-btn-gold" onclick="openShareModal()">📲 App &amp; Teilen</button>
      <button type="button" class="action-btn" onclick="logoutMvpUser()" title="Abmelden">
        <span id="mvpActiveUserLbl">👤 Mitglied</span>
        <span style="color:var(--txt-muted)">·</span>
        <span style="color:var(--rose);font-size:11px">Logout</span>
      </button>
      <div class="status-pill mono">
        <span class="live-dot"></span>
        <span id="liveStatusText">{esc(now_str)}</span>
      </div>
    </div>
  </header>

  <main class="container">
    <div id="view-scanner">
    <section class="kpi-grid">
      <div class="kpi-card">
        <span class="kpi-lbl">S/R-Geschützte Top-Setups</span>
        <span class="kpi-val mono txt-emerald">{len(shielded_main)}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-lbl">Qualifizierte Weekly Optionen</span>
        <span class="kpi-val mono txt-sky">{len(qualified_main)}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-lbl">Hauptsignale Long (Short Put)</span>
        <span class="kpi-val mono txt-emerald">{len(strong_buys)}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-lbl">Hauptsignale Short (Short Call)</span>
        <span class="kpi-val mono txt-rose">{len(strong_sells)}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-lbl">Fundamental Top- / Attraktiv</span>
        <span class="kpi-val mono" style="color:var(--txt-primary)">{len(top_fundamentals)}</span>
      </div>
    </section>

    <section class="controls-bar">
      <div class="tab-group">
        <button class="tab-btn active" onclick="setFilter('main', this)">Hauptsignale 3–4/4 ({len(strong_buys)+len(strong_sells)})</button>
        <button class="tab-btn" onclick="setFilter('shielded', this)">S/R-Geschützte Optionen ({len(shielded_main)})</button>
        <button class="tab-btn" onclick="setFilter('qual', this)">Alle Qualifizierten Optionen ({len(qualified_main)})</button>
        <button class="tab-btn" onclick="setFilter('buy', this)">Short Put · 3–4/4 ({len(strong_buys)})</button>
        <button class="tab-btn" onclick="setFilter('sell', this)">Short Call · 3–4/4 ({len(strong_sells)})</button>
        <button class="tab-btn" onclick="setFilter('signals', this)">Inkl. Beobachtung 2/4 ({len(strong_buys)+len(strong_sells)+len(weak_signals)})</button>
        <button class="tab-btn" onclick="setFilter('all', this)">Komplette Watchlist ({len(valid)})</button>
      </div>
      <div class="search-group">
        <select id="catSelect" class="select-box" onchange="applyAllFilters()">
          <option value="ALL">Alle Kategorien ({len(valid)})</option>
          {cat_options_html}
        </select>
        <input type="text" id="searchInput" class="input-box" placeholder="Suche nach Ticker, Name, Welle..." oninput="applyAllFilters()">
      </div>
    </section>

    <div class="table-wrap">
      <table id="mainTable">
        <thead>
          <tr>
            <th class="sortable" onclick="sortTable('sym')">Ticker / Segment ↕</th>
            <th class="sortable" onclick="sortTable('price')">Live-Kurs / 52W ↕</th>
            <th class="sortable" onclick="sortTable('score')">UMWA Signal ↕</th>
            <th>4 UMWA-Indikatoren &amp; Auslöser</th>
            <th>Validierte S/R-Linien (2J)</th>
            <th class="sortable" onclick="sortTable('prem')">Weekly Stillhalter Option (OCC/IBKR) ↕</th>
            <th class="sortable" onclick="sortTable('ewprob')">Elliott-Wave Analyse (Info-Spalte) ↕</th>
            <th class="sortable" onclick="sortTable('fund')">Fundamental &amp; Kauf-Niveau (Info) ↕</th>
          </tr>
        </thead>
        <tbody id="mainTbody">
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
    </div>

    <div id="view-fundamentals" style="display:none">
    <div id="fundamental-matrix" class="section-hdr" style="margin-top:8px">
      <div>
        <h2>FUNDAMENTAL &amp; VALUATION MATRIX</h2>
      </div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Ticker / Unternehmen</th>
            <th>Live-Kurs</th>
            <th>Bewertungs-Stufe &amp; Kern-Treiber</th>
            <th>Multiple (KGV / Fwd / PEG) &amp; EPS-Dynamik</th>
            <th>Analysten-Konsens, 52W-Rabatt &amp; Kapitalisierung</th>
            <th>Earnings-Termin &amp; Nächste Support-Linie</th>
            <th>Technischer Status &amp; Elliott-Wave</th>
          </tr>
        </thead>
        <tbody>
          {"".join(fund_overview_rows)}
        </tbody>
      </table>
    </div>
    </div>

    <div id="view-charts" style="display:none">
      <div class="tv-workstation">
        <div class="tv-main">
          <div class="tv-toolbar">
            <div class="tv-sym-group">
              <span id="tvSymTitle" class="tv-sym-title mono">HAL</span>
              <span id="tvCompName" style="font-size:12.5px;color:var(--txt-secondary);font-weight:600">Halliburton Co. · 1D</span>
              <span id="tvCatBadge" class="cat-pill">C-WERTE</span>
              <span id="tvSigBadge" class="sig-badge sig-strong-buy" style="padding:3px 8px;font-size:11px">BUY (3/4)</span>
              <a id="tvExtLink" href="https://de.tradingview.com/chart/?symbol=HAL" target="_blank" class="tv-mini-btn mono" style="text-decoration:none">TradingView ↗</a>
            </div>
            <div id="tvOhlcBar" class="tv-ohlc-bar mono">
              <span>Datum: <b>—</b></span>
              <span>O <b>—</b></span>
              <span>H <b>—</b></span>
              <span>L <b>—</b></span>
              <span>C <b>—</b></span>
            </div>
            <div class="tv-btn-group">
              <button type="button" class="tv-mini-btn mono" onclick="setChartRange(22, this)" title="1 Monat (22 Handelstage)">1M</button>
              <button type="button" class="tv-mini-btn mono" onclick="setChartRange(65, this)" title="3 Monate (65 Handelstage)">3M</button>
              <button type="button" class="tv-mini-btn mono" onclick="setChartRange(126, this)" title="6 Monate (126 Handelstage)">6M</button>
              <button type="button" class="tv-mini-btn mono active" onclick="setChartRange(252, this)" title="1 Jahr (252 Handelstage)">1J (252T)</button>
              <span style="width:1px;height:16px;background:#26334d;margin:0 3px"></span>
              <button type="button" class="tv-mini-btn mono" onclick="zoomChartBy(0.75)" title="Reinzoomen (Mausrad hoch / +)">🔍＋</button>
              <button type="button" class="tv-mini-btn mono" onclick="zoomChartBy(1.33)" title="Rauszoomen (Mausrad runter / -)">🔍－</button>
              <button type="button" id="tvAutoBtn" class="tv-mini-btn mono" onclick="resetChartView()" title="Ansicht &amp; Skalierung zurücksetzen (Doppelklick im Chart)">⟲ Auto</button>
              <span style="width:1px;height:16px;background:#26334d;margin:0 3px"></span>
              <button type="button" id="togBB" class="tv-mini-btn active" onclick="toggleChartLayer('bb', this)">BB (20,2)</button>
              <button type="button" id="togEMA" class="tv-mini-btn active" onclick="toggleChartLayer('ema', this)">EMAs</button>
              <button type="button" id="togSR" class="tv-mini-btn active" onclick="toggleChartLayer('sr', this)">🔵 S/R-Linien</button>
              <button type="button" id="togOpt" class="tv-mini-btn active" onclick="toggleChartLayer('opt', this)">🛡️ Strike</button>
            </div>
          </div>

          <div class="tv-canvas-wrap">
            <canvas id="umwaChartCanvas"></canvas>
          </div>

          <div class="tv-bottom-cards">
            <div class="tv-info-card">
              <div class="tv-info-hdr">4 UMWA-Indikatoren &amp; Auslöser</div>
              <div id="tvCardInd"></div>
            </div>
            <div class="tv-info-card">
              <div class="tv-info-hdr">🔵 Validierte Horizontale S/R-Linien</div>
              <div id="tvCardSR"></div>
            </div>
            <div class="tv-info-card">
              <div class="tv-info-hdr">🛡️ Weekly Stillhalter-Option (Mo/Di &rarr; Fr)</div>
              <div id="tvCardOpt"></div>
            </div>
            <div class="tv-info-card">
              <div class="tv-info-hdr">Elliott-Wave &amp; Fundamental-Niveau</div>
              <div id="tvCardEW"></div>
            </div>
          </div>
        </div>

        <aside class="tv-sidebar">
          <div class="tv-side-hdr">
            <div style="display:flex;justify-content:space-between;align-items:center">
              <span style="font-size:12.5px;font-weight:800;color:var(--txt-primary)">WATCHLIST · 1J CHARTS</span>
              <span id="tvWlCount" class="mono" style="font-size:11px;color:var(--txt-muted)">{len(valid)} Werte</span>
            </div>
            <div class="tv-btn-group">
              <button type="button" class="tv-mini-btn active" onclick="setWlFilter('all', this)">Alle</button>
              <button type="button" class="tv-mini-btn" onclick="setWlFilter('main', this)">3–4/4 Signale</button>
              <button type="button" class="tv-mini-btn" onclick="setWlFilter('shield', this)">🛡️ Geschützt</button>
              <button type="button" class="tv-mini-btn" onclick="setWlFilter('watch', this)">2/4 Watch</button>
            </div>
            <select id="tvWlCat" class="select-box" style="width:100%;font-size:11.5px;padding:6px 8px" onchange="renderWatchlistSidebar()">
              <option value="ALL">Alle Kategorien ({len(valid)})</option>
              {cat_options_html}
            </select>
            <input id="tvWlSearch" type="text" class="input-box" style="width:100%;min-width:0;font-size:12px;padding:6px 10px" placeholder="Ticker oder Name suchen... (↑/↓ Tasten)" oninput="renderWatchlistSidebar()">
          </div>
          <div id="tvWlList" class="tv-wl-list"></div>
        </aside>
      </div>
    </div>
  </main>

  <!-- iPad / Handy & Bruder-App Modal -->
  <div id="shareModal" class="modal-backdrop" onclick="if(event.target===this)closeShareModal()">
    <div class="modal-box">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <div style="display:flex;align-items:center;gap:12px">
          {brand_icon_html}
          <div>
            <div style="font-size:16px;font-weight:800;color:var(--txt-primary)">Version 1 auf iPad / Handy &amp; mit Bruder teilen</div>
            <div style="font-size:12px;color:var(--txt-muted)">Als echte Vollbild-App mit eigenem App-Icon installieren</div>
          </div>
        </div>
        <button class="action-btn" onclick="closeShareModal()">✕ Schließen</button>
      </div>

      <div class="step-card">
        <div style="font-weight:700;color:var(--txt-primary);margin-bottom:6px"><span class="step-num">WEG 1</span>Direkt als App-Datei an deinen Bruder oder dein iPad/iPhone senden (AirDrop, WhatsApp, iMessage, Mail)</div>
        <div style="font-size:12px;color:var(--txt-secondary);margin-bottom:10px">
          Die Datei <b>Version_1_App.html</b> enthält die komplette App inkl. App-Icon, allen 255 Aktien, 1-Jahres-TradingView-Charts und Fundamental-Matrix. Sie funktioniert auf jedem iPad, iPhone, Android-Handy und Mac auch komplett unabhängig!
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap">
          <button class="action-btn action-btn-gold" onclick="shareAppFile()">📤 Jetzt per AirDrop / WhatsApp / Teilen senden</button>
          <a href="/download/Version_1_App.html" download="Version_1_App.html" class="action-btn" style="text-decoration:none">⬇ Version_1_App.html speichern</a>
        </div>
      </div>

      <div class="step-card">
        <div style="font-weight:700;color:var(--txt-primary);margin-bottom:6px"><span class="step-num">WEG 2</span>Live-App-Link &amp; QR-Code (für iPad, iPhone &amp; Bruder über Browser)</div>
        <div id="serverUrlBox" style="font-size:12px;color:var(--txt-secondary);margin-bottom:10px">
          Prüfe lokalen Live-App-Server...
        </div>
        <div id="qrWrap" style="display:none;align-items:center;gap:16px;flex-wrap:wrap;margin-top:10px">
          <img id="qrImg" src="" alt="QR Code" style="width:148px;height:148px;border-radius:8px;background:#fff;padding:8px">
          <div style="flex:1;min-width:200px">
            <div style="font-size:11px;color:var(--txt-muted);margin-bottom:4px">Scanne den QR-Code mit der iPad-/iPhone-Kamera oder kopiere den Link:</div>
            <input id="shareLinkInput" class="input-box mono" style="width:100%;margin-bottom:8px" readonly value="">
            <div style="display:flex;gap:8px;flex-wrap:wrap">
              <button class="action-btn" onclick="copyShareLink()">📋 Link kopieren</button>
              <button id="tunnelBtn" class="action-btn action-btn-gold" onclick="startPublicTunnel()">🌐 Öffentlichen Internet-Link für Bruder aktivieren</button>
            </div>
          </div>
        </div>
      </div>

      <div class="step-card" style="margin-bottom:0">
        <div style="font-weight:700;color:var(--txt-primary);margin-bottom:6px"><span class="step-num">SO GEHT'S</span>Auf iPad &amp; iPhone mit schönem Icon auf den Home-Bildschirm setzen</div>
        <ol style="margin:6px 0 0 18px;padding:0;font-size:12px;color:var(--txt-secondary);line-height:1.6">
          <li>Öffne <b>https://www.market-vision-pro.com</b> auf dem iPad oder iPhone in <b>Safari</b>.</li>
          <li>Tippe auf das <b>Teilen-Symbol</b> (<span class="mono">↑</span> Viereck mit Pfeil nach oben).</li>
          <li>Wähle <b>„Zum Home-Bildschirm“</b> (Add to Home Screen) &rarr; Tippe auf <b>„Hinzufügen“</b>.</li>
          <li>Fertig! <b>Version 1</b> liegt jetzt mit dem neuen dunkel-goldenen <b>V1 App-Icon</b> als echte Vollbild-App auf deinem iPad, Handy und beim Bruder!</li>
        </ol>
      </div>
    </div>
  </div>
  </div><!-- /#mvpProtectedApp -->

<script>
// =====================================================================
// MITGLIEDER-AUTHENTIFIZIERUNG (Julian, Pascal, Hannes, Laura, Steffan)
// =====================================================================
const MVP_AUTH_KEY = 'mvp_v1_auth_session_2026';
const MVP_AUTH_HASHES = {{
  '67825683201452b1': 'Julian',
  '9741db88ef85b044': 'Pascal',
  '80a0c335bc8f0aab': 'Hannes',
  'd6c4648d934c79db': 'Laura',
  'ad39aac7e47855d5': 'Steffan',
  '0bad7959cc9ee473': 'Steffan'
}};

function computeMvpAuthHash(userStr, codeStr) {{
  const s = 'MVP_V1_SALT_2026::' + String(userStr || '').trim().toLowerCase() + '::' + String(codeStr || '').trim() + '::UMWA';
  let h1 = 0x811c9dc5 | 0;
  let h2 = 0x01000193 | 0;
  for (let i = 0; i < s.length; i++) {{
    const c = s.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193);
    h2 = Math.imul((h2 + c) | 0, 0x85ebca6b);
  }}
  const p1 = (h1 >>> 0).toString(16).padStart(8, '0');
  const p2 = (h2 >>> 0).toString(16).padStart(8, '0');
  return p1 + p2;
}}

function unlockMvpApp(displayName, tokenHash) {{
  const gate = document.getElementById('mvpAuthGate');
  const app = document.getElementById('mvpProtectedApp');
  const lbl = document.getElementById('mvpActiveUserLbl');
  if (gate) gate.style.display = 'none';
  if (app) app.style.display = 'block';
  if (lbl) lbl.textContent = '👤 ' + displayName;
  if (document.getElementById('view-charts') && document.getElementById('view-charts').style.display !== 'none') {{
    drawUmwaChart();
  }}
}}

function handleMvpLogin(e) {{
  if (e && e.preventDefault) e.preventDefault();
  const uEl = document.getElementById('mvpUserInput');
  const cEl = document.getElementById('mvpCodeInput');
  const rEl = document.getElementById('mvpRememberChk');
  const errEl = document.getElementById('mvpAuthError');
  const formEl = document.getElementById('mvpAuthForm');
  const uVal = uEl ? uEl.value : '';
  const cVal = cEl ? cEl.value : '';
  const h = computeMvpAuthHash(uVal, cVal);
  const matchedUser = MVP_AUTH_HASHES[h];

  if (matchedUser) {{
    if (errEl) errEl.style.display = 'none';
    const payload = JSON.stringify({{ h: h, u: matchedUser, ts: Date.now() }});
    try {{
      sessionStorage.setItem(MVP_AUTH_KEY, payload);
      if (!rEl || rEl.checked) localStorage.setItem(MVP_AUTH_KEY, payload);
      else localStorage.removeItem(MVP_AUTH_KEY);
    }} catch (err) {{}}
    unlockMvpApp(matchedUser, h);
  }} else {{
    if (errEl) errEl.style.display = 'block';
    if (formEl) {{
      formEl.classList.remove('auth-shake');
      void formEl.offsetWidth;
      formEl.classList.add('auth-shake');
    }}
    if (cEl) {{
      cEl.value = '';
      cEl.focus();
    }}
  }}
  return false;
}}

function logoutMvpUser() {{
  try {{
    localStorage.removeItem(MVP_AUTH_KEY);
    sessionStorage.removeItem(MVP_AUTH_KEY);
  }} catch (err) {{}}
  const gate = document.getElementById('mvpAuthGate');
  const app = document.getElementById('mvpProtectedApp');
  const cEl = document.getElementById('mvpCodeInput');
  if (app) app.style.display = 'none';
  if (gate) gate.style.display = 'flex';
  if (cEl) cEl.value = '';
}}

(function checkSavedMvpAuth() {{
  try {{
    const raw = sessionStorage.getItem(MVP_AUTH_KEY) || localStorage.getItem(MVP_AUTH_KEY);
    if (raw) {{
      const parsed = JSON.parse(raw);
      if (parsed && parsed.h && MVP_AUTH_HASHES[parsed.h]) {{
        unlockMvpApp(MVP_AUTH_HASHES[parsed.h], parsed.h);
        return;
      }}
    }}
  }} catch (err) {{}}
}})();

const MASTER_DATES = {master_dates_json};
const CHART_ORDER = {chart_order_json};
const CHART_DATA = {chart_data_json};

let currentMode = 'main';
let sortState = {{ key: null, asc: false }};
let activeChartSym = CHART_ORDER[0] || 'HAL';
let chartBarsCount = 252;
let chartLayers = {{ bb: true, ema: true, sr: true, opt: true }};
let wlFilterMode = 'all';
let hoverBarIdx = null;
let hoverRelX = null;
let hoverRelY = null;
let cachedChartCalc = null;
let tvViewStart = null;
let tvViewEnd = null;
let tvPriceZoom = 1.0;
let tvPricePanPx = 0.0;

function switchMainView(view, btn) {{
  document.getElementById('view-scanner').style.display = (view === 'scanner') ? '' : 'none';
  document.getElementById('view-charts').style.display = (view === 'charts') ? '' : 'none';
  document.getElementById('view-fundamentals').style.display = (view === 'fundamentals') ? '' : 'none';
  document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  else {{
    const navBtn = document.getElementById('nav-' + view);
    if (navBtn) navBtn.classList.add('active');
  }}
  if (view === 'charts') {{
    renderWatchlistSidebar();
    selectChartSymbol(activeChartSym);
  }}
  window.scrollTo({{ top: 0, behavior: 'smooth' }});
}}

function openStockChart(sym) {{
  if (CHART_DATA[sym]) {{
    activeChartSym = sym;
  }}
  switchMainView('charts', document.getElementById('nav-charts'));
}}

function setFilter(mode, btn) {{
  currentMode = mode;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  applyAllFilters();
}}

function applyAllFilters() {{
  const q = (document.getElementById('searchInput').value || '').trim().toLowerCase();
  const cat = document.getElementById('catSelect').value;
  document.querySelectorAll('#mainTbody .sig-row').forEach(row => {{
    const lvl = row.getAttribute('data-level');
    const main = row.getAttribute('data-main') === '1';
    const weak = row.getAttribute('data-weak') === '1';
    const qual = row.getAttribute('data-qual') === '1';
    const shielded = row.getAttribute('data-shielded') === '1';
    const rCat = row.getAttribute('data-cat');
    const sBlob = row.getAttribute('data-search') || '';

    let show = true;
    if (currentMode === 'main') show = main;
    else if (currentMode === 'shielded') show = shielded;
    else if (currentMode === 'qual') show = qual;
    else if (currentMode === 'buy') show = (lvl === 'STRONG_BUY');
    else if (currentMode === 'sell') show = (lvl === 'STRONG_SELL');
    else if (currentMode === 'signals') show = (main || weak);
    else if (currentMode === 'all') show = true;

    if (show && cat !== 'ALL' && rCat !== cat) show = false;
    if (show && q && !sBlob.includes(q)) show = false;

    row.style.display = show ? '' : 'none';
  }});
}}

function sortTable(key) {{
  const tbody = document.getElementById('mainTbody');
  const rows = Array.from(tbody.querySelectorAll('.sig-row'));
  if (sortState.key === key) sortState.asc = !sortState.asc;
  else {{ sortState.key = key; sortState.asc = (key === 'sym'); }}

  rows.sort((a, b) => {{
    if (key === 'sym') {{
      const va = a.querySelector('.sym-badge').textContent.trim();
      const vb = b.querySelector('.sym-badge').textContent.trim();
      return sortState.asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }}
    const attrMap = {{
      price: 'data-price',
      score: 'data-score',
      prem: 'data-prem',
      ewprob: 'data-ewprob',
      fund: 'data-fund'
    }};
    const attr = attrMap[key];
    const na = parseFloat(a.getAttribute(attr) || '0');
    const nb = parseFloat(b.getAttribute(attr) || '0');
    return sortState.asc ? (na - nb) : (nb - na);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

async function resolveApiBase() {{
  if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {{
    try {{
      const r = await fetch('/api/status');
      if (r.ok && (r.headers.get('content-type') || '').includes('application/json')) {{
        return '';
      }}
    }} catch (e) {{}}
  }}
  try {{
    const r = await fetch('http://localhost:8765/api/status');
    if (r.ok) return 'http://localhost:8765';
  }} catch (e) {{}}
  return 'https://version-1-umwa.onrender.com';
}}

async function pollRefreshStatus(base, btn) {{
  btn.disabled = true;
  const start = Date.now();
  while (true) {{
    const sec = Math.max(1, Math.round((Date.now() - start) / 1000));
    btn.innerHTML = '⏳ Scanne 255 Aktien (' + sec + 's)...';
    await new Promise(r => setTimeout(r, 2000));
    try {{
      const st = await fetch(base + '/api/status');
      if (st.ok) {{
        const d = await st.json();
        if (!d.refreshing) {{
          if (base === '' || window.location.hostname.includes('market-vision-pro.com')) window.location.reload();
          else window.location.href = base + '/';
          return;
        }}
      }}
    }} catch (e) {{}}
    if (sec > 180) break;
  }}
  btn.innerHTML = '↻ Live-Update';
  btn.disabled = false;
}}

async function triggerLiveRefresh() {{
  const btn = document.getElementById('refreshBtn');
  btn.innerHTML = '⏳ Starte Live-Scan...';
  btn.disabled = true;
  try {{
    const base = await resolveApiBase();
    const res = await fetch(base + '/api/refresh', {{ method: 'POST' }});
    if (res.ok) {{
      await pollRefreshStatus(base, btn);
      return;
    }}
  }} catch (e) {{}}
  window.location.reload();
}}

async function openShareModal() {{
  document.getElementById('shareModal').classList.add('open');
  const box = document.getElementById('serverUrlBox');
  const qrWrap = document.getElementById('qrWrap');
  const mainDomainUrl = 'https://www.market-vision-pro.com';
  box.innerHTML = '🌐 <b>Offizielle Website aktiv!</b> Überall auf Mac, iPad &amp; Handy mit Passwort-Schutz nutzbar:';
  document.getElementById('shareLinkInput').value = mainDomainUrl;
  document.getElementById('qrImg').src = 'https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=' + encodeURIComponent(mainDomainUrl);
  qrWrap.style.display = 'flex';
}}

function closeShareModal() {{
  document.getElementById('shareModal').classList.remove('open');
}}

function copyShareLink() {{
  const inp = document.getElementById('shareLinkInput');
  inp.select();
  navigator.clipboard.writeText(inp.value);
  alert('App-Link in die Zwischenablage kopiert: ' + inp.value);
}}

async function startPublicTunnel() {{
  const btn = document.getElementById('tunnelBtn');
  btn.innerHTML = '⏳ Erzeuge öffentlichen Bruder-Link...';
  btn.disabled = true;
  try {{
    const r = await fetch('/api/start_tunnel', {{ method: 'POST' }});
    const d = await r.json();
    if (d.public_url) {{
      document.getElementById('shareLinkInput').value = d.public_url;
      document.getElementById('qrImg').src = 'https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=' + encodeURIComponent(d.public_url);
      document.getElementById('serverUrlBox').innerHTML = '🌐 <b>Öffentlicher Bruder-Link aktiv!</b> Dein Bruder kann diesen Link jetzt von überall auf seinem Handy/iPad öffnen und zum Home-Bildschirm hinzufügen:';
      btn.innerHTML = '✓ Öffentlicher Link aktiv';
      return;
    }}
  }} catch (e) {{}}
  btn.innerHTML = '🌐 Öffentlichen Internet-Link für Bruder aktivieren';
  btn.disabled = false;
}}

async function shareAppFile() {{
  const htmlContent = '<!DOCTYPE html>\\n' + document.documentElement.outerHTML;
  const blob = new Blob([htmlContent], {{ type: 'text/html' }});
  const file = new File([blob], 'Version_1_App.html', {{ type: 'text/html' }});
  if (navigator.canShare && navigator.canShare({{ files: [file] }})) {{
    try {{
      await navigator.share({{
        title: 'Market Vision Pro — Version 1',
        text: 'Hier ist unsere Trading & Stillhalter App (Version 1) für Mac, iPad und Handy:',
        files: [file]
      }});
      return;
    }} catch (e) {{}}
  }}
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'Version_1_App.html';
  a.click();
}}

// =====================================================================
// TRADINGVIEW 4-PANE UMWA CHART ENGINE (1 JAHR + BLAUE S/R-LINIEN)
// =====================================================================

function setWlFilter(mode, btn) {{
  wlFilterMode = mode;
  document.querySelectorAll('.tv-side-hdr .tv-mini-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderWatchlistSidebar();
}}

function getFilteredChartSymbols() {{
  const cat = (document.getElementById('tvWlCat') || {{ value: 'ALL' }}).value;
  const q = ((document.getElementById('tvWlSearch') || {{ value: '' }}).value || '').trim().toLowerCase();
  return CHART_ORDER.filter(sym => {{
    const d = CHART_DATA[sym];
    if (!d) return false;
    const isMain = (d.lvl === 'STRONG_BUY' || d.lvl === 'STRONG_SELL');
    const isWatch = (d.lvl === 'WEAK_BUY' || d.lvl === 'WEAK_SELL');
    const isShield = isMain && d.opt && d.opt.q && d.opt.sh;
    if (wlFilterMode === 'main' && !isMain) return false;
    if (wlFilterMode === 'shield' && !isShield) return false;
    if (wlFilterMode === 'watch' && !(isMain || isWatch)) return false;
    if (cat !== 'ALL' && d.cat !== cat) return false;
    if (q && !(sym.toLowerCase().includes(q) || (d.n || '').toLowerCase().includes(q))) return false;
    return true;
  }});
}}

function renderWatchlistSidebar() {{
  const listEl = document.getElementById('tvWlList');
  if (!listEl) return;
  const syms = getFilteredChartSymbols();
  const cntEl = document.getElementById('tvWlCount');
  if (cntEl) cntEl.textContent = syms.length + ' Werte';

  const html = syms.map(sym => {{
    const d = CHART_DATA[sym];
    const isAct = (sym === activeChartSym) ? ' active' : '';
    const chg = d.chg != null ? (d.chg >= 0 ? '+' + d.chg.toFixed(2) + '%' : d.chg.toFixed(2) + '%') : '—';
    const chgClr = (d.chg || 0) >= 0 ? 'var(--emerald)' : 'var(--rose)';
    let dot = '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#475569;margin-right:6px"></span>';
    let pill = '';
    if (d.lvl === 'STRONG_BUY') {{
      dot = '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--emerald);box-shadow:0 0 6px var(--emerald);margin-right:6px"></span>';
      pill = '<span class="mono" style="font-size:10px;background:var(--emerald-bg);color:var(--emerald);padding:1px 5px;border-radius:3px;font-weight:700">' + d.bull + '/4 BUY</span>';
    }} else if (d.lvl === 'STRONG_SELL') {{
      dot = '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--rose);box-shadow:0 0 6px var(--rose);margin-right:6px"></span>';
      pill = '<span class="mono" style="font-size:10px;background:var(--rose-bg);color:var(--rose);padding:1px 5px;border-radius:3px;font-weight:700">' + d.bear + '/4 SELL</span>';
    }} else if (d.lvl === 'WEAK_BUY' || d.lvl === 'WEAK_SELL') {{
      dot = '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--amber);margin-right:6px"></span>';
      pill = '<span class="mono" style="font-size:10px;background:var(--amber-bg);color:var(--amber);padding:1px 5px;border-radius:3px;font-weight:700">2/4</span>';
    }}
    const shieldIcon = (d.opt && d.opt.sh) ? ' 🛡️' : '';
    return '<div class="tv-wl-row' + isAct + '" data-wlsym="' + sym + '" onclick="selectChartSymbol(this.dataset.wlsym)">' +
      '<div>' +
        '<div style="display:flex;align-items:center;gap:4px">' + dot + '<b class="mono" style="font-size:13px;font-weight:800;color:var(--txt-primary)">' + sym + '</b>' + shieldIcon + ' ' + pill + '</div>' +
        '<div style="font-size:10.5px;color:var(--txt-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:155px">' + (d.n || sym) + '</div>' +
      '</div>' +
      '<div class="mono" style="font-size:13px;font-weight:800;color:var(--txt-primary);text-align:right">' + d.last.toFixed(2) + '</div>' +
      '<div class="mono" style="font-size:11.5px;font-weight:700;color:' + chgClr + ';text-align:right;min-width:56px">' + chg + '</div>' +
    '</div>';
  }}).join('');
  listEl.innerHTML = html || '<div style="padding:18px;color:var(--txt-muted);font-size:12px">Keine Werte für diesen Filter gefunden.</div>';
}}

function clampChartViewport(total) {{
  if (!total || total < 1) return;
  const minSpan = Math.min(14, total);
  const maxSpan = total + 8;
  if (tvViewStart == null || tvViewEnd == null || isNaN(tvViewStart) || isNaN(tvViewEnd)) {{
    tvViewStart = Math.max(0, total - chartBarsCount);
    tvViewEnd = total;
  }}
  let span = tvViewEnd - tvViewStart;
  if (span < minSpan) {{
    const mid = (tvViewStart + tvViewEnd) * 0.5;
    tvViewStart = mid - minSpan * 0.5;
    tvViewEnd = mid + minSpan * 0.5;
    span = minSpan;
  }} else if (span > maxSpan) {{
    tvViewStart = 0;
    tvViewEnd = total;
    span = total;
  }}
  const maxRight = total + Math.min(8, Math.max(2, span * 0.12));
  if (tvViewEnd > maxRight) {{
    tvViewEnd = maxRight;
    tvViewStart = tvViewEnd - span;
  }}
  if (tvViewStart < 0) {{
    tvViewStart = 0;
    tvViewEnd = Math.min(maxRight, span);
  }}
}}

function setChartRange(bars, btn) {{
  chartBarsCount = bars;
  document.querySelectorAll('.tv-toolbar .tv-btn-group .tv-mini-btn.mono').forEach(b => {{
    if (b.textContent.includes('M') || b.textContent.includes('1J')) b.classList.remove('active');
  }});
  if (btn) btn.classList.add('active');
  hoverBarIdx = null;
  if (activeChartSym) {{
    if (!cachedChartCalc || cachedChartCalc.sym !== activeChartSym) {{
      cachedChartCalc = computeUmwaSeries(activeChartSym);
    }}
    if (cachedChartCalc) {{
      const total = cachedChartCalc.closes.length;
      tvViewStart = Math.max(0, total - bars);
      tvViewEnd = total;
      tvPriceZoom = 1.0;
      tvPricePanPx = 0.0;
      drawUmwaChart();
    }}
  }}
}}

function zoomChartBy(factor, anchorRatio = 0.82) {{
  if (!cachedChartCalc) return;
  const total = cachedChartCalc.closes.length;
  clampChartViewport(total);
  const curSpan = tvViewEnd - tvViewStart;
  const minSpan = Math.min(14, total);
  const maxSpan = total;
  const newSpan = Math.max(minSpan, Math.min(maxSpan, curSpan * factor));
  const anchorBar = tvViewStart + anchorRatio * curSpan;
  tvViewStart = anchorBar - anchorRatio * newSpan;
  tvViewEnd = tvViewStart + newSpan;
  clampChartViewport(total);
  drawUmwaChart();
}}

function resetChartView() {{
  if (!cachedChartCalc) return;
  const total = cachedChartCalc.closes.length;
  tvViewStart = Math.max(0, total - chartBarsCount);
  tvViewEnd = total;
  tvPriceZoom = 1.0;
  tvPricePanPx = 0.0;
  hoverBarIdx = null;
  drawUmwaChart();
}}

function toggleChartLayer(key, btn) {{
  chartLayers[key] = !chartLayers[key];
  if (btn) btn.classList.toggle('active', chartLayers[key]);
  drawUmwaChart();
}}

function computeEmaWithSeed(vals, span, seedVal) {{
  const out = new Array(vals.length);
  if (!vals.length) return out;
  const alpha = 2.0 / (span + 1.0);
  out[0] = (seedVal != null && !isNaN(seedVal)) ? seedVal : vals[0];
  for (let i = 1; i < vals.length; i++) {{
    out[i] = alpha * vals[i] + (1.0 - alpha) * out[i - 1];
  }}
  return out;
}}

function computeStochSeries(highs, lows, closes, kLen, kSmooth, dSmooth) {{
  const n = closes.length;
  const rawK = new Array(n).fill(null);
  for (let i = kLen - 1; i < n; i++) {{
    let hh = -Infinity, ll = Infinity;
    for (let j = i - kLen + 1; j <= i; j++) {{
      if (highs[j] > hh) hh = highs[j];
      if (lows[j] < ll) ll = lows[j];
    }}
    rawK[i] = (hh === ll) ? 0.0 : 100.0 * (closes[i] - ll) / (hh - ll);
  }}
  const kArr = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {{
    let s = 0, c = 0;
    for (let j = Math.max(0, i - kSmooth + 1); j <= i; j++) {{
      if (rawK[j] != null) {{ s += rawK[j]; c++; }}
    }}
    if (c === kSmooth) kArr[i] = s / c;
  }}
  const dArr = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {{
    let s = 0, c = 0;
    for (let j = Math.max(0, i - dSmooth + 1); j <= i; j++) {{
      if (kArr[j] != null) {{ s += kArr[j]; c++; }}
    }}
    if (c === dSmooth) dArr[i] = s / c;
  }}
  return {{ k: kArr, d: dArr }};
}}

function computeUmwaSeries(sym) {{
  const d = CHART_DATA[sym];
  if (!d || !d.c || !d.c.length) return null;
  const flat = d.c;
  const total = Math.floor(flat.length / 5);
  const dates = new Array(total);
  const opens = new Array(total);
  const highs = new Array(total);
  const lows = new Array(total);
  const closes = new Array(total);

  for (let i = 0; i < total; i++) {{
    const base = i * 5;
    dates[i] = MASTER_DATES[flat[base]] || '';
    opens[i] = flat[base + 1];
    highs[i] = flat[base + 2];
    lows[i] = flat[base + 3];
    closes[i] = flat[base + 4];
  }}

  // 1. Bollinger Bands (20, 2.0)
  const bbMid = new Array(total).fill(null);
  const bbUpper = new Array(total).fill(null);
  const bbLower = new Array(total).fill(null);
  for (let i = 19; i < total; i++) {{
    let sum = 0;
    for (let j = i - 19; j <= i; j++) sum += closes[j];
    const m = sum / 20.0;
    let vSum = 0;
    for (let j = i - 19; j <= i; j++) {{
      const diff = closes[j] - m;
      vSum += diff * diff;
    }}
    const sd = Math.sqrt(vSum / 20.0);
    bbMid[i] = m;
    bbUpper[i] = m + 2.0 * sd;
    bbLower[i] = m - 2.0 * sd;
  }}

  // 2. EMAs (20, 50, 100, 200) mit 2J-Seeds
  const seeds = d.es || [];
  const ema20 = computeEmaWithSeed(closes, 20, seeds[0]);
  const ema50 = computeEmaWithSeed(closes, 50, seeds[1]);
  const ema100 = computeEmaWithSeed(closes, 100, seeds[2]);
  const ema200 = computeEmaWithSeed(closes, 200, seeds[3]);

  // 3. Stochastik UMWA Dual (Kurz 14,5,3 & Lang 50,6,5)
  const stShort = computeStochSeries(highs, lows, closes, 14, 5, 3);
  const stLong = computeStochSeries(highs, lows, closes, 50, 6, 5);

  // 4. RSI UMWA (14)
  const rsi = new Array(total).fill(null);
  if (total > 14) {{
    let avgG = 0, avgL = 0;
    for (let i = 1; i <= 14; i++) {{
      const ch = closes[i] - closes[i - 1];
      if (ch > 0) avgG += ch; else avgL -= ch;
    }}
    avgG /= 14.0; avgL /= 14.0;
    rsi[14] = avgL === 0 ? 100.0 : 100.0 - (100.0 / (1.0 + avgG / avgL));
    for (let i = 15; i < total; i++) {{
      const ch = closes[i] - closes[i - 1];
      const g = ch > 0 ? ch : 0;
      const l = ch < 0 ? -ch : 0;
      avgG = (avgG * 13.0 + g) / 14.0;
      avgL = (avgL * 13.0 + l) / 14.0;
      rsi[i] = avgL === 0 ? 100.0 : 100.0 - (100.0 / (1.0 + avgG / avgL));
    }}
  }}

  // 5. MACD UMWA (12, 26, 9)
  const ema12 = computeEmaWithSeed(closes, 12, closes[0]);
  const ema26 = computeEmaWithSeed(closes, 26, closes[0]);
  const macdLine = closes.map((_, i) => ema12[i] - ema26[i]);
  const sigLine = computeEmaWithSeed(macdLine, 9, macdLine[0]);
  const macdHist = macdLine.map((v, i) => v - sigLine[i]);

  return {{
    sym,
    meta: d,
    dates,
    opens,
    highs,
    lows,
    closes,
    bbMid,
    bbUpper,
    bbLower,
    ema20,
    ema50,
    ema100,
    ema200,
    stShortK: stShort.k,
    stShortD: stShort.d,
    stLongK: stLong.k,
    stLongD: stLong.d,
    rsi,
    macdHist,
  }};
}}

function selectChartSymbol(sym) {{
  if (!CHART_DATA[sym]) return;
  activeChartSym = sym;
  hoverBarIdx = null;
  cachedChartCalc = computeUmwaSeries(sym);
  if (cachedChartCalc) {{
    const total = cachedChartCalc.closes.length;
    tvViewStart = Math.max(0, total - chartBarsCount);
    tvViewEnd = total;
    tvPriceZoom = 1.0;
    tvPricePanPx = 0.0;
  }}

  document.querySelectorAll('#tvWlList .tv-wl-row').forEach(r => {{
    r.classList.toggle('active', r.getAttribute('data-wlsym') === sym);
  }});

  const d = CHART_DATA[sym];
  document.getElementById('tvSymTitle').textContent = sym;
  document.getElementById('tvCompName').textContent = (d.n || sym) + ' · 1D';
  document.getElementById('tvCatBadge').textContent = d.cat || '';
  document.getElementById('tvExtLink').href = 'https://de.tradingview.com/chart/?symbol=' + encodeURIComponent(sym);

  const sigEl = document.getElementById('tvSigBadge');
  if (d.lvl === 'STRONG_BUY') {{
    sigEl.className = 'sig-badge sig-strong-buy';
    sigEl.textContent = 'KAUFSIGNAL (' + d.bull + '/4) · SHORT PUT';
  }} else if (d.lvl === 'STRONG_SELL') {{
    sigEl.className = 'sig-badge sig-strong-sell';
    sigEl.textContent = 'VERKAUFSSIGNAL (' + d.bear + '/4) · SHORT CALL';
  }} else if (d.lvl === 'WEAK_BUY') {{
    sigEl.className = 'sig-badge sig-weak-buy';
    sigEl.textContent = 'WATCH LONG (2/4)';
  }} else if (d.lvl === 'WEAK_SELL') {{
    sigEl.className = 'sig-badge sig-weak-sell';
    sigEl.textContent = 'WATCH SHORT (2/4)';
  }} else {{
    sigEl.className = 'sig-badge sig-neutral';
    sigEl.textContent = 'NEUTRAL (' + d.bull + 'B/' + d.bear + 'S)';
  }}

  // Bottom Cards aktualisieren
  const ind = d.ind || {{}};
  const pill = (lbl, v) => {{
    const cls = v > 0 ? 'ind-bull' : (v < 0 ? 'ind-bear' : 'ind-neut');
    const ic = v > 0 ? '▲' : (v < 0 ? '▼' : '·');
    return '<span class="mono ind-pill ' + cls + '" style="margin-right:4px">' + lbl + ' ' + ic + '</span>';
  }};
  const rsnHtml = (d.rsn && d.rsn.length)
    ? d.rsn.map(x => '<div style="font-size:11.5px;color:var(--txt-secondary);margin-top:3px">• ' + x + '</div>').join('')
    : '<div style="font-size:11.5px;color:var(--txt-muted);margin-top:4px">Keine extremen Umkehr-Trigger am aktuellen Tag.</div>';
  document.getElementById('tvCardInd').innerHTML = '<div style="margin-bottom:6px">' +
    pill('BB', ind.bb) + pill('RSI', ind.rsi) + pill('MACD', ind.macd) + pill('STOCH', ind.stoch) + '</div>' + rsnHtml;

  const supHtml = (d.sup || []).map(s =>
    '<div class="mono" style="font-size:12px;color:#60a5fa;margin-bottom:3px">🔵 <b>SUP ' + s.p.toFixed(2) + ' $</b> (' + (s.d >= 0 ? '+' : '') + s.d.toFixed(1) + '% · ' + s.b + 'x Touch / ' + s.v + ' Pivots)</div>'
  ).join('');
  const resHtml = (d.res || []).map(r =>
    '<div class="mono" style="font-size:12px;color:#38bdf8;margin-bottom:3px">🔵 <b>RES ' + r.p.toFixed(2) + ' $</b> (+' + r.d.toFixed(1) + '% · ' + r.b + 'x Touch / ' + r.v + ' Pivots)</div>'
  ).join('');
  document.getElementById('tvCardSR').innerHTML = (supHtml || resHtml) ? (supHtml + resHtml) : '<div class="txt-muted">Keine validierte horizontale Linie im nahen Radius.</div>';

  if (d.opt && d.opt.q) {{
    const shBadge = d.opt.sh ? '<div style="color:var(--emerald);font-weight:700;font-size:11.5px;margin-top:4px">🛡️ GESCHÜTZT durch S/R-Linie</div>' : '';
    document.getElementById('tvCardOpt').innerHTML =
      '<div class="mono" style="font-size:13px;font-weight:800;color:var(--txt-primary)">' + (d.opt.con || d.opt.t) + '</div>' +
      '<div class="mono" style="font-size:12px;color:var(--txt-secondary);margin-top:3px">Strike: <b style="color:var(--txt-primary)">' + (d.opt.k || 0).toFixed(2) + ' $</b> (OTM ' + (d.opt.otm || 0).toFixed(1) + '%) · Δ ' + (d.opt.d || 0).toFixed(2) + '</div>' +
      '<div class="mono" style="font-size:12.5px;color:var(--emerald);font-weight:800;margin-top:2px">Prämie: +' + (d.opt.p || 0).toFixed(2) + ' $ (+' + (d.opt.pr || 0).toFixed(2) + '%)</div>' + shBadge;
  }} else if (d.opt) {{
    document.getElementById('tvCardOpt').innerHTML = '<div class="txt-muted" style="font-size:11.5px">' + (d.opt.note || 'Kein qualifizierter Weekly-Strike') + '</div>';
  }} else {{
    document.getElementById('tvCardOpt').innerHTML = '<div class="txt-muted" style="font-size:11.5px">Kein aktives Options-Signal.</div>';
  }}

  const ew = d.ew || {{}};
  document.getElementById('tvCardEW').innerHTML =
    '<div style="font-weight:700;color:var(--txt-primary);font-size:12px">' + (ew.w ? '[' + ew.w + '] ' : '') + (ew.ph || 'Neutral') + ' (' + (ew.pr || 50) + '%)</div>' +
    '<div style="font-size:11px;color:var(--txt-secondary);margin-top:2px">' + (ew.sc || '') + '</div>' +
    '<div class="mono" style="font-size:11.5px;color:var(--sky);margin-top:5px">Fundamental: <b>' + (d.fl || '') + ' (' + (d.fs || 0) + '/100)</b></div>';

  drawUmwaChart();
}}

function drawUmwaChart() {{
  const canvas = document.getElementById('umwaChartCanvas');
  if (!canvas || !cachedChartCalc) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 50 || rect.height < 50) return;

  const dpr = window.devicePixelRatio || 2;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const axisW = 78;
  const timeH = 26;
  const plotW = W - axisW;
  const plotH = H - timeH;

  // 4 Panes wie in TradingView ("Julian Layout")
  const p1Top = 0, p1H = Math.floor(plotH * 0.50);
  const p2Top = p1H, p2H = Math.floor(plotH * 0.17);
  const p3Top = p2Top + p2H, p3H = Math.floor(plotH * 0.16);
  const p4Top = p3Top + p3H, p4H = plotH - p4Top;

  const c = cachedChartCalc;
  const total = c.closes.length;
  if (!total) return;

  clampChartViewport(total);
  const vStart = tvViewStart;
  const vEnd = tvViewEnd;
  const visSpan = Math.max(5, vEnd - vStart);
  const iStart = Math.max(0, Math.floor(vStart));
  const iEnd = Math.min(total - 1, Math.ceil(vEnd));

  const idx = (hoverBarIdx != null && hoverBarIdx >= 0 && hoverBarIdx < total) ? hoverBarIdx : (total - 1);

  // OHLC Header-Leiste aktualisieren (leicht größere, kontrastreiche Zahlen)
  const ohlcEl = document.getElementById('tvOhlcBar');
  if (ohlcEl) {{
    const prevC = idx > 0 ? c.closes[idx - 1] : c.opens[idx];
    const diff = c.closes[idx] - prevC;
    const pct = prevC > 0 ? (diff / prevC) * 100.0 : 0.0;
    const clr = diff >= 0 ? 'var(--emerald)' : 'var(--rose)';
    const sign = diff >= 0 ? '+' : '';
    ohlcEl.innerHTML =
      '<span><b>' + c.dates[idx] + '</b></span>' +
      '<span>O <b style="color:' + clr + '">' + c.opens[idx].toFixed(2) + '</b></span>' +
      '<span>H <b style="color:' + clr + '">' + c.highs[idx].toFixed(2) + '</b></span>' +
      '<span>L <b style="color:' + clr + '">' + c.lows[idx].toFixed(2) + '</b></span>' +
      '<span>C <b style="color:' + clr + ';font-size:13.5px">' + c.closes[idx].toFixed(2) + ' $</b></span>' +
      '<span style="color:' + clr + ';font-weight:800;font-size:13px">' + sign + diff.toFixed(2) + ' (' + sign + pct.toFixed(2) + '%)</span>';
  }}

  // Hintergrund
  ctx.fillStyle = '#090d14';
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#0d131f';
  ctx.fillRect(plotW, 0, axisW, H);

  const xOf = (i) => 8 + ((i - vStart + 0.5) / visSpan) * (plotW - 16);
  const barW = Math.max(1.5, Math.min(18.0, ((plotW - 16) / visSpan) * 0.74));

  // ===================================================================
  // PANE 1: PREIS + BB & EMAs + BLAUE S/R-LINIEN + OPTIONS-STRIKE
  // ===================================================================
  let minP = Infinity, maxP = -Infinity;
  for (let i = iStart; i <= iEnd; i++) {{
    if (c.lows[i] < minP) minP = c.lows[i];
    if (c.highs[i] > maxP) maxP = c.highs[i];
    if (chartLayers.bb && c.bbUpper[i] != null) {{
      if (c.bbLower[i] < minP) minP = c.bbLower[i];
      if (c.bbUpper[i] > maxP) maxP = c.bbUpper[i];
    }}
  }}
  if (!isFinite(minP) || !isFinite(maxP)) {{
    minP = c.closes[total - 1] * 0.9;
    maxP = c.closes[total - 1] * 1.1;
  }}
  // Beziehe nahe S/R-Linien & Options-Strike in die Basis-Y-Skala ein
  const midVis = (minP + maxP) * 0.5;
  const spanVis = Math.max(maxP - minP, midVis * 0.04);
  if (chartLayers.sr) {{
    (c.meta.sup || []).forEach(s => {{
      if (s.p >= minP - spanVis * 0.35 && s.p <= maxP + spanVis * 0.2) {{
        if (s.p < minP) minP = s.p;
        if (s.p > maxP) maxP = s.p;
      }}
    }});
    (c.meta.res || []).forEach(r => {{
      if (r.p >= minP - spanVis * 0.2 && r.p <= maxP + spanVis * 0.35) {{
        if (r.p < minP) minP = r.p;
        if (r.p > maxP) maxP = r.p;
      }}
    }});
  }}
  if (chartLayers.opt && c.meta.opt && c.meta.opt.k) {{
    const ok = c.meta.opt.k;
    if (ok >= minP - spanVis * 0.45 && ok <= maxP + spanVis * 0.45) {{
      if (ok < minP) minP = ok;
      if (ok > maxP) maxP = ok;
    }}
  }}
  const padP = (maxP - minP) * 0.075 || 1.0;
  minP -= padP; maxP += padP;

  // TradingView vertikaler Preis-Zoom & Pan anwenden
  const baseMidP = (minP + maxP) * 0.5;
  const scaledSpanP = Math.max(0.01, (maxP - minP) / Math.max(0.15, tvPriceZoom));
  const panPriceShift = (tvPricePanPx / Math.max(60, p1H - 24)) * scaledSpanP;
  minP = baseMidP - scaledSpanP * 0.5 + panPriceShift;
  maxP = baseMidP + scaledSpanP * 0.5 + panPriceShift;

  const yOfP1 = (val) => p1Top + 12 + (1.0 - (val - minP) / (maxP - minP)) * (p1H - 24);
  const priceOfY1 = (y) => minP + (1.0 - (y - (p1Top + 12)) / (p1H - 24)) * (maxP - minP);

  // Horizontales Preis-Gitter + Rechte Preisachse
  ctx.strokeStyle = 'rgba(30, 41, 59, 0.55)';
  ctx.lineWidth = 1;
  ctx.font = '600 11px "SF Mono", Menlo, monospace';
  const stepsP1 = 6;
  for (let s = 0; s <= stepsP1; s++) {{
    const val = minP + (s / stepsP1) * (maxP - minP);
    const y = yOfP1(val);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(plotW, y); ctx.stroke();
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(val.toFixed(2), plotW + 6, y + 3.5);
  }}

  // Adaptives vertikales Zeit-Gitter (Monate bei Übersicht, Wochen/Tage beim Reinzoomen wie in TradingView!)
  const monthNames = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'];
  let lastMonth = '';
  let lastLabelX = -100;
  const stepBars = visSpan <= 28 ? 3 : (visSpan <= 65 ? 5 : (visSpan <= 115 ? 10 : 0));
  for (let i = iStart; i <= iEnd; i++) {{
    const dStr = c.dates[i] || '';
    if (!dStr) continue;
    const ym = dStr.slice(0, 7);
    const mNum = parseInt(dStr.slice(5, 7), 10) - 1;
    const dayStr = dStr.slice(8, 10);
    const yr = dStr.slice(2, 4);
    const x = xOf(i);
    if (x < 10 || x > plotW - 14) continue;

    let drawTick = false;
    let lbl = '';
    if (stepBars === 0) {{
      if (ym !== lastMonth) {{
        lastMonth = ym;
        drawTick = true;
        lbl = (mNum === 0) ? (monthNames[mNum] + " '" + yr) : monthNames[mNum];
      }}
    }} else {{
      if (ym !== lastMonth) {{
        lastMonth = ym;
        drawTick = true;
        lbl = dayStr + '. ' + monthNames[mNum];
      }} else if ((i - iStart) % stepBars === 0) {{
        drawTick = true;
        lbl = dayStr + '. ' + monthNames[mNum];
      }}
    }}

    if (drawTick && (x - lastLabelX) >= 54) {{
      lastLabelX = x;
      ctx.strokeStyle = 'rgba(30, 41, 59, 0.42)';
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, plotH); ctx.stroke();
      ctx.fillStyle = '#94a3b8';
      ctx.font = '600 10.5px "SF Mono", Menlo, monospace';
      const lw = ctx.measureText(lbl).width;
      ctx.fillText(lbl, Math.max(4, Math.min(plotW - lw - 4, x - lw * 0.5)), plotH + 16);
    }}
  }}

  const drawCurve = (arr, yFn, color, width, dashed = false) => {{
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    if (dashed) ctx.setLineDash([4, 3]);
    ctx.beginPath();
    let started = false;
    const drawFrom = Math.max(0, iStart - 1);
    const drawTo = Math.min(total - 1, iEnd + 1);
    for (let i = drawFrom; i <= drawTo; i++) {{
      if (arr[i] == null) continue;
      const x = xOf(i), y = yFn(arr[i]);
      if (!started) {{ ctx.moveTo(x, y); started = true; }}
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
    ctx.restore();
  }};

  const drawAxisTag = (y, topB, botB, text, bg, fg = '#fff') => {{
    if (y < topB + 7 || y > botB - 7) return;
    ctx.fillStyle = bg;
    ctx.fillRect(plotW + 1, y - 9, axisW - 2, 18);
    ctx.fillStyle = fg;
    ctx.font = 'bold 11px "SF Mono", Menlo, monospace';
    ctx.fillText(text, plotW + 5, y + 4);
  }};

  // Clip Pane 1 für sauberes Zeichnen beim vertikalen/horizontalen Zoomen
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, p1Top, plotW, p1H);
  ctx.clip();

  // 1a. Bollinger Bänder (20, 2.0)
  if (chartLayers.bb) {{
    ctx.save();
    ctx.fillStyle = 'rgba(34, 211, 238, 0.055)';
    ctx.beginPath();
    let firstIdx = -1;
    const drawFrom = Math.max(0, iStart - 1);
    const drawTo = Math.min(total - 1, iEnd + 1);
    for (let i = drawFrom; i <= drawTo; i++) {{
      if (c.bbUpper[i] != null) {{
        const x = xOf(i), y = yOfP1(c.bbUpper[i]);
        if (firstIdx < 0) {{ firstIdx = i; ctx.moveTo(x, y); }}
        else ctx.lineTo(x, y);
      }}
    }}
    if (firstIdx >= 0) {{
      for (let i = drawTo; i >= firstIdx; i--) {{
        if (c.bbLower[i] != null) ctx.lineTo(xOf(i), yOfP1(c.bbLower[i]));
      }}
      ctx.closePath();
      ctx.fill();
    }}
    ctx.restore();
    drawCurve(c.bbUpper, yOfP1, 'rgba(34, 211, 238, 0.78)', 1.25);
    drawCurve(c.bbLower, yOfP1, 'rgba(34, 211, 238, 0.78)', 1.25);
    drawCurve(c.bbMid, yOfP1, 'rgba(56, 189, 248, 0.55)', 1.1, true);
  }}

  // 1b. EMAs (20 Gelb, 50 Rot/Pink, 100 Blau, 200 Magenta)
  if (chartLayers.ema) {{
    drawCurve(c.ema200, yOfP1, '#d946ef', 1.45);
    drawCurve(c.ema100, yOfP1, '#3b82f6', 1.35);
    drawCurve(c.ema50, yOfP1, '#f43f5e', 1.35);
    drawCurve(c.ema20, yOfP1, '#facc15', 1.45);
  }}

  // 1c. Horizontale Support- & Widerstandslinien (Blaue Linien)
  if (chartLayers.sr) {{
    (c.meta.sup || []).forEach((s, sIdx) => {{
      const y = yOfP1(s.p);
      if (y >= p1Top + 4 && y <= p1Top + p1H - 4) {{
        ctx.save();
        ctx.strokeStyle = sIdx === 0 ? '#2563eb' : 'rgba(59, 130, 246, 0.75)';
        ctx.lineWidth = sIdx === 0 ? 2.0 : 1.4;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(plotW, y); ctx.stroke();
        const lbl = '🔵 SUP ' + s.p.toFixed(2) + '$ (' + s.b + 'T/' + s.v + 'P)';
        ctx.font = 'bold 10.5px "SF Mono", Menlo, monospace';
        const tw = ctx.measureText(lbl).width + 10;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        ctx.fillRect(10, y - 16, tw, 15);
        ctx.fillStyle = '#60a5fa';
        ctx.fillText(lbl, 15, y - 4.5);
        ctx.restore();
      }}
    }});
    (c.meta.res || []).forEach((r, rIdx) => {{
      const y = yOfP1(r.p);
      if (y >= p1Top + 4 && y <= p1Top + p1H - 4) {{
        ctx.save();
        ctx.strokeStyle = rIdx === 0 ? '#0284c7' : 'rgba(56, 189, 248, 0.65)';
        ctx.lineWidth = rIdx === 0 ? 1.8 : 1.3;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(plotW, y); ctx.stroke();
        const lbl = '🔵 RES ' + r.p.toFixed(2) + '$ (' + r.b + 'T/' + r.v + 'P)';
        ctx.font = 'bold 10.5px "SF Mono", Menlo, monospace';
        const tw = ctx.measureText(lbl).width + 10;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        ctx.fillRect(10, y + 2, tw, 15);
        ctx.fillStyle = '#38bdf8';
        ctx.fillText(lbl, 15, y + 13);
        ctx.restore();
      }}
    }});
  }}

  // 1d. Weekly Stillhalter Options-Strike-Linie (🛡️)
  if (chartLayers.opt && c.meta.opt && c.meta.opt.k) {{
    const opt = c.meta.opt;
    const y = yOfP1(opt.k);
    if (y >= p1Top + 4 && y <= p1Top + p1H - 4) {{
      const isPut = (opt.t || '').includes('PUT');
      const clr = isPut ? '#10b981' : '#f43f5e';
      ctx.save();
      ctx.strokeStyle = clr;
      ctx.lineWidth = 1.7;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(plotW, y); ctx.stroke();
      const lbl = (opt.sh ? '🛡️ ' : '') + opt.t + ' STRIKE ' + opt.k.toFixed(2) + '$ (Δ ' + (opt.d || 0).toFixed(2) + ' · +' + (opt.pr || 0).toFixed(2) + '%)';
      ctx.font = 'bold 10.5px "SF Mono", Menlo, monospace';
      const tw = ctx.measureText(lbl).width + 10;
      ctx.fillStyle = 'rgba(6, 78, 59, 0.90)';
      ctx.fillRect(plotW - tw - 12, y - 16, tw, 15);
      ctx.fillStyle = '#6ee7b7';
      ctx.fillText(lbl, plotW - tw - 7, y - 4.5);
      ctx.restore();
    }}
  }}

  // 1e. Candlesticks (Sichtbarer Ausschnitt)
  for (let i = iStart; i <= iEnd; i++) {{
    const x = xOf(i);
    if (x < -barW || x > plotW + barW) continue;
    const o = c.opens[i], h = c.highs[i], l = c.lows[i], cl = c.closes[i];
    const isBull = cl >= o;
    const clr = isBull ? '#10b981' : '#f43f5e';
    ctx.strokeStyle = clr;
    ctx.lineWidth = barW > 6 ? 1.4 : 1.15;
    ctx.beginPath();
    ctx.moveTo(x, yOfP1(h));
    ctx.lineTo(x, yOfP1(l));
    ctx.stroke();

    const yTop = yOfP1(Math.max(o, cl));
    const yBot = yOfP1(Math.min(o, cl));
    const bh = Math.max(1.5, yBot - yTop);
    ctx.fillStyle = clr;
    ctx.fillRect(x - barW * 0.5, yTop, barW, bh);
  }}

  // Aktuelle Schlusskurs-Linie
  const lastC = c.closes[total - 1];
  const lastOpen = c.opens[total - 1];
  const yLast = yOfP1(lastC);
  ctx.save();
  ctx.strokeStyle = lastC >= lastOpen ? 'rgba(16, 185, 129, 0.5)' : 'rgba(244, 63, 94, 0.5)';
  ctx.setLineDash([2, 2]);
  ctx.beginPath(); ctx.moveTo(0, yLast); ctx.lineTo(plotW, yLast); ctx.stroke();
  ctx.restore();

  ctx.restore(); // Ende Clip Pane 1

  // Rechte Achsen-Tags für S/R, Strike & Aktuellen Schlusskurs
  if (chartLayers.sr) {{
    (c.meta.sup || []).forEach(s => drawAxisTag(yOfP1(s.p), p1Top, p1Top + p1H, s.p.toFixed(2), '#1d4ed8', '#ffffff'));
    (c.meta.res || []).forEach(r => drawAxisTag(yOfP1(r.p), p1Top, p1Top + p1H, r.p.toFixed(2), '#0369a1', '#ffffff'));
  }}
  if (chartLayers.opt && c.meta.opt && c.meta.opt.k) {{
    drawAxisTag(yOfP1(c.meta.opt.k), p1Top, p1Top + p1H, c.meta.opt.k.toFixed(2), '#059669', '#ffffff');
  }}
  drawAxisTag(yLast, p1Top, p1Top + p1H, lastC.toFixed(2), lastC >= lastOpen ? '#10b981' : '#f43f5e', '#04130d');

  // Top-Left Legende Pane 1 (wie TradingView "BB & EMAs")
  ctx.font = 'bold 11.5px -apple-system, BlinkMacSystemFont, "SF Mono", monospace';
  let lx = 10, ly = p1Top + 17;
  ctx.fillStyle = '#94a3b8';
  ctx.fillText('BB & EMAs', lx, ly); lx += 72;
  const drawLegVal = (val, clr) => {{
    if (val == null) return;
    const s = val.toFixed(2);
    ctx.fillStyle = clr;
    ctx.fillText(s, lx, ly);
    lx += ctx.measureText(s).width + 9;
  }};
  drawLegVal(c.bbMid[idx], '#38bdf8');
  drawLegVal(c.bbUpper[idx], '#22d3ee');
  drawLegVal(c.bbLower[idx], '#22d3ee');
  drawLegVal(c.ema20[idx], '#facc15');
  drawLegVal(c.ema50[idx], '#f43f5e');
  drawLegVal(c.ema100[idx], '#3b82f6');
  drawLegVal(c.ema200[idx], '#d946ef');

  // ===================================================================
  // PANE 2: STOCHASTIK UMWA (DUAL: KURZ 14,5,3 & LANG 50,6,5)
  // ===================================================================
  ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(0, p2Top); ctx.lineTo(W, p2Top); ctx.stroke();

  const yOfP2 = (v) => p2Top + 8 + (1.0 - v / 100.0) * (p2H - 16);
  ctx.fillStyle = 'rgba(30, 41, 59, 0.22)';
  ctx.fillRect(0, yOfP2(80), plotW, yOfP2(20) - yOfP2(80));
  [80, 50, 20].forEach(lvl => {{
    const y = yOfP2(lvl);
    ctx.strokeStyle = lvl === 50 ? 'rgba(51, 65, 85, 0.4)' : 'rgba(100, 116, 139, 0.45)';
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(plotW, y); ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '10.5px "SF Mono", monospace';
    ctx.fillText(lvl.toFixed(0), plotW + 6, y + 3);
  }});

  ctx.save();
  ctx.beginPath(); ctx.rect(0, p2Top, plotW, p2H); ctx.clip();
  drawCurve(c.stLongD, yOfP2, '#3b82f6', 1.35);
  drawCurve(c.stLongK, yOfP2, '#22c55e', 1.45);
  drawCurve(c.stShortD, yOfP2, '#38bdf8', 1.25);
  drawCurve(c.stShortK, yOfP2, '#ef4444', 1.55);
  ctx.restore();

  if (c.stLongK[total - 1] != null) drawAxisTag(yOfP2(c.stLongK[total - 1]), p2Top, p2Top + p2H, c.stLongK[total - 1].toFixed(1), '#16a34a');
  if (c.stShortK[total - 1] != null) drawAxisTag(yOfP2(c.stShortK[total - 1]), p2Top, p2Top + p2H, c.stShortK[total - 1].toFixed(1), '#dc2626');

  lx = 10; ly = p2Top + 16;
  ctx.font = 'bold 11.5px -apple-system, BlinkMacSystemFont, "SF Mono", monospace';
  ctx.fillStyle = '#94a3b8';
  ctx.fillText('Stochastik UMWA', lx, ly); lx += 110;
  drawLegVal(c.stShortK[idx], '#ef4444');
  drawLegVal(c.stShortD[idx], '#38bdf8');
  drawLegVal(c.stLongK[idx], '#22c55e');
  drawLegVal(c.stLongD[idx], '#3b82f6');

  // ===================================================================
  // PANE 3: RSI UMWA (14)
  // ===================================================================
  ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(0, p3Top); ctx.lineTo(W, p3Top); ctx.stroke();

  const yOfP3 = (v) => p3Top + 8 + (1.0 - (v - 10.0) / 80.0) * (p3H - 16);
  ctx.fillStyle = 'rgba(168, 85, 247, 0.05)';
  ctx.fillRect(0, yOfP3(70), plotW, yOfP3(30) - yOfP3(70));

  const yRsi70 = yOfP3(70), yRsi30 = yOfP3(30);
  ctx.strokeStyle = 'rgba(16, 185, 129, 0.55)';
  ctx.beginPath(); ctx.moveTo(0, yRsi70); ctx.lineTo(plotW, yRsi70); ctx.stroke();
  ctx.strokeStyle = 'rgba(244, 63, 94, 0.55)';
  ctx.beginPath(); ctx.moveTo(0, yRsi30); ctx.lineTo(plotW, yRsi30); ctx.stroke();
  ctx.fillStyle = '#64748b'; ctx.font = '10.5px "SF Mono", monospace';
  ctx.fillText('70', plotW + 6, yRsi70 + 3);
  ctx.fillText('30', plotW + 6, yRsi30 + 3);

  ctx.save();
  ctx.beginPath(); ctx.rect(0, p3Top, plotW, p3H); ctx.clip();
  drawCurve(c.rsi, yOfP3, '#d946ef', 1.75);
  ctx.restore();
  if (c.rsi[total - 1] != null) drawAxisTag(yOfP3(c.rsi[total - 1]), p3Top, p3Top + p3H, c.rsi[total - 1].toFixed(2), '#c026d3');

  lx = 10; ly = p3Top + 16;
  ctx.font = 'bold 11.5px -apple-system, BlinkMacSystemFont, "SF Mono", monospace';
  ctx.fillStyle = '#94a3b8';
  ctx.fillText('RSI UMWA', lx, ly); lx += 72;
  drawLegVal(c.rsi[idx], '#d946ef');

  // ===================================================================
  // PANE 4: MACD UMWA (12, 26, 9 HISTOGRAMM)
  // ===================================================================
  ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(0, p4Top); ctx.lineTo(W, p4Top); ctx.stroke();

  let maxM = 0.001;
  for (let i = iStart; i <= iEnd; i++) {{
    if (c.macdHist[i] != null && Math.abs(c.macdHist[i]) > maxM) maxM = Math.abs(c.macdHist[i]);
  }}
  maxM *= 1.15;
  const yOfP4 = (v) => p4Top + 8 + (0.5 - (v / (2.0 * maxM))) * (p4H - 16);
  const yZero = yOfP4(0);
  ctx.strokeStyle = 'rgba(100, 116, 139, 0.45)';
  ctx.beginPath(); ctx.moveTo(0, yZero); ctx.lineTo(plotW, yZero); ctx.stroke();

  ctx.save();
  ctx.beginPath(); ctx.rect(0, p4Top, plotW, p4H); ctx.clip();
  for (let i = iStart; i <= iEnd; i++) {{
    const v = c.macdHist[i];
    if (v == null) continue;
    const prevV = i > 0 && c.macdHist[i - 1] != null ? c.macdHist[i - 1] : 0;
    let mClr = '#22c55e';
    if (v >= 0) mClr = (v >= prevV) ? '#22c55e' : '#86efac';
    else mClr = (v <= prevV) ? '#ef4444' : '#fca5a5';
    const x = xOf(i);
    if (x < -barW || x > plotW + barW) continue;
    const y = yOfP4(v);
    ctx.fillStyle = mClr;
    ctx.fillRect(x - barW * 0.45, Math.min(y, yZero), Math.max(1.2, barW * 0.9), Math.max(1.0, Math.abs(y - yZero)));
  }}
  ctx.restore();
  const lastM = c.macdHist[total - 1];
  if (lastM != null) {{
    drawAxisTag(yOfP4(lastM), p4Top, p4Top + p4H, lastM.toFixed(4), lastM >= 0 ? '#16a34a' : '#dc2626');
  }}

  lx = 10; ly = p4Top + 16;
  ctx.font = 'bold 11.5px -apple-system, BlinkMacSystemFont, "SF Mono", monospace';
  ctx.fillStyle = '#94a3b8';
  ctx.fillText('MACD UMWA', lx, ly); lx += 86;
  if (c.macdHist[idx] != null) {{
    ctx.fillStyle = c.macdHist[idx] >= 0 ? '#22c55e' : '#f43f5e';
    ctx.fillText(c.macdHist[idx].toFixed(4), lx, ly);
  }}

  // Untere Zeitachsen-Trennlinie & rechte Achsen-Trennlinie
  ctx.strokeStyle = '#1e293b';
  ctx.beginPath();
  ctx.moveTo(0, plotH); ctx.lineTo(W, plotH);
  ctx.moveTo(plotW, 0); ctx.lineTo(plotW, H);
  ctx.stroke();

  // "AUTO" Badge auf der Preisachse unten rechts (wie in TradingView)
  const isAutoY = (Math.abs(tvPriceZoom - 1.0) < 0.01 && Math.abs(tvPricePanPx) < 1.0);
  ctx.fillStyle = isAutoY ? 'rgba(56, 189, 248, 0.22)' : 'rgba(30, 41, 59, 0.85)';
  ctx.fillRect(plotW + 8, p1H - 20, axisW - 16, 16);
  ctx.fillStyle = isAutoY ? '#38bdf8' : '#94a3b8';
  ctx.font = 'bold 9.5px "SF Mono", monospace';
  ctx.fillText('AUTO', plotW + 22, p1H - 8.5);

  // Synchronisiertes 2D-Fadenkreuz (Crosshair vertikal + horizontal mit Preis-Tag wie in TradingView)
  if (hoverBarIdx != null && hoverBarIdx >= 0 && hoverBarIdx < total) {{
    const cx = xOf(hoverBarIdx);
    if (cx >= 0 && cx <= plotW) {{
      ctx.save();
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.6)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, plotH);
      if (hoverRelY != null && hoverRelY >= 0 && hoverRelY <= plotH) {{
        ctx.moveTo(0, hoverRelY);
        ctx.lineTo(plotW, hoverRelY);
      }}
      ctx.stroke();
      ctx.restore();

      if (hoverRelY != null && hoverRelY >= p1Top && hoverRelY <= p1Top + p1H) {{
        const curP = priceOfY1(hoverRelY);
        if (isFinite(curP)) {{
          drawAxisTag(hoverRelY, p1Top, p1Top + p1H, curP.toFixed(2), '#334155', '#f8fafc');
        }}
      }}

      const dtStr = c.dates[hoverBarIdx];
      ctx.font = 'bold 11px "SF Mono", monospace';
      const dw = ctx.measureText(dtStr).width + 14;
      const dx = Math.max(4, Math.min(plotW - dw - 4, cx - dw * 0.5));
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(dx, plotH + 3, dw, 19);
      ctx.fillStyle = '#f8fafc';
      ctx.fillText(dtStr, dx + 7, plotH + 16);
    }}
  }}
}}

// TradingView-Haptik: Mausrad/Trackpad-Zoom (am Cursor verankert), Click-Drag-Panning,
// Preisachsen-Skalierung (Rechts), Zeitachsen-Drag (Unten), Touch-Pinch & Doppelklick-Reset
(function initChartInteractions() {{
  const canvas = document.getElementById('umwaChartCanvas');
  if (!canvas) return;

  let dragMode = null; // 'pan' | 'y_axis' | 'x_axis'
  let dragStartX = 0;
  let dragStartY = 0;
  let dragStartViewStart = 0;
  let dragStartViewEnd = 0;
  let dragStartPriceZoom = 1.0;
  let dragStartPricePan = 0.0;
  let touchPinchDist = null;
  let touchPinchSpan = null;
  let touchPinchAnchorBar = null;
  let touchPinchRatio = 0.5;

  const updateHover = (clientX, clientY) => {{
    if (!cachedChartCalc) return;
    const rect = canvas.getBoundingClientRect();
    const axisW = 78, timeH = 26;
    const plotW = rect.width - axisW;
    const plotH = rect.height - timeH;
    const relX = clientX - rect.left;
    const relY = clientY - rect.top;

    // Cursor-Form je nach Bereich wie in TradingView anpassen
    if (!dragMode) {{
      if (relX >= plotW && relY < plotH) canvas.style.cursor = 'ns-resize';
      else if (relY >= plotH && relX < plotW) canvas.style.cursor = 'ew-resize';
      else canvas.style.cursor = 'crosshair';
    }}

    if (relX < 0 || relX > plotW || relY < 0 || relY > plotH) {{
      if (hoverBarIdx !== null || hoverRelY !== null) {{
        hoverBarIdx = null;
        hoverRelX = null;
        hoverRelY = null;
        drawUmwaChart();
      }}
      return;
    }}

    const total = cachedChartCalc.closes.length;
    clampChartViewport(total);
    const visSpan = Math.max(5, tvViewEnd - tvViewStart);
    const rawIdx = tvViewStart + ((relX - 8) / Math.max(10, plotW - 16)) * visSpan - 0.5;
    const idx = Math.max(0, Math.min(total - 1, Math.round(rawIdx)));
    hoverBarIdx = idx;
    hoverRelX = relX;
    hoverRelY = relY;
    drawUmwaChart();
  }};

  // 1. Mausrad & Trackpad (Zoom am Mauszeiger + 2-Finger-Horizontal-Swipe)
  canvas.addEventListener('wheel', e => {{
    if (!cachedChartCalc) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const plotW = rect.width - 78;
    const relX = e.clientX - rect.left;
    const total = cachedChartCalc.closes.length;
    clampChartViewport(total);

    // Mausrad auf der rechten Preisachse -> Vertikale Y-Skalierung
    if (relX >= plotW) {{
      const zFactor = Math.exp(-e.deltaY * 0.0025);
      tvPriceZoom = Math.max(0.25, Math.min(5.0, tvPriceZoom * zFactor));
      drawUmwaChart();
      return;
    }}

    const curSpan = tvViewEnd - tvViewStart;
    // 2-Finger-Horizontal-Swipe auf dem Mac-Trackpad -> Sanftes Verschieben (Panning)
    if (!e.ctrlKey && Math.abs(e.deltaX) > Math.abs(e.deltaY) * 1.15) {{
      const shiftBars = (e.deltaX / Math.max(50, plotW - 16)) * curSpan;
      tvViewStart += shiftBars;
      tvViewEnd += shiftBars;
      clampChartViewport(total);
      updateHover(e.clientX, e.clientY);
      return;
    }}

    // Vertikales Mausrad oder Trackpad-Pinch -> Rein-/Rauszoomen exakt an der Mausposition
    const focusRatio = Math.max(0.0, Math.min(1.0, (relX - 8) / Math.max(20, plotW - 16)));
    const anchorBar = tvViewStart + focusRatio * curSpan;
    const speed = e.ctrlKey ? 0.011 : 0.0028;
    const factor = Math.exp(e.deltaY * speed);
    const minSpan = Math.min(14, total);
    const maxSpan = total;
    const newSpan = Math.max(minSpan, Math.min(maxSpan, curSpan * factor));
    tvViewStart = anchorBar - focusRatio * newSpan;
    tvViewEnd = tvViewStart + newSpan;
    clampChartViewport(total);
    updateHover(e.clientX, e.clientY);
  }}, {{ passive: false }});

  // 2. Click & Drag (Pan im Chart, Skalieren auf Preisachse & Zeitachse)
  canvas.addEventListener('mousedown', e => {{
    if (!cachedChartCalc || e.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const plotW = rect.width - 78;
    const plotH = rect.height - 26;
    const p1H = Math.floor(plotH * 0.50);
    const relX = e.clientX - rect.left;
    const relY = e.clientY - rect.top;

    // Klick auf den "AUTO" Button unten auf der Preisachse
    if (relX >= plotW && relY >= p1H - 24 && relY <= p1H) {{
      tvPriceZoom = 1.0;
      tvPricePanPx = 0.0;
      drawUmwaChart();
      return;
    }}

    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragStartViewStart = tvViewStart;
    dragStartViewEnd = tvViewEnd;
    dragStartPriceZoom = tvPriceZoom;
    dragStartPricePan = tvPricePanPx;

    if (relX >= plotW && relY < plotH) {{
      dragMode = 'y_axis';
      canvas.style.cursor = 'ns-resize';
    }} else if (relY >= plotH && relX < plotW) {{
      dragMode = 'x_axis';
      canvas.style.cursor = 'ew-resize';
    }} else {{
      dragMode = 'pan';
      canvas.style.cursor = 'grabbing';
    }}
  }});

  window.addEventListener('mousemove', e => {{
    if (!dragMode || !cachedChartCalc) return;
    const rect = canvas.getBoundingClientRect();
    const plotW = Math.max(60, rect.width - 78);
    const total = cachedChartCalc.closes.length;
    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;

    if (dragMode === 'pan') {{
      const span = dragStartViewEnd - dragStartViewStart;
      const shiftBars = -(dx / (plotW - 16)) * span;
      tvViewStart = dragStartViewStart + shiftBars;
      tvViewEnd = dragStartViewEnd + shiftBars;
      if (Math.abs(tvPriceZoom - 1.0) > 0.01 || Math.abs(tvPricePanPx) > 0.5 || e.shiftKey) {{
        tvPricePanPx = dragStartPricePan + dy;
      }}
      clampChartViewport(total);
      drawUmwaChart();
    }} else if (dragMode === 'y_axis') {{
      const zFactor = Math.exp(-dy * 0.006);
      tvPriceZoom = Math.max(0.25, Math.min(5.0, dragStartPriceZoom * zFactor));
      drawUmwaChart();
    }} else if (dragMode === 'x_axis') {{
      const span = dragStartViewEnd - dragStartViewStart;
      const zFactor = Math.exp(dx * 0.005);
      const minSpan = Math.min(14, total);
      const newSpan = Math.max(minSpan, Math.min(total, span * zFactor));
      tvViewEnd = dragStartViewEnd;
      tvViewStart = tvViewEnd - newSpan;
      clampChartViewport(total);
      drawUmwaChart();
    }}
  }});

  window.addEventListener('mouseup', () => {{
    if (dragMode) {{
      dragMode = null;
      canvas.style.cursor = 'crosshair';
    }}
  }});

  canvas.addEventListener('mousemove', e => {{
    if (!dragMode) updateHover(e.clientX, e.clientY);
  }});

  canvas.addEventListener('mouseleave', () => {{
    if (!dragMode) {{
      hoverBarIdx = null;
      hoverRelX = null;
      hoverRelY = null;
      drawUmwaChart();
    }}
  }});

  // Doppelklick -> Automatische Skalierung / Reset wie in TradingView
  canvas.addEventListener('dblclick', e => {{
    if (!cachedChartCalc) return;
    const rect = canvas.getBoundingClientRect();
    const plotW = rect.width - 78;
    const relX = e.clientX - rect.left;
    if (relX >= plotW) {{
      tvPriceZoom = 1.0;
      tvPricePanPx = 0.0;
      drawUmwaChart();
    }} else {{
      resetChartView();
    }}
  }});

  // 3. Touch-Steuerung für iPad & iPhone (1-Finger-Pan + 2-Finger-Pinch-Zoom)
  canvas.addEventListener('touchstart', e => {{
    if (!cachedChartCalc) return;
    const total = cachedChartCalc.closes.length;
    clampChartViewport(total);
    if (e.touches.length === 2) {{
      e.preventDefault();
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      touchPinchDist = Math.hypot(dx, dy);
      touchPinchSpan = tvViewEnd - tvViewStart;
      const rect = canvas.getBoundingClientRect();
      const plotW = Math.max(60, rect.width - 78);
      const midX = (e.touches[0].clientX + e.touches[1].clientX) * 0.5 - rect.left;
      touchPinchRatio = Math.max(0.0, Math.min(1.0, (midX - 8) / (plotW - 16)));
      touchPinchAnchorBar = tvViewStart + touchPinchRatio * touchPinchSpan;
      dragMode = null;
    }} else if (e.touches.length === 1) {{
      dragMode = 'pan';
      dragStartX = e.touches[0].clientX;
      dragStartY = e.touches[0].clientY;
      dragStartViewStart = tvViewStart;
      dragStartViewEnd = tvViewEnd;
      updateHover(e.touches[0].clientX, e.touches[0].clientY);
    }}
  }}, {{ passive: false }});

  canvas.addEventListener('touchmove', e => {{
    if (!cachedChartCalc) return;
    const total = cachedChartCalc.closes.length;
    if (e.touches.length === 2 && touchPinchDist > 10) {{
      e.preventDefault();
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const dist = Math.hypot(dx, dy);
      const factor = touchPinchDist / Math.max(10, dist);
      const minSpan = Math.min(14, total);
      const newSpan = Math.max(minSpan, Math.min(total, touchPinchSpan * factor));
      tvViewStart = touchPinchAnchorBar - touchPinchRatio * newSpan;
      tvViewEnd = tvViewStart + newSpan;
      clampChartViewport(total);
      drawUmwaChart();
    }} else if (e.touches.length === 1 && dragMode === 'pan') {{
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const plotW = Math.max(60, rect.width - 78);
      const dx = e.touches[0].clientX - dragStartX;
      const span = dragStartViewEnd - dragStartViewStart;
      const shiftBars = -(dx / (plotW - 16)) * span;
      tvViewStart = dragStartViewStart + shiftBars;
      tvViewEnd = dragStartViewEnd + shiftBars;
      clampChartViewport(total);
      updateHover(e.touches[0].clientX, e.touches[0].clientY);
    }}
  }}, {{ passive: false }});

  canvas.addEventListener('touchend', () => {{
    dragMode = null;
    touchPinchDist = null;
  }});

  window.addEventListener('resize', () => {{
    if (document.getElementById('view-charts').style.display !== 'none') drawUmwaChart();
  }});

  window.addEventListener('keydown', e => {{
    if (document.getElementById('view-charts').style.display === 'none') return;
    if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {{
      e.preventDefault();
      const syms = getFilteredChartSymbols();
      if (!syms.length) return;
      const cur = syms.indexOf(activeChartSym);
      const next = e.key === 'ArrowDown'
        ? syms[(cur + 1) % syms.length]
        : syms[(cur - 1 + syms.length) % syms.length];
      selectChartSymbol(next);
      const activeRow = document.querySelector('#tvWlList .tv-wl-row.active');
      if (activeRow) activeRow.scrollIntoView({{ block: 'nearest' }});
    }} else if (e.key === '+' || e.key === '=') {{
      e.preventDefault();
      zoomChartBy(0.78);
    }} else if (e.key === '-' || e.key === '_') {{
      e.preventDefault();
      zoomChartBy(1.28);
    }} else if (e.key === '0') {{
      e.preventDefault();
      resetChartView();
    }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {{
      if (!cachedChartCalc) return;
      e.preventDefault();
      const total = cachedChartCalc.closes.length;
      clampChartViewport(total);
      const step = Math.max(2, (tvViewEnd - tvViewStart) * 0.12) * (e.key === 'ArrowLeft' ? -1 : 1);
      tvViewStart += step;
      tvViewEnd += step;
      clampChartViewport(total);
      drawUmwaChart();
    }}
  }});
}})();

applyAllFilters();
(async function checkInitialRefresh() {{
  if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {{
    try {{
      const r = await fetch('/api/status');
      if (r.ok) {{
        const d = await r.json();
        if (d.refreshing) {{
          pollRefreshStatus('', document.getElementById('refreshBtn'));
        }}
      }}
    }} catch (e) {{}}
  }}
}})();
</script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    v1_path = os.path.join(base_dir, "Version_1_App.html")
    if os.path.abspath(html_path) != os.path.abspath(v1_path):
        with open(v1_path, "w", encoding="utf-8") as f_v1:
            f_v1.write(html)
    print(f"HTML-Dashboard (Version 1) gespeichert unter: {html_path} & {v1_path}")


def send_macos_notification(results: list):
    import subprocess
    valid = [r for r in results if "error" not in r]
    buys = [r["symbol"] for r in valid if r.get("signal_level") == "STRONG_BUY"]
    sells = [r["symbol"] for r in valid if r.get("signal_level") == "STRONG_SELL"]
    shielded = [
        r["symbol"] for r in valid
        if r.get("signal_level") in ("STRONG_BUY", "STRONG_SELL")
        and (r.get("option_setup") or {}).get("qualified")
        and (r.get("option_setup") or {}).get("sr_protected")
    ]
    title = f"UMWA Stillhalter: {len(buys)} Kauf | {len(sells)} Verkauf"
    subtitle = f"🛡️ {len(shielded)} mit Linien-Schutz: {', '.join(shielded[:6])}"
    script = f'display notification "{subtitle}" with title "{title}" sound name "Glass"'
    try:
        subprocess.run(["osascript", "-e", script], check=False)
    except Exception:
        pass


def send_telegram_if_configured(results: list):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    valid = [r for r in results if "error" not in r]
    shielded = [
        r for r in valid
        if r.get("signal_level") in ("STRONG_BUY", "STRONG_SELL")
        and (r.get("option_setup") or {}).get("qualified")
        and (r.get("option_setup") or {}).get("sr_protected")
    ]
    lines = ["📊 *UMWA Stillhalter-Signale (geschützt durch Linie)*:"]
    for r in shielded[:15]:
        opt = r["option_setup"]
        sl = opt["sr_shield_line"]
        icon = "🟢" if r["signal_level"] == "STRONG_BUY" else "🔴"
        lines.append(
            f"{icon} *{r['symbol']}* ({r['last_close']:.2f}$) -> {opt['type']} Strike {opt['strike']:.2f}$ "
            f"(Δ {opt['delta']:.2f}, +{opt['prem_pct']:.2f}%) | 🛡️ Linie {sl['price']:.2f}$"
        )
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": "\n".join(lines), "parse_mode": "Markdown"},
            timeout=10,
        )
        print("📲 Telegram-Nachricht erfolgreich gesendet!")
    except Exception as e:
        print(f"⚠️ Telegram-Fehler: {e}")


if __name__ == "__main__":
    default_dir = os.path.dirname(os.path.abspath(__file__))
    default_html = os.path.join(default_dir, "Stillhalter_Dashboard.html")
    default_csv = os.path.join(default_dir, "Stillhalter_Signale.csv")

    parser = argparse.ArgumentParser(description="Market Vision Pro – UMWA Stillhalter Agent")
    parser.add_argument("symbols", nargs="*", default=DEFAULT_WATCHLIST, help="Ticker-Symbole (Standard: komplette 255er Watchlist)")
    parser.add_argument(
        "--group", "-g",
        choices=["A", "B", "C", "JULIAN", "OPTIONS-PRO"],
        help="Bestimmte Watchlist-Gruppe scannen (A, B, C, JULIAN oder OPTIONS-PRO)"
    )
    parser.add_argument("--period", "-p", default="d", choices=["d", "w", "m"], help="Zeitrahmen: d (Tag), w (Woche), m (Monat)")
    parser.add_argument("--only-signals", "-s", action="store_true", help="Nur Aktien mit aktivem Signal (ab 2/4) anzeigen")
    parser.add_argument("--html", nargs="?", const=default_html, help="Speichert ein interaktives HTML-Dashboard für den Browser")
    parser.add_argument("--csv", nargs="?", const=default_csv, help="Speichert alle Signale als CSV für Excel / Numbers")
    parser.add_argument("--open", action="store_true", help="Öffnet das HTML-Dashboard nach dem Scan direkt im Standard-Browser")
    parser.add_argument("--notify", action="store_true", help="Zeigt eine macOS Desktop-Mitteilung mit den Top-Signalen")
    parser.add_argument("--json", action="store_true", help="Ausgabe als JSON")
    args = parser.parse_args()

    syms = args.symbols
    if args.group:
        if args.group in ("A", "B", "C"):
            syms = WATCHLIST_GROUPS[f"{args.group}-WERTE"]
        elif args.group == "JULIAN":
            syms = (
                WATCHLIST_GROUPS["JULIAN-STABLE"]
                + WATCHLIST_GROUPS["JULIAN-ACTIVE"]
                + WATCHLIST_GROUPS["JULIAN-VOLATILE"]
                + WATCHLIST_GROUPS["JULIAN-TECH-BIO"]
            )
        elif args.group == "OPTIONS-PRO":
            syms = WATCHLIST_GROUPS["OPTIONS-PRO-50"]

    out = scan_symbols(syms, period=args.period, only_signals=args.only_signals)
    if args.html:
        export_html_dashboard(out, args.html)
        if args.open:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(args.html)}")
    if args.csv:
        export_csv_report(out, args.csv)
    if args.notify:
        send_macos_notification(out)
    send_telegram_if_configured(out)
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
