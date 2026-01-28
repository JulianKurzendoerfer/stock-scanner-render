import os,time
from datetime import datetime,timedelta,timezone
import requests
import pandas as pd
import pandas_ta as ta
import pandas_market_calendars as mcal
import yfinance as yf
from fastapi import FastAPI

WATCHLIST_PATH="watchlist.txt"
EODHD_KEY=os.environ.get("EODHD_API_KEY","")
BASE="https://eodhd.com/api"
CACHE={}
app=FastAPI()

def wl_raw():
    with open(WATCHLIST_PATH,"r",encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

def norm_symbol(sym:str)->str:
    s=sym.strip().upper()
    if s.startswith("$"): s=s[1:]
    if ":" in s: s=s.split(":")[-1]
    return s

def yf_symbol(sym:str)->str:
    s=norm_symbol(sym)
    if s.endswith(".US"): s=s[:-3]
    return s

def eod_symbol(sym:str)->str:
    s=norm_symbol(sym)
    if "." in s: return s
    if s.endswith(".US"): return s
    return f"{s}.US"

def times():
    cal=mcal.get_calendar("XNYS")
    d=datetime.now(timezone.utc).date()
    sch=cal.schedule(d,d)
    if sch.empty:
        return {"isTradingDay":False}
    o=sch.iloc[0]["market_open"].to_pydatetime().astimezone(timezone.utc)
    return {"isTradingDay":True,"open":o.isoformat(),"run1":(o+timedelta(minutes=30)).isoformat(),"run2":(o+timedelta(hours=3,minutes=30)).isoformat()}

def cget(k):
    v=CACHE.get(k)
    if not v: return None
    e,d=v
    if time.time()>e:
        CACHE.pop(k,None)
        return None
    return d

def cset(k,d,ttl):
    CACHE[k]=(time.time()+ttl,d)

def df_from_yf(sym:str, interval="5m"):
    bucket=300 if interval=="5m" else 900
    k=f"yf:{sym}:{interval}:{int(time.time())//bucket}"
    d=cget(k)
    if d is None:
        t=yf.Ticker(sym)
        df=t.history(period="5d", interval=interval, auto_adjust=False, actions=False)
        cset(k,df,240)
    else:
        df=d
    if df is None or df.empty:
        return pd.DataFrame()
    df=df.reset_index()
    dtcol="Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else None)
    if not dtcol:
        return pd.DataFrame()
    df=df.rename(columns={dtcol:"datetime","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df["datetime"]=pd.to_datetime(df["datetime"],utc=True,errors="coerce")
    df=df.dropna(subset=["datetime"]).sort_values("datetime").tail(260).reset_index(drop=True)
    return df[["datetime","open","high","low","close","volume"]]

def df_from_eod(sym:str, interval="5m"):
    if not EODHD_KEY:
        raise RuntimeError("Missing EODHD_API_KEY")
    bucket=300 if interval=="5m" else 900
    k=f"eod:{sym}:{interval}:{int(time.time())//bucket}"
    d=cget(k)
    if d is None:
        r=requests.get(f"{BASE}/intraday/{sym}",params={"api_token":EODHD_KEY,"interval":interval,"fmt":"json"},timeout=30)
        if r.status_code!=200:
            raise RuntimeError(f"HTTP {r.status_code} {r.text[:200]}")
        d=r.json()
        cset(k,d,240)
    df=pd.DataFrame(d)
    if df.empty: return df
    df["datetime"]=pd.to_datetime(df["datetime"],utc=True,errors="coerce")
    df=df.dropna(subset=["datetime"]).sort_values("datetime").tail(260).reset_index(drop=True)
    return df

def intraday(sym_raw:str, interval="5m"):
    sym_y=yf_symbol(sym_raw)
    df=df_from_yf(sym_y,interval)
    if df is not None and not df.empty and len(df)>=60:
        return df, "yfinance", sym_y
    sym_e=eod_symbol(sym_raw)
    df2=df_from_eod(sym_e,interval)
    if df2 is not None and not df2.empty and len(df2)>=60:
        return df2, "eodhd", sym_e
    return pd.DataFrame(), "none", sym_y

@app.get("/times")
def _t():
    return times()

@app.get("/scan_debug")
def scan_debug():
    syms=wl_raw()
    total=len(syms)
    ok=0
    ok_y=0
    ok_e=0
    too_short=0
    empty_or_err=0
    samples=[]
    for s in syms:
        try:
            df,src,used=intraday(s,"5m")
            if df is None or df.empty:
                empty_or_err+=1
                if len(samples)<5:
                    samples.append({"symbol":s,"src":src,"used":used,"rows":0})
                continue
            if len(df)<60:
                too_short+=1
                if len(samples)<5:
                    samples.append({"symbol":s,"src":src,"used":used,"rows":int(len(df))})
                continue
            ok+=1
            if src=="yfinance": ok_y+=1
            if src=="eodhd": ok_e+=1
        except Exception as e:
            empty_or_err+=1
            if len(samples)<5:
                samples.append({"symbol":s,"error":str(e)[:200]})
    return {"total":total,"ok_len_ge_60":ok,"ok_yfinance":ok_y,"ok_eodhd":ok_e,"too_short":too_short,"empty_or_err":empty_or_err,"samples":samples}

@app.get("/scan")
def scan():
    syms=wl_raw()
    signals=[]
    for s in syms:
        try:
            df,src,used=intraday(s,"5m")
            if df is None or df.empty or len(df)<60:
                continue
            close=df["close"]
            rsi=ta.rsi(close, length=14)
            if rsi is None or rsi.empty:
                continue
            last=float(rsi.iloc[-1])
            if last<=30:
                signals.append({"symbol":s,"source":src,"rsi":round(last,2),"type":"WEAK_BUY"})
        except:
            pass
    return {"signals":signals}
