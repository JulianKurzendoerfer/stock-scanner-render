import os,time,math
from datetime import datetime,timedelta,timezone
import requests,pandas as pd,pandas_ta as ta,pandas_market_calendars as mcal
from fastapi import FastAPI

EODHD_KEY=os.environ["EODHD_API_KEY"]
WATCHLIST_PATH="watchlist.txt"
BASE="https://eodhd.com/api"
CACHE={}
app=FastAPI()

def norm(sym):
    return sym if "." in sym else f"{sym}.US"

def wl():
    with open(WATCHLIST_PATH,"r",encoding="utf-8") as f:
        return [norm(l.strip()) for l in f if l.strip() and not l.strip().startswith("#")]

def times():
    cal=mcal.get_calendar("XNYS")
    d=datetime.now(timezone.utc).date()
    s=cal.schedule(d,d)
    if s.empty:return {"isTradingDay":False}
    o=s.iloc[0]["market_open"].to_pydatetime().astimezone(timezone.utc)
    return {"isTradingDay":True,"open":o.isoformat(),"run1":(o+timedelta(minutes=30)).isoformat(),"run2":(o+timedelta(hours=3,minutes=30)).isoformat()}

def cget(k):
    v=CACHE.get(k)
    if not v:return None
    e,d=v
    if time.time()>e:
        CACHE.pop(k,None)
        return None
    return d

def cset(k,d,t):
    CACHE[k]=(time.time()+t,d)

def intraday(sym):
    k=f"{sym}:{int(time.time())//300}"
    d=cget(k)
    if d is None:
        r=requests.get(f"{BASE}/intraday/{sym}",params={"api_token":EODHD_KEY,"interval":"5m","fmt":"json"},timeout=30)
        r.raise_for_status()
        d=r.json()
        cset(k,d,240)
    df=pd.DataFrame(d)
    if df.empty:return df
    df["datetime"]=pd.to_datetime(df["datetime"],utc=True)
    return df.sort_values("datetime").tail(260).reset_index(drop=True)

def sig(df,sym):
    if df is None or df.empty or len(df)<80:
        return None

    close=df["close"]; high=df["high"]; low=df["low"]
    last=float(close.iloc[-1])

    bb=ta.bbands(close,20,2)
    bbl=float(bb["BBL_20_2.0"].iloc[-1])
    if bbl<=0:return None
    bb_dist=(last-bbl)/bbl
    bb_ok=bb_dist<=0.35

    rsi=ta.rsi(close,14)
    rsi_now=float(rsi.iloc[-1]); rsi_prev=float(rsi.iloc[-2])
    rsi_ok=(rsi_prev<30 and rsi_now>=30) or (rsi_now>=28.7 and rsi_now-rsi_prev>=2.0)

    st=ta.stoch(high,low,close,14,3,3)
    kcol=[c for c in st.columns if c.startswith("STOCHk_")][0]
    k=st[kcol].dropna()
    k_now=float(k.iloc[-1]); k_prev=float(k.iloc[-2])
    stoch_ok=(k_prev<20 and k_now-k_prev>=10) or (k_now-k_prev>=15)

    if not (rsi_ok and stoch_ok):
        return None

    macd=ta.macd(close,12,26,9)
    h=macd["MACDh_12_26_9"].dropna()
    if len(h)<3:return None
    h1,h2,h3=float(h.iloc[-1]),float(h.iloc[-2]),float(h.iloc[-3])
    macd_ok=(h1>h2>h3 and abs(h1)<abs(h2)<abs(h3)) or h1>=0

    strength="BUY_STRONG" if (macd_ok and bb_ok) else "BUY_WEAK"

    return {
        "symbol":sym,
        "signal":strength,
        "price":round(last,4),
        "rsi":round(rsi_now,2),
        "bb_distance_pct":round(bb_dist*100,2),
        "macdh":[round(h3,5),round(h2,5),round(h1,5)],
        "stoch_jump":round(k_now-k_prev,2)
    }

@app.get("/times")
def t():
    return times()

@app.get("/scan")
def scan():
    out=[]
    for s in wl():
        try:
            df=intraday(s)
            r=sig(df,s)
            if r:out.append(r)
        except:
            pass
    return {"signals":out}
