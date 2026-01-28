import os,time,math
from datetime import datetime,timedelta,timezone
import requests,pandas as pd,pandas_ta as ta,pandas_market_calendars as mcal
from fastapi import FastAPI,Query

EODHD_KEY=os.environ["EODHD_API_KEY"]
WATCHLIST_PATH="watchlist.txt"
BASE="https://eodhd.com/api"
CACHE={}
app=FastAPI()

def wl():
    with open(WATCHLIST_PATH,"r",encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

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

def cset(k,d,ttl):
    CACHE[k]=(time.time()+ttl,d)

def intraday(sym,interval="5m"):
    bucket=300 if interval=="5m" else 900
    key=f"intra:{sym}:{interval}:{int(time.time())//bucket}"
    d=cget(key)
    if d is None:
        r=requests.get(f"{BASE}/intraday/{sym}",params={"api_token":EODHD_KEY,"interval":interval,"fmt":"json"},timeout=25)
        r.raise_for_status()
        d=r.json()
        cset(key,d,240)
    df=pd.DataFrame(d)
    if df.empty:return df
    df["datetime"]=pd.to_datetime(df["datetime"],utc=True)
    df=df.sort_values("datetime").tail(260).reset_index(drop=True)
    return df

def daily_1y(sym):
    key=f"daily:{sym}:{datetime.now(timezone.utc).date().isoformat()}"
    d=cget(key)
    if d is None:
        to_=datetime.now(timezone.utc).date()
        fr=to_-timedelta(days=370)
        r=requests.get(f"{BASE}/eod/{sym}",params={"api_token":EODHD_KEY,"from":fr.isoformat(),"to":to_.isoformat(),"fmt":"json"},timeout=30)
        r.raise_for_status()
        d=r.json()
        cset(key,d,6*3600)
    df=pd.DataFrame(d)
    if df.empty:return df
    df["date"]=pd.to_datetime(df["date"],utc=True,errors="coerce")
    df=df.dropna(subset=["date"]).sort_values("date").tail(260).reset_index(drop=True)
    return df

def drawdown_ok(sym,max_dd=0.35):
    ddf=daily_1y(sym)
    if ddf.empty:return True
    hi=float(ddf["high"].max())
    last=float(ddf["close"].iloc[-1])
    if hi<=0:return True
    dd=(hi-last)/hi
    return dd<=max_dd

def strong_rise(x,thr):
    if len(x)<2:return False
    return float(x.iloc[-1]-x.iloc[-2])>=thr

def sig(df,sym):
    if df is None or df.empty or len(df)<80:return None
    if not drawdown_ok(sym,0.35):return None

    close=df["close"]; high=df["high"]; low=df["low"]

    bb=ta.bbands(close,length=20,std=2)
    if bb is None or bb.empty:return None
    bbl=bb["BBL_20_2.0"].iloc[-1]
    bbu=bb["BBU_20_2.0"].iloc[-1]
    last=float(close.iloc[-1])
    near_lower = (not math.isnan(bbl)) and (last <= float(bbl)*1.005)

    rsi=ta.rsi(close,length=14)
    if rsi is None or rsi.isna().all():return None
    rsi_now=float(rsi.iloc[-1]); rsi_prev=float(rsi.iloc[-2])
    rsi_cross = (rsi_prev<30 and rsi_now>=30)
    rsi_pre29 = (rsi_now>=28.7 and rsi_now<30 and strong_rise(rsi,2.0))

    macd=ta.macd(close,fast=12,slow=26,signal=9)
    if macd is None or macd.empty:return None
    h=macd["MACDh_12_26_9"].dropna()
    if len(h)<4:return None
    h1=float(h.iloc[-1]); h2=float(h.iloc[-2]); h3=float(h.iloc[-3])
    macd_improving_2 = (h1>h2>h3) and (abs(h1)<abs(h2)<abs(h3))
    macd_ok = macd_improving_2 or (h1>h2 and h1>=0)

    st=ta.stoch(high,low,close,k=14,d=3,smooth_k=3)
    if st is None or st.empty:return None
    kcol=[c for c in st.columns if c.startswith("STOCHk_")][0]
    dcol=[c for c in st.columns if c.startswith("STOCHd_")][0]
    k=st[kcol].dropna(); d=st[dcol].dropna()
    if len(k)<3:return None
    k_now=float(k.iloc[-1]); k_prev=float(k.iloc[-2]); k_prev2=float(k.iloc[-3])
    k_jump=float(k_now-k_prev)
    stoch_from_low = (k_prev<20 and k_jump>=10)
    stoch_sudden = (k_jump>=15) or (k_prev2<30 and k_now-k_prev2>=20)
    stoch_ok = stoch_from_low or stoch_sudden
    d_bonus = (len(d)>=2 and float(d.iloc[-1])>float(d.iloc[-2]))

    if not (near_lower and (rsi_cross or rsi_pre29) and macd_ok and stoch_ok):
        return None

    return {
        "symbol": sym,
        "signal": "BUY",
        "price": round(last,4),
        "rsi": round(rsi_now,2),
        "bb_lower": round(float(bbl),4),
        "bb_upper": round(float(bbu),4),
        "macdh": [round(h3,6), round(h2,6), round(h1,6)],
        "stoch_k": [round(k_prev2,2), round(k_prev,2), round(k_now,2)],
        "stoch_d_bonus": bool(d_bonus)
    }

@app.get("/times")
def t():return times()

@app.get("/scan")
def scan(interval:str="5m"):
    out=[]
    for s in wl():
        try:
            df=intraday(s,interval=interval)
            r=sig(df,s)
            if r:out.append(r)
        except:pass
    return {"signals":out}
