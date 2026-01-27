import os,time
from datetime import datetime,timedelta,timezone
import requests,pandas as pd,pandas_ta as ta,pandas_market_calendars as mcal
from fastapi import FastAPI,Query

EODHD_KEY=os.environ["EODHD_API_KEY"]
WATCHLIST_PATH="watchlist.txt"
BASE="https://eodhd.com/api"
CACHE={}
app=FastAPI()

def wl():
    with open(WATCHLIST_PATH) as f:
        return [l.strip() for l in f if l.strip()]

def times():
    cal=mcal.get_calendar("XNYS")
    d=datetime.now(timezone.utc).date()
    s=cal.schedule(d,d)
    if s.empty:return {"isTradingDay":False}
    o=s.iloc[0]["market_open"].to_pydatetime().astimezone(timezone.utc)
    return {"isTradingDay":True,"open":o.isoformat(),"run1":(o+timedelta(minutes=30)).isoformat(),"run2":(o+timedelta(hours=3,minutes=30)).isoformat()}

def get(k):
    v=CACHE.get(k)
    if not v:return None
    e,d=v
    if time.time()>e:
        CACHE.pop(k,None)
        return None
    return d

def setc(k,d,t):
    CACHE[k]=(time.time()+t,d)

def data(sym):
    k=f"{sym}:{int(time.time())//300}"
    d=get(k)
    if d is None:
        r=requests.get(f"{BASE}/intraday/{sym}",params={"api_token":EODHD_KEY,"interval":"5m","fmt":"json"},timeout=20)
        r.raise_for_status()
        d=r.json()
        setc(k,d,240)
    df=pd.DataFrame(d)
    if df.empty:return df
    df["datetime"]=pd.to_datetime(df["datetime"],utc=True)
    return df.sort_values("datetime").tail(200).reset_index(drop=True)

def sig(df):
    if len(df)<60:return None
    c=df["close"]
    rsi=ta.rsi(c,14)
    macd=ta.macd(c)
    if rsi.iloc[-1]<30 and macd["MACDh_12_26_9"].iloc[-1]>0:return "BUY"
    if rsi.iloc[-1]>70 and macd["MACDh_12_26_9"].iloc[-1]<0:return "SELL"
    return None

@app.get("/times")
def t():return times()

@app.get("/scan")
def scan():
    res=[]
    for s in wl():
        try:
            df=data(s)
            sg=sig(df)
            if sg:res.append({"symbol":s,"signal":sg})
        except:pass
    return {"signals":res}
