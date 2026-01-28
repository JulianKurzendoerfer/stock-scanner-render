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

def intraday(sym,interval="5m"):
    bucket=300 if interval=="5m" else 900
    k=f"{sym}:{interval}:{int(time.time())//bucket}"
    d=cget(k)
    if d is None:
        r=requests.get(f"{BASE}/intraday/{sym}",params={"api_token":EODHD_KEY,"interval":interval,"fmt":"json"},timeout=30)
        r.raise_for_status()
        d=r.json()
        cset(k,d,240)
    df=pd.DataFrame(d)
    if df.empty:return df
    df["datetime"]=pd.to_datetime(df["datetime"],utc=True,errors="coerce")
    df=df.dropna(subset=["datetime"]).sort_values("datetime").tail(260).reset_index(drop=True)
    return df

def rsi_eval(close):
    rsi=ta.rsi(close,14)
    if rsi is None or rsi.isna().all(): return (False,None,None,None)
    now=float(rsi.iloc[-1]); prev=float(rsi.iloc[-2])
    weak=(prev<30 and now>=30) or (now>=28.7 and now-prev>=2.0) or (now<30)
    return (weak,now,prev,float(now-prev))

def stoch_ok(high,low,close):
    st=ta.stoch(high,low,close,14,3,3)
    if st is None or st.empty:return (False,None)
    kcol=[c for c in st.columns if c.startswith("STOCHk_")][0]
    k=st[kcol].dropna()
    if len(k)<3:return (False,None)
    kn=float(k.iloc[-1]); kp=float(k.iloc[-2]); k2=float(k.iloc[-3])
    jump=kn-kp
    ok=(kp<20 and jump>=10) or (jump>=15) or (k2<30 and kn-k2>=20)
    return (ok,jump)

def macd_ok(close):
    m=ta.macd(close,12,26,9)
    if m is None or m.empty:return (False,None,None,None)
    h=m["MACDh_12_26_9"].dropna()
    if len(h)<4:return (False,None,None,None)
    h1=float(h.iloc[-1]); h2=float(h.iloc[-2]); h3=float(h.iloc[-3])
    improving2=(h1>h2>h3) and (abs(h1)<abs(h2)<abs(h3))
    ok=improving2 or (h1>h2 and h1>=0)
    return (ok,h3,h2,h1)

def bb_ok(close):
    bb=ta.bbands(close,20,2)
    if bb is None or bb.empty:return (False,None)
    bbl=float(bb["BBL_20_2.0"].iloc[-1])
    if bbl<=0 or math.isnan(bbl):return (False,None)
    last=float(close.iloc[-1])
    dist=(last-bbl)/bbl
    ok=dist<=0.35
    return (ok,dist*100)

def sig(df,sym):
    if df is None or df.empty or len(df)<60:return None
    close=df["close"]; high=df["high"]; low=df["low"]
    price=float(close.iloc[-1])

    r_ok,r_now,r_prev,r_d=rsi_eval(close)
    if not r_ok:return None

    s_ok,s_jump=stoch_ok(high,low,close)
    m_ok,h3,h2,h1=macd_ok(close)
    b_ok,b_dist=bb_ok(close)

    level="BUY_STRONG" if (s_ok and m_ok and b_ok) else "BUY_WEAK"

    out={"symbol":sym,"signal":level,"price":round(price,4),"rsi":round(float(r_now),2),"rsi_delta":round(float(r_d),2)}
    if b_dist is not None: out["bb_distance_pct"]=round(float(b_dist),2)
    if s_jump is not None: out["stoch_jump"]=round(float(s_jump),2)
    if h1 is not None: out["macdh"]=[round(float(h3),5),round(float(h2),5),round(float(h1),5)]
    return out

@app.get("/times")
def t():
    return times()

@app.get("/scan")
def scan():
    out=[]
    for s in wl():
        try:
            df=intraday(s,"5m")
            r=sig(df,s)
            if r: out.append(r)
        except:
            pass
    return {"signals":out}

@app.get("/scan_debug")
def scan_debug():
    syms=wl()
    total=len(syms)
    ok_df=0
    short_df=0
    rsi_ok_ct=0
    strong_ct=0
    errs=0
    lowest=[]
    for s in syms:
        try:
            df=intraday(s,"5m")
            if df is None or df.empty:
                errs+=1
                continue
            if len(df)<60:
                short_df+=1
                continue
            ok_df+=1
            close=df["close"]
            r_ok,r_now,r_prev,r_d=rsi_eval(close)
            if r_now is not None:
                lowest.append((float(r_now),s,round(float(close.iloc[-1]),4),round(float(r_d),2)))
            if not r_ok:
                continue
            rsi_ok_ct+=1
            high=df["high"]; low=df["low"]
            s_ok,_=stoch_ok(high,low,close)
            m_ok,_,_,_=macd_ok(close)
            b_ok,_=bb_ok(close)
            if s_ok and m_ok and b_ok:
                strong_ct+=1
        except:
            errs+=1
    lowest_sorted=sorted(lowest, key=lambda x: x[0])[:15]
    return {
        "total": total,
        "ok_df_len_ge_60": ok_df,
        "too_short_len_lt_60": short_df,
        "errors_or_empty": errs,
        "rsi_weak_count": rsi_ok_ct,
        "strong_count": strong_ct,
        "lowest_rsi_15": [{"rsi":x[0],"symbol":x[1],"price":x[2],"rsi_delta":x[3]} for x in lowest_sorted]
    }
