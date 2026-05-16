import os
import uvicorn
import warnings
import time
import numpy as np
import pandas as pd
import feedparser
import requests
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

app = FastAPI(title="MarketCast API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── CACHE STORE ───────────────────────────────────────────────────────────────
_cache = {}

def cache_get(key):
    """Get value from cache if not expired"""
    if key in _cache:
        value, expiry = _cache[key]
        if time.time() < expiry:
            return value
    return None

def cache_set(key, value, ttl_seconds):
    """Save value to cache with TTL"""
    _cache[key] = (value, time.time() + ttl_seconds)

# Cache TTL constants (seconds)
TTL_GLOBAL   = 300   # 5 minutes  — global indices
TTL_NEWS     = 600   # 10 minutes — news articles
TTL_GIFT     = 120   # 2 minutes  — gift nifty
TTL_SECTOR   = 300   # 5 minutes  — sector momentum
TTL_FII      = 300   # 5 minutes  — FII/DII
TTL_STATUS   = 60    # 1 minute   — market status
TTL_MARKET   = 120   # 2 minutes  — index prices

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

NEWS_FEEDS = [
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^NSEI&region=IN&lang=en-US",     "source": "Yahoo Finance", "tag": "NIFTY"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^BSESN&region=IN&lang=en-US",    "source": "Yahoo Finance", "tag": "SENSEX"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=RELIANCE.NS&region=IN&lang=en-US","source": "Yahoo Finance", "tag": "RELIANCE"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TCS.NS&region=IN&lang=en-US",    "source": "Yahoo Finance", "tag": "TCS"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=HDFCBANK.NS&region=IN&lang=en-US","source": "Yahoo Finance", "tag": "HDFCBANK"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=INFY.NS&region=IN&lang=en-US",   "source": "Yahoo Finance", "tag": "INFY"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SBIN.NS&region=IN&lang=en-US",   "source": "Yahoo Finance", "tag": "SBIN"},
    {"url": "https://economictimes.indiatimes.com/markets/stocks/rss.cms",         "source": "ET Markets",    "tag": "STOCKS"},
    {"url": "https://economictimes.indiatimes.com/markets/rss.cms",                "source": "ET Markets",    "tag": "MARKET"},
    {"url": "https://economictimes.indiatimes.com/markets/mutual-funds/rss.cms",   "source": "ET Markets",    "tag": "MF"},
    {"url": "https://economictimes.indiatimes.com/markets/commodities/rss.cms",    "source": "ET Markets",    "tag": "COMMODITIES"},
    {"url": "https://economictimes.indiatimes.com/markets/forex/rss.cms",          "source": "ET Markets",    "tag": "FOREX"},
    {"url": "https://economictimes.indiatimes.com/news/economy/rss.cms",           "source": "ET Economy",    "tag": "ECONOMY"},
    {"url": "https://www.livemint.com/rss/markets",                                "source": "Live Mint",     "tag": "MARKET"},
    {"url": "https://www.livemint.com/rss/companies",                              "source": "Live Mint",     "tag": "STOCKS"},
    {"url": "https://www.livemint.com/rss/money",                                  "source": "Live Mint",     "tag": "ECONOMY"},
    {"url": "https://www.business-standard.com/rss/markets-106.rss",              "source": "Business Std",  "tag": "MARKET"},
    {"url": "https://www.business-standard.com/rss/finance-109.rss",              "source": "Business Std",  "tag": "FINANCE"},
    {"url": "https://www.business-standard.com/rss/companies-101.rss",            "source": "Business Std",  "tag": "STOCKS"},
    {"url": "https://www.thehindu.com/business/markets/?service=rss",             "source": "Hindu Business", "tag": "MARKET"},
    {"url": "https://feeds.reuters.com/reuters/INbusinessNews",                    "source": "Reuters India",  "tag": "GLOBAL"},
    {"url": "https://feeds.reuters.com/reuters/businessNews",                      "source": "Reuters",        "tag": "GLOBAL"},
    {"url": "https://www.moneycontrol.com/rss/MCtopnews.xml",                     "source": "Moneycontrol",   "tag": "MARKET"},
    {"url": "https://www.moneycontrol.com/rss/stocksmarket.xml",                  "source": "Moneycontrol",   "tag": "STOCKS"},
    {"url": "https://www.moneycontrol.com/rss/results.xml",                       "source": "Moneycontrol",   "tag": "RESULTS"},
    {"url": "https://www.moneycontrol.com/rss/economy.xml",                       "source": "Moneycontrol",   "tag": "ECONOMY"},
    {"url": "https://www.moneycontrol.com/rss/ipo.xml",                           "source": "Moneycontrol",   "tag": "IPO"},
]

GLOBAL_INDICES = {
    "DOW":       "^DJI",
    "NASDAQ":    "^IXIC",
    "SP500":     "^GSPC",
    "FTSE":      "^FTSE",
    "DAX":       "^GDAXI",
    "NIKKEI":    "^N225",
    "HANGSENG":  "^HSI",
    "SHANGHAI":  "000001.SS",
    "GIFTNIFTY": "NIFTY50.NS",
    "CAC40":     "^FCHI",
    "ASX":       "^AXJO",
    "KOSPI":     "^KS11",
    "SENSEX":    "^BSESN",
    "BANKNIFTY": "^NSEBANK",
}

SECTOR_INDICES = {
    "IT":       "^CNXIT",
    "Banking":  "^NSEBANK",
    "Pharma":   "^CNXPHARMA",
    "Auto":     "^CNXAUTO",
    "FMCG":     "^CNXFMCG",
    "Energy":   "^CNXENERGY",
    "Metals":   "^CNXMETAL",
    "Infra":    "^CNXINFRA",
    "Realty":   "^CNXREALTY",
    "Media":    "^CNXMEDIA",
}

def nse(symbol):
    m = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN",
         "FINNIFTY":"NIFTY_FIN_SERVICE.NS","MIDCPNIFTY":"^NSEMDCP50"}
    return m.get(symbol.upper(), f"{symbol.upper()}.NS")

def safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if (v != v) else round(v, 2)
    except:
        return default

def safe_int(val, default=0):
    try:
        v = float(val)
        return default if (v != v) else int(v)
    except:
        return default

def arima_forecast(prices, days=30):
    try:
        clean = [p for p in prices if p == p]
        model = ARIMA(clean, order=(2,1,2))
        fit   = model.fit()
        fc    = fit.forecast(steps=days)
        ci    = fit.get_forecast(steps=days).conf_int(alpha=0.2)
        return {
            "predicted": [safe_float(v) for v in fc],
            "upper":     [safe_float(v) for v in ci.iloc[:,1]],
            "lower":     [safe_float(v) for v in ci.iloc[:,0]],
        }
    except:
        last  = prices[-1] if prices else 100
        trend = (prices[-1]-prices[-5])/5 if len(prices)>=5 else 0
        pred  = [round(last+trend*i,2) for i in range(1,days+1)]
        return {
            "predicted": pred,
            "upper": [round(p*1.03,2) for p in pred],
            "lower": [round(p*0.97,2) for p in pred],
        }

def forecast_dates(days):
    dates=[]; d=datetime.today()
    while len(dates)<days:
        d+=timedelta(days=1)
        if d.weekday()<5:
            dates.append(d.strftime("%d %b"))
    return dates

def intraday_forecast(price, pct):
    times=["9:15","9:30","9:45","10:00","10:30","11:00","11:30",
           "12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:30"]
    p=price; step=(pct/100)/len(times)
    np.random.seed(int(abs(price))%9999); result=[]
    for i,t in enumerate(times):
        p += p*step + np.random.normal(0,price*0.001)
        conf=price*(0.003+i*0.0005)
        result.append({"time":t,"predicted":round(p,2),"upper":round(p+conf,2),"lower":round(p-conf,2)})
    return result

def get_technicals(df):
    try:
        close  = df["Close"].squeeze().dropna()
        volume = df["Volume"].squeeze().dropna()
        high   = df["High"].squeeze().dropna()
        low    = df["Low"].squeeze().dropna()
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rsi    = safe_float((100-100/(1+gain/loss)).iloc[-1])
        ema12  = close.ewm(span=12).mean()
        ema26  = close.ewm(span=26).mean()
        macd   = safe_float((ema12-ema26).iloc[-1])
        signal = safe_float((ema12-ema26).ewm(span=9).mean().iloc[-1])
        sma20  = safe_float(close.rolling(20).mean().iloc[-1])
        sma50  = safe_float(close.rolling(50).mean().iloc[-1])
        sma200 = safe_float(close.rolling(200).mean().iloc[-1])
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = safe_float((bb_mid+2*bb_std).iloc[-1])
        bb_lower = safe_float((bb_mid-2*bb_std).iloc[-1])
        bb_width = round((bb_upper-bb_lower)/safe_float(bb_mid.iloc[-1],1)*100,2)
        avg_vol   = safe_float(volume.rolling(20).mean().iloc[-1])
        last_vol  = safe_float(volume.iloc[-1])
        vol_ratio = round(last_vol/avg_vol,2) if avg_vol else 1.0
        resistance = safe_float(high.rolling(20).max().iloc[-1])
        support    = safe_float(low.rolling(20).min().iloc[-1])
        last_close = safe_float(close.iloc[-1])
        tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
        atr = safe_float(tr.rolling(14).mean().iloc[-1])
        return {
            "rsi":rsi,"macd":macd,"macd_signal":signal,
            "sma20":sma20,"sma50":sma50,"sma200":sma200,
            "bb_upper":bb_upper,"bb_lower":bb_lower,"bb_width":bb_width,
            "volume_ratio":vol_ratio,"avg_volume":safe_int(avg_vol),
            "resistance":resistance,"support":support,
            "near_resistance":abs(last_close-resistance)/last_close<0.02,
            "near_support":abs(last_close-support)/last_close<0.02,
            "broke_resistance":last_close>=resistance*0.99,
            "broke_support":last_close<=support*1.01,
            "atr":atr,"atr_pct":round(atr/last_close*100,2) if last_close else 0,
        }
    except:
        return {"rsi":50,"macd":0,"macd_signal":0,"sma20":0,"sma50":0,"sma200":0,
                "bb_upper":0,"bb_lower":0,"bb_width":0,"volume_ratio":1.0,"avg_volume":0,
                "resistance":0,"support":0,"near_resistance":False,"near_support":False,
                "broke_resistance":False,"broke_support":False,"atr":0,"atr_pct":0}

def get_sentiment(text):
    pos=["surge","rally","gain","profit","growth","record","strong","beat","rise","high",
         "up","bull","positive","upgrade","buy","jump","soar","outperform","boost",
         "recover","rebound","peak","milestone","exceed","robust","optimistic","breakout","accumulate"]
    neg=["fall","drop","loss","decline","weak","miss","down","bear","negative","crash",
         "slump","concern","risk","downgrade","sell","plunge","underperform","cut","warn",
         "fear","worry","crisis","volatile","pressure","selloff","correction","breakdown","bearish","caution","avoid"]
    t=text.lower()
    p=sum(1 for w in pos if w in t)
    n=sum(1 for w in neg if w in t)
    total=p+n
    if total==0: return 0.0
    return round((p-n)/total,2)

def fetch_feed(feed_info, symbol=""):
    articles=[]
    try:
        resp = requests.get(feed_info["url"], headers=HEADERS, timeout=6)
        feed = feedparser.parse(resp.content)
        if not feed.entries:
            feed = feedparser.parse(feed_info["url"])
        for entry in feed.entries[:10]:
            title   = entry.get("title","")
            summary = entry.get("summary",entry.get("description",""))
            if not title: continue
            if symbol and symbol.upper() not in title.upper() and symbol.upper() not in summary.upper():
                continue
            score = get_sentiment(title+" "+summary)
            articles.append({
                "title":title,"source":feed_info["source"],"tag":feed_info.get("tag","MARKET"),
                "link":entry.get("link",""),"time":entry.get("published",""),
                "sentiment":"positive" if score>0.1 else "negative" if score<-0.1 else "neutral",
                "score":score,
            })
    except:
        pass
    return articles

def get_news_sentiment(symbol):
    """Cached news sentiment per symbol"""
    cache_key = f"news_sentiment_{symbol}"
    cached = cache_get(cache_key)
    if cached: return cached

    all_articles = []
    for feed_info in NEWS_FEEDS[:8]:
        all_articles.extend(fetch_feed(feed_info, symbol))
    scores = [a["score"] for a in all_articles]
    avg_score = round(sum(scores)/len(scores),2) if scores else 0.0
    result = (avg_score, all_articles[:5])
    cache_set(cache_key, result, TTL_NEWS)
    return result

def get_global_mood():
    """Cached global indices mood"""
    cached = cache_get("global_mood")
    if cached: return cached

    results={}; bullish_count=0; bearish_count=0; total_weight=0; weighted_pct=0.0
    weights = {k:1.0 for k in GLOBAL_INDICES}  # Equal weight for all

    for name,sym in GLOBAL_INDICES.items():
        try:
            h = yf.Ticker(sym).history(period="2d")
            if h.empty: continue
            close = h["Close"].dropna()
            if len(close)<2: continue
            price = safe_float(close.iloc[-1])
            prev  = safe_float(close.iloc[-2])
            pct   = round((price-prev)/prev*100,2) if prev else 0
            w     = weights.get(name,1.0)
            results[name] = {"price":price,"pct":pct,"bullish":pct>=0}
            weighted_pct += pct*w; total_weight += w
            if pct>=0: bullish_count+=1
            else:      bearish_count+=1
        except: continue

    avg_pct      = round(weighted_pct/total_weight,2) if total_weight else 0
    mood_score   = max(-1.0,min(1.0,round(avg_pct/2,2)))
    overall_mood = "Bullish" if avg_pct>0.3 else "Bearish" if avg_pct<-0.3 else "Neutral"
    result = {"indices":results,"avg_pct":avg_pct,"mood_score":mood_score,
              "overall_mood":overall_mood,"bullish_count":bullish_count,"bearish_count":bearish_count}
    cache_set("global_mood", result, TTL_GLOBAL)
    return result

def get_gift_nifty():
    """Cached Gift Nifty data"""
    cached = cache_get("gift_nifty")
    if cached: return cached

    try:
        hist    = yf.Ticker("^NSEI").history(period="2d",interval="1h")
        if hist.empty:
            return {"price":0,"pct":0,"expected_gap":0,"signal":"Neutral","available":False}
        close   = hist["Close"].dropna()
        price   = safe_float(close.iloc[-1])
        prev    = safe_float(close.iloc[0])
        pct     = round((price-prev)/prev*100,2) if prev else 0
        nifty_h = yf.Ticker("^NSEI").history(period="1d")
        nifty_c = safe_float(nifty_h["Close"].iloc[-1]) if not nifty_h.empty else price
        gap_pts = round(price-nifty_c,2)
        gap_pct = round(gap_pts/nifty_c*100,2) if nifty_c else 0
        signal  = "Strong Bull" if gap_pct>0.5 else "Bull" if gap_pct>0.1 else "Strong Bear" if gap_pct<-0.5 else "Bear" if gap_pct<-0.1 else "Neutral"
        result  = {"price":price,"pct":pct,"expected_gap":gap_pts,"gap_pct":gap_pct,"signal":signal,"available":True}
        cache_set("gift_nifty", result, TTL_GIFT)
        return result
    except Exception as e:
        return {"price":0,"pct":0,"expected_gap":0,"signal":"Neutral","available":False,"error":str(e)}

def get_sector_momentum(sector):
    """Cached sector momentum"""
    cache_key = f"sector_{sector}"
    cached = cache_get(cache_key)
    if cached: return cached

    try:
        sym = SECTOR_INDICES.get(sector)
        if not sym: return 0.0,"Neutral"
        h = yf.Ticker(sym).history(period="5d")
        if h.empty: return 0.0,"Neutral"
        close = h["Close"].dropna()
        if len(close)<2: return 0.0,"Neutral"
        pct_5d   = round((close.iloc[-1]-close.iloc[0])/close.iloc[0]*100,2)
        momentum = "Strong Bull" if pct_5d>2 else "Bull" if pct_5d>0.5 else "Strong Bear" if pct_5d<-2 else "Bear" if pct_5d<-0.5 else "Neutral"
        result   = (pct_5d, momentum)
        cache_set(cache_key, result, TTL_SECTOR)
        return result
    except:
        return 0.0,"Neutral"

def get_fii_dii_sentiment():
    """Cached FII/DII approximation"""
    cached = cache_get("fii_dii")
    if cached: return cached

    try:
        hist  = yf.Ticker("^NSEI").history(period="5d")
        if hist.empty:
            return {"fii_sentiment":"Neutral","fii_score":0}
        close = hist["Close"].dropna()
        vol   = hist["Volume"].dropna()
        price_trend = (close.iloc[-1]-close.iloc[0])/close.iloc[0]*100
        vol_trend   = (vol.iloc[-1]-vol.mean())/vol.mean()*100 if vol.mean() else 0
        fii_score   = max(-1.0,min(1.0,round((price_trend*0.6+vol_trend*0.4)/10,2)))
        sentiment   = "Buying" if fii_score>0.1 else "Selling" if fii_score<-0.1 else "Neutral"
        result = {"fii_sentiment":sentiment,"fii_score":fii_score,
                  "price_trend_5d":round(price_trend,2),"volume_trend":round(vol_trend,2)}
        cache_set("fii_dii", result, TTL_FII)
        return result
    except:
        return {"fii_sentiment":"Neutral","fii_score":0}

def get_options_oi(symbol):
    """Cached options OI"""
    cache_key = f"oi_{symbol}"
    cached = cache_get(cache_key)
    if cached: return cached

    try:
        ticker      = yf.Ticker(nse(symbol))
        expirations = ticker.options
        if not expirations:
            return {"pcr":1.0,"max_pain":0,"oi_signal":"Neutral"}
        chain         = ticker.option_chain(expirations[0])
        calls,puts    = chain.calls, chain.puts
        total_call_oi = calls["openInterest"].sum() if "openInterest" in calls else 0
        total_put_oi  = puts["openInterest"].sum()  if "openInterest" in puts  else 0
        pcr           = round(total_put_oi/total_call_oi,2) if total_call_oi>0 else 1.0
        max_pain = 0
        try:
            strikes  = calls["strike"].tolist()
            min_pain = float("inf")
            for s in strikes:
                cp = ((calls[calls["strike"]<=s]["strike"]-s).abs()*calls[calls["strike"]<=s]["openInterest"]).sum()
                pp = ((puts[puts["strike"]>=s]["strike"]-s).abs()*puts[puts["strike"]>=s]["openInterest"]).sum()
                if cp+pp < min_pain:
                    min_pain=cp+pp; max_pain=s
        except: pass
        oi_signal = "Bullish" if pcr>1.2 else "Bearish" if pcr<0.7 else "Neutral"
        result = {"pcr":pcr,"max_pain":max_pain,"oi_signal":oi_signal,
                  "call_oi":safe_int(total_call_oi),"put_oi":safe_int(total_put_oi),
                  "expiry":expirations[0]}
        cache_set(cache_key, result, TTL_GLOBAL)
        return result
    except:
        return {"pcr":1.0,"max_pain":0,"oi_signal":"Neutral","call_oi":0,"put_oi":0}

def compute_confidence(bullish,tech,news_score,global_mood_score,sector_pct,fii_score,oi_signal,price,gift_nifty_pct=0):
    score = 50
    score += 5 if bullish else -5
    rsi = tech.get("rsi",50)
    if bullish: score += 8 if 40<=rsi<=60 else 4 if rsi<70 else -5
    else:       score += 8 if rsi>60 else 4 if rsi>50 else -5
    if tech.get("macd",0)>tech.get("macd_signal",0): score += 8 if bullish else -4
    else: score += -4 if bullish else 8
    price_val = price
    if price_val>tech.get("sma20",0):  score += 5 if bullish else -3
    if price_val>tech.get("sma50",0):  score += 5 if bullish else -3
    if price_val>tech.get("sma200",0): score += 4 if bullish else -2
    bb_upper=tech.get("bb_upper",0); bb_lower=tech.get("bb_lower",0)
    if bb_upper and bb_lower:
        bb_pos=(price_val-bb_lower)/(bb_upper-bb_lower) if (bb_upper-bb_lower)>0 else 0.5
        if bullish and bb_pos<0.5:    score+=6
        elif bullish and bb_pos>0.8:  score-=4
        elif not bullish and bb_pos>0.5: score+=6
        elif not bullish and bb_pos<0.2: score-=4
    vol_ratio=tech.get("volume_ratio",1.0)
    if vol_ratio>1.5:   score+=8
    elif vol_ratio>1.2: score+=4
    elif vol_ratio<0.7: score-=3
    if tech.get("broke_resistance") and bullish:     score+=10
    if tech.get("broke_support") and not bullish:    score+=10
    if tech.get("near_resistance") and bullish:      score-=5
    if tech.get("near_support") and not bullish:     score-=5
    if news_score>0.3:    score+=8 if bullish else -4
    elif news_score>0.1:  score+=4 if bullish else -2
    elif news_score<-0.3: score-=4 if bullish else 8
    elif news_score<-0.1: score-=2 if bullish else 4
    if global_mood_score>0.3:    score+=6 if bullish else -3
    elif global_mood_score>0.1:  score+=3 if bullish else -1
    elif global_mood_score<-0.3: score-=3 if bullish else 6
    elif global_mood_score<-0.1: score-=1 if bullish else 3
    if gift_nifty_pct>0.5:    score+=6 if bullish else -3
    elif gift_nifty_pct>0.1:  score+=3 if bullish else -1
    elif gift_nifty_pct<-0.5: score-=3 if bullish else 6
    elif gift_nifty_pct<-0.1: score-=1 if bullish else 3
    if sector_pct>2:     score+=6 if bullish else -3
    elif sector_pct>0.5: score+=3 if bullish else -1
    elif sector_pct<-2:  score-=3 if bullish else 6
    elif sector_pct<-0.5:score-=1 if bullish else 3
    if fii_score>0.2:    score+=5 if bullish else -2
    elif fii_score<-0.2: score-=2 if bullish else 5
    if oi_signal=="Bullish":   score+=5 if bullish else -2
    elif oi_signal=="Bearish": score-=2 if bullish else 5
    return min(97,max(40,round(score)))

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status":"MarketCast API is running!","cache_keys":len(_cache)}

@app.get("/cache-status")
def cache_status():
    """Show what's cached and when it expires"""
    now = time.time()
    result = {}
    for key,(value,expiry) in _cache.items():
        remaining = round(expiry-now)
        result[key] = {"expires_in_seconds":remaining,"fresh":remaining>0}
    return {"cached_items":len(_cache),"items":result}

@app.get("/cache-clear")
def cache_clear():
    """Clear all cache manually"""
    _cache.clear()
    return {"status":"Cache cleared","message":"All data will be fetched fresh on next request"}

@app.get("/status")
def get_market_status():
    cached = cache_get("market_status")
    if cached: return cached
    try:
        ticker  = yf.Ticker("^NSEI")
        hist    = ticker.history(period="1d",interval="1m")
        now_ist = datetime.utcnow()+timedelta(hours=5,minutes=30)
        is_weekday   = now_ist.weekday()<5
        market_open  = now_ist.replace(hour=9,minute=15,second=0,microsecond=0)
        market_close = now_ist.replace(hour=15,minute=30,second=0,microsecond=0)
        is_open = is_weekday and market_open<=now_ist<=market_close
        if hist.empty:
            result = {"open":is_open,"status":"Open" if is_open else "Closed","time_ist":now_ist.strftime("%I:%M %p"),"price":0,"change":0,"pct":0}
        else:
            last_price = safe_float(hist["Close"].iloc[-1])
            prev_price = safe_float(hist["Close"].iloc[0])
            change     = round(last_price-prev_price,2)
            pct        = round((change/prev_price)*100,2) if prev_price else 0
            result     = {"open":is_open,"status":"Open" if is_open else "Closed","time_ist":now_ist.strftime("%I:%M %p"),"price":last_price,"change":change,"pct":pct}
        cache_set("market_status", result, TTL_STATUS)
        return result
    except Exception as e:
        now_ist = datetime.utcnow()+timedelta(hours=5,minutes=30)
        is_weekday   = now_ist.weekday()<5
        market_open  = now_ist.replace(hour=9,minute=15,second=0,microsecond=0)
        market_close = now_ist.replace(hour=15,minute=30,second=0,microsecond=0)
        is_open = is_weekday and market_open<=now_ist<=market_close
        return {"open":is_open,"status":"Open" if is_open else "Closed","time_ist":now_ist.strftime("%I:%M %p"),"error":str(e)}

@app.get("/gift-nifty")
def gift_nifty():
    return get_gift_nifty()

@app.get("/global-mood")
def global_mood():
    return get_global_mood()

@app.get("/fii-dii")
def fii_dii():
    return get_fii_dii_sentiment()

@app.get("/options-oi/{symbol}")
def options_oi(symbol:str):
    return get_options_oi(symbol)

@app.get("/sector-momentum")
def sector_momentum():
    cached = cache_get("all_sectors")
    if cached: return cached
    result = {}
    for sector,sym in SECTOR_INDICES.items():
        pct,momentum = get_sector_momentum(sector)
        result[sector] = {"pct":pct,"momentum":momentum,"symbol":sym}
    cache_set("all_sectors", result, TTL_SECTOR)
    return result

@app.get("/stock/{symbol}")
def get_stock(symbol:str):
    try:
        ticker = yf.Ticker(nse(symbol))
        hist   = ticker.history(period="6mo")
        if hist.empty: return {"error":"Symbol not found"}
        close  = hist["Close"].dropna()
        high   = hist["High"].dropna()
        low    = hist["Low"].dropna()
        volume = hist["Volume"].dropna()
        price  = safe_float(close.iloc[-1])
        prev   = safe_float(close.iloc[-2])
        change = round(price-prev,2)
        pct    = round((change/prev)*100,2) if prev else 0
        prices = close.tolist()
        dates  = forecast_dates(30)
        fc30   = arima_forecast(prices,30)
        fc7    = {k:v[:7] for k,v in fc30.items()}
        tech   = get_technicals(hist)
        bullish= fc30["predicted"][0]>price

        # All cached — super fast from 2nd call onwards
        news_score,news_articles = get_news_sentiment(symbol)
        global_data  = get_global_mood()
        global_score = global_data.get("mood_score",0)
        gift_data    = get_gift_nifty()
        gift_pct     = gift_data.get("pct",0)
        ticker_info  = ticker.info
        sector       = ticker_info.get("sector","")
        sector_pct,sector_mood = get_sector_momentum(sector)
        fii_data     = get_fii_dii_sentiment()
        fii_score    = fii_data.get("fii_score",0)
        oi_data      = get_options_oi(symbol)
        oi_signal    = oi_data.get("oi_signal","Neutral")

        confidence = compute_confidence(bullish,tech,news_score,global_score,sector_pct,fii_score,oi_signal,price,gift_nifty_pct=gift_pct)

        reasons = []
        if bullish: reasons.append("ARIMA predicts upward movement")
        else:       reasons.append("ARIMA predicts downward movement")
        if tech["macd"]>tech["macd_signal"]: reasons.append("MACD bullish crossover")
        else: reasons.append("MACD bearish crossover")
        reasons.append(f"RSI at {tech['rsi']:.0f} — {'overbought' if tech['rsi']>70 else 'oversold' if tech['rsi']<30 else 'neutral'}")
        if tech.get("broke_resistance"): reasons.append("Price broke above resistance — strong bullish signal")
        elif tech.get("broke_support"):  reasons.append("Price broke below support — strong bearish signal")
        if tech.get("volume_ratio",1)>1.5: reasons.append(f"High volume ({tech['volume_ratio']:.1f}x avg) confirms move")
        if news_score>0.1: reasons.append(f"News sentiment positive ({news_score:+.2f})")
        elif news_score<-0.1: reasons.append(f"News sentiment negative ({news_score:+.2f})")
        if global_score>0.2: reasons.append(f"Global markets bullish (avg {global_data.get('avg_pct',0):+.2f}%)")
        elif global_score<-0.2: reasons.append(f"Global markets bearish (avg {global_data.get('avg_pct',0):+.2f}%)")
        if gift_pct>0.1: reasons.append(f"Gift Nifty positive ({gift_pct:+.2f}%) — bullish opening expected")
        elif gift_pct<-0.1: reasons.append(f"Gift Nifty negative ({gift_pct:+.2f}%) — bearish opening expected")
        if sector_pct!=0: reasons.append(f"Sector momentum: {sector_mood} ({sector_pct:+.2f}%)")
        reasons.append(f"FII activity: {fii_data.get('fii_sentiment','Neutral')}")
        reasons.append(f"Options OI: {oi_signal} (PCR: {oi_data.get('pcr',1.0):.2f})")

        return {
            "symbol":symbol.upper(),"price":price,"change":change,"pct":pct,
            "high":safe_float(high.iloc[-1]),"low":safe_float(low.iloc[-1]),
            "volume":safe_int(volume.iloc[-1]),
            "week52high":safe_float(high.max()),"week52low":safe_float(low.min()),
            "sector":sector,"technicals":tech,
            "forecast":{
                "intraday":intraday_forecast(price,pct),
                "day7":{"dates":dates[:7],"data":fc7},
                "day30":{"dates":dates,"data":fc30},
            },
            "recommendation":{
                "action":"BUY" if bullish else "SELL","confidence":confidence,
                "entry":round(price*(1.002 if bullish else 0.998),2),
                "target":round(price*(1.08 if bullish else 0.92),2),
                "stop_loss":round(price*(0.96 if bullish else 1.04),2),
                "bullish":bullish,"reasons":reasons,
            },
            "signals":{
                "news_sentiment":news_score,"global_mood":global_data.get("overall_mood","Neutral"),
                "global_avg_pct":global_data.get("avg_pct",0),
                "gift_nifty_pct":gift_pct,"gift_nifty_signal":gift_data.get("signal","Neutral"),
                "gift_nifty_gap":gift_data.get("expected_gap",0),
                "sector_momentum":sector_mood,"sector_pct":sector_pct,
                "fii_sentiment":fii_data.get("fii_sentiment","Neutral"),
                "options_pcr":oi_data.get("pcr",1.0),"options_signal":oi_signal,
                "volume_ratio":tech.get("volume_ratio",1.0),
                "broke_resistance":tech.get("broke_resistance",False),
                "broke_support":tech.get("broke_support",False),
            },
            "options_oi":oi_data,"news_headlines":news_articles,
        }
    except Exception as e:
        return {"error":str(e)}

@app.get("/market")
def get_market():
    cached = cache_get("market_prices")
    if cached: return cached
    symbols={"NIFTY":"^NSEI","SENSEX":"^BSESN","BANKNIFTY":"^NSEBANK","NIFTYIT":"^CNXIT","DOW":"^DJI","NASDAQ":"^IXIC"}
    result={}
    for name,sym in symbols.items():
        try:
            h=yf.Ticker(sym).history(period="2d")
            if h.empty: continue
            close=h["Close"].dropna()
            price=safe_float(close.iloc[-1]); prev=safe_float(close.iloc[-2])
            chg=round(price-prev,2)
            result[name]={"price":price,"change":chg,"pct":round((chg/prev)*100,2) if prev else 0}
        except: continue
    cache_set("market_prices", result, TTL_MARKET)
    return result

@app.get("/news")
def get_news(symbol:str="", tag:str="", limit:int=50):
    cache_key = f"news_{symbol}_{tag}_{limit}"
    cached = cache_get(cache_key)
    if cached: return cached
    seen=set(); articles=[]
    for feed_info in NEWS_FEEDS:
        if tag and feed_info.get("tag","").upper()!=tag.upper(): continue
        for a in fetch_feed(feed_info,symbol):
            if a["title"] not in seen:
                seen.add(a["title"]); articles.append(a)
        if len(articles)>=limit: break
    articles=sorted(articles,key=lambda x:abs(x["score"]),reverse=True)
    result={"news":articles[:limit],"count":len(articles[:limit]),"sources":list(set(a["source"] for a in articles[:limit]))}
    cache_set(cache_key, result, TTL_NEWS)
    return result

@app.get("/fo/{symbol}")
def get_fo(symbol:str):
    try:
        ticker=yf.Ticker(nse(symbol))
        hist=ticker.history(period="3mo")
        if hist.empty: return {"error":"Symbol not found"}
        close=hist["Close"].dropna()
        price=safe_float(close.iloc[-1]); prev=safe_float(close.iloc[-2])
        change=round(price-prev,2); pct=round((change/prev)*100,2) if prev else 0
        prices=close.tolist(); dates=forecast_dates(30)
        fc30=arima_forecast(prices,30); fc7={k:v[:7] for k,v in fc30.items()}
        tech=get_technicals(hist); bullish=fc30["predicted"][0]>price

        # All cached
        news_score,_ = get_news_sentiment(symbol)
        global_data  = get_global_mood()
        global_score = global_data.get("mood_score",0)
        gift_data    = get_gift_nifty()
        gift_pct     = gift_data.get("pct",0)
        fii_data     = get_fii_dii_sentiment()
        fii_score    = fii_data.get("fii_score",0)
        oi_data      = get_options_oi(symbol)
        oi_signal    = oi_data.get("oi_signal","Neutral")
        confidence   = compute_confidence(bullish,tech,news_score,global_score,0,fii_score,oi_signal,price,gift_nifty_pct=gift_pct)
        atm=round(price/100)*100; step=100

        return {
            "symbol":symbol.upper(),"price":price,"change":change,"pct":pct,
            "bullish":bullish,"confidence":confidence,
            "expected_move":round(fc30["predicted"][0]-price,2),"atm":atm,
            "forecast":{
                "intraday":intraday_forecast(price,pct),
                "day7":{"dates":dates[:7],"data":fc7},
                "day30":{"dates":dates,"data":fc30},
            },
            "strategy":{
                "action":"BUY CALL (CE)" if bullish else "BUY PUT (PE)","strike":atm,
                "target":atm+(step*2) if bullish else atm-(step*2),
                "stop_loss":atm-(step*2) if bullish else atm+(step*2),
            },
            "levels":{
                "resistance2":round(price*1.04,2),"resistance1":round(price*1.02,2),
                "support1":round(price*0.98,2),"support2":round(price*0.96,2),
            },
            "signals":{
                "news_sentiment":news_score,"global_mood":global_data.get("overall_mood","Neutral"),
                "gift_nifty_pct":gift_pct,"gift_nifty_signal":gift_data.get("signal","Neutral"),
                "gift_nifty_gap":gift_data.get("expected_gap",0),
                "fii_sentiment":fii_data.get("fii_sentiment","Neutral"),
                "options_pcr":oi_data.get("pcr",1.0),"options_signal":oi_signal,
                "volume_ratio":tech.get("volume_ratio",1.0),
                "broke_resistance":tech.get("broke_resistance",False),
                "broke_support":tech.get("broke_support",False),
            },
            "options_oi":oi_data,"technicals":tech,
        }
    except Exception as e:
        return {"error":str(e)}

if __name__ == "__main__":
    port=int(os.environ.get("PORT",8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
