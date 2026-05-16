import os
import uvicorn
import warnings
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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

# 25+ news sources covering all market segments
NEWS_FEEDS = [
    # Yahoo Finance - Multiple feeds
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^NSEI&region=IN&lang=en-US",     "source": "Yahoo Finance", "tag": "NIFTY"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^BSESN&region=IN&lang=en-US",    "source": "Yahoo Finance", "tag": "SENSEX"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=RELIANCE.NS&region=IN&lang=en-US","source": "Yahoo Finance", "tag": "RELIANCE"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TCS.NS&region=IN&lang=en-US",    "source": "Yahoo Finance", "tag": "TCS"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=HDFCBANK.NS&region=IN&lang=en-US","source": "Yahoo Finance", "tag": "HDFCBANK"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=INFY.NS&region=IN&lang=en-US",   "source": "Yahoo Finance", "tag": "INFY"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SBIN.NS&region=IN&lang=en-US",   "source": "Yahoo Finance", "tag": "SBIN"},
    # Economic Times
    {"url": "https://economictimes.indiatimes.com/markets/stocks/rss.cms",         "source": "ET Markets",    "tag": "STOCKS"},
    {"url": "https://economictimes.indiatimes.com/markets/rss.cms",                "source": "ET Markets",    "tag": "MARKET"},
    {"url": "https://economictimes.indiatimes.com/markets/mutual-funds/rss.cms",   "source": "ET Markets",    "tag": "MF"},
    {"url": "https://economictimes.indiatimes.com/markets/commodities/rss.cms",    "source": "ET Markets",    "tag": "COMMODITIES"},
    {"url": "https://economictimes.indiatimes.com/markets/forex/rss.cms",          "source": "ET Markets",    "tag": "FOREX"},
    {"url": "https://economictimes.indiatimes.com/news/economy/rss.cms",           "source": "ET Economy",    "tag": "ECONOMY"},
    # Live Mint
    {"url": "https://www.livemint.com/rss/markets",                                "source": "Live Mint",     "tag": "MARKET"},
    {"url": "https://www.livemint.com/rss/companies",                              "source": "Live Mint",     "tag": "STOCKS"},
    {"url": "https://www.livemint.com/rss/money",                                  "source": "Live Mint",     "tag": "ECONOMY"},
    {"url": "https://www.livemint.com/rss/industry",                               "source": "Live Mint",     "tag": "SECTOR"},
    # Business Standard
    {"url": "https://www.business-standard.com/rss/markets-106.rss",              "source": "Business Std",  "tag": "MARKET"},
    {"url": "https://www.business-standard.com/rss/finance-109.rss",              "source": "Business Std",  "tag": "FINANCE"},
    {"url": "https://www.business-standard.com/rss/economy-policy-102.rss",       "source": "Business Std",  "tag": "ECONOMY"},
    {"url": "https://www.business-standard.com/rss/companies-101.rss",            "source": "Business Std",  "tag": "STOCKS"},
    # The Hindu Business
    {"url": "https://www.thehindu.com/business/markets/?service=rss",             "source": "Hindu Business", "tag": "MARKET"},
    {"url": "https://www.thehindu.com/business/?service=rss",                     "source": "Hindu Business", "tag": "ECONOMY"},
    # Reuters
    {"url": "https://feeds.reuters.com/reuters/INbusinessNews",                    "source": "Reuters India",  "tag": "GLOBAL"},
    {"url": "https://feeds.reuters.com/reuters/businessNews",                      "source": "Reuters",        "tag": "GLOBAL"},
    # Financial Express
    {"url": "https://www.financialexpress.com/market/feed/",                       "source": "Financial Express","tag": "MARKET"},
    # CNBC TV18
    {"url": "https://www.cnbctv18.com/commonfeeds/v1/eng/rss/market.xml",         "source": "CNBC TV18",      "tag": "MARKET"},
    {"url": "https://www.cnbctv18.com/commonfeeds/v1/eng/rss/economy.xml",        "source": "CNBC TV18",      "tag": "ECONOMY"},
    # Moneycontrol
    {"url": "https://www.moneycontrol.com/rss/MCtopnews.xml",                     "source": "Moneycontrol",   "tag": "MARKET"},
    {"url": "https://www.moneycontrol.com/rss/marketreports.xml",                 "source": "Moneycontrol",   "tag": "MARKET"},
    {"url": "https://www.moneycontrol.com/rss/stocksmarket.xml",                  "source": "Moneycontrol",   "tag": "STOCKS"},
    {"url": "https://www.moneycontrol.com/rss/results.xml",                       "source": "Moneycontrol",   "tag": "RESULTS"},
    {"url": "https://www.moneycontrol.com/rss/economy.xml",                       "source": "Moneycontrol",   "tag": "ECONOMY"},
    {"url": "https://www.moneycontrol.com/rss/ipo.xml",                           "source": "Moneycontrol",   "tag": "IPO"},
]

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
        result.append({
            "time":t,
            "predicted":round(p,2),
            "upper":round(p+conf,2),
            "lower":round(p-conf,2),
        })
    return result

def get_technicals(df):
    try:
        close = df["Close"].squeeze().dropna()
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain/loss
        rsi   = safe_float((100-100/(1+rs)).iloc[-1])
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = safe_float((ema12-ema26).iloc[-1])
        signal= safe_float((ema12-ema26).ewm(span=9).mean().iloc[-1])
        sma20 = safe_float(close.rolling(20).mean().iloc[-1])
        sma50 = safe_float(close.rolling(50).mean().iloc[-1])
        bb_mid= close.rolling(20).mean()
        bb_std= close.rolling(20).std()
        return {
            "rsi":rsi,"macd":macd,"macd_signal":signal,
            "sma20":sma20,"sma50":sma50,
            "bb_upper":safe_float((bb_mid+2*bb_std).iloc[-1]),
            "bb_lower":safe_float((bb_mid-2*bb_std).iloc[-1]),
        }
    except:
        return {"rsi":50,"macd":0,"macd_signal":0,"sma20":0,"sma50":0,"bb_upper":0,"bb_lower":0}

def get_sentiment(text):
    pos=["surge","rally","gain","profit","growth","record","strong","beat",
         "rise","high","up","bull","positive","upgrade","buy","jump","soar",
         "outperform","boost","recover","rebound","peak","milestone","exceed"]
    neg=["fall","drop","loss","decline","weak","miss","down","bear",
         "negative","crash","slump","concern","risk","downgrade","sell","plunge",
         "underperform","cut","warn","fear","worry","crisis","volatile","pressure"]
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
            summary = entry.get("summary", entry.get("description",""))
            if not title: continue
            if symbol and symbol.upper() not in title.upper() and symbol.upper() not in summary.upper():
                continue
            score = get_sentiment(title+" "+summary)
            articles.append({
                "title":     title,
                "source":    feed_info["source"],
                "tag":       feed_info.get("tag","MARKET"),
                "link":      entry.get("link",""),
                "time":      entry.get("published",""),
                "sentiment": "positive" if score>0.1 else "negative" if score<-0.1 else "neutral",
                "score":     score,
            })
    except:
        pass
    return articles

@app.get("/")
def root():
    return {"status":"MarketCast API is running!"}

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
        confidence=min(95,max(55,50
            +(10 if bullish and tech["rsi"]<60 else -5)
            +(10 if tech["macd"]>tech["macd_signal"] else -5)
            +(10 if price>tech["sma20"] else -5)
            +(5  if price>tech["sma50"] else -3)))
        return {
            "symbol":symbol.upper(),"price":price,"change":change,"pct":pct,
            "high":safe_float(high.iloc[-1]),"low":safe_float(low.iloc[-1]),
            "volume":safe_int(volume.iloc[-1]),
            "week52high":safe_float(high.max()),"week52low":safe_float(low.min()),
            "technicals":tech,
            "forecast":{
                "intraday":intraday_forecast(price,pct),
                "day7":{"dates":dates[:7],"data":fc7},
                "day30":{"dates":dates,"data":fc30},
            },
            "recommendation":{
                "action":"BUY" if bullish else "SELL",
                "confidence":confidence,
                "entry":round(price*(1.002 if bullish else 0.998),2),
                "target":round(price*(1.08 if bullish else 0.92),2),
                "stop_loss":round(price*(0.96 if bullish else 1.04),2),
                "bullish":bullish,
            }
        }
    except Exception as e:
        return {"error":str(e)}

@app.get("/market")
def get_market():
    symbols={"NIFTY":"^NSEI","SENSEX":"^BSESN","BANKNIFTY":"^NSEBANK",
             "NIFTYIT":"^CNXIT","DOW":"^DJI","NASDAQ":"^IXIC"}
    result={}
    for name,sym in symbols.items():
        try:
            h=yf.Ticker(sym).history(period="2d")
            if h.empty: continue
            close=h["Close"].dropna()
            price=safe_float(close.iloc[-1])
            prev=safe_float(close.iloc[-2])
            chg=round(price-prev,2)
            result[name]={"price":price,"change":chg,"pct":round((chg/prev)*100,2) if prev else 0}
        except: continue
    return result

@app.get("/news")
def get_news(symbol:str="", tag:str="", limit:int=50):
    seen   = set()
    articles = []
    for feed_info in NEWS_FEEDS:
        # Filter by tag if provided
        if tag and feed_info.get("tag","").upper() != tag.upper():
            continue
        fetched = fetch_feed(feed_info, symbol)
        for a in fetched:
            if a["title"] not in seen:
                seen.add(a["title"])
                articles.append(a)
        if len(articles) >= limit:
            break
    # Sort: most recent sentiment first
    articles = sorted(articles, key=lambda x: abs(x["score"]), reverse=True)
    return {
        "news":    articles[:limit],
        "count":   len(articles[:limit]),
        "sources": list(set(a["source"] for a in articles[:limit])),
    }

@app.get("/fo/{symbol}")
def get_fo(symbol:str):
    try:
        ticker=yf.Ticker(nse(symbol))
        hist=ticker.history(period="3mo")
        if hist.empty: return {"error":"Symbol not found"}
        close=hist["Close"].dropna()
        price=safe_float(close.iloc[-1])
        prev=safe_float(close.iloc[-2])
        change=round(price-prev,2)
        pct=round((change/prev)*100,2) if prev else 0
        prices=close.tolist()
        dates=forecast_dates(30)
        fc30=arima_forecast(prices,30)
        fc7={k:v[:7] for k,v in fc30.items()}
        bullish=fc30["predicted"][0]>price
        atm=round(price/100)*100
        step=100
        return {
            "symbol":symbol.upper(),"price":price,"change":change,"pct":pct,
            "bullish":bullish,"expected_move":round(fc30["predicted"][0]-price,2),"atm":atm,
            "forecast":{
                "intraday":intraday_forecast(price,pct),
                "day7":{"dates":dates[:7],"data":fc7},
                "day30":{"dates":dates,"data":fc30},
            },
            "strategy":{
                "action":"BUY CALL (CE)" if bullish else "BUY PUT (PE)",
                "strike":atm,
                "target":atm+(step*2) if bullish else atm-(step*2),
                "stop_loss":atm-(step*2) if bullish else atm+(step*2),
            },
            "levels":{
                "resistance2":round(price*1.04,2),"resistance1":round(price*1.02,2),
                "support1":round(price*0.98,2),"support2":round(price*0.96,2),
            }
        }
    except Exception as e:
        return {"error":str(e)}

if __name__ == "__main__":
    port=int(os.environ.get("PORT",8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
