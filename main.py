import os
import uvicorn
import warnings
import numpy as np
import pandas as pd
import feedparser
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

app = FastAPI(title="MarketCast API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def nse(symbol):
    m = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN",
         "FINNIFTY":"NIFTY_FIN_SERVICE.NS","MIDCPNIFTY":"^NSEMDCP50"}
    return m.get(symbol.upper(), f"{symbol.upper()}.NS")

def arima_forecast(prices, days=30):
    try:
        model = ARIMA(prices, order=(2,1,2))
        fit = model.fit()
        fc = fit.forecast(steps=days)
        ci = fit.get_forecast(steps=days).conf_int(alpha=0.2)
        return {
            "predicted": [round(float(v),2) for v in fc],
            "upper": [round(float(v),2) for v in ci.iloc[:,1]],
            "lower": [round(float(v),2) for v in ci.iloc[:,0]],
        }
    except:
        last = prices[-1]
        trend = (prices[-1]-prices[-5])/5 if len(prices)>=5 else 0
        pred = [round(last+trend*i,2) for i in range(1,days+1)]
        return {"predicted":pred,"upper":[round(p*1.03,2) for p in pred],"lower":[round(p*0.97,2) for p in pred]}

def forecast_dates(days):
    dates=[]; d=datetime.today()
    while len(dates)<days:
        d+=timedelta(days=1)
        if d.weekday()<5: dates.append(d.strftime("%d %b"))
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
    close=df["Close"].squeeze()
    delta=close.diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    rsi=round(float((100-100/(1+gain/loss)).iloc[-1]),2)
    ema12=close.ewm(span=12).mean(); ema26=close.ewm(span=26).mean()
    macd=round(float((ema12-ema26).iloc[-1]),2)
    signal=round(float((ema12-ema26).ewm(span=9).mean().iloc[-1]),2)
    sma20=round(float(close.rolling(20).mean().iloc[-1]),2)
    sma50=round(float(close.rolling(50).mean().iloc[-1]),2)
    bb_mid=close.rolling(20).mean(); bb_std=close.rolling(20).std()
    return {
        "rsi":rsi,"macd":macd,"macd_signal":signal,"sma20":sma20,"sma50":sma50,
        "bb_upper":round(float((bb_mid+2*bb_std).iloc[-1]),2),
        "bb_lower":round(float((bb_mid-2*bb_std).iloc[-1]),2),
    }

def get_sentiment(text):
    pos=["surge","rally","gain","profit","growth","record","strong","beat","rise","high","up","bull","positive","upgrade","buy"]
    neg=["fall","drop","loss","decline","weak","miss","down","bear","negative","crash","slump","concern","risk","downgrade","sell"]
    t=text.lower()
    p=sum(1 for w in pos if w in t); n=sum(1 for w in neg if w in t)
    total=p+n
    if total==0: return 0.0
    return round((p-n)/total,2)

@app.get("/")
def root():
    return {"status":"MarketCast API is running!"}

@app.get("/stock/{symbol}")
def get_stock(symbol:str):
    try:
        ticker=yf.Ticker(nse(symbol))
        hist=ticker.history(period="6mo")
        if hist.empty: return {"error":"Symbol not found"}
        price=round(float(hist["Close"].iloc[-1]),2)
        prev=round(float(hist["Close"].iloc[-2]),2)
        change=round(price-prev,2)
        pct=round((change/prev)*100,2)
        prices=hist["Close"].tolist()
        dates=forecast_dates(30)
        fc30=arima_forecast(prices,30)
        fc7={k:v[:7] for k,v in fc30.items()}
        tech=get_technicals(hist)
        bullish=fc30["predicted"][0]>price
        confidence=min(95,max(55,50
            +(10 if bullish and tech["rsi"]<60 else -5)
            +(10 if tech["macd"]>tech["macd_signal"] else -5)
            +(10 if price>tech["sma20"] else -5)
            +(5 if price>tech["sma50"] else -3)))
        return {
            "symbol":symbol.upper(),"price":price,"change":change,"pct":pct,
            "high":round(float(hist["High"].iloc[-1]),2),
            "low":round(float(hist["Low"].iloc[-1]),2),
            "volume":int(hist["Volume"].iloc[-1]),
            "week52high":round(float(hist["High"].max()),2),
            "week52low":round(float(hist["Low"].min()),2),
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
            price=round(float(h["Close"].iloc[-1]),2)
            prev=round(float(h["Close"].iloc[-2]),2)
            chg=round(price-prev,2)
            result[name]={"price":price,"change":chg,"pct":round((chg/prev)*100,2)}
        except: continue
    return result

@app.get("/news")
def get_news(symbol:str=""):
    feeds=["https://www.moneycontrol.com/rss/MCtopnews.xml",
           "https://economictimes.indiatimes.com/markets/rss.cms",
           "https://feeds.reuters.com/reuters/INbusinessNews"]
    articles=[]
    for url in feeds:
        try:
            feed=feedparser.parse(url)
            for entry in feed.entries[:6]:
                title=entry.get("title","")
                if symbol and symbol.upper() not in title.upper(): continue
                score=get_sentiment(title)
                articles.append({
                    "title":title,
                    "source":feed.feed.get("title","News"),
                    "link":entry.get("link",""),
                    "time":entry.get("published",""),
                    "sentiment":"positive" if score>0.1 else "negative" if score<-0.1 else "neutral",
                    "score":score,
                })
        except: continue
    return {"news":articles,"count":len(articles)}

@app.get("/fo/{symbol}")
def get_fo(symbol:str):
    try:
        ticker=yf.Ticker(nse(symbol))
        hist=ticker.history(period="3mo")
        if hist.empty: return {"error":"Symbol not found"}
        price=round(float(hist["Close"].iloc[-1]),2)
        prev=round(float(hist["Close"].iloc[-2]),2)
        change=round(price-prev,2)
        pct=round((change/prev)*100,2)
        prices=hist["Close"].tolist()
        dates=forecast_dates(30)
        fc30=arima_forecast(prices,30)
        fc7={k:v[:7] for k,v in fc30.items()}
        bullish=fc30["predicted"][0]>price
        atm=round(price/100)*100
        step=100
        return {
            "symbol":symbol.upper(),"price":price,"change":change,"pct":pct,
            "bullish":bullish,
            "expected_move":round(fc30["predicted"][0]-price,2),
            "atm":atm,
            "forecast":{
                "intraday":intraday_forecast(price,pct),
                "day7":{"dates":dates[:7],"data":fc7},
                "day30":{"dates":dates,"data":fc30},
            },
            "strategy":{
                "action":"BUY CALL (CE)" if bullish else "BUY PUT (PE)",
                "strike":atm,
                "target":atm+(step*2 if bullish else -(step*2)),
                "stop_loss":atm-(step*2 if bullish else -(step*2)),
            },
            "levels":{
                "resistance2":round(price*1.04,2),
                "resistance1":round(price*1.02,2),
                "support1":round(price*0.98,2),
                "support2":round(price*0.96,2),
            }
        }
    except Exception as e:
        return {"error":str(e)}

if __name__ == "__main__":
    port=int(os.environ.get("PORT",8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
