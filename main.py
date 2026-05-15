from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import feedparser
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="MarketCast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def nse(symbol):
    index_map = {
        "NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK",
        "SENSEX":"^BSESN","FINNIFTY":"NIFTY_FIN_SERVICE.NS",
    }
    if symbol.upper() in index_map:
        return index_map[symbol.upper()]
    return f"{symbol.upper()}.NS"

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
        return {
            "predicted": pred,
            "upper": [round(p*1.03,2) for p in pred],
            "lower": [round(p*0.97,2) for p in pred],
        }

def forecast_dates(days):
    dates = []
    d = datetime.today()
    while len(dates) < days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d.strftime("%d %b"))
    return dates

def intraday_forecast(last_price, pct_change):
    times = ["9:15","9:30","9:45","10:00","10:30","11:00",
             "11:30","12:00","12:30","13:00","13:30","14:00",
             "14:30","15:00","15:30"]
    price = last_price
    step = (pct_change/100) / len(times)
    np.random.seed(int(abs(last_price)) % 9999)
    result = []
    for i, t in enumerate(times):
        noise = np.random.normal(0, last_price*0.001)
        price += price*step + noise
        conf = last_price*(0.003 + i*0.0005)
        result.append({
            "time": t,
            "predicted": round(price,2),
            "upper": round(price+conf,2),
            "lower": round(price-conf,2),
        })
    return result

def get_sentiment(text):
    pos_words = ["surge","rally","gain","profit","growth","record",
                 "strong","beat","rise","high","up","bull","positive"]
    neg_words = ["fall","drop","loss","decline","weak","miss","down",
                 "bear","negative","crash","slump","concern","risk"]
    t = text.lower()
    pos = sum(1 for w in pos_words if w in t)
    neg = sum(1 for w in neg_words if w in t)
    total = pos+neg
    if total == 0: return 0.0
    return round((pos-neg)/total, 2)

def get_technicals(df):
    close = df["Close"].squeeze()
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain/loss
    rsi   = round(float((100 - 100/(1+rs)).iloc[-1]), 2)

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = round(float((ema12-ema26).iloc[-1]), 2)
    signal= round(float((ema12-ema26).ewm(span=9).mean().iloc[-1]), 2)

    sma20 = round(float(close.rolling(20).mean().iloc[-1]), 2)
    sma50 = round(float(close.rolling(50).mean().iloc[-1]), 2)
    bb_mid= close.rolling(20).mean()
    bb_std= close.rolling(20).std()
    bb_upper = round(float((bb_mid+2*bb_std).iloc[-1]), 2)
    bb_lower = round(float((bb_mid-2*bb_std).iloc[-1]), 2)

    return {
        "rsi": rsi,
        "macd": macd,
        "macd_signal": signal,
        "sma20": sma20,
        "sma50": sma50,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
    }

# ── ROUTES ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "MarketCast API is running!"}

@app.get("/stock/{symbol}")
def get_stock(symbol: str):
    try:
        ticker = yf.Ticker(nse(symbol))
        info   = ticker.info
        hist   = ticker.history(period="6mo")
        if hist.empty:
            return {"error": "Symbol not found"}

        price  = round(float(hist["Close"].iloc[-1]), 2)
        prev   = round(float(hist["Close"].iloc[-2]), 2)
        change = round(price - prev, 2)
        pct    = round((change/prev)*100, 2)

        prices = hist["Close"].tolist()
        dates  = forecast_dates(30)
        fc30   = arima_forecast(prices, 30)
        fc7    = {k: v[:7] for k, v in fc30.items()}
        fc1day = intraday_forecast(price, pct)

        technicals = get_technicals(hist)
        bullish = fc30["predicted"][0] > price

        # Recommendation
        confidence = min(95, max(55,
            50 +
            (10 if bullish and technicals["rsi"] < 60 else -5) +
            (10 if technicals["macd"] > technicals["macd_signal"] else -5) +
            (10 if price > technicals["sma20"] else -5) +
            (5  if price > technicals["sma50"] else -3)
        ))

        action = "BUY" if bullish else "SELL"
        entry  = round(price * (1.002 if bullish else 0.998), 2)
        target = round(price * (1.08  if bullish else 0.92 ), 2)
        sl     = round(price * (0.96  if bullish else 1.04 ), 2)

        return {
            "symbol":   symbol.upper(),
            "price":    price,
            "change":   change,
            "pct":      pct,
            "high":     round(float(hist["High"].iloc[-1]), 2),
            "low":      round(float(hist["Low"].iloc[-1]),  2),
            "volume":   int(hist["Volume"].iloc[-1]),
            "week52high": round(float(hist["High"].max()), 2),
            "week52low":  round(float(hist["Low"].min()),  2),
            "technicals": technicals,
            "forecast": {
                "intraday":  fc1day,
                "day7":  {
                    "dates": dates[:7],
                    "data":  fc7,
                },
                "day30": {
                    "dates": dates,
                    "data":  fc30,
                },
            },
            "recommendation": {
                "action":     action,
                "confidence": confidence,
                "entry":      entry,
                "target":     target,
                "stop_loss":  sl,
                "bullish":    bullish,
            },
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/news")
def get_news(symbol: str = ""):
    feeds = [
        "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "https://economictimes.indiatimes.com/markets/rss.cms",
        "https://feeds.reuters.com/reuters/INbusinessNews",
    ]
    articles = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                title   = entry.get("title","")
                summary = entry.get("summary","")
                score   = get_sentiment(title+" "+summary)
                sentiment = "positive" if score>0.1 else "negative" if score<-0.1 else "neutral"
                if symbol and symbol.upper() not in title.upper():
                    continue
                articles.append({
                    "title":     title,
                    "source":    feed.feed.get("title","News"),
                    "link":      entry.get("link",""),
                    "time":      entry.get("published",""),
                    "sentiment": sentiment,
                    "score":     score,
                })
        except:
            continue
    return {"news": articles, "count": len(articles)}

@app.get("/market")
def get_market():
    symbols = {
        "NIFTY":    "^NSEI",
        "SENSEX":   "^BSESN",
        "BANKNIFTY":"^NSEBANK",
        "NIFTYIT":  "^CNXIT",
        "DOW":      "^DJI",
        "NASDAQ":   "^IXIC",
        "SGXNIFTY": "^NSEI",
    }
    result = {}
    for name, sym in symbols.items():
        try:
            t    = yf.Ticker(sym)
            h    = t.history(period="2d")
            if h.empty: continue
            price= round(float(h["Close"].iloc[-1]),2)
            prev = round(float(h["Close"].iloc[-2]),2)
            chg  = round(price-prev,2)
            pct  = round((chg/prev)*100,2)
            result[name] = {"price":price,"change":chg,"pct":pct}
        except:
            continue
    return result
if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
