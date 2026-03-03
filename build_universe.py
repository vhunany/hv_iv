import requests
import pandas as pd
import yfinance as yf

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}
sp500 = pd.read_html(requests.get(url, headers=headers).text)[0]

tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()

no_div = []

for t in tickers:
    try:
        if yf.Ticker(t).dividends.empty:
            no_div.append(t)
    except:
        pass

pd.Series(no_div).to_csv("sp500_no_dividends.csv", index=False)

print("Saved universe.")