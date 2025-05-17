import yfinance as yf

def fetch_data(ticker, period="30d", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    return df
