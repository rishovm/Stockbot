def add_technical_indicators(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    return df

def predict_trend(df):
    if df['MA10'].iloc[-1] > df['MA50'].iloc[-1]:
        return "BUY"
    return "HOLD"
