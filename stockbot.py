import yfinance as yf
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import warnings
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")


def add_technical_indicators(df):
    df['momentum'] = df['Close'].diff(5)
    df['volatility'] = df['Close'].rolling(window=5).std()
    df['pct_return'] = df['Close'].pct_change()
    df = df.dropna()
    return df


def generate_target(df):
    df['future_return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['target'] = (df['future_return'] > 0).astype(int)
    return df.dropna()


def train_model(df):
    features = ['momentum', 'volatility', 'pct_return']
    target = 'target'

    model = LGBMClassifier()
    model.fit(df[features], df[target])
    return model


def estimate_hold_duration(df):
    if 'momentum' not in df.columns or 'volatility' not in df.columns:
        df = add_technical_indicators(df)

    df['signal'] = (df['momentum'] > 0) & (df['volatility'] < df['volatility'].median())

    for i in range(len(df)):
        if df['signal'].iloc[i]:
            entry_price = df['Close'].iloc[i]
            for j in range(i + 1, min(i + 30, len(df))):
                future_price = df['Close'].iloc[j]
                if future_price >= entry_price * 1.05 or future_price <= entry_price * 0.97:
                    return j - i
            return 30
    return None


def generate_recommendation(ticker):
    print(f"Processing {ticker}...")

    df = yf.download(ticker, period='6mo', interval='1d', auto_adjust=True)

    # Flatten columns if multiindex (just in case)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == ticker else col[0] + '_' + col[1] for col in df.columns]

    df = add_technical_indicators(df)
    df = generate_target(df)

    if df.empty:
        return f"{ticker}: Not enough data to analyze."

    model = train_model(df)

    latest_data = df.iloc[-1:]
    features = ['momentum', 'volatility', 'pct_return']
    prediction = model.predict(latest_data[features])[0]

    recommendation = "BUY" if prediction == 1 else "SELL"
    hold_days = estimate_hold_duration(df)
    hold_message = f"Estimated hold duration: {hold_days} days." if recommendation == "BUY" and hold_days else ""

    return f"{ticker}: Recommendation: {recommendation}. {hold_message}"


def save_output_as_image(lines, filename='recommendations.jpeg'):
    # Prepare image size
    width = 800
    line_height = 30
    margin = 20
    height = margin * 2 + line_height * len(lines)

    # Create white image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Load a default font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    y_text = margin
    for line in lines:
        draw.text((margin, y_text), line, font=font, fill='black')
        y_text += line_height

    img.save(filename)
    print(f"Saved recommendations as image: {filename}")


def run_once():
    print("Hey! I am StockBot 🤖 - I help you invest for your future savings!\n")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

    results = []
    for ticker in tickers:
        results.append(generate_recommendation(ticker))

    for res in results:
        print(res)

    save_output_as_image(results)


if __name__ == "__main__":
    run_once()
