import yfinance as yf
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import warnings
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import os

warnings.filterwarnings("ignore")

# Set output directory
OUTPUT_DIR = r"C:\Users\pharm\OneDrive\Desktop\stockbot_project\stock_ai_whatsapp_bot\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure the directory exists


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

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model = LGBMClassifier()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {acc:.2f}")

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

    try:
        df = yf.download(ticker, period='6mo', interval='1d', auto_adjust=True)
    except Exception as e:
        return f"{ticker}: Failed to fetch data. Error: {e}"

    if df.empty or len(df) < 30:
        return f"{ticker}: Not enough data to analyze."

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == ticker else col[0] + '_' + col[1] for col in df.columns]

    df = add_technical_indicators(df)
    df = generate_target(df)

    if df.empty:
        return f"{ticker}: Not enough data after processing."

    model = train_model(df)

    latest_data = df.iloc[-1:]
    features = ['momentum', 'volatility', 'pct_return']
    prediction = model.predict(latest_data[features])[0]

    recommendation = "BUY" if prediction == 1 else "SELL"
    hold_days = estimate_hold_duration(df)
    hold_message = f" Estimated hold duration: {hold_days} days." if recommendation == "BUY" and hold_days else ""

    return f"{ticker}: Recommendation: {recommendation}.{hold_message}"


def save_output_as_image(lines, filename='recommendations.jpeg'):
    lines.insert(0, f"Stock Recommendations - {datetime.today().strftime('%Y-%m-%d')}")

    margin = 20
    line_height = 30

    temp_img = Image.new('RGB', (1, 1), color='white')
    draw = ImageDraw.Draw(temp_img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    max_width = max(draw.textsize(line, font=font)[0] for line in lines) + margin * 2
    height = margin * 2 + line_height * len(lines)

    img = Image.new('RGB', (max_width, height), color='white')
    draw = ImageDraw.Draw(img)

    y_text = margin
    for line in lines:
        fill = 'green' if 'BUY' in line else 'red' if 'SELL' in line else 'black'
        draw.text((margin, y_text), line, font=font, fill=fill)
        y_text += line_height

    output_path = os.path.join(OUTPUT_DIR, filename)
    img.save(output_path)
    print(f"Saved recommendations as image: {output_path}")


def run_once():
    print("Hey! I am StockBot 🤖 - I help you invest for your future savings!\n")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

    results = []
    for ticker in tickers:
        result = generate_recommendation(ticker)
        print(result)
        results.append(result)

    save_output_as_image(results)


if __name__ == "__main__":
    run_once()
