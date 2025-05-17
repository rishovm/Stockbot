from modules.data_fetcher import fetch_data
from modules.model import add_technical_indicators, predict_trend

def generate_recommendations(tickers):
    recommendations = []
    for ticker in tickers:
        df = fetch_data(ticker)
        df = add_technical_indicators(df)
        action = predict_trend(df)
        if action == "BUY":
            recommendations.append(f"{ticker}: {action}")
    return recommendations
