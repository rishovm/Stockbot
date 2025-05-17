# Stockbot
StockBot is an AI-powered tool that analyzes stock data to provide buy/sell recommendations and estimated hold durations.

StockBot is an AI-powered stock analysis and recommendation tool built in Python. It automates the process of fetching historical stock data, engineering technical indicators, training a machine learning model, and generating buy/sell recommendations with estimated hold durations. Results are saved as clear, readable JPEG images for easy review.
Features
•	Fetches historical stock data from Yahoo Finance
•	Computes key technical indicators: momentum, volatility, and percentage return
•	Uses LightGBM classifier for predictive modeling
•	Provides buy/sell recommendations with confidence-based hold duration estimates
•	Outputs results as JPEG images with clean, readable text
•	Easily scalable to analyze hundreds of stocks
Installation
1.	Clone this repository:
2.	git clone https://github.com/yourusername/stockbot.git
3.	cd stockbot
4.	Create and activate a Python virtual environment (recommended):
5.	python -m venv venv
6.	source venv/bin/activate   # On Windows: venv\Scripts\activate
7.	Install required packages:
8.	pip install -r requirements.txt
If you don't have a requirements.txt, install manually:
pip install yfinance pandas numpy lightgbm pillow
Usage
Run the StockBot script to generate stock recommendations:
python stockbot.py
By default, StockBot analyzes a preset list of stocks (e.g., AAPL, MSFT, GOOGL). You can customize the ticker list inside the run_once() function.
The bot outputs:
•	Console logs showing progress and recommendations
•	A JPEG image saved in your project directory with all recommendations
Customization
•	Ticker list: Modify the tickers list inside run_once() to analyze your preferred stocks.
•	Output path: Change the file path in the save_output_as_image() function to save the output image elsewhere.
•	Model features: Adjust technical indicators or machine learning model parameters inside the script for deeper customization.
Example Output
AAPL: Recommendation: BUY. Estimated hold duration: 15 days.  
MSFT: Recommendation: SELL.  
GOOGL: Recommendation: BUY. Estimated hold duration: 10 days.  
...
Dependencies
•	yfinance — for stock market data
•	pandas — data manipulation
•	numpy — numerical operations
•	lightgbm — machine learning model
•	Pillow — image generation
Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve StockBot.
License
This project is licensed under the MIT License.
