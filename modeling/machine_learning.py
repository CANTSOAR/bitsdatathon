import yfinance as yf
import pandas as pd
from datetime import datetime
import predict_stock  # Import predict_stock.py

def get_stock_data(ticker, start_date, end_date):
    """Retrieves stock data from Yahoo Finance, including historical data."""
    try:
        stock = yf.Ticker(ticker)

        # Retrieve general stock info
        info = stock.info

        # Calculate period from start and end dates
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        delta = end_datetime - start_datetime
        period_days = delta.days

        # Retrieve historical data using the calculated period.
        historical_data = stock.history(period=f"{period_days}d", start=start_date, end=end_date)

        # Combine info and historical data into one dictionary
        combined_data = {
            "info": info,
            "historical": historical_data
        }

        return combined_data

    except Exception as e:
        print(f"Error retrieving stock data: {e}")
        return None

def display_stock_data(stock_data):
    """Displays key stock data and historical data."""
    if not stock_data:
        return

    info = stock_data["info"]
    historical = stock_data["historical"]

    print(f"Stock Information for {info.get('longName', 'N/A')} ({info.get('symbol', 'N/A')})")
    print("-" * 40)

    print(f"Current Price: {info.get('currentPrice', 'N/A')}")
    print(f"Previous Close: {info.get('previousClose', 'N/A')}")
    print(f"Open: {info.get('open', 'N/A')}")
    print(f"Day High: {info.get('dayHigh', 'N/A')}")
    print(f"Day Low: {info.get('dayLow', 'N/A')}")
    print(f"Volume: {info.get('volume', 'N/A')}")

    print("\nKey Financials:")
    print(f"Market Cap: {info.get('marketCap', 'N/A')}")
    print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
    print(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")

    print("\nCompany Information:")
    print(f"Industry: {info.get('industry', 'N/A')}")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Summary: {info.get('longBusinessSummary', 'N/A')}")

    if not historical.empty:
        print("\nHistorical Data:")
        print(historical)  # Display the historical data (pandas DataFrame)
    else:
        print("\nNo historical data available for the specified date range.")

# User Input and Execution
ticker = input("Enter a stock ticker: ")
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")

stock_data = get_stock_data(ticker, start_date, end_date)
display_stock_data(stock_data)

# Call the prediction function from predict_stock.py
prediction, stock_data_pred, articles_pred = predict_stock.make_prediction(ticker, start_date, end_date)
print(f"\n📈 Prediction for {ticker}: {prediction}")
predict_stock.display_results(stock_data_pred, articles_pred) #display results