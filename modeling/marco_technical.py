import pymongo
from pymongo import MongoClient
import yfinance as yf
import pandas as pd

uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri)
db = client["stocks_db"]
stock_collection = db["stock_data"]

macro_csv_path = "macro_data.csv"  # Replace with actual path

etf_tickers = [
    "IEF", "LQD", "QQQ", "SPY", "SPYG", "SPYV", "TLT", "^VIX",
    "XLE", "XLF", "XLK", "XLV", "XLY"
]

def read_macro_data(csv_path):
    macro_data = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    # Select only relevant columns
    macro_columns = [
        "Fed_Funds_Rate", "CPI", "GDP", "Unemployment_Rate",
        "Money_Supply", "Retail_Sales", "Oil_Price"
    ]
    macro_data = macro_data[macro_columns].ffill().dropna()
    macro_data.index = pd.to_datetime(macro_data.index)

    return macro_data

def fetch_etf_data(tickers, start_date, end_date):
    etf_data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]

    # Fill forward and drop any missing values
    etf_data = etf_data.ffill().dropna()
    etf_data.index = pd.to_datetime(etf_data.index)

    # Rename columns to be consistent
    etf_data.rename(columns={
        "IEF": "IEF_Close", 
        "LQD": "LQD_Close", 
        "QQQ": "QQQ_Close",
        "SPY": "SPY_Close", 
        "SPYG": "SPYG_Close", 
        "SPYV": "SPYV_Close",
        "TLT": "TLT_Close", 
        "^VIX": "VIX_Close",
        "XLE": "XLE_Close", 
        "XLF": "XLF_Close", 
        "XLK": "XLK_Close",
        "XLV": "XLV_Close", 
        "XLY": "XLY_Close"
    }, inplace=True)

    return etf_data

def calculate_technical_indicators(data):
    # Simple Moving Average (SMA)
    data["SMA_20"] = data["Close"].rolling(window=20).mean()

    # Exponential Moving Average (EMA)
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    def calculate_rsi(data, window=14):
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        data["RSI"] = 100 - (100 / (1 + rs))
        return data

    data = calculate_rsi(data)

    # MACD and Signal Line
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data["BB_Middle"] = data["Close"].rolling(window=20).mean()
    data["BB_Upper"] = data["BB_Middle"] + 2 * data["Close"].rolling(window=20).std()
    data["BB_Lower"] = data["BB_Middle"] - 2 * data["Close"].rolling(window=20).std()

    data = data.dropna()
    return data

def merge_all_data(stock_data, macro_data, etf_data):
    # Merge stock and macro data
    enriched_data = stock_data.merge(macro_data, left_index=True, right_index=True, how="inner")

    # Merge with ETF/sector data
    enriched_data = enriched_data.merge(etf_data, left_index=True, right_index=True, how="inner")

    # Calculate technical indicators and merge
    enriched_data = calculate_technical_indicators(enriched_data)

    return enriched_data

ticker = "AAPL"  # Change to your desired stock
start_date = "2024-01-01"
end_date = "2024-03-01"