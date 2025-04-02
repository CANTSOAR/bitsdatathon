import pymongo
from pymongo import MongoClient
import yfinance as yf
import pandas as pd

# ✅ MongoDB Connection
uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri)
db = client["stocks_db"]
stock_collection = db["stock_data"]

# ✅ Macro Indicators CSV Path
macro_csv_path = "macro_data.csv"  # Replace with actual path

# ✅ List of ETF/Index and Sector Tickers to Fetch
etf_tickers = [
    "IEF", "LQD", "QQQ", "SPY", "SPYG", "SPYV", "TLT", "^VIX",
    "XLE", "XLF", "XLK", "XLV", "XLY"
]

# ✅ Read macro indicators from CSV (updated selection)
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

# ✅ Fetch ETF/Index and Sector Data using yfinance
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

# ✅ Fetch stock data from MongoDB or fallback to yfinance
def fetch_stock_data(ticker, start_date, end_date):
    query = {"ticker": ticker, "date": {"$gte": start_date, "$lte": end_date}}
    stock_data_cursor = stock_collection.find(query)

    # Convert MongoDB documents to DataFrame
    stock_data_list = list(stock_data_cursor)
    if stock_data_list:
        data = pd.DataFrame(stock_data_list)
        data.set_index("date", inplace=True)
        data.index = pd.to_datetime(data.index)
        print(f"✅ Data found in MongoDB for {ticker}")
    else:
        print(f"⚠️ No data found in MongoDB. Fetching data from Yahoo Finance for {ticker}.")
        data = yf.download(ticker, start=start_date, end=end_date)

    if "Close" not in data.columns:
        raise ValueError("Missing 'Close' data. Please check your source.")

    return data

# ✅ Calculate Technical Indicators
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

# ✅ Merge stock, macro, ETF/sector, and technical data
def merge_all_data(stock_data, macro_data, etf_data):
    # Merge stock and macro data
    enriched_data = stock_data.merge(macro_data, left_index=True, right_index=True, how="inner")

    # Merge with ETF/sector data
    enriched_data = enriched_data.merge(etf_data, left_index=True, right_index=True, how="inner")

    # Calculate technical indicators and merge
    enriched_data = calculate_technical_indicators(enriched_data)

    return enriched_data

# ✅ Store enriched data with macro, ETF/sector, and technical indicators in MongoDB
def store_enriched_data(ticker, data):
    enriched_data = data.reset_index().to_dict(orient="records")

    for record in enriched_data:
        record["ticker"] = ticker
        record["date"] = record["Date"]
        del record["Date"]

    # Insert enriched data into MongoDB
    enriched_collection = db["enriched_stock_data"]
    enriched_collection.insert_many(enriched_data)
    print(f"✅ Enriched data with macro, ETF/sector, and technical indicators stored successfully in MongoDB for {ticker}")

# ✅ Main process to update stock data with macro, ETF/sector, and technical indicators
def update_stock_data_with_all(ticker, start_date, end_date):
    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # Read macro data from CSV
    macro_data = read_macro_data(macro_csv_path)

    # Fetch ETF/Index and Sector data from yfinance
    etf_data = fetch_etf_data(etf_tickers, start_date, end_date)

    # Merge all data
    enriched_data = merge_all_data(stock_data, macro_data, etf_data)

    # Store enriched data in MongoDB
    store_enriched_data(ticker, enriched_data)

# 🔥 Run the process for a stock
ticker = "AAPL"  # Change to your desired stock
start_date = "2024-01-01"
end_date = "2024-03-01"

update_stock_data_with_all(ticker, start_date, end_date)