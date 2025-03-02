import yfinance as yf

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        print(f"Error retrieving stock data: {e}")
        return None
    
def display_stock_data(stock_info):
    if stock_info:
        print(f"Stock Information for {stock_info.get('longName', 'N/A')} ({stock_info.get('symbol', 'N/A')}")
        print("-" * 40)

        print(f"Current Price: {stock_info.get('currentPrice', 'N/A')}")
        print(f"Previous Close: {stock_info.get('previousClose', 'N/A')}")
        print(f"Open: {stock_info.get('open', 'N/A')}")
        print(f"Day High: {stock_info.get('dayHigh', 'N/A')}")
        print(f"Day Low: {stock_info.get('dayLow', 'N/A')}")
        print(f"Volume: {stock_info.get('volume', 'N/A')}")

        print("\nKey Financials:")
        print(f"Market Cap: {stock_info.get('marketCap', 'N/A')}")
        print(f"P/E Ratio: {stock_info.get('trailingPE', 'N/A')}")
        print(f"Dividend Yield: {stock_info.get('dividendYield', 'N/A')}")

        print("\nCompany Information:")
        print(f"Industry: {stock_info.get('industry', 'N/A')}")
        print(f"Sector: {stock_info.get('sector', 'N/A')}")
        print(f"Summary: {stock_info.get('longBusinessSummary', 'N/A')}")
    else:
        print("No stock data to display.")
        
# User Input and Execution
ticker = input("Enter a stock ticker: ")
stock_data = get_stock_data(ticker)
display_stock_data(stock_data)