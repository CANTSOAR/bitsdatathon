import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import re

# Headers to avoid bot detection
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

def get_article_content(url):
    """Fetches the article title, content, and published date from Yahoo Finance."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()  # Raises an error for HTTP failures
        
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title_tag = soup.find("h1")
        title = title_tag.text.strip() if title_tag else "No Title"

        # Extract article paragraphs
        paragraphs = soup.find_all("p")
        content = " ".join(p.text for p in paragraphs) if paragraphs else "No Content"

        # Extract date (if available)
        date_tag = soup.find("time")
        date = date_tag.text.strip() if date_tag else "Unknown Date"

        return title, date, content

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Error fetching article: {e}")

    return None, None, None

def extract_stock_symbols(text):
    """Extracts stock tickers in the format (NASDAQ:AAPL) or (NYSE:MSFT) from the text."""
    stock_symbols = []

    # Regular expression to find tickers in the specified format
    matches = re.findall(r'\((NASDAQ|NYSE):([A-Z]{1,5})\)', text)

    for match in matches:
        ticker = match[1]  # Extract the ticker symbol (second capturing group)
        try:
            # Verify if it's a valid stock symbol using Yahoo Finance API
            stock = yf.Ticker(ticker)
            if stock.info.get("shortName"):  # Check if the stock exists
                stock_symbols.append(ticker)
        except:
            continue  # Skip invalid tickers

    return ", ".join(stock_symbols) if stock_symbols else "Unknown Stock"

# Define the Yahoo Finance article URL
ARTICLE_URL = "https://finance.yahoo.com/news/apple-inc-aapl-best-money-163954518.html"

# Fetch article data
title, date, content = get_article_content(ARTICLE_URL)

if content:
    # Extract stock ticker from the title
    stock = extract_stock_symbols(title)

    # Create DataFrame
    article_df = pd.DataFrame([{
        "title": title,
        "stock": stock,
        "date": date,
        "url": ARTICLE_URL,
        "content": content
    }])

    # Save to CSV
    article_df.to_csv("yahoo_finance_article.csv", index=False)

    # Display results
    print(article_df.to_string())  # Prints all rows properly

    print("✅ Article scraped successfully!")
else:
    print("❌ Failed to scrape the article.")

