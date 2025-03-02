import yfinance as yf
from newsapi import NewsApiClient
import ollama
import pymongo
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import pandas as pd
import os
import configparser

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main", tlsAllowInvalidCertificates=True)
db = client["stocks_db"]
collection = db["stock_data"]


def get_stock_data(ticker, start_date, end_date):
    """Retrieves stock data from Yahoo Finance, including historical data."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        delta = end_datetime - start_datetime
        period_days = delta.days
        historical_data = stock.history(period=f"{period_days}d", start=start_date, end=end_date)
        combined_data = {"info": info, "historical": historical_data}
        return combined_data
    except Exception as e:
        print(f"Error retrieving stock data: {e}")
        return None

def display_stock_data(stock_data):
    """Displays key stock data and historical data."""
    if not stock_data:
        print("No stock data to display.")
        return

    info = stock_data["info"]
    historical = stock_data["historical"]

    print(f"\nStock Information for {info.get('longName', 'N/A')} ({info.get('symbol', 'N/A')})")
    print("-" * 40)

    print("\nGeneral Stock Information:")
    general_info = {
        "Current Price": info.get('currentPrice', 'N/A'),
        "Previous Close": info.get('previousClose', 'N/A'),
        "Open": info.get('open', 'N/A'),
        "Day High": info.get('dayHigh', 'N/A'),
        "Day Low": info.get('dayLow', 'N/A'),
        "Volume": info.get('volume', 'N/A')
    }
    for key, value in general_info.items():
        print(f"{key}: {value}")

    print("\nKey Financials:")
    financials = {
        "Market Cap": info.get('marketCap', 'N/A'),
        "P/E Ratio": info.get('trailingPE', 'N/A'),
        "Dividend Yield": info.get('dividendYield', 'N/A')
    }
    for key, value in financials.items():
        print(f"{key}: {value}")

    print("\nCompany Information:")
    company_info = {
        "Industry": info.get('industry', 'N/A'),
        "Sector": info.get('sector', 'N/A'),
        "Summary": info.get('longBusinessSummary', 'N/A')
    }
    for key, value in company_info.items():
        print(f"{key}: {value}")

    if not historical.empty:
        print("\nHistorical Data:")
        print(historical)
    else:
        print("\nNo historical data available for the specified date range.")

def generate_stock_description(stock_info):
    """Generates a text description for a stock."""
    company_name = stock_info.get("longName", "")
    industry = stock_info.get("industry", "")
    return f"{company_name} ({stock_info.get('symbol')}), a {industry} company..."

def get_stock_news(ticker, from_date, to_date):
    """Retrieves news articles related to a stock ticker."""
    news_api_key = "e9e401a98e26446ca6bfbc87145ef498"
    if not news_api_key:
        raise ValueError("NEWSAPI_KEY environment variable not set.")
    try:
        newsapi = NewsApiClient(api_key=news_api_key)
        stock = yf.Ticker(ticker)
        company_name = stock.info.get('longName', ticker)
        all_articles = newsapi.get_everything(q=company_name, from_param=from_date, to=to_date, language='en', sort_by='relevancy')
        if all_articles['status'] == 'ok':
            return all_articles['articles']
        else:
            print(f"NewsAPI error: {all_articles['message']}")
            return []
    except Exception as e:
        print(f"Error retrieving news: {e}")
        return []

def store_article_embeddings(articles):
    """Generates and stores embeddings for news articles."""
    for article in articles:
        try:
            text = f"{article['title']} {article['description']}"
            embedding = ollama.embeddings(model="llama3.2", prompt=text)["embedding"]
            collection.insert_one({
                "title": article['title'],
                "description": article['description'],
                "url": article['url'],
                "embedding": embedding
            })
        except Exception as e:
            print(f"Error storing embedding: {e}")

def find_similar_articles(user_embedding, company_name):
    """Finds similar articles based on vector similarity."""
    articles = list(collection.find({"title": {"$regex": company_name, "$options": "i"}}))
    similarities = []
    for article in articles:
        if "embedding" in article and article["embedding"]:
            article_embedding = np.array(article["embedding"]).reshape(1, -1)
            user_embedding_np = np.array(user_embedding).reshape(1, -1)
            similarity = cosine_similarity(user_embedding_np, article_embedding)[0][0]
            similarities.append((similarity, article))
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities

# User Input
ticker = input("Enter a stock ticker: ")
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")

# Stock Data
stock_data = get_stock_data(ticker, start_date, end_date)
if stock_data:
    display_stock_data(stock_data)
else:
    print("Failed to retrieve stock data")

# News Articles
news_articles = get_stock_news(ticker, start_date, end_date)
if news_articles:
    store_article_embeddings(news_articles)

# Embeddings and Similarity Search
if stock_data:
    user_embedding = ollama.embeddings(model="llama3.2", prompt=generate_stock_description(stock_data['info']))["embedding"]
    similar_articles = find_similar_articles(user_embedding, stock_data['info']["longName"])
    print("\nRelevant Articles:")
    for similarity, article in similar_articles[:5]:
        print(f"  {article['title']} (Similarity: {similarity:.4f})")
        print(f"  {article['url']}")
        print("-" * 20)
else:
    print("Cannot perform similarity search, stock data was not retrieved")