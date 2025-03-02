import pymongo
from pymongo import MongoClient
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ MongoDB Connection
uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri)
db = client["stocks_db"]  # ✅ Ensure we select the correct database
stock_collection = db["stock_data"]  # Historical news collection
news_collection = db["mi_data"]  # Recent news collection

# ✅ Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_stock_data(ticker, start_date, end_date):
    """Retrieve historical stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(start=start_date, end=end_date)
        return historical_data
    except Exception as e:
        print(f"Error retrieving stock data: {e}")
        return None

def embed_text(text):
    """Convert text into a 384-dimensional vector using Sentence Transformers."""
    return embedding_model.encode(text).tolist()

def search_mongodb_articles(ticker, start_date, end_date):
    """Retrieve relevant articles from MongoDB collections based on stock and date range."""
    print(f"🔍 Searching MongoDB for {ticker} news between {start_date} and {end_date}...")

    query = {"stock": ticker, "time": {"$gte": start_date, "$lte": end_date}}
    historical_articles = list(stock_collection.find(query))
    recent_articles = list(news_collection.find(query))

    print(f"✅ Found {len(historical_articles)} historical articles & {len(recent_articles)} recent articles.")
    return historical_articles + recent_articles  # Merge both datasets

def vector_search_articles(query_text, top_k=5):
    """Perform vector search for articles using MongoDB."""
    print(f"🔍 Generating embedding for query: {query_text}...")
    query_embedding = embed_text(query_text)  # Convert query to vector

    print("🔍 Fetching all articles with embeddings from MongoDB...")
    all_articles = list(stock_collection.find({}, {"_id": 1, "title": 1, "body": 1, "embedding": 1}))
    all_articles += list(news_collection.find({}, {"_id": 1, "title": 1, "body": 1, "embedding": 1}))

    print(f"✅ Retrieved {len(all_articles)} total articles.")

    # ✅ Ensure embeddings are correctly formatted
    valid_articles = [article for article in all_articles if "embedding" in article and len(article["embedding"]) == 384]
    
    if not valid_articles:
        print("⚠️ No valid articles with 384-dim embeddings found!")
        return []

    print(f"✅ {len(valid_articles)} articles have correct embeddings.")

    # Extract embeddings & compute similarity
    embeddings = np.array([article["embedding"] for article in valid_articles])
    query_vector = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(query_vector, embeddings)[0]

    # ✅ Sort results by similarity
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    similar_articles = [valid_articles[i] for i in sorted_indices]

    print(f"✅ Found {len(similar_articles)} similar articles using vector search.")
    return similar_articles

def display_results(stock_data, articles):
    """Display stock data and relevant news articles."""
    print("\n📈 Stock Historical Data:")
    if stock_data is not None and not stock_data.empty:
        print(stock_data.tail(5))  # Display last 5 rows
    else:
        print("⚠️ No stock data available.")

    print("\n📰 Related News Articles:")
    if articles:
        for article in articles[:5]:  # Show top 5 articles
            print(f"- {article['title']} ({article.get('time', 'N/A')})")
            print(f"  {article['body'][:200]}...")  # Show first 200 characters
            print("-" * 40)
    else:
        print("⚠️ No relevant articles found.")

# ✅ User Inputs
ticker = input("Enter a stock ticker: ").strip().upper()
start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
end_date = input("Enter the end date (YYYY-MM-DD): ").strip()

# ✅ Fetch stock data
stock_data = get_stock_data(ticker, start_date, end_date)

# ✅ Search news using MongoDB query + Local Vector Search
articles_mongo = search_mongodb_articles(ticker, start_date, end_date)
articles_vector = vector_search_articles(f"News about {ticker} stock performance")

# ✅ Merge & Display Results
all_articles = articles_mongo + articles_vector
display_results(stock_data, all_articles)
