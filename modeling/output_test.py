import pymongo
from pymongo import MongoClient
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ✅ MongoDB Connection
uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri)
db = client["stocks_db"]  # ✅ Ensure we select the correct database
stock_collection = db["stock_data"]  # Historical news collection
news_collection = db["mi_data"]  # Recent news collection

# ✅ Load Embedding Model
embedding_model = SentenceTransformer("ProsusAI/finbert")

def get_stock_data(ticker, start_date, end_date):
    """Retrieve historical stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(start=start_date, end=end_date)
        summary = stock.info.get("longBusinessSummary")
        return historical_data, summary
    except Exception as e:
        print(f"Error retrieving stock data: {e}")
        return None

def embed_text(text):
    """Convert text into a 768-dimensional vector using Sentence Transformers."""
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
    pipeline = [
        {
            "$addFields": {
                "dot_product": {
                    "$sum": {
                        "$map": {
                            "input": {
                                "$zip": {
                                    "inputs": ["$embedding", query_embedding]
                                }
                            },
                            "as": "pair",
                            "in": {
                                "$multiply": [
                                    {"$arrayElemAt": ["$$pair", 0]},
                                    {"$arrayElemAt": ["$$pair", 1]}
                                ]
                            }
                        }
                    }
                },
                "query_norm": {
                    "$sqrt": {
                        "$max": [
                            0.000001,  # Small epsilon to avoid zero
                            {
                                "$sum": {
                                    "$map": {
                                        "input": "$embedding",
                                        "as": "val",
                                        "in": { "$multiply": ["$$val", "$$val"] }
                                    }
                                }
                            }
                        ]
                    }
                },
                "embedding_norm": {
                    "$sqrt": {
                        "$max": [
                            0.000001,  # Small epsilon to avoid zero
                            {
                                "$sum": {
                                    "$map": {
                                        "input": query_embedding,
                                        "as": "val",
                                        "in": { "$multiply": ["$$val", "$$val"] }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        },
        {
            "$addFields": {
                "cosine_similarity": {
                    "$cond": {
                        "if": {
                            "$or": [
                                {"$eq": ["$query_norm", 0]},
                                {"$eq": ["$embedding_norm", 0]}
                            ]
                        },
                        "then": 0,  # Default to 0 similarity if either norm is zero
                        "else": {
                            "$divide": ["$dot_product", {"$multiply": ["$query_norm", "$embedding_norm"]}]
                        }
                    }
                }
            }
        },
        {
            "$sort": {"cosine_similarity": -1}
        },
        {
            "$limit": top_k  # Top 5 results
        }
    ]

    results = list(news_collection.aggregate(pipeline))
    return results

def display_results(stock_data, articles):
    """Display stock data and relevant news articles."""
    print("\n📈 Stock Historical Data:")
    if stock_data is not None and not stock_data.empty:
        print(stock_data.tail(5))  # Display last 5 rows
    else:
        print("⚠️ No stock data available.")

    print("\n📰 Related News Articles:")
    if articles:
        for article in articles[-5:]:  # Show top 5 articles
            print(f"- {article['title']} ({article.get('time', 'N/A')})")
            print(f"  {article['body'][:200]}...")  # Show first 200 characters
            print("-" * 40)
    else:
        print("⚠️ No relevant articles found.")

# ✅ User Inputs
ticker = "AAPL"#input("Enter a stock ticker: ").strip().upper()
start_date = "2025-01-01"#input("Enter the start date (YYYY-MM-DD): ").strip()
end_date = "2025-03-01"#input("Enter the end date (YYYY-MM-DD): ").strip()

# ✅ Fetch stock data
stock_data, summary = get_stock_data(ticker, start_date, end_date)

# ✅ Search news using MongoDB query + Local Vector Search
#articles_mongo = search_mongodb_articles(ticker, start_date, end_date)
print(summary)
articles_vector = vector_search_articles(f"{ticker} {summary}")

# ✅ Merge & Display Results
all_articles =  articles_vector
display_results(stock_data, all_articles)
