import pymongo
from pymongo import MongoClient
import yfinance as yf
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle

# MongoDB Connection
uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri)
db = client["stocks_db"]
stock_collection = db["stock_data"]
news_collection = db["mi_data"]

# Load Embedding Model
embedding_model = SentenceTransformer("ProsusAI/finbert")

# Functions from output_test.py (copied here)
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
    """Convert text into a 768-dimensional vector using Sentence Transformers."""
    return embedding_model.encode(text).tolist()

def search_mongodb_articles(ticker, start_date, end_date):
    """Retrieve relevant articles from MongoDB collections based on stock and date range."""
    query = {"stock": ticker, "time": {"$gte": start_date, "$lte": end_date}}
    historical_articles = list(stock_collection.find(query))
    recent_articles = list(news_collection.find(query))
    return historical_articles + recent_articles

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

def calculate_sentiment(articles):
    """Calculate average sentiment score from article embeddings."""
    if not articles:
        return 0.0
    embeddings = [article['embedding'] for article in articles if 'embedding' in article]
    if not embeddings:
      return 0.0
    sentiment_scores = [np.mean(embedding) for embedding in embeddings] #Simplified, improve later
    return np.mean(sentiment_scores)

def prepare_features(stock_data, articles):
    """Prepare features for the SVM model."""
    if stock_data is None or stock_data.empty:
        return None

    # Technical Indicators (Example)
    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['Sentiment'] = calculate_sentiment(articles)

    # Prepare features
    features = stock_data[['Close', 'SMA_20', 'Sentiment']].dropna()
    return features

def predict_buy_sell(features):
    """Predict buy/sell signal using the SVM model."""
    if features is None or features.empty:
        return "Insufficient data for prediction."

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Load the model
    with open('svm_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Make prediction
    prediction = model.predict(scaled_features[-1].reshape(1, -1))[0]

    return "Buy" if prediction == 1 else "Sell/Hold"

# Prediction Function (receives arguments)
def make_prediction(ticker, start_date, end_date):
    """Makes a buy/sell prediction for a given stock."""
    stock_data = get_stock_data(ticker, start_date, end_date)
    articles = search_mongodb_articles(ticker, start_date, end_date)
    features = prepare_features(stock_data, articles)
    prediction = predict_buy_sell(features)
    return prediction, stock_data, articles #return all for display

if __name__ == "__main__":
    # This block will only execute if predict_stock.py is run directly
    print("This file should not be run directly. Please run your main script.")