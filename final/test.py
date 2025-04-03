from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import spacy
from concurrent.futures import ProcessPoolExecutor
import time

# MongoDB URI - Replace <password> with your actual password
uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri)
db = client["stocks_db"]
collection = db["mi_data"]

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Load the pre-trained NER model
nlp = spacy.load("en_core_web_trf")

# Extract stock tickers (companies) from the article
def extract_tickers(article):
    doc = nlp(article)
    tickers = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return tickers

# Function to process articles in parallel
def process_articles(articles):
    results = []
    for article in articles:
        results.append(extract_tickers(article))
        #print(results[-1])
    return results

# Fetch articles from MongoDB collection (you can adjust the query as needed)
def fetch_articles_from_db():
    articles = []
    # Adjust this query if necessary based on how your data is stored
    cursor = collection.find({}, {"_id": 0, "body": 1})  # Assuming "article_text" is the field name for articles
    for document in cursor[:10]:
        articles.append(document["body"])
    return articles

# Fetch articles from the database
articles = fetch_articles_from_db()

# Process the articles to extract tickers in parallel
t1 = time.time()
tickers = process_articles(articles)

print(time.time() - t1)