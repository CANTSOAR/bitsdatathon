import pymongo
import numpy as np
from pymongo import UpdateOne
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ✅ MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main", tlsAllowInvalidCertificates=True)
db = client["stocks_db"]
collection = db["mi_data"]

# ✅ Load Embedding Model
embedding_model = SentenceTransformer("ProsusAI/finbert")

def embed_text(text):
    return embedding_model.encode(text).tolist()

def convert_date_to_timestamp(date_value):
    """Converts date string to UNIX timestamp (integer) only if needed."""
    if isinstance(date_value, int):
        return date_value  # ✅ Already a timestamp, no conversion needed
    
    if isinstance(date_value, str):  
        try:
            return int(datetime.strptime(date_value, "%b. %d, %Y, %I:%M %p").timestamp())  # Example: "Mar. 1, 2025, 05:14 PM"
        except ValueError:
            try:
                return int(datetime.strptime(date_value, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())  # ISO format fallback
            except ValueError:
                print(f"⚠️ Date conversion failed for: {date_value}")
                return None
    
    print(f"⚠️ Unexpected time format: {date_value} (type: {type(date_value)})")
    return None


BATCH_SIZE = 1
total_articles = 20  # collection.count_documents({})
processed_count = 0

print(f"Starting to process {total_articles} articles in batches of {BATCH_SIZE}...")

while processed_count < total_articles:
    articles = list(collection.find().skip(processed_count).limit(BATCH_SIZE))
    
    if not articles:
        break  # No more articles

    bulk_operations = []
    for article in articles:
        print(f"Processing: {article.get('title', 'No title')}")

        text = article.get("body", "")
        if not text:
            print(f"⚠️ No text found for article: {article.get('title', 'No title')}")
            continue

        embedding = embed_text(text)

        # ✅ Convert date to UNIX timestamp
        date_str = article.get("time", "")
        timestamp = convert_date_to_timestamp(date_str) if date_str else None

        update_data = {"embedding": embedding}
        if timestamp is not None:
            update_data["time"] = timestamp  # Replace string date with number
        
        bulk_operations.append(UpdateOne(
            {"_id": article["_id"]},
            {"$set": update_data}
        ))

    if bulk_operations:
        collection.bulk_write(bulk_operations)
        print(f"✅ Batch of {len(bulk_operations)} embeddings & timestamps added.") 

    processed_count += BATCH_SIZE
    print(f"✅ Processed {processed_count} of {total_articles} articles.")

print("✅ Embeddings & timestamps stored successfully!")
