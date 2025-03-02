import pymongo
import ollama
import numpy as np
from pymongo import UpdateOne
from sentence_transformers import SentenceTransformer

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main", tlsAllowInvalidCertificates=True)
db = client["stocks_db"]
collection = db["mi_data"]

embedding_model = SentenceTransformer('yiyanghkust/finbert')

def embed_text(text):
    return embedding_model.encode(text).tolist()

BATCH_SIZE = 1
total_articles = 10#collection.count_documents({})
processed_count = 0

print(f"Starting to process {total_articles} articles in batches of {BATCH_SIZE}...")

while processed_count < total_articles:
    articles = list(collection.find().skip(processed_count).limit(BATCH_SIZE))
    
    if not articles:
        break  # No more articles

    bulk_operations = []
    for article in articles:
        # Print just title for debugging
        print(f"Processing: {article.get('title', 'No title')}")
        
        text = article.get("body", "")
        if not text:
            print(f"No text found for article: {article.get('title', 'No title')}")
            continue
            
        embedding = embed_text(text)

        if embedding:
            bulk_operations.append(UpdateOne(
                {"_id": article["_id"]},
                {"$set": {"embedding": embedding}}
            ))
        else:
            print(f"Embedding failed for article: {article.get('title', 'No title')}")

    if bulk_operations:
        collection.bulk_write(bulk_operations)
        print(f"Batch of {len(bulk_operations)} embeddings added.") 

    processed_count += BATCH_SIZE
    print(f"Processed {processed_count} of {total_articles} articles.")

print("Embeddings generated and stored for all articles.")