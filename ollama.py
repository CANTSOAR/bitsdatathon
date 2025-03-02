import pymongo
import ollama
import numpy as np
from pymongo import UpdateOne

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main")
db = client["stocks_db"]
collection = db["stock_data"]

def get_embedding(text, model_name="mistral"):
    """Generates an embedding for the given text using Ollama."""
    try:
        response = ollama.embeddings(model=model_name, prompt=text)
        return np.array(response["embedding"]).tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

BATCH_SIZE = 10 # Adjust as needed
total_articles = collection.count_documents({})
processed_count = 0

while processed_count < total_articles:
    articles = list(collection.find().skip(processed_count).limit(BATCH_SIZE))
    print(articles)
    
    if not articles:
        break  # No more articles

    bulk_operations = []
    for article in articles:
        text = article["body"]
        embedding = get_embedding(text)

        if embedding:
            bulk_operations.append(UpdateOne(
                {"_id": article["_id"]},
                {"$set": {"embedding": embedding}}
            ))
        else:
            print(f"Embedding failed for article: {article['title']}")

    if bulk_operations:
        collection.bulk_write(bulk_operations)
        print(f"Batch of {len(bulk_operations)} embeddings added.")

    processed_count += BATCH_SIZE

print("Embeddings generated and stored for all articles.") 