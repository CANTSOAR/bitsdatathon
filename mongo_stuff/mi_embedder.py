import pymongo
import ollama
import numpy as np
from pymongo import UpdateOne
import ssl

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main", tlsAllowInvalidCertificates=True)
db = client["stocks_db"]
collection = db["mi_data"]

def get_embedding(text, model_name="nomic-embed-text"):
    """Generates an embedding for the given text using Ollama."""
    try:
        # Generate embeddings
        response = ollama.embeddings(model=model_name, prompt=text)
        
        # The response format may vary, so let's check what we got
        if "embedding" in response:
            return response["embedding"]
        else:
            # For debugging
            print(f"Unexpected response format: {type(response)}")
            print(f"Response content: {response}")
            return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


BATCH_SIZE = 100
total_articles = collection.count_documents({})
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
            
        embedding = get_embedding(text)

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