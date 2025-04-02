import pymongo
import numpy as np
from pymongo import UpdateOne
from sentence_transformers import SentenceTransformer
from datetime import datetime
import time  # Import time for potential delays if needed

# --- Configuration ---
MONGO_URI = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
DB_NAME = "stocks_db"
COLLECTION_NAME = "mi_data"  # Use the collection specified in the first script
EMBEDDING_MODEL_NAME = "ProsusAI/finbert" # Use the model specified in the plan [cite: 11]
BATCH_SIZE = 50 # Adjusted batch size, modify as needed based on performance/memory
# --- End Configuration ---

print("--- Starting Vectorization Script ---")

# Connect to MongoDB
try:
    print(f"🔌 Connecting to MongoDB Atlas...")
    client = pymongo.MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True) # Use tlsAllowInvalidCertificates=True cautiously
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    # Test connection
    client.admin.command('ping') 
    print(f"✅ Successfully connected to DB: '{DB_NAME}', Collection: '{COLLECTION_NAME}'")
except pymongo.errors.ConnectionFailure as e:
    print(f"❌ MongoDB Connection Error: {e}")
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred during MongoDB connection: {e}")
    exit()

# Load Embedding Model
try:
    print(f"🧠 Loading embedding model: '{EMBEDDING_MODEL_NAME}'...")
    # Consider adding device='cuda' or device='mps' if GPU is available:
    # embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda') 
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ Embedding model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading Sentence Transformer model: {e}")
    exit()

def embed_text(text):
    """Converts text into an embedding vector using the loaded model."""
    if not text or not isinstance(text, str):
        print("⚠️ Skipping embedding generation for empty or invalid text.")
        return None
    try:
        # The model expects a string or list of strings
        embedding = embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        # Ensure the output is a list of floats for MongoDB compatibility
        return embedding.tolist() 
    except Exception as e:
        print(f"❌ Error during text embedding: {e}")
        return None

def convert_date_to_timestamp(date_value):
    """Converts various date string formats to UNIX timestamp (integer)."""
    if date_value is None:
        return None
    if isinstance(date_value, (int, float)):
         # Assuming it might already be a timestamp (float or int)
        return int(date_value) 
    if not isinstance(date_value, str):
        print(f"⚠️ Unexpected date type: {type(date_value)}, value: {date_value}. Skipping conversion.")
        return None

    # Add more formats here if needed
    possible_formats = [
        "%Y-%m-%dT%H:%M:%SZ",        # ISO format e.g., 2024-01-15T10:30:00Z
        "%b. %d, %Y, %I:%M %p",      # Example: "Mar. 1, 2025, 05:14 PM"
        "%Y-%m-%d %H:%M:%S",        # Example: "2024-01-15 10:30:00"
        "%a, %d %b %Y %H:%M:%S %Z",  # Example: "Mon, 15 Jan 2024 10:30:00 GMT"
        "%m/%d/%Y"                  # Example: "01/15/2024" 
    ]

    for fmt in possible_formats:
        try:
            # Use timestamp() for UTC timestamp
            return int(datetime.strptime(date_value, fmt).timestamp()) 
        except ValueError:
            continue # Try the next format

    print(f"⚠️ Date conversion failed for format: '{date_value}'. Tried formats: {possible_formats}")
    return None

# --- Main Processing Loop ---
try:
    total_articles = collection.count_documents({}) # Count all documents initially
    # Alternatively, count only documents *without* an embedding if reprocessing:
    # query_filter = {"embedding": {"$exists": False}}
    # total_articles = collection.count_documents(query_filter) 
    processed_count = 0
    print(f"🚀 Found {total_articles} articles to process in '{COLLECTION_NAME}'. Starting in batches of {BATCH_SIZE}...")

    # Use a query filter if you only want to process documents missing the embedding
    # find_filter = {"embedding": {"$exists": False}} 
    find_filter = {} # Process all documents found (as in original script)

    while True: # Loop until no more articles are found in a batch
        articles_cursor = collection.find(
            find_filter, 
            {"_id": 1, "title": 1, "body": 1, "time": 1} # Project only necessary fields
        ).skip(processed_count).limit(BATCH_SIZE)
        
        articles = list(articles_cursor) # Fetch the batch
        
        if not articles:
            print("✅ No more articles found to process.")
            break # Exit the loop if no articles are returned

        batch_start_time = time.time()
        bulk_operations = []
        processed_in_batch = 0

        for article in articles:
            article_id = article["_id"]
            title = article.get('title', 'No title provided')
            body = article.get("body")
            original_time = article.get("time")

            print(f"  🔍 Processing Article ID: {article_id} | Title: {title[:50]}...") # Log article being processed

            # 1. Generate Embedding
            embedding = embed_text(body)
            if embedding is None:
                print(f"  ⚠️ Embedding failed for Article ID: {article_id}. Skipping update for this article.")
                # Optionally, mark the article as failed instead of skipping?
                # update_data = {"embedding_failed": True, "last_attempt": datetime.utcnow()}
                # bulk_operations.append(UpdateOne({"_id": article_id}, {"$set": update_data}))
                continue # Skip this article if embedding failed

            # 2. Convert Timestamp
            timestamp = convert_date_to_timestamp(original_time)
            
            # 3. Prepare Update Operation
            update_data = {"embedding": embedding}
            if timestamp is not None:
                # Only update time if conversion was successful and it's different
                if timestamp != original_time: 
                    update_data["time_unix_ts"] = timestamp # Store as a new field to preserve original
                    # Or replace original: update_data["time"] = timestamp 
                else:
                     print(f"  ℹ️ Timestamp for Article ID: {article_id} is already in correct format or conversion yielded same value.")
            else:
                 print(f"  ⚠️ Timestamp conversion failed for Article ID: {article_id}. Original value: {original_time}")


            bulk_operations.append(UpdateOne(
                {"_id": article_id},
                {"$set": update_data},
                upsert=False # Do not insert if document is somehow missing
            ))
            processed_in_batch += 1

        # Perform Bulk Write if there are operations
        if bulk_operations:
            try:
                result = collection.bulk_write(bulk_operations, ordered=False) # ordered=False can be faster
                batch_end_time = time.time()
                print(f"  ✅ Batch Write Result: {result.bulk_api_result}")
                print(f"  ⏱️ Batch processing time: {batch_end_time - batch_start_time:.2f} seconds.")
                print(f"  ✍️ Processed and prepared updates for {processed_in_batch} articles in this batch.")
            except pymongo.errors.BulkWriteError as bwe:
                print(f"  ❌ Bulk Write Error: {bwe.details}")
                # Potentially add retry logic here or log failures
            except Exception as e:
                 print(f"  ❌ Unexpected error during bulk write: {e}")

        # Update processed count based on articles *retrieved* in the batch,
        # not necessarily successfully updated, to ensure pagination works correctly.
        processed_count += len(articles) 
        print(f"📊 Progress: Approximately {processed_count}/{total_articles} articles checked.")
        
        # Optional: Add a small delay between batches if needed
        # time.sleep(1) 

except KeyboardInterrupt:
     print("\n🛑 Process interrupted by user.")
except Exception as e:
    print(f"❌ An unexpected error occurred during the main processing loop: {e}")
finally:
    if 'client' in locals() and client:
        client.close()
        print("🚪 MongoDB connection closed.")

print("🎉 Script finished processing articles.")