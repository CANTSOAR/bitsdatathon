from pymongo import MongoClient
from pymongo.errors import OperationFailure
import numpy as np
from sentence_transformers import SentenceTransformer

# Connect to MongoDB
client = MongoClient("mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
)  # Adjust connection URI

db = client["stocks_db"]  # Replace with your database name
collection = db["mi_data"]  # Replace with your collection name

def check_vector_index():
    try:
        indexes = list(collection.aggregate([{"$listSearchIndexes": {}}]))
        for index in indexes:
            if index.get("name") == "vector_index":  # Replace with your actual index name
                print("Vector index found:", index)
                return True
        print("No vector index found.")
        return False
    except OperationFailure as e:
        print("Error checking index:", e)
        return False

# Perform a vector search
def vector_search(query_vector, top_k=2):
    search_query = {
        "$vectorSearch": {
            "index": "vector_index",  # Use your actual index name
            "path": "embedding",  # Ensure this matches your field name
            "queryVector": query_vector,
            "numCandidates": 100,  # Adjust this based on performance needs
            "limit": top_k
        }
    }

    try:
        pipeline = [
            search_query,  # $search stage must be inside an aggregation pipeline
            {"$limit": top_k}  # Optional: Ensures we don't get excessive results
        ]

        results = list(collection.aggregate(pipeline))

        print("Search Results:")
        for result in results:
            print(result)
    except OperationFailure as e:
        print("Vector search error:", e)

def vectorize_search(search_term):
    model = SentenceTransformer("ProsusAI/finbert")
    vector = model.encode(search_term, convert_to_numpy=True, show_progress_bar=False)

    return vector.tolist()

if __name__ == "__main__":
    if check_vector_index():
        test_vector = np.random.rand(768).tolist()  # Adjust size to match your vector dimension
        vector_search(test_vector)

        test = input("gimme something")
        vector = vectorize_search(test)
        vector_search(vector)
