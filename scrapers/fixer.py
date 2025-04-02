from pymongo import MongoClient, ASCENDING
from pymongo.operations import IndexModel

# MongoDB URI - Replace with your connection string
uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri, tlsAllowInvalidCertificates=True)
db = client["stocks_db"]
collection = db["mi_data"]

# Function to fix documents (flatten embeddings)
def fix_documents():
    # Find all documents in the collection
    documents = collection.find({})
    
    for doc in documents:
        # Check if the document contains the 'embedding' field
        if "embedding" in doc:
            # Flatten the embedding if it is stored incorrectly as a sub-document
            if isinstance(doc["embedding"], list) and isinstance(doc["embedding"][0], dict):
                # Flattening the embedding (e.g. from BSON-style structure)
                doc["embedding"] = [value["$numberDouble"] for value in doc["embedding"]]
            
            # Ensure that embedding is an array of floats
            if isinstance(doc["embedding"], list):
                try:
                    doc["embedding"] = [float(x) for x in doc["embedding"]]
                    # Ensure correct vector size (you can add a check here if necessary)
                    if len(doc["embedding"]) != 768:  # Replace with your vector dimension
                        print(f"Warning: Embedding of incorrect size in doc: {doc['_id']}")
                    else:
                        # Update the document in MongoDB
                        collection.update_one({"_id": doc["_id"]}, {"$set": {"embedding": doc["embedding"]}})
                        print(f"Document {doc["_id"]} updated successfully.")
                except ValueError:
                    print(f"Skipping document {doc["_id"]} due to embedding format issue.")
        else:
            print(f"No embedding found in document {doc['_id']}. Skipping.")

# Run the fix function
# Create the vector index
index_definition = {
    'name': 'embedding_index',
    'type': 'vectorSearch',
    'definition': {
        'fields': [{
            'type': 'vector',
            'path': 'embedding',
            'numDimensions': 768,  # Adjust this based on your embedding size
            'similarity': 'cosine'
        }]
    }
}

# Create the index model
index_model = IndexModel([('embedding', 1)])  # Using 1 for ascending order, you can modify this as necessary

# Create index with IndexModel instance
collection.create_indexes([index_model])

print("Index created successfully.")