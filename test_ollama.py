import ollama

try:
    response = ollama.embeddings(
        model="llama3.2",
        prompt="This is a test sentence"
    )

    if "embedding" in response:
        print("Success! Embedding length:", len(response["embedding"]))
        print("First few values:", response["embedding"][:5])
    else:
        print("No embedding found in response")
        print("Response:", response) #printing the entire response is better for debugging.
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")