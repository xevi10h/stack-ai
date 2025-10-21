import json
import cohere

co = cohere.Client("cohere_api_key")

library_id = "library_id"  # Replace

# Query text
query_text = "What is attention mechanism?"

# Generate query embedding
print("Generating embedding for query...")
response = co.embed(
    texts=[query_text],
    model="embed-english-light-v3.0",
    input_type="search_query"
)
query_embedding = response.embeddings[0]

# Create the query data
query_data = {
    "embedding": query_embedding,
    "k": 10,
    "metadata_filters": [
        {
            "field": "page_number",
            "operator": "eq",
            "value": 1
        },
        {
            "field": "tags",
            "operator": "contains",
            "value": "introduction"
        }
    ]
}

# Generate curl command
curl_command = f"""curl -X POST http://localhost:8000/libraries/{library_id}/query \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(query_data)}'"""

print("\nHere's your curl command with a real embedding:\n")
print(curl_command)
print("\n\nOr use this formatted JSON:\n")
print(json.dumps(query_data, indent=2))
