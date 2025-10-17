import requests
import cohere

co = cohere.Client("REDACTED_API_KEY_2")

library_id = "17815267-0b06-4188-a6bb-8bb820576411"

# Query text
query_text = "What is attention mechanism?"

# Generate query embedding
print(f"Query: {query_text}")
print("Generating embedding...")
response = co.embed(
    texts=[query_text],
    model="embed-english-light-v3.0",
    input_type="search_query"
)
query_embedding = response.embeddings[0]

# Query the library with metadata filters
print("\nSearching library with metadata filters...")
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

response = requests.post(
    f"http://localhost:8000/libraries/{library_id}/query",
    json=query_data
)

if response.status_code == 200:
    result = response.json()

    query_time_ms = result.get('query_time_ms', 0)
    total_results = result.get('total_results', 0)

    print(f"\nWith metadata filtering (page_number=1, tags contains 'introduction'):")
    print(f"Found {total_results} relevant chunks in {query_time_ms:.2f}ms:\n")

    if 'results' in result and result['results']:
        for i, res in enumerate(result['results'], 1):
            score = res.get('score', 0)
            text = res.get('text', 'N/A')
            metadata = res.get('metadata', {})

            # Create a brief preview
            preview = text[:80] + "..." if len(text) > 80 else text

            print(f"{i}. Score: {score:.4f}")
            print(f"   {preview}")
            print(f"   Page: {metadata.get('page_number')}, Tags: {metadata.get('tags')}")
            print()

        print("Metadata filtering allows you to combine semantic search with structured filters!")
    else:
        print("No results found matching the filters.")
else:
    print(f"\nError: {response.status_code}")
    print(f"Response: {response.text}")
