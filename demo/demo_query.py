import requests
import cohere

co = cohere.Client("rQsWxQJOK89Gp87QHo6qnGtPiWerGJOxvdg59o5f")

library_id = "17815267-0b06-4188-a6bb-8bb820576411"  # Replace

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

# Query the library
print("\nSearching library...")
query_data = {
    "embedding": query_embedding,
    "k": 3
}

response = requests.post(
    f"http://localhost:8000/libraries/{library_id}/query",
    json=query_data
)

if response.status_code == 200:
    result = response.json()

    query_time_ms = result.get('query_time_ms', 0)
    total_results = result.get('total_results', 0)

    print(f"\nExcellent! In just {query_time_ms:.2f} milliseconds, the HNSW index found {total_results} relevant chunks:\n")

    if 'results' in result and result['results']:
        for i, res in enumerate(result['results'], 1):
            score = res.get('score', 0)
            text = res.get('text', 'N/A')

            # Create a brief preview
            preview = text[:80] + "..." if len(text) > 80 else text

            print(f"{i}. Score: {score:.4f}")
            print(f"   {preview}")

            if i == 1:
                print(f"   → The top result talks about {text.lower()}")
            elif i == 2:
                print(f"   → Second result mentions {text[:50].lower()}...")
            print()

        print("These are semantically similar to our query even though they don't use the exact words.")
        print("That's the power of vector embeddings!")
    else:
        print("No results found.")
else:
    print(f"\nError: {response.status_code}")
    print(f"Response: {response.text}")