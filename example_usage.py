import json

import requests

BASE_URL = "http://localhost:8000"


def main():
    print("=" * 60)
    print("Vector Database API - Example Usage")
    print("=" * 60)

    print("\n1. Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    print("\n2. Create Library")
    library_data = {
        "name": "AI Research Papers",
        "description": "Collection of papers on artificial intelligence",
        "tags": ["ai", "research", "machine-learning"],
        "embedding_dimension": 3,
    }
    response = requests.post(f"{BASE_URL}/libraries", json=library_data)
    print(f"Status: {response.status_code}")
    library = response.json()
    library_id = library["id"]
    print(f"Created Library ID: {library_id}")

    print("\n3. Create Document")
    document_data = {
        "title": "Attention Is All You Need",
        "source": "arxiv_2017_transformers.pdf",
        "author": "Vaswani et al.",
        "tags": ["transformers", "attention"],
        "language": "en",
    }
    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/documents", json=document_data
    )
    print(f"Status: {response.status_code}")
    document = response.json()
    document_id = document["id"]
    print(f"Created Document ID: {document_id}")

    print("\n4. Create Chunks with Embeddings")
    chunks = [
        {
            "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
            "embedding": [0.8, 0.1, 0.1],
            "source": "arxiv_2017_transformers.pdf",
            "page_number": 1,
            "tags": ["introduction"],
            "position": 0,
        },
        {
            "text": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
            "embedding": [0.1, 0.9, 0.0],
            "source": "arxiv_2017_transformers.pdf",
            "page_number": 1,
            "tags": ["introduction", "architecture"],
            "position": 1,
        },
        {
            "text": "Experiments show that the Transformer can be trained significantly faster than other architectures.",
            "embedding": [0.2, 0.3, 0.5],
            "source": "arxiv_2017_transformers.pdf",
            "page_number": 8,
            "tags": ["results", "performance"],
            "position": 2,
        },
    ]

    chunk_ids = []
    for i, chunk_data in enumerate(chunks, 1):
        response = requests.post(
            f"{BASE_URL}/libraries/{library_id}/documents/{document_id}/chunks",
            json=chunk_data,
        )
        print(f"  Chunk {i} created: {response.status_code}")
        chunk_ids.append(response.json()["id"])

    print("\n5. Index Library (using HNSW)")
    index_data = {"index_type": "hnsw"}
    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/index", json=index_data
    )
    print(f"Status: {response.status_code}")
    print(f"Library indexed: {response.json()['is_indexed']}")

    print("\n6. Query Library")
    query_data = {"embedding": [0.1, 0.85, 0.05], "k": 2}
    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/query", json=query_data
    )
    print(f"Status: {response.status_code}")
    results = response.json()
    print(f"Query Time: {results['query_time_ms']:.2f}ms")
    print(f"Total Results: {results['total_results']}")
    print("\nTop Results:")
    for i, result in enumerate(results["results"], 1):
        print(f"\n  Result {i}:")
        print(f"    Score: {result['score']:.4f}")
        print(f"    Text: {result['text'][:80]}...")
        print(f"    Page: {result['metadata']['page_number']}")

    print("\n7. Query with Metadata Filter")
    query_with_filter = {
        "embedding": [0.1, 0.85, 0.05],
        "k": 10,
        "metadata_filters": [{"field": "page_number", "operator": "eq", "value": 1}],
    }
    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/query", json=query_with_filter
    )
    print(f"Status: {response.status_code}")
    results = response.json()
    print(f"Results with filter (page=1): {results['total_results']}")

    print("\n8. List All Libraries")
    response = requests.get(f"{BASE_URL}/libraries")
    print(f"Status: {response.status_code}")
    libraries = response.json()
    print(f"Total Libraries: {len(libraries)}")

    print("\n9. Get Library Details")
    response = requests.get(f"{BASE_URL}/libraries/{library_id}")
    print(f"Status: {response.status_code}")
    library = response.json()
    print(f"Library: {library['name']}")
    print(f"Documents: {library['total_documents']}")
    print(f"Chunks: {library['total_chunks']}")
    print(f"Index Type: {library['index_type']}")

    print("\n10. Update Library")
    update_data = {
        "name": "AI Research Papers (Updated)",
        "description": "Updated description",
    }
    response = requests.patch(f"{BASE_URL}/libraries/{library_id}", json=update_data)
    print(f"Status: {response.status_code}")
    print(f"Updated name: {response.json()['name']}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API.")
        print("Make sure the server is running: docker-compose up")
    except Exception as e:
        print(f"ERROR: {e}")
