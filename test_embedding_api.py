#!/usr/bin/env python3
"""
Simple script to test the embedding API integration

This script creates a library, adds a document with chunks,
indexes it, and performs a query - all using natural language.

Usage:
    export COHERE_API_KEY="your-key-here"
    python test_embedding_api.py
"""

import os
import sys
import requests
import time

API_BASE_URL = "http://localhost:8000"


def main():
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code != 200:
            print("❌ API is not healthy")
            sys.exit(1)
        print("✅ API is running")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API at", API_BASE_URL)
        print("   Make sure the API is running: uvicorn app.main:app --reload")
        sys.exit(1)

    # Check if COHERE_API_KEY is set
    if not os.getenv("COHERE_API_KEY"):
        print("❌ COHERE_API_KEY environment variable is not set")
        print("   Set it with: export COHERE_API_KEY='your-key-here'")
        sys.exit(1)
    print("✅ COHERE_API_KEY is set")

    print("\n" + "=" * 60)
    print("Testing Embedding API Integration")
    print("=" * 60 + "\n")

    # 1. Create a library (dimension auto-determined)
    print("1️⃣  Creating library...")
    library_response = requests.post(
        f"{API_BASE_URL}/libraries",
        json={
            "name": "Test Library - Machine Learning Papers",
            "description": "A collection of ML research papers",
            "tags": ["ml", "ai", "research"],
        },
    )

    if library_response.status_code != 201:
        print(f"❌ Failed to create library: {library_response.text}")
        sys.exit(1)

    library = library_response.json()
    library_id = library["id"]
    print(f"✅ Library created: {library['name']}")
    print(f"   ID: {library_id}")
    print(f"   Embedding dimension: {library['embedding_dimension']} (auto-determined)")

    # 2. Create a document
    print("\n2️⃣  Creating document...")
    document_response = requests.post(
        f"{API_BASE_URL}/libraries/{library_id}/documents",
        json={
            "title": "Attention Is All You Need",
            "source": "arxiv.org/abs/1706.03762",
            "author": "Vaswani et al.",
            "tags": ["transformers", "attention"],
            "language": "en",
        },
    )

    if document_response.status_code != 201:
        print(f"❌ Failed to create document: {document_response.text}")
        sys.exit(1)

    document = document_response.json()
    document_id = document["id"]
    print(f"✅ Document created: {document['title']}")
    print(f"   ID: {document_id}")

    # 3. Create chunks (embeddings generated automatically)
    print("\n3️⃣  Creating chunks with automatic embedding generation...")

    chunks_data = [
        {
            "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
            "source": "arxiv.org/abs/1706.03762",
            "page_number": 1,
            "tags": ["introduction"],
            "position": 0,
        },
        {
            "text": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
            "source": "arxiv.org/abs/1706.03762",
            "page_number": 1,
            "tags": ["architecture"],
            "position": 1,
        },
        {
            "text": "The Transformer allows for significantly more parallelization and can reach a new state of the art.",
            "source": "arxiv.org/abs/1706.03762",
            "page_number": 1,
            "tags": ["results"],
            "position": 2,
        },
    ]

    for i, chunk_data in enumerate(chunks_data, 1):
        chunk_response = requests.post(
            f"{API_BASE_URL}/libraries/{library_id}/documents/{document_id}/chunks",
            json=chunk_data,
        )

        if chunk_response.status_code != 201:
            print(f"❌ Failed to create chunk {i}: {chunk_response.text}")
            sys.exit(1)

        chunk = chunk_response.json()
        print(f"   ✅ Chunk {i} created (embedding: {len(chunk['embedding'])} dimensions)")

    # 4. Index the library
    print("\n4️⃣  Indexing library with HNSW algorithm...")
    index_response = requests.post(
        f"{API_BASE_URL}/libraries/{library_id}/index",
        json={"index_type": "hnsw"},
    )

    if index_response.status_code != 200:
        print(f"❌ Failed to index library: {index_response.text}")
        sys.exit(1)

    indexed_library = index_response.json()
    print(f"✅ Library indexed successfully")
    print(f"   Index type: {indexed_library['index_type']}")
    print(f"   Total chunks: {indexed_library['total_chunks']}")

    # 5. Query the library using natural language
    print("\n5️⃣  Querying library with natural language...")

    queries = [
        "What is the Transformer architecture?",
        "Tell me about attention mechanisms",
        "What are the benefits of the Transformer?",
    ]

    for query_text in queries:
        print(f"\n   Query: '{query_text}'")

        query_response = requests.post(
            f"{API_BASE_URL}/libraries/{library_id}/query",
            json={
                "query_text": query_text,
                "k": 3,
            },
        )

        if query_response.status_code != 200:
            print(f"   ❌ Query failed: {query_response.text}")
            continue

        query_result = query_response.json()
        print(f"   ✅ Found {query_result['total_results']} results in {query_result['query_time_ms']:.2f}ms")

        for j, result in enumerate(query_result['results'][:2], 1):
            print(f"      {j}. Score: {result['score']:.4f}")
            print(f"         Text: {result['text'][:80]}...")

    # 6. Cleanup
    print("\n6️⃣  Cleaning up...")
    delete_response = requests.delete(f"{API_BASE_URL}/libraries/{library_id}")

    if delete_response.status_code != 204:
        print(f"❌ Failed to delete library: {delete_response.text}")
    else:
        print("✅ Library deleted successfully")

    print("\n" + "=" * 60)
    print("✅ All tests passed! Embedding API is working correctly.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
