"""
Example usage of the Vector Database Python SDK

This script demonstrates how to use the SDK to interact with the Vector Database API.
"""

import random

from vector_db_client import MetadataFilter, VectorDBClient


def generate_random_embedding(dimension: int) -> list[float]:
    """Generate a random embedding vector for demo purposes"""
    return [random.random() for _ in range(dimension)]


def main():
    # Initialize client
    print("Initializing Vector Database client...")
    client = VectorDBClient(base_url="http://localhost:8000")

    # Check API health
    health = client.health_check()
    print(f"API Status: {health['status']}")
    print(f"Available indices: {health['available_indices']}")
    print()

    # Create a library
    print("Creating library...")
    library = client.create_library(
        name="Research Papers",
        description="Collection of ML research papers",
        tags=["machine-learning", "research"],
        embedding_dimension=128,  # Using 128 dimensions for this example
    )
    print(f"✓ Created library: {library.metadata.name} (ID: {library.id})")
    print()

    # Create a document
    print("Creating document...")
    document = client.create_document(
        library_id=library.id,
        title="Attention Is All You Need",
        source="arxiv.org/abs/1706.03762",
        author="Vaswani et al.",
        tags=["transformers", "attention"],
        language="en",
    )
    print(f"✓ Created document: {document.metadata.title} (ID: {document.id})")
    print()

    # Create chunks with embeddings
    print("Creating chunks...")
    chunks = []

    chunk_texts = [
        "The Transformer architecture revolutionized natural language processing.",
        "Self-attention mechanisms allow the model to weigh the importance of different words.",
        "Multi-head attention provides the model with multiple representation subspaces.",
        "Positional encoding is used to inject information about token positions.",
        "The Transformer achieved state-of-the-art results on machine translation tasks.",
    ]

    for i, text in enumerate(chunk_texts):
        chunk = client.create_chunk(
            document_id=document.id,
            text=text,
            embedding=generate_random_embedding(128),
            source="arxiv.org/abs/1706.03762",
            position=i,
            page_number=i + 1,
            author="Vaswani et al.",
            tags=["transformers"],
        )
        chunks.append(chunk)
        print(f"  ✓ Created chunk {i + 1}: {text[:50]}...")

    print()

    # Index the library with HNSW
    print("Indexing library with HNSW...")
    indexed_library = client.index_library(library.id, index_type="hnsw")
    print(f"✓ Library indexed: {indexed_library.index_type}")
    print()

    # Query the library
    print("Querying library...")
    query_embedding = generate_random_embedding(128)
    results, query_time = client.query_library(
        library_id=library.id, query_embedding=query_embedding, k=3
    )

    print(f"✓ Query completed in {query_time:.2f}ms")
    print(f"✓ Found {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Score: {result.score:.4f}")
        print(f"  Text: {result.chunk.text}")
        print(f"  Source: {result.chunk.metadata.source}")
        print(f"  Page: {result.chunk.metadata.page_number}")
        print()

    # Query with metadata filtering
    print("Querying with metadata filters...")
    filters = [MetadataFilter(field="page_number", operator="gte", value=3)]

    filtered_results, query_time = client.query_library(
        library_id=library.id,
        query_embedding=query_embedding,
        k=5,
        metadata_filters=filters,
    )

    print(f"✓ Filtered query completed in {query_time:.2f}ms")
    print(f"✓ Found {len(filtered_results)} results (page_number >= 3)")
    print()

    # List all libraries
    print("Listing all libraries...")
    all_libraries = client.list_libraries()
    print(f"✓ Total libraries: {len(all_libraries)}")
    for lib in all_libraries:
        print(
            f"  - {lib.metadata.name}: {lib.metadata.total_documents} docs, {lib.metadata.total_chunks} chunks"
        )
    print()

    # Update library metadata
    print("Updating library metadata...")
    updated_library = client.update_library(
        library_id=library.id,
        description="Updated: Collection of seminal ML research papers",
        tags=["machine-learning", "research", "nlp"],
    )
    print(f"✓ Updated library description: {updated_library.metadata.description}")
    print()

    # Cleanup
    print("Cleaning up...")
    client.delete_library(library.id)
    print("✓ Deleted library")
    print()

    print("Example completed successfully! ✓")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        raise
