import requests
import cohere

# Cohere client
co = cohere.Client("REDACTED_API_KEY_2")

# API endpoint
API_BASE = "http://localhost:8000"

print("Setting up Temporal demo...")
print()

# Create library
print("1. Creating library...")
library_response = requests.post(
    f"{API_BASE}/libraries",
    json={
        "name": "Temporal Demo Library",
        "description": "Library for Temporal workflow demonstrations",
        "tags": ["demo", "temporal"],
        "embedding_dimension": 1024,  # embed-english-v3.0 produces 1024 dimensions
    }
)

if library_response.status_code != 201:
    print(f"Error creating library: {library_response.text}")
    exit(1)

library = library_response.json()
library_id = library["id"]
print(f"✓ Created library: {library_id}")
print()

# Create document
print("2. Creating document...")
doc_response = requests.post(
    f"{API_BASE}/libraries/{library_id}/documents",
    json={
        "title": "Attention Is All You Need",
        "source": "arxiv.org/abs/1706.03762",
        "author": "Vaswani et al.",
        "tags": ["transformers", "nlp"],
    }
)

if doc_response.status_code != 201:
    print(f"Error creating document: {doc_response.text}")
    exit(1)

document = doc_response.json()
document_id = document["id"]
print(f"✓ Created document: {document_id}")
print()

# Sample texts from the paper
texts = [
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    "Attention mechanisms have become an integral part of compelling sequence modeling.",
    "We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on attention.",
    "The Transformer allows for significantly more parallelization and can reach a new state of the art.",
    "Self-attention relates different positions of a single sequence to compute a representation.",
]

# Generate embeddings
print("3. Generating embeddings with Cohere...")
response = co.embed(
    texts=texts,
    model="embed-english-v3.0",
    input_type="search_document"
)
embeddings = response.embeddings
print(f"✓ Generated {len(embeddings)} embeddings")
print()

# Create chunks
print("4. Creating chunks...")
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    chunk_data = {
        "text": text,
        "embedding": embedding,
        "source": "arxiv.pdf",
        "page_number": 1,
        "author": "Vaswani et al.",
        "tags": ["introduction"],
        "position": i
    }

    response = requests.post(
        f"{API_BASE}/libraries/{library_id}/documents/{document_id}/chunks",
        json=chunk_data
    )

    if response.status_code == 201:
        print(f"  ✓ Chunk {i+1}: {text[:60]}...")
    else:
        print(f"  ✗ Error: {response.text}")

print()

# Index the library
print("5. Indexing library with HNSW...")
index_response = requests.post(
    f"{API_BASE}/libraries/{library_id}/index",
    json={"index_type": "hnsw"}
)

if index_response.status_code != 200:
    print(f"Error indexing library: {index_response.text}")
    exit(1)

print("✓ Library indexed")
print()

print("=" * 60)
print("Setup complete!")
print()
print(f"Library ID: {library_id}")
print()
print("Update temporal_example.py with this library_id:")
print(f'  library_id="{library_id}",')
print()
print("Then run:")
print("  ./run_worker.sh")
print("  python3 temporal_example.py")
print("=" * 60)
