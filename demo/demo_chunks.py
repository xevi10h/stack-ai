import requests
import cohere
from uuid import UUID

# Cohere client
co = cohere.Client("cohere_api_key")

library_id = "library_id"  # Replace
document_id = "document_id"  # Replace

# Sample texts from the paper
texts = [
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    "Attention mechanisms have become an integral part of compelling sequence modeling.",
    "We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on attention.",
    "The Transformer allows for significantly more parallelization and can reach a new state of the art.",
    "Self-attention relates different positions of a single sequence to compute a representation."
]

# Generate embeddings
print("Generating embeddings with Cohere...")
response = co.embed(
    texts=texts,
    model="embed-english-light-v3.0",
    input_type="search_document"
)
embeddings = response.embeddings

# Create chunks
print(f"Creating {len(texts)} chunks...")
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    chunk_data = {
        "text": text,
        "source": "arxiv.pdf",
        "page_number": i + 1,  # Each chunk on a different page (1, 2, 3, 4, 5)
        "author": "Vaswani et al.",
        "tags": ["introduction"],
        "position": i
    }

    response = requests.post(
        f"http://localhost:8000/libraries/{library_id}/documents/{document_id}/chunks",
        json=chunk_data
    )

    if response.status_code == 201:
        print(f"✓ Chunk {i+1} created: {text[:50]}...")
    else:
        print(f"✗ Error: {response.text}")

print("\nAll chunks created successfully!")