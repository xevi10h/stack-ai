# Vector Database API

A high-performance REST API for indexing and querying documents using vector embeddings. Built with FastAPI, following SOLID principles and Domain-Driven Design.

## Features

- **CRUD Operations**: Full support for creating, reading, updating, and deleting libraries, documents, and chunks
- **Multiple Indexing Algorithms**:
  - Flat Index (brute-force, exact search)
  - HNSW (Hierarchical Navigable Small World, approximate nearest neighbor)
  - IVF (Inverted File Index, clustering-based)
- **Thread-Safe**: Concurrent read/write operations with Read-Write locks
- **Metadata Filtering**: Filter search results based on chunk metadata
- **RESTful Design**: Clean, intuitive API endpoints
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Comprehensive Testing**: Unit and integration tests included

## Architecture

The project follows a layered architecture based on Domain-Driven Design:

```
app/
├── domain/          # Domain models and business logic
├── application/     # Application services (use cases)
├── infrastructure/  # Infrastructure implementations
│   ├── indexing/    # Vector indexing algorithms
│   └── repositories/ # Data access layer
└── api/             # REST API layer (controllers)
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.13 (for local development)

### Running with Docker

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
uvicorn app.main:app --reload
```

3. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Usage

### 1. Create a Library

```bash
curl -X POST http://localhost:8000/libraries \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Research Papers",
    "description": "Collection of AI research papers",
    "tags": ["research", "ai"],
    "embedding_dimension": 384
  }'
```

### 2. Create a Document

```bash
curl -X POST http://localhost:8000/libraries/{library_id}/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Attention Is All You Need",
    "source": "arxiv.pdf",
    "author": "Vaswani et al.",
    "tags": ["transformers"],
    "language": "en"
  }'
```

### 3. Create Chunks with Embeddings

```bash
curl -X POST http://localhost:8000/libraries/{library_id}/documents/{document_id}/chunks \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The dominant sequence transduction models...",
    "embedding": [0.1, 0.2, 0.3, ...],
    "source": "arxiv.pdf",
    "page_number": 1,
    "tags": ["introduction"],
    "position": 0
  }'
```

### 4. Index the Library

```bash
curl -X POST http://localhost:8000/libraries/{library_id}/index \
  -H "Content-Type: application/json" \
  -d '{
    "index_type": "hnsw"
  }'
```

Available index types: `flat`, `hnsw`, `ivf`

### 5. Query the Library

```bash
curl -X POST http://localhost:8000/libraries/{library_id}/query \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.1, 0.2, 0.3, ...],
    "k": 10,
    "metadata_filters": [
      {
        "field": "page_number",
        "operator": "gte",
        "value": 1
      }
    ]
  }'
```

## Generating Embeddings

Use the provided Cohere API key to generate embeddings for testing:

```python
import cohere

co = cohere.Client(<cohere_api_key>)

response = co.embed(
    texts=["Your text here"],
    model="embed-english-light-v3.0"
)

embedding = response.embeddings[0]
```

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=app --cov-report=html
```

## Indexing Algorithms

### Flat Index
- **Time Complexity**: O(n * d) for search
- **Space Complexity**: O(n * d)
- **Best for**: Small datasets, exact search requirements
- **Pros**: Simple, exact results, no training required
- **Cons**: Slow for large datasets

### HNSW (Hierarchical Navigable Small World)
- **Time Complexity**: O(log(n) * d) for search (approximate)
- **Space Complexity**: O(n * m * log(n))
- **Best for**: Large datasets requiring fast approximate search
- **Pros**: Very fast queries, good recall
- **Cons**: Memory intensive, complex implementation

### IVF (Inverted File Index)
- **Time Complexity**: O((n/k) * d) for search
- **Space Complexity**: O(n * d + k * d)
- **Best for**: Medium to large datasets with clustering properties
- **Pros**: Balanced speed/accuracy tradeoff, memory efficient
- **Cons**: Requires training, may miss results in other clusters

## Project Structure

```
stack-ai/
├── app/
│   ├── domain/                 # Domain layer
│   │   ├── models.py          # Domain entities
│   │   └── exceptions.py      # Domain exceptions
│   ├── application/            # Application layer
│   │   └── services.py        # Business logic services
│   ├── infrastructure/         # Infrastructure layer
│   │   ├── indexing/          # Vector indexing
│   │   │   ├── base.py        # Index interface
│   │   │   ├── flat_index.py  # Flat index implementation
│   │   │   ├── hnsw_index.py  # HNSW implementation
│   │   │   ├── ivf_index.py   # IVF implementation
│   │   │   ├── factory.py     # Index factory
│   │   │   └── utils.py       # Shared utilities
│   │   ├── repositories/      # Data access
│   │   │   ├── base.py        # Repository interfaces
│   │   │   └── memory.py      # In-memory implementation
│   │   └── persistence/       # Persistence layer
│   │       └── json_persister.py
│   ├── api/                    # API layer
│   │   ├── routers/           # API endpoints
│   │   │   ├── libraries.py
│   │   │   ├── documents.py
│   │   │   └── chunks.py
│   │   ├── dto.py             # Data transfer objects
│   │   ├── dependencies.py    # Dependency injection
│   │   └── error_handlers.py  # Error handling
│   └── main.py                 # Application entry point
├── tests/                      # Test suite
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── pytest.ini                  # Pytest configuration
└── README.md                   # This file
```

## SOLID Principles Applied

1. **Single Responsibility Principle (SRP)**: Each class has one reason to change
   - Services handle business logic
   - Repositories handle data access
   - Indexes handle vector operations

2. **Open-Closed Principle (OCP)**: Open for extension, closed for modification
   - New index types can be added without modifying existing code
   - Factory pattern for index creation

3. **Liskov Substitution Principle (LSP)**: Subtypes must be substitutable
   - All indexes implement the same interface
   - Any index can be swapped without breaking functionality

4. **Interface Segregation Principle (ISP)**: Clients shouldn't depend on unused interfaces
   - Separate repositories for each entity type
   - Specific DTOs for each operation

5. **Dependency Inversion Principle (DIP)**: Depend on abstractions
   - Services depend on repository interfaces, not implementations
   - Dependency injection for loose coupling
