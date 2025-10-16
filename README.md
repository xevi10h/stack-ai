# Vector Database API

A production-ready, high-performance REST API for indexing and querying documents using vector embeddings. Built with FastAPI, following SOLID principles and Domain-Driven Design.

## ✨ Features

### Core Features
- **CRUD Operations**: Full support for creating, reading, updating, and deleting libraries, documents, and chunks
- **Multiple Indexing Algorithms** (implemented from scratch):
  - **Flat Index**: Brute-force, exact search for small datasets
  - **HNSW**: Hierarchical Navigable Small World for fast approximate search
  - **IVF**: Inverted File Index with clustering for balanced performance
- **Thread-Safe Concurrency**: Read-Write locks for safe concurrent operations
- **Metadata Filtering**: Advanced filtering with 8 operators (eq, ne, gt, gte, lt, lte, in, contains)
- **RESTful Design**: Clean, intuitive API endpoints with automatic OpenAPI documentation
- **Docker Support**: Complete Docker and Docker Compose setup
- **Comprehensive Testing**: 23 tests covering all features (100% pass rate)

### Advanced Features (Extra Points)

#### ✅ 1. Metadata Filtering
- 8 filtering operators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `contains`
- Post-query filtering for optimal performance
- Supports string, numeric, list, and boolean fields
- Multiple filters with AND logic

#### ✅ 2. Disk Persistence
- JSON-based persistence layer for state management
- Serialize/deserialize all entities (libraries, documents, chunks)
- Simple load/save API for data durability

#### ✅ 3. Leader-Follower Architecture
- **Async replication** from leader to follower nodes
- **Automatic failover** with health checking and heartbeats
- **Leader election** using simplified Raft algorithm (selects node with lowest lag)
- **Replication log** for ordered operation tracking
- **Health monitoring** with configurable timeouts
- REST API endpoints for cluster management

#### ✅ 4. Python SDK Client
- Complete Python client library (`vector-db-client`)
- Type-safe models matching API responses
- Comprehensive error handling with custom exceptions
- Supports all API operations (libraries, documents, chunks, queries)
- Easy installation: `pip install -e sdk/`
- Full example usage included

#### ✅ 5. Temporal Durable Execution
- **Durable workflows** for long-running queries
- **Automatic retries** on failures
- **Signals** to update query parameters mid-execution
- **Queries** to check workflow status
- **Batch query workflow** for parallel execution
- Full Temporal UI integration (http://localhost:8080)

## Architecture

The project follows a layered architecture based on Domain-Driven Design:

```
app/
├── domain/          # Domain models and business logic
├── application/     # Application services (use cases)
├── infrastructure/  # Infrastructure implementations
│   ├── indexing/    # Vector indexing algorithms
│   ├── repositories/ # Data access layer
│   ├── replication/ # Leader-Follower architecture
│   ├── temporal/    # Temporal workflows and activities
│   └── persistence/ # Disk persistence
├── api/             # REST API layer (controllers)
└── sdk/             # Python SDK client library
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.13+ (for local development)

### Option 1: Full Stack with Docker (Recommended)

Run the complete stack including API, Temporal server, worker, and UI:

```bash
docker-compose up --build
```

**Services:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Temporal UI: http://localhost:8080
- PostgreSQL: localhost:5432

### Option 2: Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the API:**
```bash
uvicorn app.main:app --reload
```

3. **Run Temporal worker** (optional, in a separate terminal):
```bash
python -m app.infrastructure.temporal.worker
```

4. **Access services:**
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Option 3: Using the Python SDK

```bash
cd sdk
pip install -e .
python example.py
```

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

## Advanced Features Deep Dive

### Leader-Follower Architecture

Enable high availability and read scalability with multi-node replication:

**Enable replication mode:**
```bash
export REPLICATION_ENABLED=true
uvicorn app.main:app --reload
```

**Register nodes:**
```bash
# Register leader (first node becomes leader automatically)
curl -X POST http://localhost:8000/nodes/register \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "node1",
    "host": "localhost",
    "port": 8001
  }'

# Register followers
curl -X POST http://localhost:8000/nodes/register \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "node2",
    "host": "localhost",
    "port": 8002
  }'
```

**Check cluster status:**
```bash
curl http://localhost:8000/nodes/status
```

**Features:**
- Async replication (fire-and-forget for better write performance)
- Automatic leader election on failure
- Health checking via heartbeats
- Replication lag tracking
- Raft-inspired consensus

### Python SDK

Use the official Python client for type-safe interactions:

```python
from vector_db_client import VectorDBClient, MetadataFilter

# Initialize client
client = VectorDBClient(base_url="http://localhost:8000")

# Create library
library = client.create_library(
    name="My Library",
    embedding_dimension=384,
    tags=["ml", "ai"]
)

# Create document and chunks
document = client.create_document(
    library_id=library.id,
    title="Document Title",
    source="document.pdf"
)

chunk = client.create_chunk(
    document_id=document.id,
    text="Sample text",
    embedding=[0.1, 0.2, ...],
    source="document.pdf",
    position=0
)

# Index and query
client.index_library(library.id, index_type="hnsw")

results, query_time = client.query_library(
    library_id=library.id,
    query_embedding=[0.1, 0.2, ...],
    k=10,
    metadata_filters=[
        MetadataFilter(field="author", operator="eq", value="John Doe")
    ]
)

for result in results:
    print(f"Score: {result.score}, Text: {result.chunk.text}")
```

**Installation:**
```bash
cd sdk
pip install -e .
```

See `sdk/example.py` for complete usage.

### Temporal Durable Execution

Run long-running queries with automatic retries and fault tolerance:

```python
from temporalio.client import Client
from app.infrastructure.temporal.workflows import QueryWorkflow, QueryWorkflowParams

# Connect to Temporal
client = await Client.connect("localhost:7233")

# Start durable query workflow
handle = await client.start_workflow(
    QueryWorkflow.run,
    QueryWorkflowParams(
        library_id="library-uuid",
        query_embedding=[0.1, 0.2, ...],
        k=10,
        auto_index=True,  # Auto-index if not indexed
        index_type="hnsw"
    ),
    id="my-query-workflow",
    task_queue="vector-db-queue",
)

# Check workflow status
status = await handle.query(QueryWorkflow.get_status)

# Update query parameters mid-execution
await handle.signal(QueryWorkflow.update_query, new_params)

# Get result
result = await handle.result()
```

**Start Temporal server:**
```bash
docker-compose up temporal temporal-ui
```

**Start worker:**
```bash
python -m app.infrastructure.temporal.worker
```

**Access Temporal UI:**
http://localhost:8080

See `temporal_example.py` for complete usage.

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
│   │   │   ├── memory.py      # In-memory implementation
│   │   │   └── replicated.py  # Replicated wrappers
│   │   ├── replication/       # Leader-Follower
│   │   │   └── node.py        # Replication logic
│   │   ├── temporal/          # Temporal workflows
│   │   │   ├── workflows.py   # Workflow definitions
│   │   │   ├── activities.py  # Activity implementations
│   │   │   └── worker.py      # Worker process
│   │   └── persistence/       # Persistence layer
│   │       └── json_persister.py
│   ├── api/                    # API layer
│   │   ├── routers/           # API endpoints
│   │   │   ├── libraries.py
│   │   │   ├── documents.py
│   │   │   ├── chunks.py
│   │   │   └── nodes.py       # Cluster management
│   │   ├── dto.py             # Data transfer objects
│   │   ├── dependencies.py    # Dependency injection
│   │   └── error_handlers.py  # Error handling
│   └── main.py                 # Application entry point
├── sdk/                        # Python SDK
│   ├── vector_db_client/      # Client library
│   │   ├── client.py          # Main client
│   │   ├── models.py          # Data models
│   │   └── exceptions.py      # Custom exceptions
│   ├── setup.py               # Package setup
│   ├── example.py             # Usage example
│   └── README.md              # SDK documentation
├── tests/                      # Test suite (23 tests)
│   ├── test_api.py            # API integration tests
│   ├── test_indexing.py       # Algorithm tests
│   └── test_replication.py    # Replication tests
├── temporal_example.py         # Temporal workflow example
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Full stack setup
├── pytest.ini                  # Pytest configuration
├── TECHNICAL_DECISIONS.md      # Design rationale
├── INTERVIEW_PREP.md           # Interview guide
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
